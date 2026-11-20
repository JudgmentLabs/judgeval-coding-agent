"""Secret redaction in the sender, the second of two layers, and the rules table both layers share.

The hook runs the first layer in ``core/bin/prepare.jq`` before an event reaches the inbox. This
module runs the same rules, read from ``core/bin/redact-rules.json``, over each turn after transcript
text has been pulled in. ``sanitize_event`` is the jq layer step for step, and the tests check that
the two give the same result. ``compiled`` derives what the hook's jq loads (the table with every
regex composed); scripts/gen_redact.py writes it out.

Everything is plain data through pure stages:

    table          redact-rules.json, as written
    compiled       every regex composed: what jq loads, and what this module compiles
    Ruleset        the built-in rules a project kept (redactDisable), its own rules, its names and allowed literals
                   (a project setting that fails: error)
    env_secrets    (name, literal) for each secret-named environment value, and each line of it
    findings       Finding(rule, start, end, kept): each rule's matches as spans of the unchanged blob, one
                   finditer per window (linear: no match is searched for twice)
    layers         the strings and keys of JSON nested deeper in a string than the rules read, one list per depth
    pairs          findings -> Pair(found, replacement, rule), one per distinct found text, longest first; a
                   nested layer's encoded as deep as it sits
    env_found      {name, field}: the env values each field's raw text held, whatever rule replaced them
    scrub_fields   the secret_field members (exact "key":"value" tokens), then the pairs, applied to each field:
                   in its JSON when every pair can only sit inside a string there, else to each string value,
                   so no number or JSON syntax is ever touched; a Counter of what was replaced, by rule. A found
                   text still there afterwards fails closed
    records        [{rule, field, count}]: the audit, carried on the event as ``redactions``
    turn_redaction the turn's audit record

Each event ends in one state (``outcome``): clean, redacted, or error. The hook allows fewer distinct
finds than the sender (``hook_budget``, ``sender_budget``): jq matches a long window once per distinct find.

A rule set with an error (a user regex that does not compile, a bad name, an unknown rule id) redacts every text to
nothing. No function records a secret, or a hash, prefix, or length of one.
"""

from __future__ import annotations

import functools
import heapq
import json
import os
import re
import warnings
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, NamedTuple

BIN = Path(__file__).resolve().parent.parent / "bin"
RULES_FILE = BIN / "redact-rules.json"
COMPILED_FILE = BIN / "redact-compiled.json"
_NAMED = re.compile(r"\(\?<([A-Za-z_][A-Za-z0-9_]*)>")
_BACKREF = re.compile(r"\\k<([A-Za-z_][A-Za-z0-9_]*)>")
# A name a project adds with redactNames: lowercase-able parts joined by _. Anything else fails the rule set closed.
_NAME = re.compile(r"[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*")


# ---------------------------------------------------------------- the table, composed


@functools.lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


def _cases(part: str) -> list[str]:
    """A word as lower, UPPER, and Capitalized, without repeats."""
    part = part.lower()
    return list(dict.fromkeys([part, part.upper(), part.capitalize()]))


def names_re(names: Iterable[str]) -> str:
    """Secret names as one alternation: ``api_key`` -> api_key, API-KEY, apiKey, ApiKey, apikey...: each part in three
    cases, joined by _, -, ., or nothing. prepare.jq's jp_word spells a project's redactNames the same way."""
    return "|".join("[_.-]?".join("(?:" + "|".join(_cases(part)) + ")" for part in name.split("_")) for name in names)


def name_needs(names: Iterable[str], suffix: str = "") -> list[str]:
    """The literals a name rule needs: the last part of each name, in its three cases, then ``suffix``."""
    return list(dict.fromkeys(case + suffix for name in names for case in _cases(name.split("_")[-1])))


def name_rules(names: Iterable[str]) -> list[dict]:
    """secret_assignment and secret_field for these names, as compiled rules (needs only for the built-in names)."""
    table = rules()["secretNames"]
    names = list(names)
    builtin = names == table["names"]
    out = []
    for key in ("assignment", "field", "structured"):
        item = table[key]
        before, after = item["re"].split("{names}")
        rule = {"id": item["id"], "re": before + "(?:" + names_re(names) + ")" + after, "placeholder": placeholder(item["id"])}
        if builtin:
            rule["needs"] = name_needs(names, item.get("needs_suffix", ""))
        out.append(rule)
    return out


def compiled(table: Mapping | None = None) -> dict:
    """What the hook's jq loads: the table with every built-in regex composed and the comments left out, so the hook
    builds no regex for the built-in rules. The name rules also arrive as the text around {names} (``names``), so the
    hook can join a project's redactNames in. ``ids``: every built-in rule id, and every retired one, the ones
    redactDisable may name."""
    table = table or rules()
    names = table["secretNames"]

    def rule(item: Mapping, regex: str) -> dict:
        out = {"id": item["id"], "re": regex, "placeholder": placeholder(item["id"])}
        for key in ("needs", "pre", "whole", "most"):
            if item.get(key) is not None:
                out[key] = item[key]
        return out

    assignment, field, structured = name_rules(names["names"])
    patterns = [rule(item, item["re"]) for item in table["patterns"]]
    ids = [item["id"] for item in patterns] + [assignment["id"], field["id"]] + table["retiredIds"]["ids"]
    return {
        "version": table["version"],
        "limits": {key: value for key, value in table["limits"].items() if key != "_comment"},
        "env": {"name_re": table["env"]["name_re"], "min_length": table["env"]["min_length"]},
        "strict": [key for key, kind in table["settings"].items() if kind == "rule"],
        "patterns": patterns,
        "names": {
            "builtin": names_re(names["names"]),
            "list": names["names"],
            "assignment": {**assignment, "parts": names["assignment"]["re"].split("{names}")},
            "field": {**field, "parts": names["field"]["re"].split("{names}")},
            "structured": {**structured, "parts": names["structured"]["re"].split("{names}")},
        },
        "allow": names["allow"],
        "ids": {rule_id: True for rule_id in ids},
        "fields": _field_sets(table["fields"]),
    }


def _field_sets(fields: Mapping) -> dict:
    """The class table as the sets jq tests: strip (kept without text), plain (never redacted), bare (both: what an event
    whose rules failed keeps). A field not listed is text: in none of them."""
    of = {name: fields["classes"][cls] for name, cls in fields["of"].items()}
    return {
        "strip": {name: True for name, cls in of.items() if cls["strip"]},
        "plain": {name: True for name, cls in of.items() if not cls["redact"]},
        "bare": {name: True for name, cls in of.items() if cls["strip"] and not cls["redact"]},
    }


# ---------------------------------------------------------------- compiled rules


def _compile(regex: str) -> re.Pattern:
    """A rule's regex as Python runs it. Oniguruma names a group ``(?<p>...)`` and refers back to it as ``\\k<p>``; Python
    spells them ``(?P<p>...)`` and ``(?P=p)``. What Python only warns about (a POSIX class such as [[:alpha:]], which
    Oniguruma reads as one) is an error: the layers would disagree."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        try:
            return re.compile(_BACKREF.sub(r"(?P=\1)", _NAMED.sub(r"(?P<\1>", regex)))
        except Warning as exc:
            raise re.error(str(exc)) from exc


class RedactionLimit(ValueError):
    """The text needs more redaction work than the limits allow. Its event loses its text."""


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    needs: tuple = ()  # the rule can match only text holding one of these; empty means any text
    whole: bool = False  # runs over all the text, not windows of it
    most: int | None = None  # a whole rule fails on text holding more than this many of its needs
    pre: re.Pattern | None = None  # a cheap regex every match also matches

    def applies(self, text: str) -> bool:
        """Mirror of jp_holds and pre in prepare.jq: skip the regex when no literal it needs, or no match of pre, is there."""
        return (not self.needs or any(need in text for need in self.needs)) and (self.pre is None or self.pre.search(text) is not None)


def _rule(item: Mapping) -> Rule:
    pre = _compile(item["pre"]) if item.get("pre") else None
    return Rule(item["id"], _compile(item["re"]), tuple(item.get("needs") or ()), bool(item.get("whole")), item.get("most"), pre)


@functools.lru_cache(maxsize=1)
def _built() -> dict:
    done = compiled(rules())
    return {
        "patterns": tuple(_rule(item) for item in done["patterns"]),
        "names": tuple(_rule(done["names"][name]) for name in ("assignment", "field", "structured")),
        "env": (_compile(done["env"]["name_re"]), int(done["env"]["min_length"])),
        "ids": frozenset(done["ids"]),
        "allow": frozenset(done["allow"]),
    }


# Field classes. A field not in the table is text.
def field_class(name: str) -> Mapping:
    table = rules()["fields"]
    return table["classes"][table["of"].get(name, "text")]


def strip(event: Mapping, keep: Callable[[Mapping], bool] = lambda cls: cls["strip"], names: bool = True) -> dict:
    """Only the fields whose class passes ``keep``. tool_calls keep their id, and their name when ``names``:
    a name is free-form text, so an event whose rules failed keeps only the id."""
    kept = {key: value for key, value in event.items() if keep(field_class(key))}
    calls = event.get("tool_calls")
    if isinstance(calls, list):
        kept["tool_calls"] = [
            {"id": _alt(call.get("id"), call.get("tool_call_id")), **({"name": _alt(call.get("name"), call.get("tool_name"))} if names else {})}
            for call in calls
            if isinstance(call, dict)
        ]
    return kept


def _alt(value: Any, other: Any) -> Any:
    """jq's ``value // other``: only null and false fall through."""
    return other if value is None or value is False else value


def strip_text(event: Mapping) -> dict:
    """What an event keeps when it loses its text: structure and labels, which are redacted anyway."""
    return strip(event)


def _unredactable(event: Mapping) -> dict:
    """What an event keeps when its rules failed: only fields that are never redacted (env_found among them). No labels,
    no tool names."""
    return {**strip(event, lambda cls: cls["strip"] and not cls["redact"], names=False), "redaction_error": True}


@dataclass(frozen=True)
class Ruleset:
    patterns: tuple  # Rule, ... in table order, then secret_assignment, then user patterns
    members: tuple = ()
    error: str = ""  # a project setting that failed: every text is dropped, never sent raw
    allow: frozenset = frozenset()  # secretNames.allow and redactAllow: a find whose redacted text is one of these is left alone
    structured: tuple = ()


def settings() -> dict[str, str]:
    """The redaction settings of tracing.json, each "rule" or "trim" (``settings`` in redact-rules.json). A rule setting
    that is not a list of strings fails the rule set closed: the project asked for a rule."""
    return {key: kind for key, kind in rules()["settings"].items() if key != "_comment"}


def _setting(project: Mapping, key: str) -> list[str]:
    """A project's list setting: its nonempty strings. Anything else in a trim (redactAllow, redactDisable) is skipped."""
    value = project.get(key)
    return [item for item in value if isinstance(item, str) and item] if isinstance(value, list) else []


def _malformed(value: Any) -> bool:
    return value is not None and not (isinstance(value, list) and all(isinstance(item, str) for item in value))


@functools.lru_cache(maxsize=16)
def _named(names: tuple) -> tuple:
    return tuple(_rule(item) for item in name_rules(list(names)))


def ruleset(project: Mapping | None = None) -> Ruleset:
    """The built-in rules a project keeps, plus its own (tracing.json):

        redactPatterns  regexes whose matches are redacted (custom)
        redactNames     names whose values are redacted, as secretNames; !name drops a built-in name
        redactAllow     literal texts never redacted, on top of secretNames.allow
        redactDisable   built-in rule ids turned off

    A regex that does not compile, a name that is not parts joined by _, a !name no built-in has, an id no built-in rule
    has, or a rule setting that is not a list of strings fails the rule set closed (``error``), as an invalid
    redactPatterns regex always has. Other keys, such as the redactPaths and redactCommands older versions read, are
    ignored."""
    built = _built()
    project = project or {}
    off = set(_setting(project, "redactDisable"))
    builtin = rules()["secretNames"]["names"]
    entries = _setting(project, "redactNames")
    dropped = {name[1:] for name in entries if name.startswith("!")}
    names = tuple(name for name in builtin if name not in dropped) + tuple(name for name in entries if not name.startswith("!"))

    def kept(items: Iterable[Rule]) -> tuple:
        return tuple(item for item in items if item.id not in off)

    def failed(error: str) -> Ruleset:
        return Ruleset(built["patterns"], built["names"][1:2], error=error)

    bad = [key for key, kind in settings().items() if kind == "rule" and _malformed(project.get(key))]
    if bad:
        return failed(bad[0])
    if off - built["ids"]:
        return failed("redactDisable")
    if not all(_NAME.fullmatch(name) for name in names) or dropped - set(builtin):
        return failed("redactNames")
    try:
        assignment, field, structured = built["names"] if names == tuple(builtin) else _named(names)
        patterns = [Rule("custom", _compile(regex)) for regex in _setting(project, "redactPatterns")]
    except (re.error, RecursionError, OverflowError) as exc:
        return failed(type(exc).__name__)
    return Ruleset(kept(built["patterns"] + (assignment,)) + tuple(patterns), kept((field,)),
                   allow=built["allow"] | frozenset(_setting(project, "redactAllow")), structured=kept((structured,)))


def env_secrets(env: Mapping[str, str] | None = None, api_key: str = "") -> list[tuple[str, str]]:
    """Mirror of prepare.jq: (name, literal) for each secret-named variable's value, and each line of a value that
    spans lines, of at least min_length characters and not only whitespace. Longest first, then by placeholder.

    ``api_key`` is the key from the credentials file, which the hook also hands to jq as JUDGMENT_API_KEY.
    """
    source = dict(os.environ if env is None else env)
    if api_key and not source.get("JUDGMENT_API_KEY"):
        source["JUDGMENT_API_KEY"] = api_key
    name_re, shortest = _built()["env"]
    found = []
    for name, value in source.items():
        match = name_re.match(name) if isinstance(value, str) and "\n" not in name else None
        if match is None or match.end() != len(name):
            continue
        pieces = [value] + ([line.removesuffix("\r") for line in value.split("\n")] if "\n" in value else [])
        found += [(name, piece) for piece in pieces if len(piece) >= shortest and piece.strip()]
    return sorted(set(found), key=lambda pair: (-len(pair[1]), placeholder("env:" + pair[0]), pair[1]))


def placeholder(rule: str) -> str:
    return f"[REDACTED:{rule}]"


# ---------------------------------------------------------------- finding


class Pair(NamedTuple):
    found: str  # the text found
    replacement: str  # what it becomes: the kept group, then the placeholder
    rule: str  # the rule id the audit counts it under


def blob_items(value: Any) -> Iterator[str]:
    """Mirror of jq's ``.. | strings, (objects | keys_unsorted[])``: every string and key, in that order."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        yield from value.keys()
        for item in value.values():
            yield from blob_items(item)
    elif isinstance(value, list):
        for item in value:
            yield from blob_items(item)


def windows(text: str) -> list[tuple[int, str, bool]]:
    """Mirror of jp_found's windows: (start, text, last) for each window a rule runs over. See ``limits`` in redact-rules.json."""
    limits = rules()["limits"]
    size, step = limits["window"], limits["step"]
    if len(text) <= size:
        return [(0, text, True)]
    count = -(-(len(text) - size) // step) + 1
    return [(k * step, text[k * step : k * step + size], k == count - 1) for k in range(count)]


class Finding(NamedTuple):
    """One match of a rule on the unchanged text: where it is, and how much of it (the p group) is kept."""

    rule: str
    start: int
    end: int
    kept: int  # the length of the kept prefix (the p group), 0 when the rule keeps none


def findings(rule: Rule, text: str, budget: int) -> list[Finding]:
    """Every complete match of the rule on the unchanged text, as spans: one finditer per window, no rescans.

    jq matches a long window one distinct match at a time (jp_scan), since its global match counts offsets from the start
    of the string; the distinct texts it finds are the ones a global match finds, so this finds the same pairs in linear time.
    A match that reaches the end of a window that is not the last is left to the next window, which holds it from its start;
    one that starts before the next window does is longer than window - step and fails closed. More than ``budget``
    distinct texts fail closed too.
    """
    limits = rules()["limits"]
    # As jq: the first group's text is kept when that group is named p (jq counts unnamed groups too) and starts the match.
    keep = rule.pattern.groupindex.get("p") == 1
    if rule.most is not None and rule.needs and text.count(rule.needs[0]) > rule.most:
        raise RedactionLimit(rule.id)
    parts = [(0, text, True)] if rule.whole else windows(text)
    out: dict[tuple[int, int], Finding] = {}
    distinct: set[str] = set()
    for start, window, last in parts:
        for match in rule.pattern.finditer(window):
            if not match.group(0):
                continue
            if not last and match.end() == len(window):
                if match.start() < limits["step"]:
                    raise RedactionLimit(rule.id)
                continue
            kept = len(match.group("p") or "") if keep and match.start("p") == match.start() else 0
            out.setdefault((start + match.start(), start + match.end()), Finding(rule.id, start + match.start(), start + match.end(), kept))
            distinct.add(match.group(0))
            if len(distinct) > budget:
                raise RedactionLimit(rule.id)
    return sorted(out.values(), key=lambda item: (item.start, item.end))


def found(rule: Rule, text: str, budget: int | None = None) -> list[Pair]:
    """Mirror of jp_found: each distinct complete match of the rule on the unchanged text, with what it becomes."""
    budget = hook_budget() if budget is None else budget
    pairs_ = {Pair(text[item.start : item.end], text[item.start : item.start + item.kept] + placeholder(rule.id), rule.id) for item in findings(rule, text, budget)}
    return sorted(pairs_)


def hook_budget() -> int:
    """How many distinct found texts one event may hold in the hook (jq matches a long window once per distinct text)."""
    return int(rules()["limits"]["pairs"])


def sender_budget() -> int:
    """The sender's: its matching is linear, so a turn dense with genuine secrets is redacted, not dropped."""
    return int(rules()["limits"]["sender_pairs"])


def _order(pairs: Iterable[Pair]) -> list[Pair]:
    """One pair per found text, the first in rule order, then longest first: jq's unique_by(.[0]) | sort_by(-length)."""
    first: dict[str, Pair] = {}
    for pair in pairs:
        first.setdefault(pair.found, pair)
    return sorted(first.values(), key=lambda pair: (-len(pair.found), pair.found))


def _allowed(pairs: Iterable[Pair], allow: frozenset) -> list[Pair]:
    """Mirror of jp_allow: the pairs whose redacted text (the found text after its kept prefix) redactAllow does not name."""
    return [pair for pair in pairs if pair.found[len(pair.replacement) - len(placeholder(pair.rule)):] not in allow]


def _layer_pairs(blob: str, rs: Ruleset, secrets: list[tuple[str, str]], budget: int) -> list[Pair]:
    """The env values one layer's blob holds, then each rule whose needs it holds, over the unchanged blob."""
    out = [Pair(value, placeholder("env:" + name), "env:" + name) for name, value in secrets if value in blob]
    for rule in rs.patterns:
        if rule.applies(blob):
            out += found(rule, blob, budget)
    return _allowed(out, rs.allow)


def escaped(text: str, depth: int) -> str:
    """Mirror of jp_esc: the text as it reads ``depth`` JSON string encodings down (jq's tojson without its quotes)."""
    for _ in range(depth):
        text = jq_json(text)[1:-1]
    return text


# A string holds JSON nested deeper than the text rules read (they read a quote, and one escaped as \") when it holds an
# escaped backslash before a quote.
NESTED = '\\"'


def _containers(text: str) -> Iterator[Any]:
    """Mirror of jp_containers: each JSON object or array the text holds whole, or from the first { or [ of the text or of
    one of its lines (a log line's payload), decoded."""
    for piece in [text] + (text.split("\n") if "\n" in text else []):
        for opener in ("{", "["):
            at = piece.find(opener)
            if at < 0:
                continue
            try:
                value = json.loads(piece[at:])
            except (ValueError, RecursionError):
                continue
            if isinstance(value, (dict, list)):
                yield value


def layers(items: list[str]) -> list[list[str]]:
    """Mirror of jp_layers: for each depth below the text, every string and key of the JSON nested in the strings above it
    that hold an escaped backslash before a quote. Deeper than ``limits.depth`` fails closed: such nesting is not read."""
    depth = rules()["limits"]["depth"]
    out: list[list[str]] = []
    current = items
    while True:
        nxt = [item for text in current if NESTED in text for value in _containers(text) for item in blob_items(value)]
        if not nxt:
            return out
        if len(out) == depth:
            raise RedactionLimit("depth")
        out.append(nxt)
        current = nxt


def pairs(items: list[str], rs: Ruleset, secrets: list[tuple[str, str]], budget: int | None = None) -> list[Pair]:
    """Mirror of prepare.jq: the pairs of the blob of ``items`` (every string and key), then those of each layer of JSON
    nested in its strings, each found text there encoded as deep as its layer, so it is replaced where it sits. One pair per
    found text, longest first. A nested find whose encoding is not in the blob (its JSON was written another way) fails
    closed: it could not be replaced."""
    budget = max(sender_budget(), hook_budget() if budget is None else budget)
    blob = "\n".join(items)
    out = _layer_pairs(blob, rs, secrets, budget)
    for depth, layer in enumerate(layers(items), 1):
        for pair in _layer_pairs("\n".join(layer), rs, secrets, budget):
            deep = Pair(escaped(pair.found, depth), escaped(pair.replacement, depth), pair.rule)
            if deep.found not in blob:
                raise RedactionLimit("nested")
            out.append(deep)
    return merged_pairs(blob, _order(out))


class Span(NamedTuple):
    start: int
    end: int
    left: int
    right: int
    rule: str


def literal_spans(text: str, pair: Pair) -> Iterator[Span]:
    """Literal offsets on unchanged text; overlapping copies cannot identify the original regex spans."""
    pieces = pair.replacement.split(placeholder(pair.rule))
    if not pair.found or len(pieces) != 2:
        raise RedactionLimit("span")
    prefix, suffix = pieces
    if not pair.found.startswith(prefix) or not pair.found.endswith(suffix) or len(prefix) + len(suffix) > len(pair.found):
        raise RedactionLimit("span")
    at = 0
    for part in text.split(pair.found)[:-1]:
        start = at + len(part)
        at = start + len(pair.found)
        if pair.found in text[start + 1:at + len(pair.found) - 1]:
            raise RedactionLimit("span")
        yield Span(start, at, start + len(prefix), at - len(suffix), pair.rule)


def merged_spans(text: str, pairs: list[Pair]) -> Iterator[Span]:
    """Merge ordered literal streams: work is linear in text size for the bounded number of pairs."""
    prev = None
    for span in heapq.merge(*(literal_spans(text, pair) for pair in pairs), key=lambda s: (s.start, -s.end)):
        if prev is not None and span.start < prev.end:
            prev = Span(prev.start, max(prev.end, span.end), min(prev.left, span.left), max(prev.right, span.right), prev.rule)
        else:
            if prev is not None:
                yield prev
            prev = span
    if prev is not None:
        yield prev


def merged_pairs(text: str, pairs: list[Pair]) -> list[Pair]:
    out = {}
    for s in merged_spans(text, pairs):
        pair = Pair(text[s.start:s.end], text[s.start:s.left] + placeholder(s.rule) + text[s.right:s.end], s.rule)
        if pair.found in out and out[pair.found] != pair:
            raise RedactionLimit("span")
        out[pair.found] = pair
    return _order(out.values())


def apply_counted(text: str, pairs: list[Pair], counts: Counter, numbered: bool = False) -> str:
    """Replace unions on unchanged text, keeping the outermost or earliest find's label."""
    parts = []
    at = 0
    for span in merged_spans(text, pairs):
        counts[span.rule] += 1
        counts[""] += 1
        number = f"#{counts['']}" if numbered else ""
        parts.append(text[at:span.left] + placeholder(span.rule) + number + text[span.right:span.end])
        at = span.end
    parts.append(text[at:])
    return "".join(parts)


# ---------------------------------------------------------------- scrubbing


def jq_json(value: Any) -> str:
    """Compact JSON as jq's tojson writes it: raw non-ASCII, and DEL escaped."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace("\x7f", "\\u007f")


def member_pairs(fields: Mapping[str, Any], rs: Ruleset, budget: int | None = None) -> list[Pair]:
    """Mirror of prepare.jq: each member rule over the compact JSON of the fields' values (a field's name is never a
    key), as text: "key":"value" -> "key":"[REDACTED:secret_field]". One pair per found text, longest first."""
    text = jq_json(list(fields.values()))
    return _order(_allowed((pair for rule in rs.members if rule.applies(text) for pair in found(rule, text, budget)), rs.allow))


_OUTSIDE_STRING = re.compile(r"[-+.0-9eEtrufalsn{}\[\],:]*")


def in_text(json_pairs: list[Pair]) -> bool:
    """Mirror of prepare.jq: whether every pair, as JSON escapes it, can sit nowhere in compact JSON but inside a string.
    It holds no quote, and a character no number, true, false, null, or punctuation holds."""
    return not any('"' in pair.found or _OUTSIDE_STRING.fullmatch(pair.found) for pair in json_pairs)


def _objects(value: Any) -> Iterator[dict]:
    """Mirror of jq's ``.. | objects``, in the same order."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)


def _size(value: Any) -> int:
    """Mirror of jq's [.. | objects | length] | add: how many members the value's objects hold in all."""
    return sum(len(obj) for obj in _objects(value))


def _strings(value: Any, pairs: list[Pair], counts: Counter) -> Any:
    """Mirror of prepare.jq's string pass: each string value gets the pairs; keys, numbers, and the rest stay."""
    if isinstance(value, str):
        return apply_counted(value, pairs, counts)
    if isinstance(value, list):
        return [_strings(item, pairs, counts) for item in value]
    if isinstance(value, dict):
        return {key: _strings(item, pairs, counts) for key, item in value.items()}
    return value


def _loads(text: str) -> Any:
    """As jq's fromjson inside its try: JSON a replacement broke fails closed, like any limit."""
    try:
        return json.loads(text)
    except ValueError as exc:
        raise RedactionLimit("json") from exc


def _nodes(value: Any) -> Iterator[Any]:
    yield value
    if isinstance(value, dict):
        for item in value.values():
            yield from _nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _nodes(item)


def _structured_fields(fields: Mapping[str, Any], rs: Ruleset, secrets: list[tuple[str, str]], budget: int) -> tuple[dict, list[dict]]:
    used = {item[0] for item in _nodes(fields) if isinstance(item, list) and len(item) == 1 and type(item[0]) in (int, float)}
    marker = next((number for number in range(sender_budget() + 1) if number not in used), None)
    if marker is None:
        raise RedactionLimit("members")
    total = 0

    def mask(value: Any, counts: Counter) -> Any:
        nonlocal total
        if isinstance(value, dict):
            out = {}
            for key, item in value.items():
                probe = "{" + jq_json(key) + ':"x"}'
                if type(item) in (int, float, list, dict) and any(rule.applies(probe) and rule.pattern.search(probe) for rule in rs.members):
                    total += 1
                    if total > budget:
                        raise RedactionLimit("pairs")
                    out[key] = [marker]
                    counts["secret_field"] += 1
                else:
                    out[key] = mask(item, counts)
            return out
        if isinstance(value, list):
            return [mask(item, counts) for item in value]
        return value

    def restore(value: Any) -> Any:
        if isinstance(value, list):
            if len(value) == 1 and type(value[0]) in (int, float) and value[0] == marker:
                return placeholder("secret_field")
            return [restore(item) for item in value]
        if isinstance(value, dict):
            return {key: restore(item) for key, item in value.items()}
        return value

    masked, counts = {}, Counter()
    for field, value in fields.items():
        field_counts = Counter()
        masked[field] = mask(value, field_counts)
        counts.update({(field, rule): count for rule, count in field_counts.items()})
    out, records = _scrub_fields(masked, rs, secrets, budget - total)
    counts.update({(item["field"], item["rule"]): item["count"] for item in records})
    return restore(out), [{"rule": rule, "field": field, "count": count} for (field, rule), count in sorted(counts.items())]


def scrub_fields(fields: Mapping[str, Any], rs: Ruleset, secrets: list[tuple[str, str]], budget: int | None = None) -> tuple[dict, list[dict]]:
    if rs.error:
        raise RedactionLimit("rules")
    budget = hook_budget() if budget is None else budget
    if rs.structured:
        canonical = jq_json(list(fields.values()))
        if any(rule.applies(canonical) and rule.pattern.search(canonical) for rule in rs.structured):
            return _structured_fields(fields, rs, secrets, budget)
    return _scrub_fields(fields, rs, secrets, budget)


def _scrub_fields(fields: Mapping[str, Any], rs: Ruleset, secrets: list[tuple[str, str]], budget: int | None = None) -> tuple[dict, list[dict]]:
    """Mirror of prepare.jq: the fields scrubbed together, and the records of what was replaced in each.

    One blob of every string and key gives the pairs, each a complete match on the unchanged blob. The compact JSON of
    the fields gives the secret-key members as exact "key":"value" tokens. Then each field: its members, then the pairs,
    replaced in its JSON when every pair can only sit inside a string there (``in_text``), so no number or syntax is
    touched; otherwise the pairs go to each string value. Two keys that would become one are numbered instead. A found
    text still there afterwards (it spanned two strings of the blob, or sits in a key the string pass does not reach)
    fails closed. A field's name is never a member or a key.
    """
    if rs.error:
        raise RedactionLimit("rules")
    budget = hook_budget() if budget is None else budget
    found_pairs = pairs(list(blob_items(dict(fields))), rs, secrets, budget)
    members = [Pair(pair.found + '"', pair.replacement + '"', pair.rule) for pair in member_pairs(fields, rs, max(budget, sender_budget()))]
    if not found_pairs and not members:
        return dict(fields), []
    json_pairs = [Pair(jq_json(pair.found)[1:-1], jq_json(pair.replacement)[1:-1], pair.rule) for pair in found_pairs]
    canonical = jq_json(list(fields.values()))
    if any(pair.found not in canonical for pair in members + json_pairs):
        raise RedactionLimit("span")
    if len(merged_pairs(canonical, members + json_pairs)) > budget:
        raise RedactionLimit("pairs")
    text = in_text(json_pairs)
    out: dict[str, Any] = {}
    records: list[dict] = []
    for field, value in fields.items():
        counts: Counter = Counter()
        # Member pairs were found in JSON text, so they apply to it as they are.
        if text:
            before = Counter(counts)
            scrubbed = _loads(apply_counted(jq_json(value), members + json_pairs, counts))
            if _size(scrubbed) != _size(value):
                # Two keys of one object became one placeholder: number every replacement in this field instead.
                counts = before
                scrubbed = _loads(apply_counted(jq_json(value), members + json_pairs, counts, numbered=True))
                if _size(scrubbed) != _size(value):
                    raise RedactionLimit("keys")  # a key already named like a numbered placeholder: fail closed
        else:
            after_members = apply_counted(jq_json(value), members, counts)
            scrubbed = _strings(_loads(after_members), found_pairs, counts)
        out[field] = scrubbed
        records += [{"rule": rule, "field": field, "count": n} for rule, n in counts.items() if n and rule]
    after = "\n".join(blob_items(out))
    if any(pair.found in after for pair in found_pairs):
        raise RedactionLimit("span")
    return out, sort_records(records)


def sort_records(records: Iterable[Mapping]) -> list[dict]:
    """Records summed by (rule, field), sorted by field then rule, zero counts left out."""
    totals: Counter = Counter()
    for item in records:
        if not isinstance(item, Mapping):
            continue
        try:
            count = int(item.get("count") or 0)
        except (TypeError, ValueError):
            continue
        totals[(str(item.get("field") or ""), str(item.get("rule") or ""))] += count
    return [{"rule": rule, "field": field, "count": n} for (field, rule), n in sorted(totals.items()) if n > 0]


def census(event: Mapping) -> list[dict]:
    """The event's audit records, as the layers recorded them while they replaced text. Nothing is read back from placeholders."""
    return sort_records(event.get("redactions") or [])


def _with_records(event: Mapping, records: list[dict]) -> dict:
    out = dict(event)
    merged = sort_records(list(event.get("redactions") or []) + records)
    if merged:
        out["redactions"] = merged
    else:
        out.pop("redactions", None)
    return out


def env_found(event: Mapping, secrets: list[tuple[str, str]]) -> list[dict]:
    """Mirror of jp_env_found: {name, field} for each secret env value a redacted field's text holds, whatever rule then
    replaced it (a member or an assignment can swallow the value whole) and whether or not the event's rules failed.
    The sender reads it to know which env values the hook's text depended on (flush.carry_records)."""
    out = set()
    for field, value in event.items():
        if not field_class(field)["redact"]:
            continue
        blob = "\n".join(blob_items(value))
        out |= {(field, name) for name, literal in secrets if literal in blob}
    return [{"name": name, "field": field} for field, name in sorted(out)]


def _with_env(event: Mapping, found: Iterable[Mapping]) -> dict:
    """The event with ``found`` joined to the env provenance it already carries."""
    merged = {(str(item.get("field")), str(item.get("name"))) for item in list(event.get("env_found") or []) + list(found) if isinstance(item, Mapping)}
    out = {key: value for key, value in event.items() if key != "env_found"}
    if merged:
        out["env_found"] = [{"name": name, "field": field} for field, name in sorted(merged)]
    return out


def scrub_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]], budget: int | None = None) -> dict:
    """Every field whose class is redacted, scrubbed together. New records join the event's ``redactions``."""
    fields = {key: value for key, value in event.items() if field_class(key)["redact"]}
    if not fields:
        return dict(event)
    scrubbed, records = scrub_fields(fields, rs, secrets, budget)
    return _with_records({**event, **scrubbed}, records)


def redact_text(text: str, rs: Ruleset, secrets: list[tuple[str, str]], field: str = "text") -> tuple[str, list[dict]]:
    """One text, such as a history summary, and the records of what was replaced in it as ``field``.
    Nothing when the rules failed or hit a limit."""
    try:
        scrubbed, records = scrub_fields({field: text}, rs, secrets, sender_budget())
    except RedactionLimit:
        return "", []
    return scrubbed[field], records


def redact_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]]) -> dict:
    """The sender's step: scrubbed, with what it replaced added to the records the event already carries, and the env
    values its text held added to its provenance. Its budget is the sender's: its matching is linear."""
    found = env_found(event, secrets)
    if rs.error:
        return _with_env(_unredactable(event), found)
    try:
        return _with_env(scrub_event(event, rs, secrets, sender_budget()), found)
    except RedactionLimit:
        return _with_env(_unredactable(event), found)


def redact_events(events: list[dict], rs: Ruleset, secrets: list[tuple[str, str]]) -> list[dict]:
    """``redact_event`` for each of a turn's events, with each input message its requests repeat scrubbed once.

    Every request replays the messages before it, so redacting each event alone costs the square of the turn's length.
    Here the redacted fields of every event, with each distinct input message once, are scrubbed together: a text found
    anywhere in the turn is replaced everywhere in it. Each event's records count what was replaced in its own fields,
    an input message as often as the event holds it. When the turn as a whole hits a limit, each event is redacted
    alone, so one event's limit costs only its own text, as before.
    """
    if rs.error or not any(isinstance(event.get("input_messages"), list) for event in events):
        return [redact_event(event, rs, secrets) for event in events]
    pool: dict[str, str] = {}
    fields: dict[str, Any] = {}
    for number, event in enumerate(events):
        for key, value in event.items():
            if not field_class(key)["redact"]:
                continue
            if key == "input_messages" and isinstance(value, list):
                for message in value:
                    pool.setdefault(jq_json(message), f"input_messages.{len(pool)}")
                    fields.setdefault(pool[jq_json(message)], message)
            else:
                fields[f"{number}.{key}"] = value
    try:
        scrubbed, records = scrub_fields(fields, rs, secrets, sender_budget())
    except RedactionLimit:
        return [redact_event(event, rs, secrets) for event in events]
    found: dict[str, list[dict]] = {}
    for item in records:
        found.setdefault(item["field"], []).append(item)
    out = []
    for number, event in enumerate(events):
        result = dict(event)
        own: list[dict] = []
        for key, value in event.items():
            if not field_class(key)["redact"]:
                continue
            if key == "input_messages" and isinstance(value, list):
                names = [pool[jq_json(message)] for message in value]
                result[key] = [scrubbed[name] for name in names]
                own += [{**item, "field": key} for name in names for item in found.get(name, [])]
            else:
                result[key] = scrubbed[f"{number}.{key}"]
                own += [{**item, "field": key} for item in found.get(f"{number}.{key}", [])]
        out.append(_with_env(_with_records(result, own), env_found(event, secrets)))
    return out


def metadata_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]]) -> dict:
    """What metadata capture sends: the fields its class keeps, with labels redacted."""
    kept = strip(event, lambda cls: cls["metadata"])
    kept["redactions"] = [item for item in census(event) if field_class(str(item["field"]))["metadata"]]
    if not rs.error:
        try:
            return scrub_event(kept, rs, secrets, sender_budget())
        except RedactionLimit:
            pass
    return {**strip(event, lambda cls: cls["metadata"] and not cls["redact"], names=False), "redaction_error": True}


def sanitize_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]], capture: str = "sanitized") -> dict:
    """Mirror of prepare.jq's judgment_prepare for one event: capture policy, scrub, records."""
    # The env values the raw text held, recorded whatever happens next: a rule that swallows one, or a failure that drops
    # its text (the sender must not restore that text from a transcript).
    raw = event = _with_env({key: value for key, value in event.items() if key != "env_found"}, env_found(event, secrets))
    if rs.error:
        return _unredactable(event)
    try:
        if capture == "metadata":
            event = {**strip(event), "prompt": "", "output": ""}
        fields = {key: value for key, value in event.items() if field_class(key)["redact"]}
        if not fields:
            return dict(event)
        scrubbed, records = scrub_fields(fields, rs, secrets)
        # As jq: what this pass replaced is the event's record. A pass that replaced nothing leaves it as it was.
        return {**event, **scrubbed, **({"redactions": records} if records else {})}
    except RedactionLimit:
        return _unredactable(raw)


def outcome(event: Mapping) -> str:
    """The state a redacted event is in, which the layers move it into and never out of:

        clean      no rule replaced anything
        redacted   records say what was replaced (``redactions``)
        error      a rule failed or a limit was hit: the text, labels, and tool names are gone (``redaction_error``)

    error wins over redacted. Impossible: text, a label, or a tool name in an error event. That raises ValueError.
    """
    if event.get("redaction_error"):
        # tool_calls survive as ids only: checked on their own.
        text = [key for key, value in event.items() if key != "tool_calls" and not field_class(key)["strip"] and value not in ("", None)]
        labels = [key for key in event if key != "tool_calls" and field_class(key)["redact"]]
        calls = event.get("tool_calls") or []
        if text or labels or not isinstance(calls, list) or any(not isinstance(call, Mapping) or set(call) - {"id"} for call in calls):
            raise ValueError(f"error event keeps {sorted(set(text + labels)) or 'tool calls'}")
        return "error"
    return "redacted" if event.get("redactions") else "clean"


def turn_redaction(events: list[dict], extra: Iterable[Mapping] = ()) -> dict:
    """The turn's audit record: every event's records, and ``extra`` (the history's), summed by rule and field."""
    items = sort_records([item for event in events for item in (event.get("redactions") or [])] + list(extra))
    record: dict[str, Any] = {"count": sum(item["count"] for item in items), "items": items}
    if any(event.get("redaction_error") for event in events):
        record["error"] = True
    return record
