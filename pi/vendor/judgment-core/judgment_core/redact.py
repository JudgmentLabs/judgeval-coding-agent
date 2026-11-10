"""Secret redaction in the sender, the second of two layers.

The hook runs the first layer in ``core/bin/prepare.jq`` before an event reaches the inbox.
This module runs the same rules, read from the same ``core/bin/redact-rules.json``, over each
turn after transcript text has been pulled in. ``sanitize_event`` is the jq layer step for
step, and tests/test_redact.py checks that the two give the same result.

An event is a plain dict, and each step is a function of plain data:

    field_class    what a field is: kept when text goes, redacted, kept by metadata capture
    env_secrets    (name, value) for each literal to replace: secret-named env values, and their lines
    sensitive      the sensitive path or env dump a tool call names, as {rule, trigger}, or None
    pairs          the blob of every string and key -> (found text, placeholder), in rule order
    scrub_value    those pairs applied, as literals, to every string and key; a secret-named
                   member's value goes whole
    census         {rule, field, count} for each placeholder an event holds: the audit
    turn_redaction the turn's audit record, and why it was blanked

A rule set with an error (a user regex that does not compile) redacts every text to nothing.
No function records a secret, or a hash, prefix, or length of one.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

RULES_FILE = Path(__file__).resolve().parent.parent / "bin" / "redact-rules.json"
SENSITIVE_REASON = "sensitive_path"
_NAMED = re.compile(r"\(\?<([A-Za-z_][A-Za-z0-9_]*)>")
_MARK = re.compile(r"\[REDACTED:([^\]\n]{1,200})\]")


@functools.lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


def compose(regex: Any) -> str:
    """A rule's regex. A list joins its parts; {"def": name} is a shared fragment from ``defs``."""
    if isinstance(regex, list):
        return "".join(rules()["defs"][part["def"]] if isinstance(part, dict) else part for part in regex)
    return regex


def _compile(regex: Any) -> re.Pattern:
    """Oniguruma names a group ``(?<p>...)``. Python spells it ``(?P<p>...)``."""
    return re.compile(_NAMED.sub(r"(?P<\1>", compose(regex)))


class RedactionLimit(ValueError):
    """The text needs more redaction work than the limits allow. Its event loses its text."""


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    needs: tuple = ()  # the rule can match only text holding one of these; empty means any text
    whole: bool = False  # runs over all the text, not windows of it
    most: int | None = None  # a whole rule fails on text holding more than this many of its needs

    def applies(self, text: str) -> bool:
        """Mirror of jp_holds in prepare.jq: skip the regex when no literal it needs is there."""
        return not self.needs or any(need in text for need in self.needs)


def _rule(item: Mapping, regex_key: str = "re") -> Rule:
    return Rule(item["id"], _compile(item[regex_key]), tuple(item.get("needs") or ()), bool(item.get("whole")), item.get("most"))


@functools.lru_cache(maxsize=1)
def _builtin() -> tuple[tuple, tuple, tuple]:
    table = rules()
    return (
        tuple(_rule(item) for item in table["patterns"]),
        tuple(_rule(item) for item in table["sensitive"]["paths"]),
        tuple(_rule(item) for item in table["sensitive"]["commands"]),
    )


@functools.lru_cache(maxsize=1)
def _member() -> tuple[Rule, re.Pattern, re.Pattern]:
    """The structured rule: {"api_key": "..."}. Its id and needs, its key regex, and its value regex."""
    item = rules()["fields_rule"]
    return Rule(item["id"], _compile(item["json_re"]), tuple(item["needs"])), _compile(item["key_re"]), _compile(item["value_re"])


# Field classes. A field not in the table is text.
def field_class(name: str) -> Mapping:
    table = rules()["fields"]
    return table["classes"][table["of"].get(name, "text")]


def strip(event: Mapping, keep: Callable[[Mapping], bool] = lambda cls: cls["strip"]) -> dict:
    """Only the fields whose class passes ``keep``. tool_calls keep their id and name."""
    kept = {key: value for key, value in event.items() if keep(field_class(key))}
    calls = event.get("tool_calls")
    if isinstance(calls, list):
        kept["tool_calls"] = [{"id": _alt(call.get("id"), call.get("tool_call_id")), "name": _alt(call.get("name"), call.get("tool_name"))} for call in calls if isinstance(call, dict)]
    return kept


def _alt(value: Any, other: Any) -> Any:
    """jq's ``value // other``: only null and false fall through."""
    return other if value is None or value is False else value


def strip_text(event: Mapping) -> dict:
    """What an event keeps when it loses its text: structure and labels, which are redacted anyway."""
    return strip(event)


def _unredactable(event: Mapping) -> dict:
    """What an event keeps when its rules failed: only fields that are never redacted."""
    return strip(event, lambda cls: cls["strip"] and not cls["redact"])


def unblank(event: Mapping) -> dict:
    """The event without free-form labels, for the spans of a blanked turn."""
    return {key: value for key, value in event.items() if not field_class(key).get("drop_when_blanked")}


@dataclass(frozen=True)
class Ruleset:
    patterns: tuple  # Rule, ... in table order, user patterns last
    paths: tuple
    commands: tuple
    error: str = ""  # a user rule that did not compile: every text is dropped, never sent raw


def glob_re(glob: str) -> str:
    """Mirror of prepare.jq: a redactPaths glob, one character at a time through the table's glob map."""
    table = rules()["sensitive"]["glob"]
    return "(?<![^\\n/])" + "".join(table.get(char, char) for char in glob) + "(?![^\\n])"


def ruleset(project: Mapping | None = None) -> Ruleset:
    """The built-in rules plus a project's ``redactPatterns`` and ``redactPaths``. User rules only add."""
    patterns, paths, commands = _builtin()
    project = project or {}
    extra_patterns = []
    extra_paths = []
    try:
        for regex in project.get("redactPatterns") or []:
            if isinstance(regex, str) and regex:
                extra_patterns.append(Rule("custom", _compile(regex)))
        for glob in project.get("redactPaths") or []:
            if isinstance(glob, str) and glob:
                extra_paths.append(Rule("custom_path", _compile(glob_re(glob))))
    except (re.error, RecursionError, OverflowError) as exc:
        return Ruleset(patterns, paths, commands, error=type(exc).__name__)
    return Ruleset(patterns + tuple(extra_patterns), paths + tuple(extra_paths), commands)


@functools.lru_cache(maxsize=1)
def _env_res() -> tuple[re.Pattern, re.Pattern]:
    table = rules()["env"]
    return _compile(table["name_re"]), _compile(table["value_re"])


def env_secrets(env: Mapping[str, str] | None = None, api_key: str = "") -> list[tuple[str, str]]:
    """Mirror of prepare.jq: (name, literal) for each secret-named variable's value, and each line of a
    value that spans lines, that is long enough to replace. Longest first, then by placeholder.

    ``api_key`` is the key from the credentials file, which the hook also hands to jq as JUDGMENT_API_KEY.
    """
    source = dict(os.environ if env is None else env)
    if api_key and not source.get("JUDGMENT_API_KEY"):
        source["JUDGMENT_API_KEY"] = api_key
    name_re, value_re = _env_res()
    found = []
    for name, value in source.items():
        match = name_re.match(name) if isinstance(value, str) and "\n" not in name else None
        if match is None or match.end() != len(name):
            continue
        pieces = [value] + ([line.removesuffix("\r") for line in value.split("\n")] if "\n" in value else [])
        found += [(name, piece) for piece in pieces if len(piece) >= 8 and value_re.search(piece)]
    return sorted(found, key=lambda pair: (-len(pair[1]), placeholder("env:" + pair[0])))


def placeholder(rule: str) -> str:
    return f"[REDACTED:{rule}]"


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


def windows(rule: Rule, text: str) -> list[str]:
    """Mirror of jp_windows: what a rule runs over. See ``limits`` in redact-rules.json."""
    limits = rules()["limits"]
    if rule.whole:
        if rule.most is not None and sum(text.count(need) for need in rule.needs) > rule.most:
            raise RedactionLimit(rule.id)
        return [text]
    if len(text) <= limits["window"]:
        return [text]
    return [text[start : start + limits["window"]] for start in range(0, len(text), limits["step"])]


def found(rule: Rule, text: str) -> list[tuple[str, str]]:
    """Mirror of jp_found: each distinct match, longest first, with the placeholder it becomes. Group p stays.

    Each window is matched after the finds before it are replaced in it, so a repeated secret is matched once.
    """
    keep = "p" in rule.pattern.groupindex
    hits: list[tuple[str, str]] = []
    for window in windows(rule, text):
        window = apply(window, hits)
        new = {(match.group(0), ((match.group("p") or "") if keep else "") + placeholder(rule.id)) for match in rule.pattern.finditer(window) if match.group(0)}
        hits = sorted(set(hits) | new, key=lambda pair: (-len(pair[0]), pair))
        if len(hits) > rules()["limits"]["pairs"]:
            raise RedactionLimit(rule.id)
    return hits


def apply(text: str, pairs: list[tuple[str, str]]) -> str:
    for old, new in pairs:
        text = new.join(text.split(old))
    return text


def pairs(blob: str, rs: Ruleset, secrets: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Mirror of prepare.jq: env values, then each rule whose needs the blob holds, each run over the
    blob as the steps before left it. Every (found text, placeholder), in order."""
    steps: list[Any] = [(value, placeholder("env:" + name)) for name, value in secrets if value in blob]
    steps += [rule for rule in rs.patterns if rule.applies(blob)]
    out: list[tuple[str, str]] = []
    for step in steps:
        hits = ([step] if step[0] in blob else []) if isinstance(step, tuple) else found(step, blob)
        if len(out) + len(hits) > rules()["limits"]["pairs"]:
            raise RedactionLimit("pairs")
        blob = apply(blob, hits)
        out += hits
    return out


def _members(value: Any, key_re: re.Pattern, value_re: re.Pattern, mark: str) -> Any:
    if isinstance(value, dict):
        return {
            key: mark if isinstance(item, str) and key_re.search(key) and value_re.search(item) else _members(item, key_re, value_re, mark)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_members(item, key_re, value_re, mark) for item in value]
    return value


def _member_count(value: Any, key_re: re.Pattern, value_re: re.Pattern) -> int:
    """How many distinct secret members (key, value) there are: jq finds each distinct "key":"value" once."""
    seen: set[tuple[str, str]] = set()

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for key, inner in item.items():
                if isinstance(inner, str) and key_re.search(key) and value_re.search(inner):
                    seen.add((key, inner))
                else:
                    visit(inner)
        elif isinstance(item, list):
            for inner in item:
                visit(inner)

    visit(value)
    return len(seen)


def _replace(value: Any, steps: list[tuple[str, str]]) -> Any:
    if isinstance(value, str):
        return apply(value, steps)
    if isinstance(value, dict):
        return {apply(key, steps): _replace(item, steps) for key, item in value.items()}
    if isinstance(value, list):
        return [_replace(item, steps) for item in value]
    return value


def scrub_value(value: Any, rs: Ruleset, secrets: list[tuple[str, str]]) -> Any:
    """Every string and key with its secrets replaced. A secret-named member's string value goes whole."""
    if rs.error:
        return "" if isinstance(value, str) else None
    blob = "\n".join(blob_items(value))
    member, key_re, value_re = _member()
    found_pairs = pairs(blob, rs, secrets)
    members = _member_count(value, key_re, value_re) if member.applies(blob) else 0
    if len(found_pairs) + members > rules()["limits"]["pairs"]:
        raise RedactionLimit("pairs")
    if members:
        value = _members(value, key_re, value_re, placeholder(member.id))
    return _replace(value, found_pairs) if found_pairs else value


def redact_text(text: str, rs: Ruleset, secrets: list[tuple[str, str]]) -> str:
    """One text, such as a history summary. Nothing when the rules failed or hit a limit."""
    try:
        return scrub_value(text, rs, secrets)
    except RedactionLimit:
        return ""


def scrub_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]]) -> dict:
    """Mirror of prepare.jq: every field whose class is redacted, scrubbed together."""
    fields = {key: value for key, value in event.items() if field_class(key)["redact"]}
    return {**event, **scrub_value(fields, rs, secrets)} if fields else dict(event)


def census(event: Mapping) -> list[dict]:
    """{rule, field, count} for each placeholder in the event's redacted fields, keys included, by field then rule."""
    audit = []
    for field, value in event.items():
        if not field_class(field)["redact"]:
            continue
        counts: dict[str, int] = {}
        for text in blob_items(value):
            for rule in _MARK.findall(text):
                counts[rule] = counts.get(rule, 0) + 1
        audit.extend({"rule": rule, "field": field, "count": counts[rule]} for rule in sorted(counts))
    return audit


def _with_audit(event: dict) -> dict:
    audit = census(event)
    if audit:
        event["redactions"] = audit
    else:
        event.pop("redactions", None)
    return event


def redact_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]]) -> dict:
    """The sender's step: scrubbed, with ``redactions`` counted from the placeholders it then holds,
    so text the hook already redacted is counted once."""
    if rs.error:
        return {**_unredactable(event), "redaction_error": True}
    try:
        return _with_audit(scrub_event(event, rs, secrets))
    except RedactionLimit:
        return {**_unredactable(event), "redaction_error": True}


def metadata_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]]) -> dict:
    """What metadata capture sends: the fields its class keeps, with labels redacted."""
    if not rs.error:
        try:
            return _with_audit(scrub_event(strip(event, lambda cls: cls["metadata"]), rs, secrets))
        except RedactionLimit:
            pass
    return {**strip(event, lambda cls: cls["metadata"] and not cls["redact"]), "redaction_error": True}


def _objects(value: Any) -> Iterator[dict]:
    """Mirror of jq's ``.. | objects``, in the same order."""
    if isinstance(value, dict):
        yield value
        for item in value.values():
            yield from _objects(item)
    elif isinstance(value, list):
        for item in value:
            yield from _objects(item)


def _strings(value: Any) -> Iterator[str]:
    """Mirror of ``strings, (arrays | .. | strings)``: a string, or every string in a list."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)


def _all_strings(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _all_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _all_strings(item)


def tool_texts(event: Mapping) -> list[str]:
    """Mirror of prepare.jq: the path and command values in the tool input, at any depth, by the table's keys."""
    keys = rules()["sensitive"]["keys"]
    raw = [event.get("tool_input")]
    calls = event.get("tool_calls")
    for call in calls if isinstance(calls, list) else calls.values() if isinstance(calls, dict) else []:
        if isinstance(call, dict):
            raw += [call.get("arguments"), call.get("input")]
    texts: list[str] = []
    for item in raw:
        if item is None:
            continue
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except ValueError:
                pass
        if isinstance(item, str):
            item = {"command": item}
        for obj in _objects(item):
            for key, value in obj.items():
                if keys.get(key):
                    texts.extend(_strings(value))
    return texts


def lines(text: str, at: list[str]) -> str:
    """Mirror of jp_lines: the text split at each of the literals, one piece per line."""
    for literal in at:
        text = "\n".join(text.split(literal))
    return text


def words(event: Mapping, texts: list[str]) -> str:
    """Mirror of prepare.jq: the words of the path and command values, then the prompt's @mentions, one per line."""
    table = rules()["sensitive"]
    found = lines("\n".join(texts), table["split"]) + "\n"
    prompt = event.get("prompt")
    if isinstance(prompt, str) and "@" in prompt:
        if prompt.count("@") > rules()["limits"]["mentions"]:
            raise RedactionLimit("mentions")
        mentions = [lines(word[1:], table["mention"]["end"]).split("\n")[0] for word in lines(prompt, table["mention"]["split"]).split("\n") if word.startswith("@")]
        found += "\n".join(mention for mention in mentions if mention)
    return found


@functools.lru_cache(maxsize=64)
def _whole_word(regex: str) -> re.Pattern:
    """The word (line) holding the first match of a path rule: what the audit names as the trigger."""
    return re.compile("(?<![^\n])[^\n]*?(?:" + regex + ")[^\n]*")


def sensitive(event: Mapping, rs: Ruleset) -> dict | None:
    """The first sensitive path or env dump the event's tool input names, as {rule, trigger}."""
    texts = tool_texts(event)
    blob = words(event, texts)
    for rule in rs.paths:
        if rule.applies(blob) and rule.pattern.search(blob):
            return {"rule": rule.id, "trigger": _whole_word(rule.pattern.pattern).search(blob).group(0)}
    commands = "\n".join(texts)
    for rule in rs.commands:
        if rule.applies(commands) and rule.pattern.search(commands):
            return {"rule": rule.id, "trigger": f"cmd:{rule.id}"}
    return None


def sanitize_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]], capture: str = "sanitized") -> dict:
    """Mirror of prepare.jq's judgment_prepare for one event: capture policy, sensitive check, scrub.
    It records no audit; the sender counts that later from the placeholders."""
    if rs.error:
        return {**_unredactable(event), "redaction_error": True}
    raw = event
    try:
        if capture == "metadata":
            event = {**strip(event), "prompt": "", "output": ""}
        else:
            hit = sensitive(event, rs)
            if hit is not None:
                event = {**strip(event), "sensitive": True, "sensitive_rule": hit["rule"], "sensitive_trigger": hit["trigger"]}
        return scrub_event(event, rs, secrets)
    except RedactionLimit:
        return {**_unredactable(raw), "redaction_error": True}


def show_trigger(trigger: str, home: str = "") -> str:
    """The trigger as the audit shows it: the home folder as ``~``. It was redacted with the event."""
    if home and home != "/" and trigger.startswith(home.rstrip("/") + "/"):
        return "~" + trigger[len(home.rstrip("/")) :]
    return trigger


def first_sensitive(events: list[dict], rs: Ruleset, secrets: list[tuple[str, str]], home: str = "") -> dict | None:
    """Why this turn is blanked: the first event the hook marked, or the first tool call found here.

    Run it on the events before they are redacted: a placeholder can hide what made a path sensitive.
    """
    for event in events:
        if event.get("sensitive"):
            return {"rule": str(event.get("sensitive_rule") or ""), "trigger": show_trigger(str(event.get("sensitive_trigger") or ""), home)}
    for event in events:
        try:
            hit = sensitive(event, rs)
        except RedactionLimit:
            # Too much to check is treated as sensitive: the turn loses its text, never sends it unchecked.
            return {"rule": "redaction_limit", "trigger": ""}
        if hit is not None:
            return {"rule": hit["rule"], "trigger": show_trigger(redact_text(hit["trigger"], rs, secrets), home)}
    return None


def turn_redaction(events: list[dict], blank: Mapping | None = None) -> dict:
    """The turn's audit record. Counts add up across events by rule and field."""
    totals: dict[tuple[str, str], int] = {}
    for event in events:
        for item in event.get("redactions") or []:
            if not isinstance(item, dict):
                continue
            try:
                count = int(item.get("count") or 0)
            except (TypeError, ValueError):
                continue
            key = (str(item.get("rule") or ""), str(item.get("field") or ""))
            totals[key] = totals.get(key, 0) + count
    items = [{"rule": rule, "field": field, "count": count} for (rule, field), count in sorted(totals.items())]
    record: dict[str, Any] = {"count": sum(item["count"] for item in items), "items": items, "blanked": bool(blank)}
    if blank:
        record["reason"] = SENSITIVE_REASON
        record["rule"] = str(blank.get("rule") or "")
        record["trigger"] = str(blank.get("trigger") or "")
    if any(event.get("redaction_error") for event in events):
        record["error"] = True
    return record
