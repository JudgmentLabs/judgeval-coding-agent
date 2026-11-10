"""Secret redaction in the sender, the second of two layers.

The hook runs the first layer in ``core/bin/prepare.jq`` before an event reaches the inbox.
This module runs the same rules, read from the same ``core/bin/redact-rules.json``, over each
turn after transcript text has been pulled in. ``sanitize_event`` is the jq layer step for
step, and tests/test_redact.py checks that the two give the same result.

An event is a plain dict. Every step takes events and returns events:

    redact_event   secrets in text -> ``[REDACTED:<rule>]``; ``redactions`` lists {rule, field, count}
                   of the placeholders the event then holds
    sensitive      the sensitive path or env dump a tool call names, or None
    strip_text     only the structural fields, the way metadata capture keeps them
    turn_redaction the turn's audit record: counts by rule and field, and why it was blanked

No function records a secret, or a hash, prefix, or length of one.
"""

from __future__ import annotations

import functools
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

RULES_FILE = Path(__file__).resolve().parent.parent / "bin" / "redact-rules.json"
SENSITIVE_REASON = "sensitive_path"
_NAMED = re.compile(r"\(\?<([A-Za-z_][A-Za-z0-9_]*)>")


@functools.lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(RULES_FILE.read_text(encoding="utf-8"))


def _compile(regex: str) -> re.Pattern:
    """Oniguruma names a group ``(?<p>...)``. Python spells it ``(?P<p>...)``."""
    return re.compile(_NAMED.sub(r"(?P<\1>", regex))


@dataclass(frozen=True)
class Rule:
    id: str
    pattern: re.Pattern
    needs: tuple = ()  # the rule can match only text holding one of these; empty means any text

    def applies(self, text: str) -> bool:
        """Mirror of gate in prepare.jq: skip the regex when no literal it needs is there."""
        return not self.needs or any(need in text for need in self.needs)


def _rule(item: Mapping) -> Rule:
    return Rule(item["id"], _compile(item["re"]), tuple(item.get("needs") or ()))


@functools.lru_cache(maxsize=1)
def _builtin() -> tuple[tuple, tuple, tuple]:
    table = rules()
    return (
        tuple(_rule(item) for item in table["patterns"]),
        tuple(_rule(item) for item in table["sensitive"]["paths"]),
        tuple(_rule(item) for item in table["sensitive"]["commands"]),
    )


def keep_fields() -> frozenset:
    return frozenset(rules()["keep_fields"])


def metadata_fields() -> frozenset:
    """What metadata capture keeps after transcripts were read: the structure, without transcript paths."""
    return keep_fields() - frozenset(rules()["transport_fields"])


@dataclass(frozen=True)
class Ruleset:
    patterns: tuple  # Rule, ... in table order, user patterns last
    paths: tuple
    commands: tuple
    error: str = ""  # a user rule that did not compile: text is dropped, never sent raw


def glob_re(glob: str) -> str:
    """Mirror of prepare.jq: a redactPaths glob, where * is any run of characters, / included, and ? is one."""
    body = "".join({"*": "[^\\n]*", "?": "[^\\n]"}.get(char) or ("\\" + char if char in ".\\+^$()[]{}|" else char) for char in glob)
    return "(?<![^\\n/])" + body + "(?![^\\n])"


def _user_patterns(project: Mapping) -> list[tuple[str, str]]:
    return [("custom", item) for item in project.get("redactPatterns") or [] if isinstance(item, str) and item]


def ruleset(project: Mapping | None = None) -> Ruleset:
    """The built-in rules plus a project's ``redactPatterns`` and ``redactPaths``. User rules only add."""
    patterns, paths, commands = _builtin()
    project = project or {}
    extra_patterns = []
    extra_paths = []
    try:
        for rule, regex in _user_patterns(project):
            extra_patterns.append(Rule(rule, _compile(regex)))
        for glob in project.get("redactPaths") or []:
            if isinstance(glob, str) and glob:
                extra_paths.append(Rule("custom_path", _compile(glob_re(glob))))
    except (re.error, RecursionError, OverflowError) as exc:
        return Ruleset(patterns, paths, commands, error=type(exc).__name__)
    return Ruleset(patterns + tuple(extra_patterns), paths + tuple(extra_paths), commands)


@functools.lru_cache(maxsize=1)
def _env_re() -> re.Pattern:
    return _compile(rules()["env"]["re"])


def env_secrets(env: Mapping[str, str] | None = None, api_key: str = "") -> list[tuple[str, str]]:
    """Mirror of prepare.jq: the table's regex over NAME=VALUE lines, longest value first.

    ``api_key`` is the key from the credentials file, which the hook also hands to jq as JUDGMENT_API_KEY.
    """
    source = dict(os.environ if env is None else env)
    if api_key and not source.get("JUDGMENT_API_KEY"):
        source["JUDGMENT_API_KEY"] = api_key
    lines = "".join(f"{name}={value}\n" for name, value in source.items() if isinstance(value, str))
    found = [(match.group("name"), match.group("value")) for match in _env_re().finditer(lines)]
    return sorted(found, key=lambda pair: (-len(pair[1]), pair[0]))


def _add(counts: dict, more: Mapping) -> dict:
    for rule, count in more.items():
        counts[rule] = counts.get(rule, 0) + count
    return counts


def redact_string(text: str, rs: Ruleset, secrets: list[tuple[str, str]]) -> tuple[str, dict]:
    """``text`` with env values, then patterns, replaced. Returns the text and {rule: count}."""
    counts: dict[str, int] = {}
    for name, value in secrets:
        if value in text:
            parts = text.split(value)
            text = f"[REDACTED:env:{name}]".join(parts)
            _add(counts, {f"env:{name}": len(parts) - 1})
    for rule in rs.patterns:
        if not rule.applies(text):
            continue
        keep = "p" in rule.pattern.groupindex
        # Mirror of prepare.jq: each matched text is replaced as a literal, longest first.
        found = {(match.group(0), (match.group("p") or "") if keep else "") for match in rule.pattern.finditer(text)}
        for whole, prefix in sorted(found, key=lambda pair: (-len(pair[0]), pair)):
            parts = text.split(whole)
            if len(parts) > 1:
                text = (prefix + f"[REDACTED:{rule.id}]").join(parts)
                _add(counts, {rule.id: len(parts) - 1})
    return text, counts


def redact_value(value: Any, rs: Ruleset, secrets: list[tuple[str, str]]) -> tuple[Any, dict]:
    """Any JSON value, walked to its strings. Object keys are left alone."""
    if isinstance(value, str):
        return redact_string(value, rs, secrets)
    counts: dict[str, int] = {}
    if isinstance(value, list):
        out = []
        for item in value:
            new, found = redact_value(item, rs, secrets)
            out.append(new)
            _add(counts, found)
        return out, counts
    if isinstance(value, dict):
        out_map = {}
        for key, item in value.items():
            new, found = redact_value(item, rs, secrets)
            out_map[key] = new
            _add(counts, found)
        return out_map, counts
    return value, counts


def _strings(value: Any):
    if isinstance(value, str):
        yield value
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _strings(item)


def strip_text(event: Mapping, keep: frozenset | None = None) -> dict:
    """Only the structural fields. tool_calls keep their id and name."""
    keep = keep_fields() if keep is None else keep
    kept = {key: value for key, value in event.items() if key in keep}
    calls = event.get("tool_calls")
    if isinstance(calls, list):
        kept["tool_calls"] = [
            {"id": call.get("id") or call.get("tool_call_id") or "", "name": call.get("name") or call.get("tool_name") or "tool"}
            for call in calls
            if isinstance(call, dict)
        ]
    return kept


def redact_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]]) -> dict:
    """Every text field redacted. ``redactions`` gains one {rule, field, count} per rule and field."""
    if rs.error:
        return {**strip_text(event), "redaction_error": True}
    keep = keep_fields()
    fields = [key for key in event if key not in keep]
    blob = "\n".join(text for key in fields for text in _strings(event[key]))
    live = [(name, value) for name, value in secrets if value in blob]
    hit = [rule for rule in rs.patterns if rule.applies(blob) and rule.pattern.search(blob)]
    if not live and not hit:
        return dict(event)
    narrowed = Ruleset(tuple(hit), rs.paths, rs.commands)
    out = dict(event)
    for field in fields:
        out[field] = redact_value(event[field], narrowed, live)[0]
    # Count what the event now holds, so text the hook already redacted is not counted again.
    audit = census(out, fields)
    if audit:
        out["redactions"] = audit
    else:
        out.pop("redactions", None)
    return out


_MARK = re.compile(r"\[REDACTED:([A-Za-z0-9_:.-]+)\]")


def census(event: Mapping, fields: list[str] | None = None) -> list[dict]:
    """{rule, field, count} for each placeholder in the event's text fields, by field, then rule."""
    keep = keep_fields()
    audit = []
    for field in fields if fields is not None else [key for key in event if key not in keep]:
        counts: dict[str, int] = {}
        for text in _strings(event.get(field)):
            for rule in _MARK.findall(text):
                counts[rule] = counts.get(rule, 0) + 1
        audit.extend({"rule": rule, "field": field, "count": counts[rule]} for rule in sorted(counts))
    return audit


def _texts(event: Mapping) -> list[str]:
    """Mirror of prepare.jq: path and command values in the tool input, found by the table's regex over its JSON."""
    table = rules()["sensitive"]
    raw = [event.get("tool_input")]
    calls = event.get("tool_calls")
    for call in calls if isinstance(calls, list) else calls.values() if isinstance(calls, dict) else []:
        if isinstance(call, dict):
            raw += [call.get("arguments"), call.get("input")]
    texts: list[str] = []
    texts_re, string_re = _compile(table["texts_re"]), _compile(table["string_re"])
    for item in raw:
        if item is None:
            continue
        if isinstance(item, str):
            try:
                item = json.loads(item)
            except ValueError:
                item = {"command": item}
        if isinstance(item, str):
            item = {"command": item}
        encoded = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        for match in texts_re.finditer(encoded):
            values = [match.group("s")] if match.group("s") is not None else [inner.group("s") for inner in string_re.finditer(match.group("a"))]
            texts.extend(json.loads('"' + value + '"') for value in values)
    return texts


def sensitive(event: Mapping, rs: Ruleset) -> dict | None:
    """The first sensitive path or env dump the event's tool input names, as {rule, trigger}."""
    table = rules()["sensitive"]
    texts = _texts(event)
    token_re = _compile(table["token_re"])
    tokens = [token for text in texts for token in token_re.findall(text)]
    prompt = event.get("prompt")
    if isinstance(prompt, str) and "@" in prompt:
        tokens += [mention[1:] for mention in _compile(table["mention_re"]).findall(prompt)]
    blob = "\n".join(tokens)
    for rule in rs.paths:
        if rule.applies(blob) and rule.pattern.search(blob):
            return {"rule": rule.id, "trigger": next(token for token in tokens if rule.pattern.search(token))}
    joined = "\n".join(texts)
    for rule in rs.commands:
        if rule.applies(joined) and rule.pattern.search(joined):
            return {"rule": rule.id, "trigger": f"cmd:{rule.id}"}
    return None


def show_trigger(trigger: str, rs: Ruleset, secrets: list[tuple[str, str]], home: str = "") -> str:
    """The trigger as it is recorded: the home folder as ``~``, and any secret in it redacted."""
    if home and trigger.startswith(home + "/"):
        trigger = "~" + trigger[len(home) :]
    return redact_string(trigger, rs, secrets)[0]


def sanitize_event(event: Mapping, rs: Ruleset, secrets: list[tuple[str, str]], home: str = "") -> dict:
    """Mirror of prepare.jq: a sensitive event loses its text, then the rest is redacted."""
    if rs.error:
        return {**strip_text(event), "redaction_error": True}
    hit = sensitive(event, rs)
    if hit is not None:
        event = {
            **strip_text(event),
            "sensitive": True,
            "sensitive_rule": hit["rule"],
            "sensitive_trigger": show_trigger(hit["trigger"], rs, secrets, home),
        }
    return redact_event(event, rs, secrets)


def first_sensitive(events: list[dict], rs: Ruleset, secrets: list[tuple[str, str]], home: str = "") -> dict | None:
    """Why this turn is blanked: the first event the hook marked, or the first tool call found here."""
    for event in events:
        if event.get("sensitive"):
            return {"rule": str(event.get("sensitive_rule") or ""), "trigger": str(event.get("sensitive_trigger") or "")}
    for event in events:
        hit = sensitive(event, rs)
        if hit is not None:
            return {"rule": hit["rule"], "trigger": show_trigger(hit["trigger"], rs, secrets, home)}
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
