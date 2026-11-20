"""Build one OTLP/JSON trace for one user turn.

The root span has no parent and carries ``judgment.session_id``.
Child spans are listed before that root, and every span has a positive duration.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, NamedTuple

from judgment_core import __version__
from judgment_core.messages import assistant_message, text_message, tool_message

_NS = 1_000_000


def _ident(size: int, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(("judgment-tracing-v1", *parts)).encode("utf-8")).hexdigest()
    return digest[:size]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


# An empty input or output is still sent, so the span keeps the key.
_KEEP_EMPTY = {"judgment.input", "judgment.output"}


def _attr(key: str, value: Any) -> dict | None:
    if value is None:
        return None
    if isinstance(value, str):
        if value == "" and key not in _KEEP_EMPTY:
            return None
        return {"key": key, "value": {"stringValue": value}}
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    if isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    if isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    return {"key": key, "value": {"stringValue": _text(value)}}


def _attrs(pairs: list[tuple[str, Any]]) -> list[dict]:
    out = []
    for key, value in pairs:
        item = _attr(key, value)
        if item is not None:
            out.append(item)
    return out


def _ms(event: dict, default: int) -> int:
    try:
        return int(event.get("ts") or default)
    except (TypeError, ValueError):
        return default


def _stamp_start(event: dict, default: int) -> int:
    if event.get("start_ts") is not None:
        try:
            return int(event["start_ts"])
        except (TypeError, ValueError):
            pass
    return _ms(event, default)


def _stamp_end(event: dict, default: int) -> int:
    if event.get("end_ts") is not None:
        try:
            return int(event["end_ts"])
        except (TypeError, ValueError):
            pass
    return _ms(event, default)


# Router names cannot be priced. A thought or stop payload may name the model that ran.
_UNPRICED_MODELS = {"", "auto", "auto-smart", "default"}


def _catalog_model(model: str) -> str:
    """Cursor appends reasoning effort to Grok ids. xAI prices `grok-4.7`."""
    name = (model or "").strip()
    lower = name.lower()
    if lower.startswith("grok-"):
        for suffix in ("-xhigh", "-high", "-medium", "-low"):
            if lower.endswith(suffix):
                return name[: -len(suffix)]
    return name


def _concrete_model(events: list[dict]) -> str:
    for event in events:
        name = str(event.get("model") or "").strip()
        if name.lower() not in _UNPRICED_MODELS:
            return name
    return ""


def _bounds(event: dict, default_start: int) -> tuple[int, int]:
    """Start and end in milliseconds. Prefer explicit timestamps over a 1 ms placeholder."""
    if event.get("start_ts") is not None:
        try:
            start = int(event["start_ts"])
        except (TypeError, ValueError):
            start = _ms(event, default_start)
    else:
        start = _ms(event, default_start)
    if event.get("end_ts") is not None:
        try:
            end = int(event["end_ts"])
        except (TypeError, ValueError):
            end = start + 1
    else:
        try:
            extra = int(event["duration_ms"]) if event.get("duration_ms") is not None else 1
        except (TypeError, ValueError):
            extra = 1
        end = start + max(extra, 1)
    if end <= start:
        end = start + 1
    return start, end


def _provider(model: str, harness: str, explicit: str) -> str:
    if explicit:
        return explicit
    name = (model or "").lower()
    if "claude" in name or name.startswith("anthropic"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")) or "codex" in name:
        return "openai"
    if "gemini" in name:
        return "google"
    if "grok" in name:
        return "xai"
    return {"claude-code": "anthropic", "codex": "openai"}.get(harness, "")


def _grok_cost(non_cached: int, cached: int, output: int) -> float:
    """Published grok-4.7 rates from xAI's pricing page.

    Below 200k prompt tokens the rates are $2, $0.50, and $6 per million
    input, cached input, and output tokens. At 200k and above they are $4,
    $1, and $12. Cursor's ``grok-4.7-high`` is that model at the default
    reasoning effort, not the separate Fast SKU.
    """
    prompt = non_cached + cached
    if prompt >= 200_000:
        input_rate, cache_rate, output_rate = 4.0, 1.0, 12.0
    else:
        input_rate, cache_rate, output_rate = 2.0, 0.50, 6.0
    return (non_cached * input_rate + cached * cache_rate + output * output_rate) / 1_000_000


def _usage_pairs(usage: Any, model: str = "") -> list[tuple[str, Any]]:
    if not isinstance(usage, dict):
        return []
    data = dict(usage)
    includes_cache = bool(data.pop("input_includes_cache", False))
    cost = data.get("cost")
    if "total_cost_usd" not in data and isinstance(cost, dict) and cost.get("total") is not None:
        data["total_cost_usd"] = cost.get("total")
    if includes_cache and data.get("input_tokens") is not None:
        cached = None
        for key in ("cached_input_tokens", "cache_read_input_tokens", "cacheRead"):
            if data.get(key) is not None:
                cached = data.get(key)
                break
        try:
            raw_input = int(data["input_tokens"])
            cached_n = int(cached) if cached is not None else 0
        except (TypeError, ValueError):
            raw_input = None
            cached_n = 0
        if raw_input is not None and cached_n > 0 and raw_input >= cached_n:
            data["input_tokens"] = raw_input - cached_n
            data.setdefault("cache_read_input_tokens", cached_n)
    if str(model).lower().startswith("grok-4.7") and data.get("total_cost_usd") is None:
        try:
            non_cached = int(data.get("input_tokens") or data.get("non_cached_input_tokens") or 0)
            cached_total = int(data.get("cache_read_input_tokens") or data.get("cached_input_tokens") or 0)
            output = int(data.get("output_tokens") or 0)
        except (TypeError, ValueError):
            non_cached, cached_total, output = 0, 0, 0
        if non_cached or cached_total or output:
            data["total_cost_usd"] = _grok_cost(non_cached, cached_total, output)
    sources = (
        ("non_cached_input_tokens", "non_cached_input_tokens", "gen_ai.usage.input_tokens"),
        ("input_tokens", "non_cached_input_tokens", "gen_ai.usage.input_tokens"),
        ("input", "non_cached_input_tokens", "gen_ai.usage.input_tokens"),
        ("output_tokens", "output_tokens", "gen_ai.usage.output_tokens"),
        ("output", "output_tokens", "gen_ai.usage.output_tokens"),
        ("cache_read_input_tokens", "cache_read_input_tokens", "gen_ai.usage.cache_read_input_tokens"),
        ("cached_input_tokens", "cache_read_input_tokens", "gen_ai.usage.cache_read_input_tokens"),
        ("cacheRead", "cache_read_input_tokens", "gen_ai.usage.cache_read_input_tokens"),
        ("cache_creation_input_tokens", "cache_creation_input_tokens", "gen_ai.usage.cache_creation_input_tokens"),
        ("cache_write_input_tokens", "cache_creation_input_tokens", "gen_ai.usage.cache_creation_input_tokens"),
        ("cacheWrite", "cache_creation_input_tokens", "gen_ai.usage.cache_creation_input_tokens"),
        ("reasoning_tokens", "reasoning_tokens", ""),
        ("reasoning_output_tokens", "reasoning_tokens", ""),
        ("total_cost_usd", "total_cost_usd", ""),
    )
    seen: set[str] = set()
    pairs: list[tuple[str, Any]] = []
    for source, dest, gen_key in sources:
        if dest in seen or source not in data or data[source] is None:
            continue
        raw = data[source]
        try:
            if dest == "total_cost_usd":
                value: Any = float(raw)
            else:
                value = int(raw)
        except (TypeError, ValueError):
            continue
        seen.add(dest)
        pairs.append((f"judgment.usage.{dest}", value))
        if gen_key:
            pairs.append((gen_key, value))
    return pairs


def _span(
    *,
    trace_id: str,
    span_id: str,
    parent: str | None,
    name: str,
    start_ms: int,
    end_ms: int,
    attributes: list[dict],
    error: bool = False,
) -> dict:
    start_ns = start_ms * _NS
    end_ns = end_ms * _NS
    if end_ns <= start_ns:
        end_ns = start_ns + _NS
    body = {
        "traceId": trace_id,
        "spanId": span_id,
        "name": name[:200] or "span",
        "kind": 1,
        "startTimeUnixNano": str(start_ns),
        "endTimeUnixNano": str(end_ns),
        "attributes": attributes,
    }
    if parent:
        body["parentSpanId"] = parent
    if error:
        body["status"] = {"code": 2}
    return body


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _calls_of(event: dict) -> list[dict]:
    raw = event.get("tool_calls")
    if not isinstance(raw, list):
        return []
    calls = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        arguments = item.get("arguments")
        if arguments is None:
            arguments = item.get("input", item.get("tool_input"))
        calls.append(
            {
                "id": str(item.get("id") or item.get("tool_call_id") or ""),
                "name": str(item.get("name") or item.get("tool_name") or "tool"),
                "arguments": arguments if arguments is not None else {},
            }
        )
    return calls


def _call_from_tool(event: dict) -> dict:
    return {
        "id": str(event.get("tool_call_id") or ""),
        "name": str(event.get("tool_name") or "tool"),
        "arguments": event.get("tool_input") if event.get("tool_input") is not None else {},
    }


def _unique(texts) -> list[str]:
    """Texts in order, each once. Cursor delivers some thoughts twice."""
    seen: list[str] = []
    for text in texts:
        if text and text not in seen:
            seen.append(text)
    return seen


def _user(text: str) -> dict:
    return text_message("user", text)


class _Rendered(NamedTuple):
    """One llm request: its input messages, its output message, and the tool calls it made."""

    input: list[dict]
    output: dict
    calls: list[dict]


def _thread_io(
    models: list[dict],
    tools: list[dict],
    prompt: str,
    *,
    context: list[dict] | None = None,
    tools_in_input: bool = False,
    final: str = "",
) -> tuple[list[_Rendered], list[dict] | None]:
    """Each llm request's messages, and the messages this turn added to the thread.

    A request read from a transcript carries the exact ``input_messages`` it
    sent and its ``output_message``. Those are used as they are, and the turn's
    messages are None: the next turn reads its history from the transcript too.
    A request that lost them (its redaction hit a limit, or an earlier turn it
    replays was not traced) is rebuilt as below; the others keep theirs.

    Otherwise the input is ``context`` (the session's earlier turns, whole),
    the user prompt, and every tool call and result that came before the
    request. The output is the assistant message, including tool_call parts
    when this request invoked tools. A result the model saw (``tool_results``
    on the request) wins over the hook's tool output. A turn with no request
    still adds its prompt, its tools, and its ``final`` answer to the thread.

    ``tools_in_input`` is for a harness that reports one span for many
    requests (Cursor, one per stop). The tools that ran before the span ends go
    in its input with their results, and the output is the final text.
    """
    context = list(context or [])
    by_id = {}
    for tool in tools:
        call_id = str(tool.get("tool_call_id") or "")
        if call_id:
            by_id[call_id] = tool
    claimed = {call["id"] for model in models for call in _calls_of(model) if call["id"]}
    # A tool run inside a claimed call, such as a Codex code-mode command, is already in that call's result.
    pending = [tool for tool in tools if str(tool.get("tool_call_id") or "") not in claimed and str(tool.get("parent_call_id") or "") not in claimed]
    pending.sort(key=lambda tool: _ms(tool, 0))
    used: set[int] = set()
    seen_calls: set[str] = set()
    history = [_user(prompt or "")]
    rendered: list[_Rendered] = []

    def _fresh(call_id: str) -> bool:
        if call_id and call_id in seen_calls:
            return False
        if call_id:
            seen_calls.add(call_id)
        return True

    def _response(model: dict, call_id: str) -> str | None:
        for result in model.get("tool_results") or []:
            if isinstance(result, dict) and str(result.get("id") or "") == call_id:
                return _text(result.get("response"))
        return None

    for index, model in enumerate(models):
        stamp = _ms(model, 0)
        nxt = _ms(models[index + 1], 10**15) if index + 1 < len(models) else 10**15
        limit = nxt if tools_in_input else stamp
        prior = [tool for tool in pending if id(tool) not in used and _ms(tool, 0) < limit]
        preamble = "\n\n".join(_text(text) for text in model.get("preamble") or [] if _text(text)) if tools_in_input else ""
        for tool in prior:
            used.add(id(tool))
            call = _call_from_tool(tool)
            if not _fresh(call["id"]):
                continue
            history.append(assistant_message(preamble, [call]))
            preamble = ""
            history.append(tool_message(call["id"], _text(tool.get("tool_output")) if tool.get("tool_output") is not None else ""))
        for received in model.get("received") or []:
            history.append(_user(_text(received)))
        calls = _calls_of(model)
        if not calls and not tools_in_input:
            for tool in pending:
                if id(tool) in used:
                    continue
                tool_stamp = _ms(tool, 0)
                if stamp <= tool_stamp < nxt:
                    calls.append(_call_from_tool(tool))
                    used.add(id(tool))
        own = _typed(_text(model.get("prompt")))
        incoming = list(history)
        if own and own != (prompt or "") and len(history) == 1:
            incoming = [_user(own)]
        elif own and own != (prompt or "") and index > 0:
            # A later request answering a new user message, such as a Cursor background wake.
            incoming.append(_user(own))
        outgoing = assistant_message(_text(model.get("output")), calls, model.get("reasoning"))
        rendered.append(_Rendered(context + incoming, outgoing, calls))
        # A failed request's error text never reached the model. The next request does not see it.
        history = [*incoming] if model.get("is_error") else [*incoming, outgoing]
        for call in calls:
            if not _fresh(call["id"]):
                continue
            tool = by_id.get(call["id"])
            if tool is not None:
                used.add(id(tool))
            response = _response(model, call["id"])
            if response is None:
                response = _text(tool.get("tool_output")) if tool is not None and tool.get("tool_output") is not None else ""
            history.append(tool_message(call["id"], response))
    if not models:
        for tool in pending:
            call = _call_from_tool(tool)
            if _fresh(call["id"]):
                history.append(assistant_message("", [call]))
                history.append(tool_message(call["id"], _text(tool.get("tool_output")) if tool.get("tool_output") is not None else ""))
        if final:
            history.append(assistant_message(final, []))
    exact = [isinstance(model.get("input_messages"), list) for model in models]
    if not any(exact):
        return rendered, history
    rendered = [
        _Rendered(model["input_messages"], _exact_output(model), _calls_of(model)) if exact[index] else rendered[index]
        for index, model in enumerate(models)
    ]
    return rendered, None


def _exact_output(model: dict) -> dict:
    """A transcript request's output message. Text the transcript lacked but the turn's end reported is added to it."""
    text = _text(model.get("output"))
    message = model.get("output_message")
    if not isinstance(message, dict):
        return assistant_message(text, _calls_of(model), model.get("reasoning"))
    parts = list(message.get("parts") or [])
    if text and not any(part.get("type") == "text" for part in parts if isinstance(part, dict)):
        # Text goes after reasoning and before tool calls, as assistant_message orders them.
        at = next((index for index, part in enumerate(parts) if isinstance(part, dict) and part.get("type") != "reasoning"), len(parts))
        parts.insert(at, {"type": "text", "content": text})
    return {**message, "parts": parts}


def _contain(spans: list[dict]) -> None:
    """Pull each parent's start and end around its children, so every child sits inside its parent."""
    children: dict[str, list[dict]] = {}
    for span in spans:
        parent = span.get("parentSpanId")
        if parent:
            children.setdefault(parent, []).append(span)

    def walk(span: dict) -> tuple[int, int]:
        start = int(span["startTimeUnixNano"])
        end = int(span["endTimeUnixNano"])
        for child in children.get(span["spanId"], []):
            child_start, child_end = walk(child)
            start = min(start, child_start)
            end = max(end, child_end)
        if end <= start:
            end = start + _NS
        span["startTimeUnixNano"] = str(start)
        span["endTimeUnixNano"] = str(end)
        return start, end

    for span in spans:
        if not span.get("parentSpanId"):
            walk(span)


def _stamp_session(spans: list[dict], session_id: str, customer_user_id: str, update_id: int | None) -> None:
    for span in spans:
        attrs = span.setdefault("attributes", [])
        keys = {item["key"] for item in attrs}
        # The person running the agent, as judgeval's set_customer_user_id puts it on every span.
        if customer_user_id and "judgment.customer_user_id" not in keys:
            attrs.append(_attr("judgment.customer_user_id", customer_user_id))
        if "judgment.session_id" not in keys:
            extra = _attr("judgment.session_id", session_id)
            if extra:
                attrs.append(extra)
        if "gen_ai.conversation.id" not in keys:
            extra = _attr("gen_ai.conversation.id", session_id)
            if extra:
                attrs.append(extra)
        # A resend rebuilds every span. The server keeps the highest version of each span id.
        if update_id is not None and "judgment.update_id" not in keys:
            attrs.append(_attr("judgment.update_id", update_id))


def _usable(events: list[dict]) -> list[dict]:
    return [event for event in events if event.get("event") not in ("session_model", "session_end", None)]


# Desktop apps wrap what the user typed in notes for the model, such as the worktree it runs in.
_REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def _typed(prompt: str) -> str:
    """What the user typed. A prompt that is only app notes, such as a session handoff, stays whole."""
    return _REMINDER.sub("", prompt).strip() or prompt


def _prompt_and_final(usable: list[dict]) -> tuple[str, str]:
    prompt = ""
    final = ""
    for event in usable:
        if event.get("event") == "turn_start" and event.get("prompt"):
            prompt = _text(event.get("prompt"))
            break
    if not prompt:
        for event in usable:
            if event.get("prompt"):
                prompt = _text(event.get("prompt"))
                break
    prompt = _typed(prompt)
    for event in reversed(usable):
        if event.get("event") == "turn_end" and event.get("output") and not event.get("agent_id"):
            final = _text(event.get("output"))
            break
    if not final:
        for event in reversed(usable):
            if event.get("event") == "model" and not event.get("is_error") and event.get("output") and not event.get("agent_id"):
                final = _text(event.get("output"))
                break
    return prompt, final


def turn_start(events: list[dict]) -> int | None:
    """When the turn started, in milliseconds, so later turns can find the turns before them."""
    usable = _usable(events)
    if not usable or not usable[0].get("turn_id"):
        return None
    return min(_stamp_start(event, 0) for event in usable)


class Turn(NamedTuple):
    """One turn's trace, and the messages it added to the main thread.

    ``messages`` is None when the turn's llm inputs came from a transcript:
    the next turn reads its history from the transcript as well.
    """

    payload: dict
    messages: list[dict] | None


def _conversation(role: str, text: str) -> str:
    """Agent span input or output as one GenAI message, so the view does not show it raw."""
    message = _user(text) if role == "user" else assistant_message(text, [])
    return _json([message])


_TEXT_KEYS = ("judgment.input", "judgment.output", "gen_ai.system_instructions", "gen_ai.tool.definitions")
_PLACEHOLDER = "[REDACTED:"


def _apply_redaction(spans: list[dict], root: dict, redaction: dict) -> None:
    """Record the turn's redactions on its root, and on each other span the placeholders its text holds."""
    for span in spans:
        if span is not root:
            attrs = span["attributes"]
            found = sum(str(attr["value"].get("stringValue", "")).count(_PLACEHOLDER) for attr in attrs if attr["key"] in _TEXT_KEYS)
            if found:
                attrs.append(_attr("judgment.redaction_count", found))
    items = redaction.get("items") or []
    pairs: list[tuple[str, Any]] = [
        ("judgment.redacted", int(redaction.get("count") or 0) > 0),
        ("judgment.redaction_count", int(redaction.get("count") or 0)),
        ("judgment.redactions", json.dumps(items, separators=(",", ":")) if items else None),
    ]
    if redaction.get("error"):
        pairs.append(("judgment.redaction_error", True))
    root["attributes"].extend(_attrs(pairs))


def build_request(
    events: list[dict],
    *,
    update_id: int | None = None,
    history: list[dict] | None = None,
    redaction: dict | None = None,
) -> Turn | None:
    """``history`` is the messages the session's earlier turns added to the main thread, oldest first.

    ``redaction`` is ``redact.turn_redaction``'s record, or None for raw (``capture: full``) text.
    """
    usable = _usable(events)
    if not usable:
        return None
    first = usable[0]
    harness = str(first.get("harness") or "agent")
    session_id = str(first.get("session_id") or "")
    turn_id = str(first.get("turn_id") or "")
    if not session_id or not turn_id:
        return None
    environment = str(first.get("environment") or "production")
    # Only the folder's name leaves the machine. The full path, and the home folder's name, hold the user's account name.
    folder = Path(str(first.get("cwd") or first.get("project_root") or ""))
    workspace = "" if any(event.get("capture") == "metadata" for event in usable) or folder == Path.home() else folder.name
    # Events the flusher reads back from a transcript carry no hook stamp, so take the first event that has one.
    customer_user_id = next((str(event["customer_user_id"]) for event in usable if event.get("customer_user_id")), "")
    harness_version = ""
    for event in usable:
        if event.get("harness_version"):
            harness_version = str(event["harness_version"])
            break

    prompt, final = _prompt_and_final(usable)

    main = [event for event in usable if not event.get("agent_id")]
    by_agent: dict[str, list[dict]] = {}
    meta: dict[str, dict] = {}
    for event in usable:
        agent_id = str(event.get("agent_id") or "")
        if not agent_id and event.get("event") == "tool" and "spawn" in str(event.get("tool_name") or ""):
            output = event.get("tool_output")
            if isinstance(output, dict):
                agent_id = str(output.get("agent_id") or output.get("agentId") or "")
            agent_id = agent_id or str(event.get("bind_agent") or "")
            if not agent_id:
                continue
            by_agent.setdefault(agent_id, [])
            slot = meta.setdefault(agent_id, {"agent_type": "", "agent_name": "", "prompt": "", "output": ""})
            if not slot["prompt"]:
                raw = event.get("tool_input")
                if isinstance(raw, dict):
                    for key in ("prompt", "task", "message", "input"):
                        value = raw.get(key)
                        if isinstance(value, str) and value and not value.startswith("gAAAA"):
                            slot["prompt"] = value
                            break
            continue
        if not agent_id:
            continue
        by_agent.setdefault(agent_id, []).append(event)
        slot = meta.setdefault(agent_id, {"agent_type": "", "agent_name": "", "prompt": "", "output": ""})
        if event.get("agent_type"):
            slot["agent_type"] = str(event["agent_type"])
        if event.get("agent_name"):
            slot["agent_name"] = str(event["agent_name"])
        if event.get("prompt") and (not slot["prompt"] or event.get("event") in ("subagent_prompt", "subagent_start", "subagent_end")):
            slot["prompt"] = _text(event.get("prompt"))
        if not slot["prompt"] and event.get("event") == "tool":
            raw = event.get("tool_input")
            if isinstance(raw, str):
                try:
                    raw = json.loads(raw)
                except json.JSONDecodeError:
                    raw = {"message": raw} if raw and not raw.startswith("gAAAA") else {}
            if isinstance(raw, dict):
                for key in ("prompt", "task", "message", "input"):
                    value = raw.get(key)
                    if isinstance(value, str) and value and not value.startswith("gAAAA"):
                        slot["prompt"] = value
                        break
        if event.get("output") and event.get("event") in ("subagent_end", "model"):
            slot["output"] = _text(event.get("output"))

    has_work = any(
        event.get("event") in ("model", "tool", "thought", "turn_start", "turn_end", "subagent_start", "subagent_end", "subagent_prompt")
        for event in usable
    )
    if not has_work:
        return None

    trace_id = _ident(32, "trace", harness, session_id, turn_id)
    turn_span_id = _ident(16, trace_id, "turn")
    starts = [_stamp_start(event, 0) for event in usable]
    ends = [_stamp_end(event, 0) for event in usable]
    start_ms = min(starts) if starts else 0
    end_ms = max(ends) if ends else start_ms
    concrete_model = _concrete_model(usable)

    spans: list[dict] = []
    tool_starts = {
        str(event.get("tool_call_id")): _ms(event, start_ms)
        for event in usable
        if event.get("event") == "tool_start" and event.get("tool_call_id")
    }
    model_starts: dict[str, list[int]] = {}
    for event in usable:
        if event.get("event") != "model_start":
            continue
        model_starts.setdefault(str(event.get("agent_id") or ""), []).append(_ms(event, start_ms))
    model_cursor = {agent: 0 for agent in model_starts}

    def timed(event: dict, kind: str) -> dict:
        item = dict(event)
        if item.get("start_ts") is not None:
            return item
        if kind == "tool":
            started = tool_starts.get(str(item.get("tool_call_id") or ""))
            if started is not None:
                item["start_ts"] = started
                item["end_ts"] = _ms(item, started)
            elif item.get("duration_ms") not in (None, "") and item.get("ts") is not None:
                # A tool event is recorded when the tool finishes, so its duration reaches back from ts.
                try:
                    item["start_ts"] = _ms(item, start_ms) - max(int(item["duration_ms"]), 1)
                    item["end_ts"] = _ms(item, start_ms)
                except (TypeError, ValueError):
                    pass
            return item
        if kind != "model":
            return item
        agent = str(item.get("agent_id") or "")
        starts = model_starts.get(agent) or []
        cursor = model_cursor.get(agent, 0)
        if cursor < len(starts):
            model_cursor[agent] = cursor + 1
            started = starts[cursor]
            item["start_ts"] = started
            item["end_ts"] = _ms(item, started)
        return item

    def add_model(event: dict, index: int, parent: str, agent_id: str, rendered: _Rendered | None) -> None:
        model = _catalog_model(str(event.get("model") or ""))
        if model.lower() in _UNPRICED_MODELS and concrete_model:
            model = _catalog_model(concrete_model)
        provider = _provider(model, harness, str(event.get("provider") or ""))
        own_prompt = _text(event.get("prompt"))
        own_output = _text(event.get("output"))
        calls = rendered.calls if rendered else _calls_of(event)
        pairs = [
            ("judgment.span_kind", "llm"),
            ("judgment.session_id", session_id),
            ("gen_ai.conversation.id", session_id),
            ("judgment.llm.model", model),
            ("judgment.llm.provider", provider),
            ("gen_ai.operation.name", "chat"),
            ("gen_ai.request.model", model),
            ("gen_ai.response.model", model),
            ("gen_ai.system", provider),
        ]
        usage_pairs = _usage_pairs(event.get("usage"), model)
        has_tokens = any(key.startswith("judgment.usage.") and key.endswith(("non_cached_input_tokens", "output_tokens")) for key, _value in usage_pairs)
        if harness == "cursor" and not has_tokens:
            return
        if not own_prompt and not own_output and not usage_pairs and not calls and not event.get("is_error"):
            return
        if rendered:
            output = rendered.output
            if event.get("finish_reason"):
                output = {**output, "finish_reason": str(event["finish_reason"])}
            pairs.append(("judgment.input", _json(rendered.input or [_user("")])))
            pairs.append(("judgment.output", _json([output])))
        if event.get("system_instructions"):
            pairs.append(("gen_ai.system_instructions", _json([{"type": "text", "content": _text(event["system_instructions"])}])))
        if event.get("tool_definitions"):
            pairs.append(("gen_ai.tool.definitions", _json(event["tool_definitions"])))
        if calls:
            pairs.append(("judgment.llm.tool_calls", _json(calls)))
        pairs.extend(usage_pairs)
        when, end = _bounds(event, start_ms)
        message_id = str(event.get("message_id") or "")
        span_key = message_id or f"{index}:{when}"
        spans.append(
            _span(
                trace_id=trace_id,
                span_id=_ident(16, trace_id, "llm", agent_id, span_key),
                parent=parent,
                name=model or "llm",
                start_ms=when,
                end_ms=end,
                attributes=_attrs(pairs),
                error=bool(event.get("is_error")),
            )
        )

    def add_tool(event: dict, index: int, parent: str, agent_id: str) -> None:
        name = str(event.get("tool_name") or "tool")
        call_id = str(event.get("tool_call_id") or index)
        incoming = event.get("tool_input")
        outgoing = event.get("tool_output")
        in_text = _text(incoming) if incoming is not None else ""
        out_text = _text(outgoing) if outgoing is not None else ""
        pairs = [
            ("judgment.span_kind", "tool"),
            ("judgment.session_id", session_id),
            ("gen_ai.conversation.id", session_id),
            ("gen_ai.tool.name", name),
            ("judgment.input", in_text or ""),
            ("judgment.output", out_text or ""),
        ]
        when, end = _bounds(event, start_ms)
        spans.append(
            _span(
                trace_id=trace_id,
                span_id=_ident(16, trace_id, "tool", agent_id, call_id),
                parent=parent,
                name=name,
                start_ms=when,
                end_ms=end,
                attributes=_attrs(pairs),
                error=bool(event.get("is_error")),
            )
        )

    def add_thought(event: dict, index: int, parent: str) -> None:
        text = _text(event.get("output"))
        # The hook fires when the thought finishes. duration_ms reaches backward from that time.
        if event.get("start_ts") is None and event.get("end_ts") is None and event.get("duration_ms") not in (None, ""):
            try:
                duration = int(event["duration_ms"])
            except (TypeError, ValueError):
                duration = 0
            if duration > 0:
                end = _ms(event, start_ms)
                when = end - duration
                if end <= when:
                    end = when + 1
            else:
                when, end = _bounds(event, start_ms)
        else:
            when, end = _bounds(event, start_ms)
        spans.append(
            _span(
                trace_id=trace_id,
                span_id=_ident(16, trace_id, "thought", str(index), str(when)),
                parent=parent,
                name="thought",
                start_ms=when,
                end_ms=end,
                attributes=_attrs(
                    [
                        # A thought is the model's reasoning: it has an output and no input of its own.
                        ("judgment.span_kind", "span"),
                        ("judgment.session_id", session_id),
                        ("gen_ai.conversation.id", session_id),
                        ("judgment.output", text or ""),
                    ]
                ),
            )
        )

    for agent_id, agent_events in by_agent.items():
        info = meta[agent_id]
        child_parent = _ident(16, trace_id, "sub", agent_id)
        agent_models = [timed(event, "model") for event in agent_events if event.get("event") == "model"]
        agent_tools = [event for event in agent_events if event.get("event") == "tool"]
        agent_io, _ = _thread_io(agent_models, agent_tools, info["prompt"] or prompt)
        model_queue = list(agent_models)
        io_at = 0
        for index, event in enumerate(agent_events):
            kind = event.get("event")
            if kind == "model":
                rendered = agent_io[io_at] if io_at < len(agent_io) else None
                io_at += 1
                add_model(model_queue.pop(0), index, child_parent, agent_id, rendered)
            elif kind == "tool":
                add_tool(timed(event, "tool"), index, child_parent, agent_id)
            elif kind == "thought":
                add_thought(event, index, child_parent)
        if agent_events:
            stamps_for_agent = [_stamp_start(event, start_ms) for event in agent_events]
            agent_ends = [_stamp_end(event, start_ms) for event in agent_events]
            agent_start = min(stamps_for_agent)
            agent_end = max(agent_ends)
        else:
            agent_start = start_ms
            agent_end = end_ms
        for event in agent_events:
            if event.get("end_ts") is None:
                continue
            try:
                agent_end = max(agent_end, int(event["end_ts"]))
            except (TypeError, ValueError):
                pass
        label = info["agent_name"] or info["agent_type"] or "subagent"
        spans.append(
            _span(
                trace_id=trace_id,
                span_id=child_parent,
                parent=turn_span_id,
                name=label,
                start_ms=agent_start,
                end_ms=agent_end,
                attributes=_attrs(
                    [
                        ("judgment.span_kind", "agent"),
                        ("judgment.session_id", session_id),
                        ("gen_ai.conversation.id", session_id),
                        ("judgment.agent_id", agent_id),
                        ("judgment.parent_agent_id", session_id),
                        ("judgment.agent_class_name", info["agent_type"]),
                        # The Conversation View draws a subagent from its own span only when it has no llm spans.
                        ("judgment.input", info["prompt"] if agent_models else _conversation("user", info["prompt"])),
                        ("judgment.output", info["output"] if agent_models else _conversation("assistant", info["output"])),
                    ]
                ),
            )
        )

    model_events = [event for event in main if event.get("event") == "model"]
    named = next(
        (str(event.get("model")) for event in reversed(main) if event.get("model") and event.get("event") in ("turn_start", "turn_end", "model")),
        "",
    )
    found_usage = next((event.get("usage") for event in reversed(main) if event.get("usage")), None)
    tokenized = False
    if isinstance(found_usage, dict):
        tokenized = any(found_usage.get(key) is not None for key in ("input_tokens", "output_tokens", "non_cached_input_tokens"))
    # Cursor reports usage only on stop. A turn that stops more than once made one request per stop.
    stops = []
    for event in main:
        if event.get("event") != "turn_end" or not isinstance(event.get("usage"), dict):
            continue
        # An aborted TUI turn sends the same stop twice. A second request never repeats the exact usage.
        if stops and stops[-1]["usage"] == event["usage"]:
            stops[-1] = event
            continue
        stops.append(event)
    if harness == "cursor" and tokenized:
        # afterAgentResponse carries text but no tokens and no start. Each stop is the request window its tokens cover.
        model_events = []
        thoughts = [event for event in main if event.get("event") == "thought" and _text(event.get("output"))]
    if not model_events and harness == "cursor" and tokenized and len(stops) > 1:
        begin = start_ms
        for number, stop in enumerate(stops):
            stamp = end_ms if number == len(stops) - 1 else _ms(stop, begin)
            model_events.append(
                {
                    "event": "model",
                    "model": str(stop.get("model") or named),
                    "prompt": prompt if number == 0 else _text(stop.get("prompt")),
                    "output": _text(stop.get("output")),
                    "preamble": stop.get("preamble") or [],
                    "reasoning": _unique(_text(event.get("output")) for event in thoughts if begin <= _ms(event, begin) <= stamp),
                    "ts": begin,
                    "duration_ms": max(stamp - begin, 1),
                    "usage": stop["usage"],
                }
            )
            begin = stamp
    elif not model_events and (prompt or final or tokenized) and ((harness == "cursor" and tokenized) or (harness != "cursor" and (named or found_usage))):
        model_events = [
            {
                "event": "model",
                "model": named,
                "provider": next((str(event.get("provider")) for event in reversed(main) if event.get("provider")), ""),
                "prompt": prompt,
                "output": final,
                "ts": start_ms,
                "duration_ms": max(end_ms - start_ms, 1),
                "usage": found_usage,
            }
        ]
        if harness == "cursor":
            stop = stops[-1] if stops else {}
            model_events[0]["preamble"] = stop.get("preamble") or []
            model_events[0]["reasoning"] = _unique(_text(event.get("output")) for event in thoughts)
    else:
        if model_events and prompt and not model_events[0].get("prompt"):
            model_events[0] = {**model_events[0], "prompt": prompt}
        if model_events and final and not model_events[-1].get("output"):
            model_events[-1] = {**model_events[-1], "output": final}
        if model_events and not model_events[-1].get("usage"):
            # Usage reported for the turn as a whole. Another request's own usage is never copied: that counts it twice.
            found = next((item.get("usage") for item in reversed(main) if item.get("usage") and item.get("event") != "model"), None)
            if found:
                model_events[-1] = {**model_events[-1], "usage": found}

    main_tools = [event for event in main if event.get("event") == "tool"]
    timed_models = [timed(event, "model") for event in model_events]
    main_io, turn_messages = _thread_io(
        timed_models,
        main_tools,
        prompt,
        context=history,
        tools_in_input=harness == "cursor",
        final=final,
    )
    for index, event in enumerate(timed_models):
        rendered = main_io[index] if index < len(main_io) else None
        add_model(event, index, turn_span_id, "", rendered)
    for index, event in enumerate(event for event in main if event.get("event") == "tool"):
        add_tool(timed(event, "tool"), index, turn_span_id, "")
    seen_thoughts: set[str] = set()
    thought_index = 0
    for event in main:
        if event.get("event") != "thought":
            continue
        text = _text(event.get("output"))
        if text and text in seen_thoughts:
            continue
        if text:
            seen_thoughts.add(text)
        add_thought(event, thought_index, turn_span_id)
        thought_index += 1

    spans.append(
        _span(
            trace_id=trace_id,
            span_id=turn_span_id,
            parent=None,
            name=f"{harness} turn",
            start_ms=start_ms,
            end_ms=end_ms,
            attributes=_attrs(
                [
                    ("judgment.span_kind", "agent"),
                    ("judgment.session_id", session_id),
                    ("gen_ai.conversation.id", session_id),
                    ("judgment.is_agent_entry_point", True),
                    ("judgment.agent_id", session_id),
                    # Plain text: the trace list and judges read the root's input and output as written.
                    ("judgment.input", prompt),
                    ("judgment.output", final),
                    ("agent.workspace_name", workspace),
                ]
            ),
        )
    )

    if not spans:
        return None
    if redaction is not None:
        _apply_redaction(spans, spans[-1], redaction)
    _contain(spans)
    _stamp_session(spans, session_id, customer_user_id, update_id)

    resource_pairs = [
        ("service.name", harness),
        ("service.version", __version__),
        ("deployment.environment.name", environment),
        ("telemetry.sdk.name", "judgment-tracing"),
        ("telemetry.sdk.version", __version__),
        ("agent.harness", harness),
    ]
    if harness_version:
        resource_pairs.append(("agent.harness.version", harness_version))
    payload = {
        "resourceSpans": [
            {
                "resource": {"attributes": _attrs(resource_pairs)},
                "scopeSpans": [
                    {
                        "scope": {"name": "judgment-tracing", "version": __version__},
                        "spans": spans,
                    }
                ],
            }
        ]
    }
    return Turn(payload, turn_messages)
