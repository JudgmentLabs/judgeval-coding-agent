"""Fill Claude Code model, tokens, and timing from the session transcript.

The hook payload for a main turn has no token counts. The transcript does,
and it is read here, after the turn has been spooled, not inside the hook.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from judgment_core.rows import records as _records


def _tool_uses(message: dict) -> list[dict]:
    content = message.get("content")
    if not isinstance(content, list):
        return []
    calls = []
    for block in content:
        if not isinstance(block, dict) or block.get("type") != "tool_use":
            continue
        calls.append(
            {
                "id": str(block.get("id") or ""),
                "name": str(block.get("name") or "tool"),
                "arguments": block.get("input") if block.get("input") is not None else {},
            }
        )
    return calls


def _blocks(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts)


def _result_text(content: object) -> str:
    """A tool_result block's content as the model read it. Images become "[image]"."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "image":
            parts.append("[image]")
        elif isinstance(block, dict) and block.get("text"):
            parts.append(str(block["text"]))
    return "\n".join(parts)


def _tool_results(rows: list[dict]) -> dict[str, str]:
    """Every tool_result in a transcript, by tool_use id. A blocked call has one too."""
    found = {}
    for row in rows:
        content = _message(row).get("content") if _role(row) == "user" else None
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result" and block.get("tool_use_id"):
                found[str(block["tool_use_id"])] = _result_text(block.get("content"))
    return found


def _attach_results(groups: list[dict], results: dict[str, str]) -> None:
    for group in groups:
        group["tool_results"] = [
            {"id": call["id"], "response": results[call["id"]]} for call in group.get("tool_calls") or [] if call.get("id") in results
        ]


def _usage(message: dict) -> dict | None:
    raw = message.get("usage")
    if not isinstance(raw, dict):
        return None
    found = {}
    for key in (
        "input_tokens",
        "output_tokens",
        "cache_read_input_tokens",
        "cache_creation_input_tokens",
    ):
        if raw.get(key) is not None:
            found[key] = raw[key]
    return found or None


def _role(row: dict) -> str:
    if row.get("type") in ("user", "assistant"):
        return str(row["type"])
    # Other row types, such as api-request-blob, copy request messages. They are not the conversation.
    if row.get("type"):
        return ""
    message = row.get("message")
    if isinstance(message, dict) and message.get("role") in ("user", "assistant"):
        return str(message["role"])
    return ""


def _is_prompt(row: dict) -> bool:
    """A user message. Tool results and compaction summaries are user rows too."""
    return _role(row) == "user" and not _is_tool_result(row) and not row.get("isCompactSummary")


def _ends_turn(row: dict, prompt_id: str) -> bool:
    """Rows of one prompt share its promptId, including messages injected mid-turn. Without ids, a meta row is not a prompt."""
    if not _is_prompt(row):
        return False
    if prompt_id and row.get("promptId"):
        return row["promptId"] != prompt_id
    return not row.get("isMeta")


def _message(row: dict) -> dict:
    message = row.get("message")
    return message if isinstance(message, dict) else {}


def _user_text(row: dict) -> str:
    return _blocks(_message(row))


def _is_tool_result(row: dict) -> bool:
    """Tool results are stored as user rows. They are not the next prompt."""
    if _role(row) != "user":
        return False
    content = _message(row).get("content")
    if isinstance(content, list) and content:
        saw = False
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                return False
            saw = True
        return saw
    return row.get("toolUseResult") is not None and not _user_text(row)


def _message_id(row: dict) -> str:
    message = _message(row)
    mid = message.get("id") or message.get("message_id") or ""
    return str(mid) if mid else ""


def _request_id(row: dict) -> str:
    message = _message(row)
    for source in (row, message):
        if not isinstance(source, dict):
            continue
        for key in ("requestId", "request_id"):
            if source.get(key):
                return str(source[key])
    return ""


def _group_keys(row: dict) -> list[str]:
    keys = []
    request_id = _request_id(row)
    message_id = _message_id(row)
    if request_id:
        keys.append("req:" + request_id)
    if message_id:
        keys.append("msg:" + message_id)
    return keys


def _extend_calls(group: dict, calls: list[dict]) -> None:
    existing = group.setdefault("tool_calls", [])
    seen = {str(call.get("id")) for call in existing if call.get("id")}
    for call in calls:
        call_id = str(call.get("id") or "")
        if call_id and call_id in seen:
            continue
        existing.append(call)
        if call_id:
            seen.add(call_id)


def _agent_of(row: dict) -> str:
    side = row.get("isSidechain") is True or row.get("is_sidechain") is True
    explicit = row.get("agentId") or row.get("agent_id") or ""
    if explicit:
        return str(explicit)
    if side:
        return "sidechain"
    return ""


def _timestamp_ms(row: dict) -> int | None:
    raw = row.get("timestamp")
    if raw is None:
        return None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = int(raw)
        if value < 10_000_000_000:
            value *= 1000
        return value
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _drop_internal_forks(events: list[dict]) -> list[dict]:
    """Drop SubagentStop for Claude's internal forks.

    Prompt suggestions and similar forks emit SubagentStop with an empty
    agent_type and no SubagentStart or Agent tool. A user-visible subagent
    has a type, or a start event, before that stop arrives.
    """
    launched: set[str] = set()
    for event in events:
        agent_id = str(event.get("agent_id") or "")
        if not agent_id:
            continue
        if event.get("event") == "subagent_start" or event.get("agent_type"):
            launched.add(agent_id)
    kept = []
    for event in events:
        agent_id = str(event.get("agent_id") or "")
        if event.get("event") == "subagent_end" and not event.get("agent_type") and agent_id not in launched:
            continue
        kept.append(event)
    return kept


def _file_sig(path: Path | None) -> tuple[int, int] | None:
    if path is None or not path.is_file():
        return None
    try:
        stat = path.stat()
    except OSError:
        return None
    return (getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)), stat.st_size)


def _contains_assistant(rows: list[dict], needle: str) -> bool:
    if not needle:
        return True
    if _contains(rows, needle):
        return True
    parts = [text for row in rows if _role(row) == "assistant" and (text := _blocks(_message(row)))]
    return needle in "\n".join(parts)


def _read_settled(resolve, final: str, attempts: int, pause: float) -> list[dict]:
    """Read until the final assistant text is present and the file has stopped growing.

    A subagent file often gains its last request just after SubagentStop. Stopping
    at the first non-empty read drops that request, and nothing flushes again.
    """
    previous = None
    rows: list[dict] = []
    path = None
    for attempt in range(max(attempts, 1)):
        path = resolve()
        rows = _records(path) if path is not None else []
        sig = _file_sig(path)
        groups = _assistant_after(rows, "", include_agents=True) if rows else []
        quiet = sig is not None and sig == previous
        if groups and _contains_assistant(rows, final) and quiet:
            return rows
        previous = sig
        if attempt + 1 < attempts:
            time.sleep(pause)
    if path is not None and _file_sig(path) != previous:
        time.sleep(pause)
        rows = _records(path)
    return rows


def _subagent_transcript(main: Path | None, agent_id: str, explicit: str) -> Path | None:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    if main is not None:
        candidates.append(main.parent / "subagents" / f"agent-{agent_id}.jsonl")
        candidates.append(main.parent / main.stem / "subagents" / f"agent-{agent_id}.jsonl")
    for path in candidates:
        if path.is_file():
            return path
    return None


def _model_event(seed: dict, assistant: dict, *, prompt: str) -> dict:
    item = {
        "v": 1,
        "harness": "claude-code",
        "provider": "anthropic",
        "event": "model",
        "session_id": seed.get("session_id"),
        "turn_id": seed.get("turn_id"),
        "cwd": seed.get("cwd"),
        "project_root": seed.get("project_root"),
        "environment": seed.get("environment") or "production",
        "model": assistant["model"],
        "prompt": prompt,
        "output": assistant["text"],
        "usage": assistant["usage"],
        "message_id": assistant["id"],
    }
    if assistant.get("tool_calls"):
        item["tool_calls"] = assistant["tool_calls"]
    if assistant.get("tool_results"):
        item["tool_results"] = assistant["tool_results"]
    if assistant.get("agent_id"):
        item["agent_id"] = assistant["agent_id"]
    start_ms = assistant.get("start_ms")
    end_ms = assistant.get("end_ms")
    if start_ms is not None:
        item["ts"] = start_ms
    if start_ms is not None and end_ms is not None and end_ms > start_ms:
        item["end_ts"] = end_ms
    return item


def _task_text(rows: list[dict], turn_id: str) -> str:
    """The invocation's task. A resumed subagent's file holds one task per turn, each stamped with that turn's prompt id."""
    tasks = [row for row in rows if _role(row) == "user" and not _is_tool_result(row)]
    own = next((row for row in tasks if turn_id and _row_turn_id(row) == turn_id), None)
    row = own or (tasks[0] if tasks else None)
    return _user_text(row) if row else ""


def enrich_claude_events(
    events: list[dict],
    *,
    attempts: int = 4,
    pause: float = 0.25,
) -> list[dict]:
    if not any(event.get("harness") == "claude-code" for event in events):
        return events
    events = _drop_internal_forks(events)
    source = events
    path_value = next((event.get("transcript_path") for event in source if event.get("transcript_path")), "")
    path = Path(str(path_value)) if path_value else None
    prompt = next(
        (str(event.get("prompt")) for event in source if event.get("event") == "turn_start" and event.get("prompt")),
        "",
    )
    final = next(
        (str(event.get("output")) for event in reversed(source) if event.get("event") == "turn_end" and event.get("output")),
        "",
    )
    anchor = next((event.get("ts") for event in source if event.get("event") == "turn_start" and event.get("ts") is not None), None)
    end_ms = next((event.get("ts") for event in reversed(source) if event.get("event") == "turn_end" and event.get("ts") is not None), None)
    turn_id = next((str(event.get("turn_id")) for event in source if event.get("turn_id")), "")
    session_id = next((str(event.get("session_id")) for event in source if event.get("session_id")), "")
    if turn_id and turn_id == session_id:
        turn_id = ""
    added: list[dict] = []
    results: dict[str, str] = {}
    seed = events[0] if events else (source[0] if source else {})
    # A late flush rebuilds the turn, so the main requests have to be in that
    # payload too. Otherwise the resent root starts at the hook time and the
    # model span already stored still starts at the earlier transcript time.
    if path is not None and path.is_file():
        rows: list[dict] = []
        since = min((_as_int(event.get("ts")) or 0 for event in source), default=0) or None
        for attempt in range(max(attempts, 1)):
            rows = _records(path, since)
            found = _assistant_after(rows, prompt, anchor_ms=_as_int(anchor), end_ms=_as_int(end_ms), turn_id=turn_id)
            if found and (not final or _contains(rows, final)):
                break
            if attempt + 1 < attempts:
                time.sleep(pause)
        assistants = [
            item
            for item in _assistant_after(rows, prompt, anchor_ms=_as_int(anchor), end_ms=_as_int(end_ms), turn_id=turn_id)
            if not item.get("agent_id")
        ]
        results.update(_tool_results(rows))
        _attach_results(assistants, {**_agent_answers(assistants, source), **results})
        first_main = True
        for assistant in assistants:
            text = prompt if first_main else ""
            first_main = False
            added.append(_model_event(seed, assistant, prompt=text))
        if added and not added[-1].get("output") and final:
            added[-1]["output"] = final

    seen_agents: set[str] = set()
    for event in source:
        agent_id = str(event.get("agent_id") or "")
        if not agent_id or agent_id in seen_agents:
            continue
        if event.get("event") not in ("subagent_start", "subagent_end", "tool", "model"):
            continue
        seen_agents.add(agent_id)
        explicit = next(
            (
                str(item.get("agent_transcript_path"))
                for item in source
                if str(item.get("agent_id") or "") == agent_id and item.get("agent_transcript_path")
            ),
            "",
        )
        directory_ready = bool(explicit) or bool(
            path and ((path.parent / "subagents").is_dir() or (path.parent / path.stem / "subagents").is_dir())
        )
        final_output = next(
            (
                str(item.get("output"))
                for item in reversed(events)
                if item.get("event") == "subagent_end"
                and str(item.get("agent_id") or "") == agent_id
                and item.get("output")
            ),
            "",
        )
        if final_output and (directory_ready or explicit):
            sub_rows = _read_settled(
                lambda: _subagent_transcript(path, agent_id, explicit),
                final_output,
                attempts,
                pause,
            )
        else:
            sub_path = _subagent_transcript(path, agent_id, explicit)
            sub_rows = _records(sub_path) if sub_path is not None else []
        groups = _assistant_after(sub_rows, "", turn_id=turn_id, include_agents=True) if sub_rows else []
        if not groups:
            continue
        events = [
            item
            for item in events
            if not (item.get("event") == "model" and str(item.get("agent_id") or "") == agent_id)
        ]
        task = _task_text(sub_rows, turn_id)
        # SubagentStart carries no prompt, and a resume through SendMessage has no Agent call that would.
        events = [
            {**item, "prompt": task}
            if item.get("event") == "subagent_start" and str(item.get("agent_id") or "") == agent_id and not item.get("prompt")
            else item
            for item in events
        ]
        sub_results = _tool_results(sub_rows)
        results.update(sub_results)
        _attach_results(groups, sub_results)
        for index, assistant in enumerate(groups):
            assistant["agent_id"] = assistant.get("agent_id") or agent_id
            added.append(_model_event(seed, assistant, prompt=task if index == 0 else ""))
    if not added:
        return events
    return [_as_seen(event, results) for event in events] + added


def _as_seen(event: dict, results: dict[str, str]) -> dict:
    """A hook tool output is the harness's envelope. Keep the text the model read instead."""
    call_id = str(event.get("tool_call_id") or "")
    if event.get("event") != "tool" or call_id not in results:
        return event
    return {**event, "tool_output": results[call_id]}


def _agent_answers(assistants: list[dict], source: list[dict]) -> dict[str, str]:
    """An Agent call with no tool_result yet still returned its subagent's answer. Match the two by prompt."""
    answers = {
        str(event.get("prompt")): str(event.get("output"))
        for event in source
        if event.get("event") == "subagent_end" and event.get("prompt") and event.get("output")
    }
    found = {}
    for assistant in assistants:
        for call in assistant.get("tool_calls") or []:
            arguments = call.get("arguments") if isinstance(call.get("arguments"), dict) else {}
            if call.get("name") in ("Agent", "Task") and arguments.get("prompt") in answers:
                found[call["id"]] = answers[arguments["prompt"]]
    return found


def _as_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _contains(rows: list[dict], needle: str) -> bool:
    if not needle:
        return True
    for row in rows:
        if needle in _blocks(_message(row)):
            return True
    return False


def _row_turn_id(row: dict) -> str:
    for key in ("promptId", "prompt_id", "uuid"):
        if row.get(key):
            return str(row[key])
    return ""


def _prompt_matches(row: dict, prompt: str, turn_id: str) -> bool:
    if turn_id and _row_turn_id(row) == turn_id:
        return True
    return bool(prompt) and prompt in _user_text(row)


def _choose_user_row(rows: list[dict], prompt: str, anchor_ms: int | None, end_ms: int | None, turn_id: str) -> tuple[int, int | None]:
    """Pick the user row for this turn. The same prompt text can occur more than once."""
    matches: list[tuple[int, int | None]] = []
    for index, row in enumerate(rows):
        if not _is_prompt(row):
            continue
        if turn_id and _row_turn_id(row) == turn_id:
            return index, _timestamp_ms(row)
        if prompt and prompt in _user_text(row):
            matches.append((index, _timestamp_ms(row)))
    if not matches:
        return 0, None
    dated = [(index, stamp) for index, stamp in matches if stamp is not None]
    if anchor_ms is not None and dated:
        slack = 15_000
        window = []
        for index, stamp in dated:
            if end_ms is not None and stamp > end_ms + slack:
                continue
            if stamp + slack >= anchor_ms:
                window.append((index, stamp))
        pool = window or dated
        return min(pool, key=lambda item: (abs(item[1] - anchor_ms), -item[0]))
    return matches[-1]


def _assistant_after(
    rows: list[dict],
    prompt: str,
    *,
    anchor_ms: int | None = None,
    end_ms: int | None = None,
    turn_id: str = "",
    include_agents: bool = False,
) -> list[dict]:
    start = 0
    boundary = None
    prompt_id = turn_id
    if (prompt or turn_id) and any(
        _is_prompt(row) and _prompt_matches(row, prompt, turn_id) for row in rows
    ):
        index, boundary = _choose_user_row(rows, prompt, anchor_ms, end_ms, turn_id)
        start = index + 1
        prompt_id = turn_id or str(rows[index].get("promptId") or "")
    groups: list[dict] = []
    index_by_id: dict[str, int] = {}
    for row in rows[start:]:
        if _ends_turn(row, prompt_id):
            if groups:
                break
            stamp = _timestamp_ms(row)
            if stamp is not None:
                boundary = stamp
            continue
        if _is_tool_result(row):
            stamp = _timestamp_ms(row)
            if stamp is not None:
                boundary = stamp
            continue
        if _role(row) != "assistant":
            continue
        message = _message(row)
        usage = _usage(message)
        model = str(message.get("model") or row.get("model") or "")
        text = _blocks(message)
        calls = _tool_uses(message)
        if not usage and not model and not text and not calls:
            continue
        agent = _agent_of(row)
        if agent and not include_agents:
            continue
        stamp = _timestamp_ms(row)
        keys = _group_keys(row)
        found = next((index_by_id[key] for key in keys if key in index_by_id), None)
        # Parallel tool calls are sometimes separate rows with no shared id.
        # They stay one request until a later row brings its own usage.
        if found is None and not keys and groups and (not usage or not groups[-1].get("usage")):
            found = len(groups) - 1
        if found is not None:
            group = groups[found]
            if text:
                group["text"] = f"{group['text']}\n{text}" if group["text"] else text
            if usage and not group["usage"]:
                group["usage"] = usage
            if model and not group["model"]:
                group["model"] = model
            if calls:
                _extend_calls(group, calls)
            for key in keys:
                index_by_id[key] = found
            if stamp is not None:
                group["end_ms"] = stamp
                if group["start_ms"] is None or (boundary is not None and boundary < group["start_ms"]):
                    group["start_ms"] = boundary if boundary is not None and boundary < stamp else group["start_ms"]
                boundary = stamp
            continue
        start_ms = boundary if boundary is not None and (stamp is None or boundary < stamp) else stamp
        group = {
            "id": _message_id(row) or _request_id(row),
            "model": model,
            "text": text,
            "usage": usage,
            "start_ms": start_ms,
            "end_ms": stamp if stamp is not None else start_ms,
            "agent_id": agent,
            "tool_calls": calls,
        }
        groups.append(group)
        for key in keys:
            index_by_id[key] = len(groups) - 1
        if stamp is not None:
            boundary = stamp
    return groups
