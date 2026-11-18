"""Codex model requests, read from the rollout file.

Each model request writes one ``token_count`` event whose ``info`` holds
``last_token_usage``. That event is the only llm span source, so llm spans map
one-to-one to requests. Hook ``model`` events only stand in when no rollout can
be read. The rows before a ``token_count`` belong to the request it closes:
assistant text, calls, and the calls' outputs. ``item_completed`` rows repeat
``response_item`` rows and are skipped, except ``SubAgentActivity``, which ties a
``spawn_agent`` call id to the child thread.

With ``multi_agent_v2`` the spawn message and the child's ``NEW_TASK`` payload
are encrypted. Only the task name is plain, so the child agent span says
``Subagent task: <name>`` and encrypted text reads ``[encrypted]``.

``transcript_path`` is not a stable interface. See
https://developers.openai.com/codex/hooks.
"""

from __future__ import annotations

import html
import json
import re
import time
from datetime import datetime
from pathlib import Path

from judgment_core.rows import records as _records

ENCRYPTED = "[encrypted]"
_CALLS = ("function_call", "custom_tool_call", "local_shell_call")
_DELEGATED = re.compile(r"<codex_delegation>.*?<input>(.*?)</input>", re.S)


def _payload(row: dict) -> dict:
    return row["payload"] if isinstance(row.get("payload"), dict) else {}


def _stamp(row: dict) -> int | None:
    value = row.get("timestamp")
    if not isinstance(value, str) or not value:
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _redact(value: object) -> object:
    if isinstance(value, str):
        return ENCRYPTED if value.startswith("gAAAA") else value
    if isinstance(value, dict):
        return {key: _redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _content_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict):
            if block.get("type") == "encrypted_content":
                parts.append(ENCRYPTED)
            elif block.get("text"):
                parts.append(str(block["text"]))
    return "".join(parts)


def _arguments(value: object) -> object:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return {"input": value}
    return _redact(value if value is not None else {})


def _usage(row: dict) -> dict | None:
    payload = _payload(row)
    if row.get("type") == "token_usage_record":
        usage = payload.get("usage")
    elif payload.get("type") == "token_count":
        usage = (payload.get("info") or {}).get("last_token_usage") if isinstance(payload.get("info"), dict) else None
    else:
        return None
    return dict(usage) if isinstance(usage, dict) else None


def _task_text(row: dict) -> str:
    """The user message, or the ``NEW_TASK`` mail, that opened a turn."""
    payload = _payload(row)
    if row.get("type") != "response_item":
        return ""
    if payload.get("type") == "message" and payload.get("role") == "user":
        text = _content_text(payload.get("content"))
        return "" if text.lstrip().startswith(("<environment_context>", "# AGENTS.md")) else text
    if payload.get("type") == "agent_message":
        text = _content_text(payload.get("content"))
        return text if "NEW_TASK" in text else ""
    if payload.get("type") == "function_call_output" and payload.get("name") == "create_thread":
        # A thread another thread created opens with the request, not a user message, and runs no prompt hook.
        found = _DELEGATED.search(_content_text(payload.get("output")))
        return html.unescape(found.group(1).strip()) if found else ""
    return ""


def _root_turn(payload: dict) -> str:
    attribution = payload.get("turn_attribution") if isinstance(payload.get("turn_attribution"), dict) else {}
    return str(payload.get("root_turn_id") or attribution.get("root_turn_id") or "")


def _turn_rows(rows: list[dict], turn_id: str, prompt: str, root_turn_id: str = "") -> list[dict]:
    """Rows of one turn. ``task_started`` and ``turn_context`` open it. A resumed or forked rollout holds older turns.

    With ``root_turn_id``, the rows of every turn a child ran for that parent turn. A child given a second task runs it as another turn.
    """
    turns: list[tuple[str, str, list[dict]]] = []
    for row in rows:
        payload = _payload(row)
        own = str(payload.get("turn_id") or "")
        if own and (row.get("type") == "turn_context" or payload.get("type") == "task_started") and (not turns or turns[-1][0] != own):
            turns.append((own, _root_turn(payload), []))
        if turns:
            turns[-1][2].append(row)
    if not turns:
        return rows
    if root_turn_id:
        # A forked child starts with a copy of the parent's turn. That copy is not the child's work.
        mine = [row for own, root, chunk in turns if root == root_turn_id and own != root_turn_id for row in chunk]
        if mine:
            return mine
    turns = [(own, chunk) for own, _root, chunk in turns]
    for own, chunk in turns:
        if turn_id and own == turn_id:
            return chunk
    if prompt:
        for _own, chunk in reversed(turns):
            if any(prompt in _task_text(row) for row in chunk):
                return chunk
    return turns[-1][1]


def _new_step() -> dict:
    return {"text": "", "tool_calls": [], "tool_results": [], "received": [], "usage": None, "start_ms": None, "end_ms": None, "call_ms": {}}


def read_thread(path: Path | None, *, turn_id: str = "", prompt: str = "", root_turn_id: str = "", since_ms: int | None = None) -> dict:
    """The model requests of one turn in one rollout.

    ``task`` is the message that opened the turn. ``task_name`` is the
    ``agent_path`` leaf of a child rollout. ``model`` comes from ``turn_context``.
    ``spawned`` maps a spawn call id to the child thread id. ``complete`` turns
    true at ``task_complete``.
    """
    rows = _records(path, since_ms)
    meta = next((_payload(row) for row in rows if row.get("type") == "session_meta"), {})
    agent_path = str(meta.get("agent_path") or "")
    chunk = _turn_rows(rows, turn_id, prompt, root_turn_id)
    model = next((str(_payload(row)["model"]) for row in chunk if row.get("type") == "turn_context" and _payload(row).get("model")), "")
    thread = {"task": "", "task_name": agent_path.rsplit("/", 1)[-1], "model": model, "steps": [], "spawned": {}, "complete": False}
    by_record = not any(_usage(row) is not None and row.get("type") != "token_usage_record" for row in chunk)
    step = _new_step()
    last_input = None
    for row in chunk:
        payload = _payload(row)
        kind = str(payload.get("type") or "")
        stamp = _stamp(row)
        is_record = row.get("type") == "token_usage_record"
        usage = _usage(row)
        if usage is not None and is_record == by_record:
            step["usage"] = usage
            step["end_ms"] = step["end_ms"] or stamp
            step["start_ms"] = step["start_ms"] if step["start_ms"] is not None else last_input
            thread["steps"].append(step)
            step = _new_step()
            last_input = stamp
        elif is_record:
            # The request finished here. Its token_count waits for the tool outputs.
            step["end_ms"] = step["end_ms"] or stamp
        elif kind == "task_complete":
            thread["complete"] = True
        elif kind == "task_started":
            last_input = stamp
        elif kind == "item_completed":
            item = payload.get("item") if isinstance(payload.get("item"), dict) else {}
            if item.get("type") == "SubAgentActivity" and item.get("kind") == "started" and item.get("agent_thread_id"):
                thread["spawned"][str(item.get("id") or "")] = str(item["agent_thread_id"])
        elif row.get("type") != "response_item":
            continue
        elif _task_text(row):
            thread["task"] = thread["task"] or _task_text(row)
            last_input = stamp
        elif kind in ("function_call_output", "custom_tool_call_output"):
            call_id = str(payload.get("call_id") or "")
            output = payload.get("output")
            owner = next((item for item in (step, *reversed(thread["steps"])) if call_id in item["call_ms"]), step)
            owner["tool_results"].append({"id": call_id, "response": _content_text(output) if isinstance(output, list) else (output if output is not None else "")})
            if call_id in owner["call_ms"]:
                owner["call_ms"][call_id] = (owner["call_ms"][call_id][0], stamp)
            last_input = stamp
        elif kind == "agent_message":
            # A child's FINAL_ANSWER, or another agent's mail. The next request reads it.
            text = _content_text(payload.get("content"))
            if text:
                step["received"].append(text)
            last_input = stamp
        elif kind in ("message", "reasoning", *_CALLS):
            if step["start_ms"] is None:
                step["start_ms"] = last_input if last_input is not None else stamp
            if kind == "message" and payload.get("role") == "assistant":
                text = _content_text(payload.get("content"))
                if text:
                    step["text"] = f"{step['text']}\n{text}" if step["text"] else text
            elif kind in _CALLS:
                call_id = str(payload.get("call_id") or payload.get("id") or "")
                if kind == "local_shell_call":
                    action = payload.get("action") if isinstance(payload.get("action"), dict) else {}
                    call = {"id": call_id, "name": "local_shell", "arguments": action}
                else:
                    raw = payload.get("arguments") if kind == "function_call" else payload.get("input")
                    call = {"id": call_id, "name": str(payload.get("name") or "tool"), "arguments": _arguments(raw)}
                step["tool_calls"].append(call)
                step["call_ms"][call_id] = (stamp, stamp)
    # Rows after the last token_count belong to a request still running. A later send picks it up.
    return thread


def _read(path: Path | None, turn_id: str, prompt: str, attempts: int, pause: float, want: int, root_turn_id: str = "", since_ms: int | None = None) -> dict:
    thread = read_thread(path, turn_id=turn_id, prompt=prompt, root_turn_id=root_turn_id, since_ms=since_ms)
    for _ in range(max(attempts, 1) - 1):
        if thread["complete"] and len(thread["steps"]) >= want:
            break
        time.sleep(pause)
        thread = read_thread(path, turn_id=turn_id, prompt=prompt, root_turn_id=root_turn_id, since_ms=since_ms)
    return thread


def _model_events(thread: dict, seed: dict, agent_id: str, model: str) -> list[dict]:
    model = thread["model"] or model
    events = []
    for step in thread["steps"]:
        usage = dict(step["usage"]) if step["usage"] else None
        if usage is not None:
            usage["input_includes_cache"] = True
        item = {
            "v": 1,
            "harness": "codex",
            "event": "model",
            "session_id": seed.get("session_id"),
            "turn_id": seed.get("turn_id"),
            "model": model,
            "provider": "openai",
            "output": step["text"],
            "tool_calls": step["tool_calls"],
            "tool_results": step["tool_results"],
            "usage": usage,
        }
        if step["received"]:
            item["received"] = step["received"]
        if agent_id:
            item["agent_id"] = agent_id
        end = step["end_ms"] if step["end_ms"] is not None else step["start_ms"]
        if step["start_ms"] is not None:
            item["start_ts"] = step["start_ms"]
        if end is not None:
            item["ts"] = item["end_ts"] = end
        events.append(item)
    return events


def _child_path(source: list[dict], agent_id: str, parent: Path | None) -> Path | None:
    for event in source:
        if str(event.get("agent_id") or "") == agent_id and event.get("agent_transcript_path"):
            return Path(str(event["agent_transcript_path"]))
    # SubagentStart carries no path, and a background child may not have stopped yet.
    if parent is not None and parent.parent.is_dir():
        found = sorted(parent.parent.glob(f"*{agent_id}.jsonl"))
        if found:
            return found[0]
    return None


def _swap_models(events: list[dict], agent_id: str, fresh: list[dict]) -> list[dict]:
    """Put rollout model events where the thread's first hook model event was."""

    def hook_model(event: dict) -> bool:
        return str(event.get("agent_id") or "") == agent_id and event.get("event") in ("model", "model_start")

    at = next((index for index, event in enumerate(events) if hook_model(event)), None)
    kept = [event for event in events if not hook_model(event)]
    if at is not None:
        at -= sum(1 for event in events[:at] if hook_model(event))
    elif agent_id:
        at = next((index + 1 for index, event in enumerate(kept) if str(event.get("agent_id") or "") == agent_id and event.get("event") == "subagent_start"), len(kept))
    else:
        at = len(kept)
    return kept[:at] + fresh + kept[at:]


def enrich_codex_events(
    events: list[dict],
    *,
    attempts: int = 20,
    pause: float = 0.1,
) -> list[dict]:
    source = events
    if not any(event.get("harness") == "codex" for event in source):
        return events
    main = [event for event in source if not event.get("agent_id")]
    seed = next((event for event in main if event.get("session_id") and event.get("turn_id")), {})
    path_value = next((event.get("transcript_path") for event in main if event.get("transcript_path")), "")
    parent_path = Path(str(path_value)) if path_value else None
    turn_id = str(seed.get("turn_id") or "")
    prompt = next((str(event.get("prompt")) for event in main if event.get("event") == "turn_start" and event.get("prompt")), "")
    model = next((str(event.get("model")) for event in main if event.get("model")), "")
    hook_models = sum(1 for event in main if event.get("event") == "model")
    since = min((int(event["ts"]) for event in main if isinstance(event.get("ts"), int)), default=None)
    parent = _read(parent_path, turn_id, prompt, attempts, pause, hook_models, since_ms=since) if parent_path else None

    out = [dict(event) for event in events]
    for event in out:
        if event.get("tool_input") is not None:
            event["tool_input"] = _redact(event["tool_input"])
    windows: dict[str, dict[str, tuple[int, int]]] = {}
    if parent and parent["steps"]:
        fresh = _model_events(parent, seed, "", model)
        if not prompt and parent["task"]:
            fresh[0]["prompt"] = parent["task"]
        out = _swap_models(out, "", fresh)
        windows[""] = {call_id: span for step in parent["steps"] for call_id, span in step["call_ms"].items()}

    spawned = parent["spawned"] if parent else {}
    names = {}
    for event in source:
        tool_input = event.get("tool_input")
        if event.get("event") == "tool" and "spawn_agent" in str(event.get("tool_name") or "") and isinstance(tool_input, dict):
            names[spawned.get(str(event.get("tool_call_id") or ""), "")] = str(tool_input.get("task_name") or "")
    agents = list(dict.fromkeys(str(event.get("agent_id")) for event in source if event.get("agent_id")))
    for agent_id in agents:
        mine = [event for event in source if str(event.get("agent_id") or "") == agent_id]
        stopped = any(event.get("event") == "subagent_end" for event in mine)
        want = sum(1 for event in mine if event.get("event") == "model")
        child_path = _child_path(source, agent_id, parent_path)
        child = _read(child_path, "", "", attempts if stopped else 1, pause, want, root_turn_id=turn_id) if child_path else None
        name = (child["task_name"] if child else "") or names.get(agent_id, "")
        asked = next((str(event["prompt"]) for event in mine if event.get("event") == "subagent_prompt" and event.get("prompt")), "")
        task = (child["task"] if child else "") or asked
        agent_type = next((str(event["agent_type"]) for event in mine if event.get("agent_type")), "subagent")
        if name:
            label = f"Subagent task: {name}"
        elif task and ENCRYPTED not in task:
            label = asked or task
        else:
            label = f"Subagent task: {agent_type}"
        for event in out:
            if str(event.get("agent_id") or "") == agent_id and event.get("event") in ("subagent_start", "subagent_end", "subagent_prompt"):
                event["prompt"] = label
                if name:
                    event["agent_name"] = name
        if child and child["steps"]:
            fresh = _model_events(child, seed, agent_id, next((str(event["model"]) for event in mine if event.get("model")), model))
            fresh[0]["prompt"] = task or label
            out = _swap_models(out, agent_id, fresh)
            windows[agent_id] = {call_id: span for step in child["steps"] for call_id, span in step["call_ms"].items()}

    starts = {(str(event.get("agent_id") or ""), str(event.get("tool_call_id") or "")): event.get("ts") for event in source if event.get("event") == "tool_start"}
    hooked = {str(event.get("tool_call_id") or "") for event in source if event.get("event") in ("tool", "tool_start")}
    for event in out:
        if event.get("event") != "tool":
            continue
        agent = str(event.get("agent_id") or "")
        started = starts.get((agent, str(event.get("tool_call_id") or ""))) or event.get("ts")
        for call_id, (begin, end) in windows.get(agent, {}).items():
            # A code-mode exec call runs its shell commands. Each command has its own hook.
            if call_id not in hooked and begin is not None and started is not None and begin <= int(started) <= (end or begin):
                event["parent_call_id"] = call_id
                break
    nested = {(str(event.get("agent_id") or ""), str(event.get("parent_call_id"))) for event in out if event.get("parent_call_id")}

    extra = []
    for event in out:
        if event.get("event") != "model" or "start_ts" not in event:
            continue
        agent = str(event.get("agent_id") or "")
        for call in event.get("tool_calls") or []:
            call_id = str(call.get("id") or "")
            if not call_id or call_id in hooked or (agent, call_id) in nested:
                continue
            hooked.add(call_id)
            result = next((item.get("response") for item in event.get("tool_results") or [] if str(item.get("id") or "") == call_id), "")
            begin, end = windows.get(agent, {}).get(call_id, (event["start_ts"], event.get("end_ts")))
            tool = {
                "v": 1,
                "harness": "codex",
                "event": "tool",
                "session_id": event.get("session_id"),
                "turn_id": event.get("turn_id"),
                "tool_name": call.get("name") or "tool",
                "tool_call_id": call_id,
                "tool_input": call.get("arguments"),
                "tool_output": result if result is not None else "",
                "start_ts": begin,
                "ts": end,
                "end_ts": end,
            }
            if agent:
                tool["agent_id"] = agent
            extra.append(tool)
    return out + extra
