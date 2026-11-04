"""Fill Cursor prompts and subagent answers from the local agent transcript.

Headless runs and Task children do not put the prompt or the final answer on
the hook payload. The transcript next to the session does. It has no tokens
and no model, so this module does not create llm spans.
"""

from __future__ import annotations

import json
from pathlib import Path

from judgment_core.rows import records as _records

_EOS = "<|eos|>"


def _text(row: dict) -> str:
    message = row.get("message") if isinstance(row.get("message"), dict) else row
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(message.get("text") or "")
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("text") and block.get("type") not in ("tool_use", "tool_result", "image"):
            # A finished subagent transcript can end with a bare end-of-sequence token.
            if str(block["text"]).strip() != _EOS:
                parts.append(str(block["text"]))
    return "\n".join(parts)


def _role(row: dict) -> str:
    if row.get("role"):
        return str(row["role"])
    message = row.get("message")
    if isinstance(message, dict) and message.get("role"):
        return str(message["role"])
    return ""


def _unwrap_prompt(text: str) -> str:
    import re

    query = re.search(r"<user_query>\s*(.*?)\s*</user_query>", text, re.DOTALL)
    if query:
        text = query.group(1)
    text = re.sub(r"<timestamp>.*?</timestamp>", "", text, flags=re.DOTALL)
    text = re.sub(r"</?user_query>", "", text)
    return text.strip()


def _turns(rows: list[dict]) -> list[tuple[str, str, list[str]]]:
    """Each user message starts a turn. The answer is the last assistant text before the next user."""
    turns: list[tuple[str, str, list[str]]] = []
    prompt = ""
    texts: list[str] = []
    for row in rows:
        role = _role(row)
        if role == "user":
            if prompt or texts:
                turns.append((prompt, texts[-1] if texts else "", list(texts)))
            prompt = _unwrap_prompt(_text(row))
            texts = []
        elif role == "assistant":
            text = _text(row)
            if text:
                texts.append(text)
    if prompt or texts:
        turns.append((prompt, texts[-1] if texts else "", list(texts)))
    return turns


def _current_turn(rows: list[dict]) -> tuple[str, str]:
    turns = _turns(rows)
    if not turns:
        return "", ""
    prompt, answer, _texts = turns[-1]
    return prompt, answer


def _answer_for(rows: list[dict], hint: str, current: str) -> str:
    """The final assistant text of the turn this hook output belongs to."""
    turns = _turns(rows)
    if hint:
        for prompt, answer, _texts in reversed(turns):
            if prompt and (hint == prompt or hint in prompt or prompt in hint):
                return answer
    if current:
        for _prompt, answer, texts in turns:
            if current in texts:
                return answer
    return ""


def _texts_for(rows: list[dict], hint: str) -> list[str]:
    """Every assistant text of the turn whose prompt matches ``hint``, in order."""
    for prompt, _answer, texts in reversed(_turns(rows)):
        if hint and prompt and (hint == prompt or hint in prompt or prompt in hint):
            return texts
    return []


def _subagent_file(main: Path | None, agent_id: str, explicit: str) -> Path | None:
    if explicit:
        path = Path(explicit)
        if path.is_file():
            return path
    if main is None:
        return None
    candidates = [
        main.parent / "subagents" / f"{agent_id}.jsonl",
        main.parent / "subagents" / f"agent-{agent_id}.jsonl",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def enrich_cursor_events(events: list[dict]) -> list[dict]:
    source = events
    if not any(event.get("harness") == "cursor" for event in source):
        return events
    path_value = next((event.get("transcript_path") for event in source if event.get("transcript_path") and not event.get("agent_id")), "")
    main = Path(str(path_value)) if path_value else None
    rows = _records(main) if main is not None and main.is_file() else []
    prompt, answer = _current_turn(rows)
    # A resent turn reads a transcript that may already hold later turns. Match its own prompt.
    turn_prompt = next((str(event["prompt"]) for event in source if event.get("event") == "turn_start" and event.get("prompt") and not event.get("agent_id")), "")
    if turn_prompt:
        answer = _answer_for(rows, turn_prompt, "")
    # A background wake is folded into the turn that launched it and stops again.
    # Each stop ends one transcript turn, so the nth stop takes the nth turn from this prompt.
    turns = _turns(rows)
    first_turn = next((index for index in reversed(range(len(turns))) if turn_prompt and turns[index][0] and (turn_prompt in turns[index][0] or turns[index][0] in turn_prompt)), None)
    stop = 0
    filled = []
    saw_prompt = bool(turn_prompt)
    for event in events:
        item = dict(event)
        hint = str(item.get("prompt") or "") or turn_prompt
        if prompt and not saw_prompt and item.get("event") in ("turn_end", "model", "thought", "tool") and not item.get("agent_id") and not item.get("prompt"):
            item["prompt"] = prompt
            hint = prompt
            saw_prompt = True
        if item.get("event") == "turn_end" and not item.get("agent_id") and first_turn is not None and stop > 0 and first_turn + stop < len(turns):
            woken, woken_answer, woken_texts = turns[first_turn + stop]
            if not item.get("prompt"):
                item["prompt"] = woken
            if woken_answer:
                item["output"] = woken_answer
            # Text the model wrote before the answer, such as "I'll run that command", came before its tools.
            item["preamble"] = woken_texts[:-1]
            stop += 1
        elif item.get("event") == "turn_end" and not item.get("agent_id"):
            stop += 1
            chosen = _answer_for(rows, hint, str(item.get("output") or ""))
            if chosen and not item.get("output"):
                item["output"] = chosen
            elif chosen and str(item.get("output") or "") != chosen:
                item["output"] = chosen
            item["preamble"] = _texts_for(rows, hint)[:-1]
        agent_id = str(item.get("agent_id") or "")
        if agent_id and item.get("event") in ("subagent_end", "subagent_start") and not item.get("output"):
            child = _subagent_file(main, agent_id, str(item.get("agent_transcript_path") or ""))
            child_rows = _records(child) if child is not None else []
            child_prompt, child_answer = _current_turn(child_rows)
            if child_answer and item.get("event") == "subagent_end":
                item["output"] = child_answer
            if not item.get("prompt") and child_prompt:
                item["prompt"] = child_prompt
        filled.append(item)
    filled.extend(_close_unbound_children(filled, source, main))
    if answer and not any(item.get("event") == "turn_end" and item.get("output") and not item.get("agent_id") for item in filled):
        seed = next((item for item in filled if item.get("session_id") and item.get("turn_id")), None)
        if seed is not None:
            filled.append(
                {
                    "v": 1,
                    "harness": "cursor",
                    "event": "turn_end",
                    "session_id": seed.get("session_id"),
                    "turn_id": seed.get("turn_id"),
                    "cwd": seed.get("cwd"),
                    "project_root": seed.get("project_root"),
                    "environment": seed.get("environment") or "production",
                    "output": answer,
                    "ts": seed.get("ts"),
                }
            )
    return filled


def _close_unbound_children(filled: list[dict], source: list[dict], main: Path | None) -> list[dict]:
    """A failed Task reports no agentId, so its child never gets a subagent_end.

    The child still ran and wrote its own transcript. Its prompt and last answer come from there.
    """
    ended = {str(event.get("agent_id")) for event in source if event.get("event") == "subagent_end" and event.get("agent_id")}
    last: dict[str, dict] = {}
    for event in filled:
        agent_id = str(event.get("agent_id") or "")
        if event.get("harness") == "cursor" and agent_id and agent_id not in ended:
            last[agent_id] = event
    closing = []
    for agent_id, event in last.items():
        child = _subagent_file(main, agent_id, "")
        child_prompt, child_answer = _current_turn(_records(child)) if child is not None else ("", "")
        if not child_prompt and not child_answer:
            continue
        closing.append(
            {
                "v": 1,
                "harness": "cursor",
                "event": "subagent_end",
                "session_id": event.get("session_id"),
                "turn_id": event.get("turn_id"),
                "cwd": event.get("cwd"),
                "project_root": event.get("project_root"),
                "environment": event.get("environment") or "production",
                "agent_id": agent_id,
                "agent_type": "subagent",
                "prompt": child_prompt,
                "output": child_answer,
                "ts": event.get("ts"),
            }
        )
    return closing
