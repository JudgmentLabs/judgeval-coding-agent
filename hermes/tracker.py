"""Hermes observer hooks -> judgment_core event records. Pure: no files, no network, no clock.

One call per hook: ``Tracker.step(hook, kwargs, now_ms)`` returns the records the hook produced,
each with the gate (the opt-in decision) of the turn it belongs to, and whether a sender should
start. Everything Hermes says is read through the tables below; the state machine is the only
place that remembers anything between hooks.

State machines (impossible states are listed so the code can refuse them):

  Turn (one per Hermes turn_id, main agent only)
      none --pre_llm_call--> OPEN --post_llm_call--> ANSWERED
      OPEN|ANSWERED --on_session_end--> closed      turn_end, interrupted unless Hermes says it completed
      OPEN|ANSWERED --pre_llm_call(other turn)--> closed (interrupted), then the new turn opens
      OPEN|ANSWERED --on_session_finalize|reset--> closed (interrupted), then session_end
      Impossible: a turn_end without a turn_start (an on_session_end with no open turn records nothing),
      two open turns in one session, an ANSWERED turn going back to OPEN.

  Gate (decided once, at turn start): None = untraced. The whole turn, its tool and model calls,
      and every child it starts, share it. ``/judgment off`` takes effect at the next turn, so a turn
      is never half recorded.

  Model call attempt (api_request_id): pending --post_api_request--> done (model record)
      pending --api_request_error--> error (model record, is_error; a retry is a new attempt id)
      pending --turn or child closes--> abandoned (model record, is_error)
      post/error without pre: recorded the same; Hermes passes started_at on both.

  Tool call (tool_call_id): pending --post_tool_call(status ok|error|blocked|cancelled)--> done
      pending --turn or child closes--> abandoned (tool record, is_error)
      post without pre (a call blocked before pre_tool_call): recorded, timed from duration_ms.

  Child (child_session_id): RUNNING --subagent_stop--> done (subagent_end)
      RUNNING --parent session finalize|reset--> done (subagent_end, interrupted)
      A child's events are folded into the turn that started it, under agent_id = its session id,
      even after that turn ended (Hermes runs top-level delegations in the background). A grandchild
      folds into the same root turn. A child of an untraced turn is remembered with no gate, so its
      own pre_llm_call never opens a turn.
"""

from __future__ import annotations

import enum
import json
import threading
from dataclasses import dataclass, field
from typing import Any, Callable

HARNESS = "hermes"
# The most text one field keeps. Spans cut at 64 KB anyway; this bounds what redaction has to scan.
TEXT_CAP = 128 * 1024
CUT_MARK = "… [truncated]"
OK_TOOL = "ok"
DONE_CHILD = ("completed", "complete", "success", "succeeded", "ok", "done")

# Hermes CanonicalUsage -> judgment_core usage keys. input_tokens there is already non-cached.
USAGE_KEYS = (
    ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"),
    ("cache_read_tokens", "cache_read_input_tokens"),
    ("cache_write_tokens", "cache_creation_input_tokens"),
    ("reasoning_tokens", "reasoning_tokens"),
)


class Phase(enum.Enum):
    OPEN = "open"
    ANSWERED = "answered"


@dataclass
class Thread:
    """One agent's calls in flight: tool_call_id or api_request_id -> start ms."""

    tools: dict = field(default_factory=dict)
    attempts: dict = field(default_factory=dict)


@dataclass
class Turn:
    session_id: str
    turn_id: str
    gate: Any
    phase: Phase = Phase.OPEN
    answer: str = ""
    model: str = ""
    thread: Thread = field(default_factory=Thread)


@dataclass
class Child:
    session_id: str  # the root session the child's events go to
    turn_id: str  # the root turn that started it
    agent_id: str  # the child's own session id
    gate: Any
    agent_type: str = "subagent"
    start: int = 0
    thread: Thread = field(default_factory=Thread)


@dataclass
class Step:
    records: list = field(default_factory=list)  # [(gate, record)]
    flush: bool = False


# ---- pure readers of Hermes kwargs ---------------------------------------------------------------


def cap(text: str) -> str:
    """At most TEXT_CAP characters. A cut drops the last partial word, so no cut-off secret survives it."""
    if len(text) <= TEXT_CAP:
        return text
    head = text[:TEXT_CAP]
    space = max(head.rfind(" "), head.rfind("\n"), head.rfind("\t"))
    if space > TEXT_CAP - 4096:
        head = head[:space]
    return head + CUT_MARK


def text_of(value: Any) -> str:
    """A message content as text: a string, or the text parts of a content list."""
    if value is None:
        return ""
    if isinstance(value, str):
        return cap(value)
    if isinstance(value, list):
        parts = []
        for part in value:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") in ("image", "image_url", "input_image"):
                    parts.append("[image]")
                elif isinstance(part.get("text"), str):
                    parts.append(part["text"])
        return cap("\n".join(p for p in parts if p))
    return cap(json.dumps(value, ensure_ascii=False, default=str))


def capped(value: Any, depth: int = 0) -> Any:
    """A JSON-shaped copy with every string capped. Unknown objects become their string."""
    if isinstance(value, str):
        return cap(value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if depth > 12:
        return "…"
    if isinstance(value, dict):
        return {str(k): capped(v, depth + 1) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [capped(v, depth + 1) for v in value]
    return cap(str(value))


def ms(seconds: Any, default: int) -> int:
    try:
        return int(float(seconds) * 1000) if seconds is not None else default
    except (TypeError, ValueError):
        return default


def usage_of(raw: Any) -> dict | None:
    if not isinstance(raw, dict):
        return None
    out = {}
    for source, dest in USAGE_KEYS:
        value = raw.get(source)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[dest] = int(value)
    return out or None


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def calls_of(raw: Any) -> list[dict]:
    """Tool calls from Hermes's response: provider objects or dicts, flat or OpenAI ``function`` shaped."""
    calls = []
    for item in raw if isinstance(raw, (list, tuple)) else []:
        fn = _get(item, "function")
        name = _get(item, "name") or _get(fn, "name") or "tool"
        args = _get(item, "arguments")
        if args is None:
            args = _get(fn, "arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except ValueError:
                pass
        calls.append({"id": str(_get(item, "id") or ""), "name": str(name), "arguments": capped(args if args is not None else {})})
    return calls


def reply_of(kw: dict) -> tuple[str, list[dict]]:
    """The assistant text and tool calls. The raw message is whole; the sanitized response is a fallback."""
    raw = kw.get("assistant_message")
    response = kw.get("response") if isinstance(kw.get("response"), dict) else {}
    cooked = response.get("assistant_message") if isinstance(response.get("assistant_message"), dict) else {}
    content = _get(raw, "content") if raw is not None else None
    if not isinstance(content, (str, list)):
        content = cooked.get("content")
    calls = _get(raw, "tool_calls") if raw is not None else None
    if not calls:
        calls = cooked.get("tool_calls")
    return text_of(content), calls_of(calls)


def error_text(kw: dict) -> str:
    error = kw.get("error") if isinstance(kw.get("error"), dict) else {}
    status = kw.get("status_code")
    head = "API request failed" + (f" ({status})" if status else "")
    kind = str(error.get("type") or kw.get("reason") or "error")
    text = f"{head}: {kind}: {error.get('message') or ''}".rstrip(": ")
    if kw.get("retry_count") is not None:
        text += f" [retry {kw.get('retry_count')}/{kw.get('max_retries')}]"
    return cap(text)


# ---- the state machine ---------------------------------------------------------------------------


class Tracker:
    """Hermes sessions, turns, children, and calls in flight. Thread-safe: children report from worker threads."""

    def __init__(self, resolve: Callable[[str], Any], version: str = "", cwd: Callable[[], str] = lambda: ""):
        self.resolve = resolve  # session id -> gate or None; called once per turn start
        self.version = version
        self.cwd = cwd
        self.turns: dict[str, Turn] = {}
        self.open: dict[str, str] = {}  # session -> its open turn
        self.children: dict[str, Child] = {}
        self.traced: dict[str, Any] = {}  # session -> the last gate it had, for session_end
        self.last_session = ""  # the newest main session, for /judgment when Hermes passes none
        self.child_ids: set[str] = set()  # every child session seen; Hermes's session context can name one
        self._lock = threading.Lock()

    def step(self, hook: str, kw: dict, now: int) -> Step:
        handler = HOOKS.get(hook)
        if handler is None:
            return Step()
        with self._lock:
            return handler(self, kw, now)

    # -- records

    def _record(self, event: str, session_id: str, turn_id: str, ts: int, **fields: Any) -> dict:
        rec = {"v": 1, "harness": HARNESS, "event": event, "session_id": session_id, "turn_id": turn_id, "ts": ts}
        if self.version:
            rec["harness_version"] = self.version
        rec.update({k: v for k, v in fields.items() if v is not None})
        return rec

    def _where(self, kw: dict) -> tuple[Any, Thread | None, dict]:
        """(gate, thread, ids) for a call event: the turn or child it belongs to."""
        sid = str(kw.get("session_id") or "")
        child = self.children.get(sid)
        if child is not None:
            return child.gate, child.thread, {"session_id": child.session_id, "turn_id": child.turn_id, "agent_id": child.agent_id}
        turn = self.turns.get(str(kw.get("turn_id") or "")) or self.turns.get(self.open.get(sid, ""))
        if turn is None:
            return None, None, {}
        return turn.gate, turn.thread, {"session_id": turn.session_id, "turn_id": turn.turn_id}

    def _abandon(self, gate: Any, thread: Thread, ids: dict, now: int) -> list:
        """Calls still in flight when their turn or child closed."""
        out = []
        for call_id, (name, args, start) in sorted(thread.tools.items(), key=lambda item: item[1][2]):
            out.append((gate, self._record("tool", ids["session_id"], ids["turn_id"], now, agent_id=ids.get("agent_id"), tool_name=name,
                                           tool_call_id=call_id, tool_input=args, tool_output="(no result: the turn ended first)",
                                           is_error=True, start_ts=start, end_ts=max(now, start + 1))))
        for attempt, (start, model, provider) in sorted(thread.attempts.items(), key=lambda item: item[1][0]):
            out.append((gate, self._record("model", ids["session_id"], ids["turn_id"], now, agent_id=ids.get("agent_id"), model=model,
                                           provider=provider, message_id=attempt, output="(no response: the turn ended first)",
                                           is_error=True, start_ts=start, end_ts=max(now, start + 1))))
        thread.tools.clear()
        thread.attempts.clear()
        return [item for item in out if item[0] is not None]

    def _close_turn(self, turn_id: str, now: int, interrupted: bool) -> Step:
        turn = self.turns.pop(turn_id, None)
        if turn is None:
            return Step()
        if self.open.get(turn.session_id) == turn_id:
            self.open.pop(turn.session_id)
        if turn.gate is None:
            return Step()
        ids = {"session_id": turn.session_id, "turn_id": turn_id}
        out = self._abandon(turn.gate, turn.thread, ids, now)
        out.append((turn.gate, self._record("turn_end", turn.session_id, turn_id, now, output=turn.answer,
                                            interrupted=True if interrupted else None, model=turn.model or None)))
        return Step(out, flush=True)

    def _close_child(self, sid: str, now: int, output: str, is_error: bool, interrupted: bool, duration: Any = None) -> Step:
        child = self.children.pop(sid, None)
        if child is None or child.gate is None:
            return Step()
        ids = {"session_id": child.session_id, "turn_id": child.turn_id, "agent_id": child.agent_id}
        out = self._abandon(child.gate, child.thread, ids, now)
        start = child.start
        if isinstance(duration, (int, float)) and duration > 0:
            start = min(start, now - int(duration))
        out.append((child.gate, self._record("subagent_end", child.session_id, child.turn_id, now, agent_id=child.agent_id,
                                             agent_type=child.agent_type, output=output, is_error=True if is_error else None,
                                             interrupted=True if interrupted else None, start_ts=start, end_ts=max(now, start + 1))))
        return Step(out, flush=True)

    # -- hooks

    def on_session_start(self, kw: dict, now: int) -> Step:
        sid = str(kw.get("session_id") or "")
        if sid and sid not in self.children and kw.get("platform") != "subagent":
            self.last_session = sid
        return Step()

    def pre_llm_call(self, kw: dict, now: int) -> Step:
        sid = str(kw.get("session_id") or "")
        if not sid or sid in self.children or kw.get("platform") == "subagent":
            return Step()  # a child's turn is part of its parent's
        tid = str(kw.get("turn_id") or f"{sid}:{now}")
        self.last_session = sid
        if tid in self.turns:
            return Step()
        step = Step()
        previous = self.open.get(sid)
        if previous:
            step = self._close_turn(previous, now, interrupted=True)
        gate = self.resolve(sid)
        model = str(kw.get("model") or "")
        self.turns[tid] = Turn(sid, tid, gate, model=model)
        self.open[sid] = tid
        if gate is None:
            return step
        self.traced[sid] = gate
        step.records.append((gate, self._record("turn_start", sid, tid, now, prompt=text_of(kw.get("user_message")),
                                                model=model or None, cwd=self.cwd() or None)))
        return step

    def post_llm_call(self, kw: dict, now: int) -> Step:
        turn = self.turns.get(str(kw.get("turn_id") or "")) or self.turns.get(self.open.get(str(kw.get("session_id") or ""), ""))
        if turn is not None and turn.phase is Phase.OPEN:
            turn.phase = Phase.ANSWERED
            turn.answer = text_of(kw.get("assistant_response"))
        return Step()

    def on_session_end(self, kw: dict, now: int) -> Step:
        """Hermes fires this at the end of every turn, and on exit when a turn was running."""
        sid = str(kw.get("session_id") or "")
        if sid in self.children:
            return Step()  # the child's turn ended; subagent_stop reports the child
        tid = str(kw.get("turn_id") or "") or self.open.get(sid, "")
        if tid not in self.turns:
            return Step()
        interrupted = bool(kw.get("interrupted")) or (kw.get("completed") is False and self.turns[tid].phase is Phase.OPEN)
        return self._close_turn(tid, now, interrupted)

    def on_session_finalize(self, kw: dict, now: int) -> Step:
        sid = str(kw.get("session_id") or "")
        if not sid or sid in self.children:
            return Step()
        step = Step()
        if self.open.get(sid):
            closed = self._close_turn(self.open[sid], now, interrupted=True)
            step.records += closed.records
        for child_sid in [key for key, child in self.children.items() if child.session_id == sid]:
            step.records += self._close_child(child_sid, now, "", False, True).records
        gate = self.traced.pop(sid, None)
        if gate is not None:
            step.records.append((gate, self._record("session_end", sid, "", now)))
        step.flush = bool(step.records)
        return step

    on_session_reset = on_session_finalize

    def pre_api_request(self, kw: dict, now: int) -> Step:
        gate, thread, _ids = self._where(kw)
        if thread is not None and gate is not None and kw.get("api_request_id"):
            thread.attempts[str(kw["api_request_id"])] = (ms(kw.get("started_at"), now), str(kw.get("model") or ""), str(kw.get("provider") or ""))
        return Step()

    def post_api_request(self, kw: dict, now: int) -> Step:
        gate, thread, ids = self._where(kw)
        if gate is None or thread is None:
            return Step()
        attempt = str(kw.get("api_request_id") or "")
        pending = thread.attempts.pop(attempt, None)
        start = ms(kw.get("started_at"), pending[0] if pending else now)
        end = ms(kw.get("ended_at"), start + ms(kw.get("api_duration"), 1))
        text, calls = reply_of(kw)
        rec = self._record("model", ids["session_id"], ids["turn_id"], end, agent_id=ids.get("agent_id"),
                           model=str(kw.get("response_model") or kw.get("model") or ""), provider=str(kw.get("provider") or "") or None,
                           message_id=attempt or None, output=text, tool_calls=calls or None, usage=usage_of(kw.get("usage")),
                           start_ts=start, end_ts=max(end, start + 1))
        return Step([(gate, rec)])

    def api_request_error(self, kw: dict, now: int) -> Step:
        gate, thread, ids = self._where(kw)
        if gate is None or thread is None:
            return Step()
        attempt = str(kw.get("api_request_id") or "")
        pending = thread.attempts.pop(attempt, None)
        start = ms(kw.get("started_at"), pending[0] if pending else now)
        end = ms(kw.get("ended_at"), now)
        rec = self._record("model", ids["session_id"], ids["turn_id"], end, agent_id=ids.get("agent_id"), model=str(kw.get("model") or ""),
                           provider=str(kw.get("provider") or "") or None, message_id=(f"{attempt}:retry{kw.get('retry_count')}" if attempt else None),
                           output=error_text(kw), is_error=True, start_ts=start, end_ts=max(end, start + 1))
        return Step([(gate, rec)])

    def pre_tool_call(self, kw: dict, now: int) -> Step:
        gate, thread, _ids = self._where(kw)
        call_id = str(kw.get("tool_call_id") or "")
        if thread is not None and gate is not None and call_id:
            thread.tools[call_id] = (str(kw.get("tool_name") or "tool"), capped(kw.get("args")), now)
        return Step()

    def post_tool_call(self, kw: dict, now: int) -> Step:
        gate, thread, ids = self._where(kw)
        if gate is None or thread is None:
            return Step()
        call_id = str(kw.get("tool_call_id") or "")
        pending = thread.tools.pop(call_id, None)
        duration = kw.get("duration_ms")
        if isinstance(duration, (int, float)) and duration >= 0:
            start = now - int(duration)
        else:
            start = pending[2] if pending else now - 1
        status = str(kw.get("status") or OK_TOOL)
        rec = self._record("tool", ids["session_id"], ids["turn_id"], now, agent_id=ids.get("agent_id"), tool_name=str(kw.get("tool_name") or "tool"),
                           tool_call_id=call_id or None, tool_input=capped(kw.get("args")), tool_output=text_of(kw.get("result")),
                           is_error=True if status != OK_TOOL else None, start_ts=start, end_ts=max(now, start + 1))
        return Step([(gate, rec)])

    def subagent_start(self, kw: dict, now: int) -> Step:
        child_sid = str(kw.get("child_session_id") or "")
        if not child_sid or child_sid in self.children:
            return Step()
        parent_sid = str(kw.get("parent_session_id") or "")
        parent = self.children.get(parent_sid)
        if parent is not None:  # a grandchild folds into the same root turn
            root_sid, root_tid, gate = parent.session_id, parent.turn_id, parent.gate
        else:
            turn = self.turns.get(str(kw.get("parent_turn_id") or "")) or self.turns.get(self.open.get(parent_sid, ""))
            root_sid, root_tid, gate = (turn.session_id, turn.turn_id, turn.gate) if turn else (parent_sid, "", None)
        agent_type = str(kw.get("child_role") or "subagent")
        self.child_ids.add(child_sid)
        self.children[child_sid] = Child(root_sid, root_tid, child_sid, gate, agent_type, now)
        if gate is None:
            return Step()
        rec = self._record("subagent_start", root_sid, root_tid, now, agent_id=child_sid, agent_type=agent_type, prompt=text_of(kw.get("child_goal")))
        return Step([(gate, rec)])

    def subagent_stop(self, kw: dict, now: int) -> Step:
        status = str(kw.get("child_status") or "")
        return self._close_child(str(kw.get("child_session_id") or ""), now, text_of(kw.get("child_summary")),
                                 bool(status) and status.lower() not in DONE_CHILD, False, kw.get("duration_ms"))


# The hook table: Hermes hook name -> handler. Every handler returns a Step; none returns a value to Hermes.
HOOKS: dict[str, Callable[[Tracker, dict, int], Step]] = {
    name: getattr(Tracker, name)
    for name in (
        "on_session_start",
        "pre_llm_call",
        "post_llm_call",
        "on_session_end",
        "on_session_finalize",
        "on_session_reset",
        "pre_api_request",
        "post_api_request",
        "api_request_error",
        "pre_tool_call",
        "post_tool_call",
        "subagent_start",
        "subagent_stop",
    )
}
