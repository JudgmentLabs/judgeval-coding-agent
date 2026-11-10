"""Hermes observer hooks -> judgment_core event records. Pure: no files, no network, no clock.

One call per hook: ``Tracker.step(hook, kwargs, now_ms)`` returns the records the hook produced,
each with the gate (the opt-in decision) of the turn it belongs to, and whether a sender should
start. The plugin adds the impure inputs to Hermes's
kwargs before the step, on pre_llm_call only: ``_workspace``, the folder Hermes resolved for that
turn, and ``_gate``, the opt-in decision resolved outside every lock for conversation ``_gate_key``.
Everything the tracker remembers between hooks is in
the plain tables of ``Tracker``; every table entry belongs to one conversation and goes when it does.

State machines. Impossible states are listed so the code can refuse them.

  Conversation identity (``ids``: any session id Hermes used -> conversation key; ``convs``: key -> Conv)
      unknown --pre_llm_call (platform != subagent)--> MAIN, key = that session id
      unknown --subagent_start(child_session_id)--> CHILD, bound to its originating root turn
      MAIN|CHILD --an event whose session id is new but whose turn_id the conversation owns--> same
            conversation, new id added as an alias (Hermes's legacy compression rotates
            agent.session_id mid-turn, conversation_compression.py:1246, and fires no hook)
      MAIN --on_session_finalize|on_session_reset--> gone, with every alias, every turn, and every
            child (and grandchild) bound to it
      CHILD --subagent_stop--> gone, with every alias and turn id
      Impossible: a session id in two conversations; an alias or turn owner pointing at a gone
      conversation; a child whose root changes after it started; a child opening a main turn.

  Turn (main conversations only; ``turns``: turn_id -> Turn; ``owner``: turn_id -> conversation key)
      none --pre_llm_call--> OPEN --post_llm_call--> ANSWERED
      OPEN|ANSWERED --on_session_end--> closed      turn_end, interrupted unless Hermes says it completed
      OPEN|ANSWERED --pre_llm_call(another turn)--> closed (interrupted), then the new turn opens
      OPEN|ANSWERED --on_session_finalize|reset--> closed (interrupted), then session_end
      Impossible: a turn_end without a turn_start; two open turns in one conversation; ANSWERED -> OPEN.

  Gate (decided once per turn, at turn start, for the conversation key and the turn's workspace):
      None = untraced. The turn, its calls, and every child it starts share it. ``/judgment off``
      takes effect at the next turn, so a turn is never half recorded.

  Model call attempt (per agent, per api_request_id; Hermes reuses one id and one started_at for
      every retry of a request, conversation_loop.py:1215-1225, and can retry without changing
      retry_count, conversation_loop.py:2804):
      pre_api_request --> PENDING(seq = n + 1, start = now)
      PENDING --post_api_request--> done (model record)          message_id = id, then id#2, id#3 ...
      PENDING --api_request_error--> error (model record, is_error)
      PENDING --pre_api_request(same id)--> abandoned (error record), then the new attempt is PENDING
      PENDING --turn or child closes--> abandoned (model record, is_error)
      post/error without pre: recorded as the next attempt, timed from Hermes's started_at.

  Tool call (tool_call_id): pending --post_tool_call(status ok|error|blocked|cancelled)--> done
      pending --turn or child closes--> abandoned (tool record, is_error)
      post without pre (a call blocked before pre_tool_call): recorded, timed from duration_ms.

  Child binding: a child's events go to the root turn that started it, under agent_id = its key,
      even after that turn ended (Hermes runs delegations in the background). A grandchild takes
      its parent's root turn. subagent_start carries ``bind_agent`` so the flusher's fold binds the
      child to that turn explicitly, and never to whatever turn is open when it folds the start.
"""

from __future__ import annotations

import enum
import json
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable

HARNESS = "hermes"
# The most of one string the tracker keeps. It only bounds memory and the work redaction does:
# the core bridge redacts first and then cuts what it keeps (core_bridge.TEXT_CAP, which is smaller).
RAW_CAP = 128 * 1024
# The most string text one captured value (tool arguments, a content list) keeps in total.
VALUE_BUDGET = 256 * 1024
OK_TOOL = "ok"
DONE_CHILD = ("completed", "complete", "success", "succeeded", "ok", "done")
# Main conversations kept without a finalize. The oldest idle one (no open turn, no child) goes first.
MAX_CONVS = 1024

# Hermes CanonicalUsage -> judgment_core usage keys. input_tokens there is already non-cached.
USAGE_KEYS = (
    ("input_tokens", "input_tokens"),
    ("output_tokens", "output_tokens"),
    ("cache_read_tokens", "cache_read_input_tokens"),
    ("cache_write_tokens", "cache_creation_input_tokens"),
    ("reasoning_tokens", "reasoning_tokens"),
)


class Kind(enum.Enum):
    MAIN = "main"
    CHILD = "child"


class Phase(enum.Enum):
    OPEN = "open"
    ANSWERED = "answered"


@dataclass
class Attempt:
    seq: int
    start: int
    model: str
    provider: str


@dataclass
class Calls:
    """One agent's calls in flight."""

    tools: dict = field(default_factory=dict)  # tool_call_id -> (name, args, start)
    attempts: dict = field(default_factory=dict)  # api_request_id -> Attempt
    seq: dict = field(default_factory=dict)  # api_request_id -> attempts started so far


@dataclass
class Conv:
    """One conversation: a main session or a child agent. ``key`` is the first session id Hermes gave it."""

    key: str
    kind: Kind
    aliases: set = field(default_factory=set)  # every session id that names it, key included
    turn_ids: set = field(default_factory=set)  # every turn id it owns (rotation is detected by these)
    # MAIN
    open_turn: str = ""
    traced: Any = None  # the last gate a turn had, for session_end
    # CHILD
    root: str = ""  # the root main conversation's key
    root_turn: str = ""  # the root turn that started it
    gate: Any = None
    agent_type: str = "subagent"
    start: int = 0
    calls: Calls = field(default_factory=Calls)


@dataclass
class Turn:
    session_id: str  # the main conversation's key
    turn_id: str
    gate: Any
    phase: Phase = Phase.OPEN
    answer: str = ""
    model: str = ""
    workspace: str = ""
    calls: Calls = field(default_factory=Calls)


@dataclass
class Step:
    records: list = field(default_factory=list)  # [(gate, record)]
    flush: bool = False
    aliases: list = field(default_factory=list)  # [(key, new session id)] for main conversations Hermes rotated


@dataclass(frozen=True)
class Where:
    """The context of a call event: whose calls, which gate, and the ids its records carry."""

    gate: Any
    calls: Calls
    session_id: str
    turn_id: str
    agent_id: str | None = None


# ---- pure readers of Hermes kwargs ---------------------------------------------------------------


def clip(text: str) -> str:
    """At most RAW_CAP characters, cut anywhere. The bridge redacts before it cuts what it keeps."""
    return text if len(text) <= RAW_CAP else text[:RAW_CAP]


def text_of(value: Any) -> str:
    """A message content as text: a string, or the text parts of a content list."""
    if value is None:
        return ""
    if isinstance(value, str):
        return clip(value)
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
        return clip("\n".join(p for p in parts if p))
    return clip(json.dumps(value, ensure_ascii=False, default=str))


def capped(value: Any) -> Any:
    """A JSON-shaped copy with at most VALUE_BUDGET characters of strings, keys included. Unknown objects become their string."""
    budget = [VALUE_BUDGET]

    def walk(item: Any, depth: int) -> Any:
        if isinstance(item, str) or not (item is None or isinstance(item, (bool, int, float, dict, list, tuple))):
            text = item if isinstance(item, str) else str(item)
            keep = clip(text[: max(budget[0], 0)])
            budget[0] -= len(keep)
            return keep
        if item is None or isinstance(item, (bool, int, float)):
            return item
        if depth > 12:
            return "…"
        if isinstance(item, dict):
            return {walk(str(k), depth + 1): walk(v, depth + 1) for k, v in item.items()}
        return [walk(v, depth + 1) for v in item]

    return walk(value, 0)


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
        calls.append({"id": clip(str(_get(item, "id") or "")), "name": clip(str(name)), "arguments": capped(args if args is not None else {})})
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
    return clip(text)


def s(kw: dict, name: str) -> str:
    """A kwarg as a string id, '' when missing."""
    return clip(str(kw.get(name) or ""))


def attempt_id(request_id: str, seq: int) -> str:
    """The first attempt keeps Hermes's id; each retry of the same request gets #2, #3, ..."""
    return request_id if seq <= 1 else f"{request_id}#{seq}"


# ---- the state machine ---------------------------------------------------------------------------


class Tracker:
    """Hermes conversations, turns, children, and calls in flight. Thread-safe: children report from worker threads."""

    def __init__(self, resolve: Callable[[str, str], Any], version: str = ""):
        self.resolve = resolve  # (conversation key, workspace) -> gate or None; called once per turn start
        self.version = version
        self.ids: dict[str, str] = {}  # session id or alias -> conversation key
        self.convs: OrderedDict[str, Conv] = OrderedDict()  # key -> Conv, least recently active first
        self.turns: dict[str, Turn] = {}  # open main turn id -> Turn
        self.owner: dict[str, str] = {}  # turn id (main or child) -> conversation key
        self._lock = threading.Lock()

    def step(self, hook: str, kw: dict, now: int) -> Step:
        handler = HOOKS.get(hook)
        if handler is None:
            return Step()
        with self._lock:
            return handler(self, kw, now)

    # -- identity

    def key_of(self, session_id: str) -> str:
        """The conversation a session id names now ('' when none). For /judgment."""
        with self._lock:
            return self.ids.get(session_id, "")

    def kind_of(self, key: str) -> Kind | None:
        with self._lock:
            conv = self.convs.get(key)
            return conv.kind if conv else None

    def _conv(self, kw: dict, step: Step, sid_name: str = "session_id") -> Conv | None:
        """The conversation an event belongs to, adding an alias when Hermes rotated its session id."""
        sid = s(kw, sid_name)
        key = self.ids.get(sid)
        if key is None:
            owner = self.owner.get(s(kw, "turn_id"))
            if not sid or owner is None:
                return None
            key = owner
            self._alias(key, sid, step)
        conv = self.convs.get(key)
        if conv is not None:
            self.convs.move_to_end(key)
        return conv

    def _alias(self, key: str, sid: str, step: Step) -> None:
        conv = self.convs[key]
        conv.aliases.add(sid)
        self.ids[sid] = key
        if conv.kind is Kind.MAIN:
            step.aliases.append((key, sid))

    def _new(self, key: str, kind: Kind, **fields: Any) -> Conv:
        conv = Conv(key, kind, aliases={key}, **fields)
        self.convs[key] = conv
        self.ids[key] = key
        return conv

    def _forget(self, conv: Conv) -> None:
        """Drop a conversation and every table entry that names it."""
        self.convs.pop(conv.key, None)
        for sid in conv.aliases:
            if self.ids.get(sid) == conv.key:
                self.ids.pop(sid)
        for tid in conv.turn_ids:
            if self.owner.get(tid) == conv.key:
                self.owner.pop(tid)
            self.turns.pop(tid, None)

    def _evict(self) -> None:
        """Keep at most MAX_CONVS conversations. Only an idle main conversation goes, oldest first."""
        if len(self.convs) <= MAX_CONVS:
            return
        busy = {conv.root for conv in self.convs.values() if conv.kind is Kind.CHILD}
        for key, conv in list(self.convs.items()):
            if len(self.convs) <= MAX_CONVS:
                return
            if conv.kind is Kind.MAIN and not conv.open_turn and key not in busy:
                self._forget(conv)

    # -- records

    def _record(self, event: str, session_id: str, turn_id: str, ts: int, **fields: Any) -> dict:
        rec = {"v": 1, "harness": HARNESS, "event": event, "session_id": session_id, "turn_id": turn_id, "ts": ts}
        if self.version:
            rec["harness_version"] = self.version
        rec.update({k: v for k, v in fields.items() if v is not None})
        return rec

    def _where(self, kw: dict, step: Step) -> Where | None:
        """The turn or child a call event belongs to, or None when it is not traced."""
        conv = self._conv(kw, step)
        if conv is None:
            return None
        if conv.kind is Kind.CHILD:
            return Where(conv.gate, conv.calls, conv.root, conv.root_turn, conv.key)
        turn = self.turns.get(s(kw, "turn_id")) or self.turns.get(conv.open_turn)
        if turn is None or turn.session_id != conv.key:
            return None
        return Where(turn.gate, turn.calls, turn.session_id, turn.turn_id)

    def _model(self, where: Where, attempt: str, start: int, end: int, model: str, provider: str, **fields: Any) -> tuple:
        rec = self._record("model", where.session_id, where.turn_id, end, agent_id=where.agent_id, model=model, provider=provider or None,
                           message_id=attempt, start_ts=start, end_ts=max(end, start + 1), **fields)
        return (where.gate, rec)

    def _abandon(self, where: Where, now: int) -> list:
        """Calls still in flight when their turn or child closed."""
        out = []
        calls = where.calls
        for call_id, (name, args, start) in sorted(calls.tools.items(), key=lambda item: item[1][2]):
            out.append((where.gate, self._record("tool", where.session_id, where.turn_id, now, agent_id=where.agent_id, tool_name=name,
                                                 tool_call_id=call_id, tool_input=args, tool_output="(no result: the turn ended first)",
                                                 is_error=True, start_ts=start, end_ts=max(now, start + 1))))
        for rid, att in sorted(calls.attempts.items(), key=lambda item: item[1].start):
            out.append(self._model(where, attempt_id(rid, att.seq), att.start, now, att.model, att.provider,
                                   output="(no response: the turn ended first)", is_error=True))
        calls.tools.clear()
        calls.attempts.clear()
        calls.seq.clear()
        return [item for item in out if item[0] is not None]

    def _close_turn(self, turn_id: str, now: int, interrupted: bool) -> Step:
        turn = self.turns.pop(turn_id, None)
        if turn is None:
            return Step()
        conv = self.convs.get(turn.session_id)
        if conv is not None:
            if conv.open_turn == turn_id:
                conv.open_turn = ""
            conv.turn_ids.discard(turn_id)
        if self.owner.get(turn_id) == turn.session_id:
            self.owner.pop(turn_id)
        if turn.gate is None:
            return Step()
        where = Where(turn.gate, turn.calls, turn.session_id, turn_id)
        out = self._abandon(where, now)
        out.append((turn.gate, self._record("turn_end", turn.session_id, turn_id, now, output=turn.answer,
                                            interrupted=True if interrupted else None, model=turn.model or None)))
        return Step(out, flush=True)

    def _close_child(self, conv: Conv, now: int, output: str, is_error: bool, interrupted: bool, duration: Any = None) -> Step:
        self._forget(conv)
        if conv.gate is None:
            return Step()
        where = Where(conv.gate, conv.calls, conv.root, conv.root_turn, conv.key)
        out = self._abandon(where, now)
        start = conv.start
        if isinstance(duration, (int, float)) and duration > 0:
            start = min(start, now - int(duration))
        out.append((conv.gate, self._record("subagent_end", conv.root, conv.root_turn, now, agent_id=conv.key, agent_type=conv.agent_type,
                                            output=output, is_error=True if is_error else None, interrupted=True if interrupted else None,
                                            start_ts=start, end_ts=max(now, start + 1))))
        return Step(out, flush=True)

    # -- hooks

    def on_session_start(self, kw: dict, now: int) -> Step:
        return Step()  # nothing to record: a conversation starts with its first turn

    def pre_llm_call(self, kw: dict, now: int) -> Step:
        step = Step()
        sid = s(kw, "session_id")
        tid = s(kw, "turn_id") or f"{sid}:{now}"
        conv = self._conv(kw, step)
        if conv is not None and conv.kind is Kind.CHILD:
            conv.turn_ids.add(tid)  # a child's own turn: its rotation is found by this id
            self.owner[tid] = conv.key
            return step
        if not sid or kw.get("platform") == "subagent":
            return step  # a child nobody started here: never a main turn
        if tid in self.turns:
            return step
        if conv is None:
            conv = self._new(sid, Kind.MAIN)
            self._evict()
        if conv.open_turn:
            closed = self._close_turn(conv.open_turn, now, interrupted=True)
            step.records += closed.records
            step.flush = closed.flush
        workspace = str(kw.get("_workspace") or "")
        # The plugin resolves the gate before the step, outside every lock (it reads files), for the key
        # it expects; when that key is not the conversation's, the gate is resolved here.
        gate = kw["_gate"] if "_gate" in kw and kw.get("_gate_key") == conv.key else self.resolve(conv.key, workspace)
        model = s(kw, "model")
        self.turns[tid] = Turn(conv.key, tid, gate, model=model, workspace=workspace)
        self.owner[tid] = conv.key
        conv.turn_ids.add(tid)
        conv.open_turn = tid
        if gate is None:
            return step
        conv.traced = gate
        step.records.append((gate, self._record("turn_start", conv.key, tid, now, prompt=text_of(kw.get("user_message")),
                                                model=model or None, cwd=workspace or None)))
        return step

    def post_llm_call(self, kw: dict, now: int) -> Step:
        step = Step()
        conv = self._conv(kw, step)
        if conv is None or conv.kind is not Kind.MAIN:
            return step
        turn = self.turns.get(s(kw, "turn_id")) or self.turns.get(conv.open_turn)
        if turn is not None and turn.session_id == conv.key and turn.phase is Phase.OPEN:
            turn.phase = Phase.ANSWERED
            turn.answer = text_of(kw.get("assistant_response"))
        return step

    def on_session_end(self, kw: dict, now: int) -> Step:
        """Hermes fires this at the end of every turn, and on exit when a turn was running."""
        step = Step()
        conv = self._conv(kw, step)
        if conv is None or conv.kind is not Kind.MAIN:
            return step  # a child's turn ended; subagent_stop reports the child
        tid = s(kw, "turn_id") or conv.open_turn
        turn = self.turns.get(tid)
        if turn is None or turn.session_id != conv.key:
            return step
        interrupted = bool(kw.get("interrupted")) or (kw.get("completed") is False and turn.phase is Phase.OPEN)
        closed = self._close_turn(tid, now, interrupted)
        closed.aliases = step.aliases
        return closed

    def on_session_finalize(self, kw: dict, now: int) -> Step:
        step = Step()
        conv = self._conv(kw, step)
        if conv is None or conv.kind is not Kind.MAIN:
            return Step()
        if conv.open_turn:
            step.records += self._close_turn(conv.open_turn, now, interrupted=True).records
        for child in [other for other in self.convs.values() if other.kind is Kind.CHILD and other.root == conv.key]:
            step.records += self._close_child(child, now, "", False, True).records
        if conv.traced is not None:
            step.records.append((conv.traced, self._record("session_end", conv.key, "", now)))
        self._forget(conv)
        step.flush = bool(step.records)
        return step

    on_session_reset = on_session_finalize

    def pre_api_request(self, kw: dict, now: int) -> Step:
        step = Step()
        where = self._where(kw, step)
        rid = s(kw, "api_request_id")
        if where is None or where.gate is None or not rid:
            return step
        calls = where.calls
        old = calls.attempts.pop(rid, None)
        if old is not None:  # the last attempt never reported: Hermes went on to the next
            step.records.append(self._model(where, attempt_id(rid, old.seq), old.start, now, old.model, old.provider,
                                            output="(no response: Hermes retried first)", is_error=True))
        seq = calls.seq.get(rid, 0) + 1
        calls.seq[rid] = seq
        calls.attempts[rid] = Attempt(seq, now, s(kw, "model"), s(kw, "provider"))
        return step

    def _finish_attempt(self, where: Where, kw: dict, now: int) -> tuple[str, int, Attempt | None]:
        rid = s(kw, "api_request_id")
        pending = where.calls.attempts.pop(rid, None) if rid else None
        if pending is not None:
            return attempt_id(rid, pending.seq), pending.start, pending
        seq = where.calls.seq.get(rid, 0) + 1 if rid else 1
        if rid:
            where.calls.seq[rid] = seq
        return (attempt_id(rid, seq) if rid else ""), ms(kw.get("started_at"), now), None

    def post_api_request(self, kw: dict, now: int) -> Step:
        step = Step()
        where = self._where(kw, step)
        if where is None or where.gate is None:
            return step
        attempt, start, _pending = self._finish_attempt(where, kw, now)
        end = ms(kw.get("ended_at"), start + ms(kw.get("api_duration"), 1))
        text, calls = reply_of(kw)
        step.records.append(self._model(where, attempt or None, start, end, s(kw, "response_model") or s(kw, "model"), s(kw, "provider"),
                                         output=text, tool_calls=calls or None, usage=usage_of(kw.get("usage"))))
        return step

    def api_request_error(self, kw: dict, now: int) -> Step:
        step = Step()
        where = self._where(kw, step)
        if where is None or where.gate is None:
            return step
        attempt, start, _pending = self._finish_attempt(where, kw, now)
        end = ms(kw.get("ended_at"), now)
        step.records.append(self._model(where, attempt or None, start, end, s(kw, "model"), s(kw, "provider"), output=error_text(kw), is_error=True))
        return step

    def pre_tool_call(self, kw: dict, now: int) -> Step:
        step = Step()
        where = self._where(kw, step)
        call_id = s(kw, "tool_call_id")
        if where is not None and where.gate is not None and call_id:
            where.calls.tools[call_id] = (s(kw, "tool_name") or "tool", capped(kw.get("args")), now)
        return step

    def post_tool_call(self, kw: dict, now: int) -> Step:
        step = Step()
        where = self._where(kw, step)
        if where is None or where.gate is None:
            return step
        call_id = s(kw, "tool_call_id")
        pending = where.calls.tools.pop(call_id, None)
        duration = kw.get("duration_ms")
        if isinstance(duration, (int, float)) and duration >= 0:
            start = now - int(duration)
        else:
            start = pending[2] if pending else now - 1
        status = str(kw.get("status") or OK_TOOL)
        rec = self._record("tool", where.session_id, where.turn_id, now, agent_id=where.agent_id, tool_name=s(kw, "tool_name") or "tool",
                           tool_call_id=call_id or None, tool_input=capped(kw.get("args")), tool_output=text_of(kw.get("result")),
                           is_error=True if status != OK_TOOL else None, start_ts=start, end_ts=max(now, start + 1))
        step.records.append((where.gate, rec))
        return step

    def subagent_start(self, kw: dict, now: int) -> Step:
        step = Step()
        child_sid = s(kw, "child_session_id")
        if not child_sid or child_sid in self.ids:
            return step
        parent = self._conv({"session_id": kw.get("parent_session_id"), "turn_id": kw.get("parent_turn_id")}, step)
        if parent is not None and parent.kind is Kind.CHILD:  # a grandchild takes its parent's root turn
            root, root_turn, gate = parent.root, parent.root_turn, parent.gate
        elif parent is not None:
            turn = self.turns.get(s(kw, "parent_turn_id")) or self.turns.get(parent.open_turn)
            if turn is None or turn.session_id != parent.key:
                return step  # no turn of a conversation known here started it
            root, root_turn, gate = parent.key, turn.turn_id, turn.gate
        else:
            return step
        agent_type = s(kw, "child_role") or "subagent"
        self._new(child_sid, Kind.CHILD, root=root, root_turn=root_turn, gate=gate, agent_type=agent_type, start=now)
        if gate is None:
            return step
        step.records.append((gate, self._record("subagent_start", root, root_turn, now, agent_id=child_sid, bind_agent=child_sid,
                                                agent_type=agent_type, prompt=text_of(kw.get("child_goal")))))
        return step

    def subagent_stop(self, kw: dict, now: int) -> Step:
        step = Step()
        conv = self._conv({"session_id": kw.get("child_session_id")}, step)
        if conv is None or conv.kind is not Kind.CHILD:
            return step
        status = str(kw.get("child_status") or "")
        closed = self._close_child(conv, now, text_of(kw.get("child_summary")), bool(status) and status.lower() not in DONE_CHILD, False,
                                   kw.get("duration_ms"))
        return closed


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
