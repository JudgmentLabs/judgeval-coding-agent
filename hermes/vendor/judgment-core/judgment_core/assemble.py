"""Assign hook events to turns.

Hooks record what happened and decide nothing. The flusher folds the events,
in the order they happened, through ``Fold.step``. The fold owns every
routing rule: which turn an event belongs to, when a child agent joins its
parent's turn, when a turn is held for a running child, and when a turn is
ready to send. Its state is one small dict saved between runs, so each run
only folds the events that arrived since the last one.
"""

from __future__ import annotations

import re
from pathlib import Path

_EMPTY = {
    "open": {},  # session -> open main turn
    "binds": {},  # "session\tagent" -> parent turn
    "pending": {},  # "session\tagent" -> parent turn, while that child runs
    "spawn": {},  # "session\ttask path" -> spawning turn (Codex v2)
    "child": {},  # Cursor child conversation -> [parent session, parent turn, agent type]
    "task_open": {},  # Cursor session -> [turn, call id] while a Task runs
    "bg": {},  # Cursor session -> turn a background wake belongs to
    "active": {},  # Cursor session -> current run
    "seq": {},  # Cursor session -> last minted run number
    "done": {},  # Cursor session -> last finished turn
    "models": {},  # session -> model announced at session start
    "closed": {},  # "harness\tsession\tturn" -> true once the turn ended
    "hidden": {},  # session -> true for a thread the agent app started on its own
    "seen": {},  # session -> last event time, for pruning
    "inbox": {},  # "last" -> [mtime_ns, name] of the last inbox file folded in
}

_KEEP_MS = 7 * 24 * 3600 * 1000


def empty_state() -> dict:
    return {key: {} for key in _EMPTY}


def _key(*parts: str) -> str:
    return "\t".join(parts)


def turn_key(event: dict) -> str:
    return _key(str(event.get("harness") or "agent"), str(event.get("session_id") or ""), str(event.get("turn_id") or ""))


def _cursor_parent_of(event: dict) -> str:
    """A Task child's transcript lives at ``<parent>/subagents/<child>.jsonl``."""
    path = str(event.get("transcript_path") or "")
    session = str(event.get("session_id") or "")
    if not session or not path.endswith(f"/subagents/{session}.jsonl"):
        return ""
    return Path(path).parent.parent.name


def _agent_path(transcript: str) -> str:
    try:
        with open(transcript, "rb") as handle:
            head = handle.read(65536).decode("utf-8", "replace")
    except OSError:
        return ""
    found = re.search(r'"agent_path":\s*"([^"]*)"', head)
    return found.group(1) if found else ""


class Fold:
    """Route one event at a time. ``step`` returns the routed event and the turns to send."""

    def __init__(self, state: dict | None = None):
        self.s = empty_state()
        for key, value in (state or {}).items():
            if key in self.s and isinstance(value, dict):
                self.s[key] = value
        self.sent: set[str] = set()  # turns sent at least once; a late event sends them again
        self.unsent_keys: set[str] = set()  # turns stored but not sent yet

    def state(self, now_ms: int, live: set[str]) -> dict:
        """The state to save. ``live`` is every turn still kept on disk.

        Routing entries for turns already dropped go, and so does everything
        for a session idle for a week.
        """
        old = {session for session, seen in self.s["seen"].items() if now_ms - int(seen) > _KEEP_MS}
        turns = {tuple(key.split("\t")[1:]) for key in live}
        self.s["closed"] = {key: value for key, value in self.s["closed"].items() if key in live}
        for name in ("binds", "pending", "spawn"):
            self.s[name] = {key: turn for key, turn in self.s[name].items() if (key.split("\t")[0], turn) in turns}
        if old:
            for name, table in self.s.items():
                self.s[name] = {key: value for key, value in table.items() if key.split("\t")[0] not in old}
        return self.s

    def running(self, key: str) -> bool:
        """Whether a child of this turn is still running."""
        _harness, session, turn = key.split("\t")
        return any(owner == turn and name.startswith(session + "\t") for name, owner in self.s["pending"].items())

    def step(self, event: dict) -> tuple[dict | None, list[str], list[dict]]:
        """Return (event to store or None, keys of turns now ready, extra events to store first)."""
        event = dict(event)
        for field in ("harness", "session_id", "turn_id", "agent_id", "bind_agent", "bind_task"):
            # Ids join routing keys with tabs. A tab or newline inside one would split a key.
            if isinstance(event.get(field), str) and ("\t" in event[field] or "\n" in event[field]):
                event[field] = event[field].replace("\t", " ").replace("\n", " ")
        harness = str(event.get("harness") or "agent")
        session = str(event.get("session_id") or "")
        kind = str(event.get("event") or "")
        if not session or not kind:
            return None, [], []
        try:
            self.s["seen"][session] = int(event.get("ts") or 0)
        except (TypeError, ValueError):
            pass
        if kind == "turn_start" and event.get("hidden"):
            self.s["hidden"][session] = True
        if self.s["hidden"].get(session):
            return None, [], []
        if not event.get("model") and self.s["models"].get(session):
            event["model"] = self.s["models"][session]

        if harness == "cursor":
            event = self._cursor_turn(event)
        routed = bool(event.get("agent_id") or event.get("bind_agent") or event.get("bind_task") or harness == "cursor" or kind == "subagent_start")
        if routed:
            event = self._route_child(event)

        if kind == "session_end" and event.get("session_id") != session:
            kind = event["event"] = "subagent_end"
            self.s["pending"].pop(_key(str(event["session_id"]), str(event.get("agent_id") or "")), None)
        session = str(event.get("session_id") or session)
        turn = str(event.get("turn_id") or "")
        ready: list[str] = []
        extra: list[dict] = []
        if kind == "turn_start" and not routed and turn:
            if harness == "claude-code":
                extra = self._close_interrupted(event, turn)
            self.s["open"][session] = turn
            self.s["bg"].pop(session, None)
        elif harness == "claude-code" and kind == "session_end":
            extra = self._close_interrupted(event, "")
        ready += [turn_key(item) for item in extra]

        if kind == "session_model":
            if event.get("model"):
                self.s["models"][session] = event["model"]
            return None, ready, extra
        if kind == "session_end":
            # Every turn of this session not sent yet goes out now.
            prefix = _key(harness, session, "")
            ready += [key for key in self.unsent_keys if key.startswith(prefix)]
            return None, ready, extra
        if not turn:
            return None, ready, extra

        key = turn_key(event)
        if key in self.sent:
            ready.append(key)  # a late line for a turn already sent: send it again
        if kind == "turn_end":
            if self.s["closed"].get(key):
                # A second stop on one turn is a Cursor background wake ending: its children are done.
                for name in [name for name, owner in self.s["pending"].items() if owner == turn and name.startswith(session + "\t")]:
                    self.s["pending"].pop(name)
            self.s["closed"][key] = True
        if kind in ("turn_end", "subagent_end") and self.s["closed"].get(key):
            ready.append(key)
        return event, ready, extra

    def _close_interrupted(self, event: dict, turn: str) -> list[dict]:
        """Claude sends no Stop when the user interrupts. Close the open turn at the next prompt or session end."""
        session = str(event.get("session_id") or "")
        previous = self.s["open"].get(session, "")
        if not previous or previous == turn:
            return []
        key = _key("claude-code", session, previous)
        if self.s["closed"].get(key) or key not in self.unsent_keys | self.sent:
            return []
        self.s["closed"][key] = True
        end = {
            "v": 1,
            "harness": "claude-code",
            "event": "turn_end",
            "session_id": session,
            "turn_id": previous,
            "interrupted": True,
            "ts": event.get("ts"),
        }
        for field in ("export_fp", "export_project", "export_url", "export_org", "capture", "environment", "project_root"):
            if event.get(field):
                end[field] = event[field]
        return [end]

    def _cursor_turn(self, event: dict) -> dict:
        """One Cursor request is one turn.

        Headless events disagree on generation id, and a headless resume fires
        no sessionStart, so an event with no open turn starts a new run.
        afterAgentResponse can land just after stop, so the finished turn's own
        generation id stays on that turn.
        """
        session = str(event.get("session_id") or "")
        kind = str(event.get("event") or "")
        turn = str(event.get("turn_id") or "")
        if event.get("agent_id") or session in self.s["child"] or _cursor_parent_of(event):
            return event
        active = self.s["active"]
        if kind == "turn_start":
            if turn:
                active[session] = turn
            return event
        done = self.s["done"].get(session, "")
        if kind == "session_model" or (
            session not in active and session not in self.s["bg"] and turn != done and kind not in ("session_end", "turn_end")
        ):
            number = int(self.s["seq"].get(session, 0)) + 1
            self.s["seq"][session] = number
            active[session] = f"{session}:run-{number}"
        if kind != "session_model" and active.get(session) and turn and active[session] != turn:
            event["turn_id"] = active[session]
        if kind == "turn_end":
            self.s["done"][session] = str(event.get("turn_id") or "")
        if kind in ("session_end", "turn_end"):
            active.pop(session, None)
        return event

    def _route_child(self, event: dict) -> dict:
        """Fold a child agent into the parent turn.

        Codex and Claude children share the parent session and carry agent_id.
        Cursor children arrive under their own conversation id, under the
        parent's ``subagents/`` folder.
        """
        s = self.s
        harness = str(event.get("harness") or "")
        kind = str(event.get("event") or "")
        turn = str(event.get("turn_id") or "")
        session = str(event.get("session_id") or "")
        agent = str(event.get("agent_id") or "")
        bind_agent = str(event.get("bind_agent") or "")
        bind_task = str(event.get("bind_task") or "")
        background = event.get("background") is True

        if kind == "turn_start" and turn:
            s["open"][session] = turn
            s["bg"].pop(session, None)
        if bind_agent and turn:
            s["binds"][_key(session, bind_agent)] = turn
            s["pending"][_key(session, bind_agent)] = turn
            if harness == "cursor":
                s["child"][bind_agent] = [session, turn, str(event.get("agent_type") or "")]
                s["task_open"].pop(session, None)
                if background:
                    s["bg"][session] = turn
        if bind_task and turn:
            s["spawn"][_key(session, bind_task)] = turn
        if harness == "cursor" and kind == "tool" and not bind_agent and event.get("tool_name") == "Task":
            s["task_open"].pop(session, None)  # a failed Task reports no agentId
        if harness == "cursor" and kind == "tool_start" and event.get("task_pending") is True:
            s["task_open"][session] = [turn, str(event.get("tool_call_id") or "")]

        parent_turn = ""
        parent_session = ""
        bound = _key(session, agent)
        # A start with no spawn waiting on it belongs to the open turn, even when the id was bound before.
        if agent and bound in s["binds"] and (kind != "subagent_start" or bound in s["pending"]):
            parent_turn = s["binds"][bound]
        if not parent_turn and kind == "subagent_start" and agent:
            path = _agent_path(str(event.get("transcript_path") or ""))
            spawn = _key(session, path)
            if path and spawn in s["spawn"]:
                parent_turn = s["spawn"].pop(spawn)
                s["binds"][bound] = parent_turn
                s["pending"][bound] = parent_turn
        if not parent_turn and kind == "subagent_start" and agent and s["open"].get(session):
            parent_turn = s["open"][session]
            s["binds"][bound] = parent_turn
            s["pending"][bound] = parent_turn
        if kind == "subagent_end" and agent and not background:
            s["pending"].pop(bound, None)

        if harness == "cursor" and session in s["child"]:
            parent_session, parent_turn = s["child"][session][0], s["child"][session][1]
        elif harness == "cursor" and _cursor_parent_of(event):
            owner = _cursor_parent_of(event)
            owner_turn = (s["task_open"].get(owner) or [""])[0] or s["active"].get(owner, "")
            if owner_turn:
                parent_session, parent_turn = owner, owner_turn
                s["child"][session] = [owner, owner_turn, "subagent"]
                agent = session
        if harness == "cursor" and not parent_turn and kind != "turn_start" and s["bg"].get(session):
            parent_turn = s["bg"][session]

        if parent_session and parent_session != session:
            event["session_id"] = parent_session
            event.setdefault("agent_id", agent or session)
        if parent_turn and turn and parent_turn != turn:
            event["turn_id"] = parent_turn
        return event
