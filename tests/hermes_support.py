"""Load the Hermes plugin the way Hermes does, and build hook kwargs shaped like Hermes 0.19.0's.

The kwargs below copy the keys and value shapes real Hermes passed to a probe plugin
(``hermes chat -q`` against tests/mock_openai.py): ids, timestamps in float seconds, usage as
CanonicalUsage, tool calls with JSON-string arguments, tool results as JSON strings.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import types
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from support import REPO

PLUGIN_DIR = REPO / "hermes"
VENDOR_CORE = PLUGIN_DIR / "vendor" / "judgment-core"
SCHEMA = "hermes.observer.v1"


def load_plugin(name: str = "judgment_tracing_under_test"):
    """Import hermes/ as a package under ``hermes_plugins``, as Hermes's loader does."""
    if "hermes_plugins" not in sys.modules:
        parent = types.ModuleType("hermes_plugins")
        parent.__path__ = []
        sys.modules["hermes_plugins"] = parent
    full = f"hermes_plugins.{name}"
    if full in sys.modules:
        return sys.modules[full]
    spec = importlib.util.spec_from_file_location(full, PLUGIN_DIR / "__init__.py", submodule_search_locations=[str(PLUGIN_DIR)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[full] = module
    spec.loader.exec_module(module)
    return module


class FakeCtx:
    """The two PluginContext methods the plugin uses, and the manager's ``_cli_ref`` (None outside the interactive CLI)."""

    def __init__(self, cli_session: str | None = None):
        self.hooks: dict[str, list] = {}
        self.commands: dict[str, dict] = {}
        cli = types.SimpleNamespace(session_id=cli_session) if cli_session is not None else None
        self._manager = types.SimpleNamespace(_cli_ref=cli)

    def cli(self, session_id: str | None) -> None:
        """Be the interactive CLI on ``session_id`` (Hermes's cli.py sets ``_cli_ref``), or the gateway with None."""
        self._manager._cli_ref = types.SimpleNamespace(session_id=session_id) if session_id is not None else None

    def register_hook(self, name, callback):
        self.hooks.setdefault(name, []).append(callback)

    def register_command(self, name, handler, description="", args_hint=""):
        self.commands[name] = {"handler": handler, "description": description, "args_hint": args_hint}

    def fire(self, hook: str, **kwargs):
        """As Hermes's invoke_hook: the schema tag added, each callback's non-None result collected."""
        kwargs.setdefault("telemetry_schema_version", SCHEMA)
        return [result for callback in self.hooks.get(hook, []) for result in [callback(**kwargs)] if result is not None]


@contextmanager
def plugin_env(sb, **extra):
    """The plugin in-process with only the sandbox env (no real variable but PATH), a fresh tracker, cwd in the project."""
    plugin = load_plugin()
    env = {**sb.env, "TERMINAL_CWD": str(sb.project), **extra}
    env = {key: value for key, value in env.items() if value is not None}
    with mock.patch.dict(os.environ, env, clear=True):
        plugin.tracker = plugin.Tracker(plugin.core.resolve, version="0.19.0")
        plugin.spool = plugin.spool_for()
        plugin.core.forget_cache()  # a fresh Hermes process
        clock = plugin.now_ms
        plugin.errors["count"] = 0
        ctx = FakeCtx()
        plugin.register(ctx)
        try:
            yield plugin, ctx
        finally:
            plugin._drain_at_exit()  # what a callback handed to the drainer, as at Hermes's exit
            plugin.spool.close()  # the helper folds every frame it was sent, then exits
            plugin.now_ms = clock


def new_session_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6]


class Turn:
    """Builds one Hermes turn's hook kwargs. ``events`` is the list of (hook, kwargs) in firing order."""

    def __init__(self, session_id: str, prompt: str, *, model: str = "mock-model-1", provider: str = "custom", platform: str = "cli", first: bool = True,
                 start: float | None = None):
        self.sid = session_id
        self.task = str(uuid.uuid4())
        self.tid = f"{session_id}:{self.task}:{uuid.uuid4().hex[:8]}"
        self.model = model
        self.provider = provider
        self.platform = platform
        self.prompt = prompt
        self.history: list[dict] = [{"role": "user", "content": prompt}]
        self.calls = 0
        self.clock = time.time() if start is None else start
        self.events: list[tuple[str, dict, float]] = []
        if first:
            self._add("on_session_start", {"session_id": session_id, "model": model, "platform": platform})
        self._add("pre_llm_call", {"session_id": session_id, "task_id": self.task, "turn_id": self.tid, "user_message": prompt,
                                             "conversation_history": list(self.history), "is_first_turn": first, "model": model,
                                             "platform": platform, "sender_id": ""})

    def _add(self, hook: str, kwargs: dict, after: float = 0.001) -> None:
        self.clock += after
        self.events.append((hook, kwargs, self.clock))

    def _ids(self) -> dict:
        return {"session_id": self.sid, "task_id": self.task, "turn_id": self.tid}

    def api(self, *, text: str | None = None, tool: tuple[str, dict] | None = None, usage: tuple[int, int, int] = (1001, 21, 0), error: dict | None = None,
            retry: bool = False) -> str:
        """One provider attempt. With ``error`` it fails. With ``retry`` it is the next attempt of the last request, as Hermes
        makes it (conversation_loop.py:1215-1225): the same api_request_id and the same started_at."""
        if not (retry and self.calls):
            self.calls += 1
            self.request_start = self.clock + 0.001
        rid = f"{self.tid}:api:{self.calls}"
        start, end = (self.request_start if retry else self.clock + 0.001), self.clock + 0.251
        common = {**self._ids(), "api_request_id": rid, "platform": self.platform, "model": self.model, "provider": self.provider,
                  "base_url": "http://127.0.0.1:9/v1", "api_mode": "chat_completions", "api_call_count": self.calls}
        request = {"method": "POST", "body": {"model": self.model, "messages": [{"role": "system", "content": "You are Hermes Agent."}] + self.history,
                                              "tools": [{"type": "function", "function": {"name": "read_file"}}], "max_tokens": None}}
        self._add("pre_api_request", {**common, "user_message": self.prompt, "conversation_history": list(self.history),
                                                "request_messages": request["body"]["messages"], "message_count": len(self.history) + 1,
                                                "tool_count": 28, "approx_input_tokens": 1708, "request_char_count": 6832, "max_tokens": None,
                                                "started_at": start, "middleware_trace": [], "request": request})
        if error is not None:
            self.clock = end
            self._add("api_request_error", {**common, "api_duration": end - start, "started_at": start, "ended_at": end,
                                                      "status_code": error.get("status_code", 500), "retry_count": error.get("retry_count", 1),
                                                      "max_retries": 3, "retryable": True, "reason": error.get("reason", "server_error"),
                                                      "error": {"type": error.get("type", "InternalServerError"), "message": error.get("message", "upstream failed")},
                                                      "request": request}, after=0)
            return rid
        calls = []
        if tool:
            calls = [{"id": f"call_{self.calls}", "name": tool[0], "arguments": json.dumps(tool[1]), "provider_data": None}]
        inp, out, cache = usage
        canonical = {"input_tokens": inp, "output_tokens": out, "cache_read_tokens": cache, "cache_write_tokens": 0, "reasoning_tokens": 0,
                     "request_count": 1, "prompt_tokens": inp + cache, "total_tokens": inp + cache + out}
        message = types.SimpleNamespace(role="assistant", content=text, tool_calls=[types.SimpleNamespace(id=c["id"], name=c["name"], arguments=c["arguments"]) for c in calls])
        self.clock = end
        self._add("post_api_request", {**common, "api_duration": end - start, "started_at": start, "ended_at": end,
                                                 "finish_reason": "tool_calls" if calls else "stop", "message_count": len(self.history) + 1,
                                                 "response_model": self.model,
                                                 "response": {"model": self.model, "finish_reason": "tool_calls" if calls else "stop",
                                                              "assistant_message": {"role": "assistant", "content": text, "tool_calls": calls}, "usage": canonical},
                                                 "usage": canonical, "assistant_message": message, "assistant_content_chars": len(text or ""),
                                                 "assistant_tool_call_count": len(calls)}, after=0)
        self.history.append({"role": "assistant", "content": text, "tool_calls": calls})
        return calls[0]["id"] if calls else rid

    def tool(self, call_id: str, name: str, args: dict, result: str, *, status: str = "ok", pre: bool = True, post: bool = True,
             error_type: str | None = None, error_message: str | None = None) -> None:
        common = {"tool_name": name, "args": args, **self._ids(), "tool_call_id": call_id, "api_request_id": f"{self.tid}:api:{self.calls}"}
        if pre:
            self._add("pre_tool_call", {**common, "middleware_trace": []})
        if post:
            self._add("post_tool_call", {**common, "result": result, "duration_ms": 47, "status": status, "error_type": error_type,
                                         "error_message": error_message, "middleware_trace": []}, after=0.047)
            self.history.append({"role": "tool", "tool_call_id": call_id, "content": result})

    def end(self, answer: str | None, *, interrupted: bool = False) -> None:
        if answer is not None and not interrupted:
            self._add("post_llm_call", {**self._ids(), "user_message": self.prompt, "assistant_response": answer,
                                                  "conversation_history": list(self.history), "model": self.model, "platform": self.platform})
        self._add("on_session_end", {**self._ids(), "completed": not interrupted, "interrupted": interrupted, "model": self.model, "platform": self.platform})


def simple_turn(sid: str, prompt: str, *, path: str, result: str, answer: str, first: bool = True, tool_name: str = "read_file", args: dict | None = None,
                start: float | None = None) -> Turn:
    """Prompt -> model asks for one tool -> tool -> model answers."""
    turn = Turn(sid, prompt, first=first, start=start)
    args = args if args is not None else {"path": path}
    call = turn.api(tool=(tool_name, args))
    turn.tool(call, tool_name, args, result)
    turn.api(text=answer, usage=(1002, 22, 512))
    turn.end(answer)
    return turn


def finalize(sid: str, reason: str = "shutdown") -> tuple[str, dict]:
    return ("on_session_finalize", {"session_id": sid, "platform": "cli", "reason": reason})


def later(events: list, seconds: float) -> float:
    """A time ``seconds`` after the last timed event."""
    return max([event[2] for event in events if len(event) > 2] or [time.time()]) + seconds


def subagent_turn(sid: str, goal: str = "count to three") -> tuple[Turn, str, list]:
    """A turn that delegates in the background: the child stops after the parent turn ended, as in Hermes 0.19.0."""
    parent = Turn(sid, "delegate the counting")
    args = {"goal": goal, "context": "be brief"}
    call = parent.api(tool=("delegate_task", args))
    child_sid = new_session_id() + "c"
    parent._add("pre_tool_call", {"tool_name": "delegate_task", "args": args, **parent._ids(), "tool_call_id": call,
                                  "api_request_id": f"{parent.tid}:api:1", "middleware_trace": []})
    parent._add("subagent_start", {"parent_session_id": sid, "parent_turn_id": parent.tid, "parent_subagent_id": None, "child_session_id": child_sid,
                                   "child_subagent_id": "sa-0-2d056562", "child_role": "leaf", "child_goal": goal})
    parent._add("post_tool_call", {"tool_name": "delegate_task", "args": args, **parent._ids(), "tool_call_id": call, "api_request_id": f"{parent.tid}:api:1",
                                   "result": json.dumps({"status": "started", "mode": "background"}), "duration_ms": 33, "status": "ok",
                                   "error_type": None, "error_message": None, "middleware_trace": []}, after=0.033)
    parent.history.append({"role": "tool", "tool_call_id": call, "content": "started"})
    # The child runs on its own thread from here, while the parent makes its second request.
    child = Turn(child_sid, goal, platform="subagent")
    child.clock = parent.clock
    child.tid = f"{child_sid}:sa-0-2d056562:1a19767f"
    child.events = []
    child._add("on_session_start", {"session_id": child_sid, "model": child.model, "platform": "subagent"})
    child._add("pre_llm_call", {"session_id": child_sid, "task_id": "sa-0-2d056562", "turn_id": child.tid, "user_message": goal,
                                "conversation_history": [], "is_first_turn": True, "model": child.model, "platform": "subagent", "sender_id": ""})
    child_call = child.api(tool=("terminal", {"command": "echo one two three"}))
    child.tool(child_call, "terminal", {"command": "echo one two three"}, json.dumps({"output": "one two three", "exit_code": 0}))
    child.api(text="one two three", usage=(400, 5, 0))
    child.end("one two three")
    parent.api(text="The child is counting in the background.")
    parent.end("The child is counting in the background.")
    stop_at = max(child.clock, parent.clock) + 0.5
    stop = ("subagent_stop", {"parent_session_id": sid, "parent_turn_id": "", "child_session_id": child_sid, "child_role": "leaf",
                              "child_summary": "one two three", "child_status": "completed", "duration_ms": 1234}, stop_at)
    events = sorted(parent.events + child.events, key=lambda event: event[2]) + [stop]
    return parent, child_sid, events


def fire_all(ctx: FakeCtx, events: list, plugin=None) -> list:
    """Fire each (hook, kwargs[, time]) in order. With ``plugin``, its clock reads each event's time."""
    results = []
    last = time.time()
    for event in events:
        hook, kwargs = event[0], event[1]
        if plugin is not None:
            last = event[2] if len(event) > 2 else last + 0.001
            plugin.now_ms = lambda at=last: int(at * 1000)
        results += ctx.fire(hook, **kwargs)
    return results


def state_files(state: Path) -> list[Path]:
    return sorted(path for path in state.rglob("*") if path.is_file()) if state.exists() else []


def flush_vendor(sb, env: dict | None = None):
    """The flusher from the plugin's vendored core, as the plugin launches it."""
    return sb.flush({"PYTHONPATH": str(VENDOR_CORE), **(env or {})})
