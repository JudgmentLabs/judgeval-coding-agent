"""Judgment tracing for Hermes Agent: a native plugin of read-only observer hooks.

Each hook runs in Hermes's process, on its thread, so each callback is short and does no I/O:

    kwargs -> normalize (the turn's workspace, as Hermes resolves it) -> tracker (records, pure)
           -> spool (frames down a pipe to one helper process, which redacts and writes the inbox)

Callbacks return None, so Hermes never blocks or changes anything because of this plugin, and
any error inside one is swallowed.
"""

from __future__ import annotations

import atexit
import collections
import logging
import os
import sys
import threading
import time

from . import core_bridge as core
from .spooler import spool_for
from .tracker import HOOKS, Kind, Tracker

log = logging.getLogger("judgment_tracing")
errors = {"count": 0}  # callbacks that failed inside; never raised to Hermes
# Nanoseconds each callback took, per hook, the newest TIMING_KEEP. Written to the file
# JUDGMENT_HERMES_STATS names, if any, when a session finalizes.
timings: dict[str, list[int]] = {}
TIMING_KEEP = 10000


def _hermes_version() -> str:
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:
        return ""


def workspace(task_id: str = "") -> str:
    """The folder Hermes works in for this task, resolved the way Hermes 0.19.0 resolves it.

    1. A cwd registered for the task (TUI, desktop, ACP, batch: tools/terminal_tool.py
       resolve_task_overrides, which tools/file_tools.py reads first too).
    2. The session's pinned cwd, then TERMINAL_CWD, then the launch folder: agent/runtime_cwd.py
       resolve_agent_cwd, which reads a per-context variable a multi-session gateway sets.
    Only modules Hermes already imported are read; outside Hermes, TERMINAL_CWD or the process folder.
    """
    terminal = sys.modules.get("tools.terminal_tool")
    if task_id and terminal is not None:
        try:
            cwd = terminal.resolve_task_overrides(task_id).get("cwd")
            if isinstance(cwd, str) and os.path.isabs(cwd) and os.path.isdir(cwd):
                return cwd
        except Exception:
            pass
    runtime = sys.modules.get("agent.runtime_cwd")
    if runtime is not None:
        try:
            return str(runtime.resolve_agent_cwd())
        except Exception:
            pass
    return os.environ.get("TERMINAL_CWD") or os.getcwd()


tracker = Tracker(core.resolve, version=_hermes_version())
spool = spool_for()


def now_ms() -> int:
    return int(time.time() * 1000)


# Hermes fires hooks from several threads (a child agent runs on its own). Frames must reach the
# pipe in the order the tracker saw the events, but no callback should wait on another's write:
#   _steps    the tracker step and the append to _pending, in one short critical section
#   _writer   held by whichever thread is writing; it writes pending batches in order, and looks
#             again after letting go, so a batch appended meanwhile is never left behind
# A callback writes for at most WRITE_BUDGET_S; past that (other threads keep appending), the
# drainer thread takes over, and the callback returns. At exit, whatever is pending is written.
_steps = threading.Lock()
_writer = threading.Lock()
_pending: collections.deque = collections.deque()
_wake = threading.Event()
_drainer: dict = {"thread": None}
WRITE_BUDGET_S = 0.001


def normalize(hook: str, kwargs: dict) -> dict:
    """The impure inputs a turn start needs, read before any lock: the workspace and the gate (files)."""
    if hook != "pre_llm_call" or kwargs.get("platform") == "subagent":
        return kwargs
    folder = workspace(str(kwargs.get("task_id") or ""))
    sid = str(kwargs.get("session_id") or "")
    key = tracker.key_of(sid) or sid
    return {**kwargs, "_workspace": folder, "_gate_key": key, "_gate": core.resolve(key, folder)}


def _write_pending(budget: float | None = -1.0) -> None:
    """Write pending batches, in order. With a budget (default WRITE_BUDGET_S), hand what is left to the drainer once it runs out."""
    budget = WRITE_BUDGET_S if budget == -1.0 else budget
    deadline = None if budget is None else time.perf_counter() + budget
    while _pending and _writer.acquire(blocking=deadline is None):
        try:
            while _pending:
                if deadline is not None and time.perf_counter() > deadline:
                    _hand_off()
                    return
                records, flush_state = _pending.popleft()
                spool.send(records, flush_state, None)
        finally:
            _writer.release()


def _hand_off() -> None:
    thread = _drainer["thread"]
    if thread is None or not thread.is_alive():
        thread = _drainer["thread"] = threading.Thread(target=_drain_forever, name="judgment-drainer", daemon=True)
        thread.start()
    _wake.set()


def _drain_forever() -> None:
    while True:
        _wake.wait()
        _wake.clear()
        try:
            _write_pending(None)
        except Exception:
            errors["count"] += 1


@atexit.register
def _drain_at_exit() -> None:
    try:
        _write_pending(None)
    except Exception:
        pass


def handle(hook: str, kwargs: dict) -> None:
    """One hook: records out of the tracker, down the pipe to the helper."""
    kwargs = normalize(hook, kwargs)
    with _steps:
        step = tracker.step(hook, kwargs, now_ms())
        if step.records:
            _pending.append((step.records, step.records[-1][0].state if step.flush else None))
    if step.records:
        _write_pending()
    for key, alias in step.aliases:
        core.copy_session_mode(key, alias)


def _dump_timings() -> None:
    path = os.environ.get("JUDGMENT_HERMES_STATS")
    if not path:
        return
    import json

    with open(path, "a", encoding="utf-8") as handle_:
        handle_.write(json.dumps({"pid": os.getpid(), "errors": errors["count"], "ns": timings}) + "\n")


def _callback(hook: str):
    def callback(**kwargs):
        start = time.perf_counter_ns()
        try:
            handle(hook, kwargs)
        except Exception as exc:  # fail open: Hermes keeps going, and nothing of the payload is logged
            errors["count"] += 1
            log.debug("judgment tracing: %s failed: %s", hook, type(exc).__name__)
        took = time.perf_counter_ns() - start
        try:
            kept = timings.setdefault(hook, [])
            if len(kept) >= TIMING_KEEP:
                del kept[: TIMING_KEEP // 2]
            kept.append(took)
            if hook == "on_session_finalize":
                _dump_timings()
        except Exception:
            pass
        return None

    callback.__name__ = f"judgment_{hook}"
    return callback


_ctx = {"ctx": None}


def requester() -> str:
    """The conversation a /judgment command came from, or '' when Hermes doesn't say.

    Hermes passes a plugin command only its arguments (hermes_cli/plugins.py register_command). The
    interactive CLI runs one conversation and its PluginManager holds it (``_cli_ref``, cli.py), so
    its session id names the requester; a compression rotation maps back to the conversation's key.
    The gateway and the TUI serve many conversations and dispatch plugin commands before binding the
    message's session (gateway/run.py: dispatch at 11194, binding at 12128), so nothing there names
    the requester, and any process-wide id (HERMES_SESSION_ID, the newest session) could name another
    chat. Those get ''.
    """
    try:
        cli = getattr(getattr(_ctx["ctx"], "_manager", None), "_cli_ref", None)
        sid = str(getattr(cli, "session_id", "") or "")
    except Exception:
        return ""
    if not sid:
        return ""
    key = tracker.key_of(sid) or sid
    return "" if tracker.kind_of(key) is Kind.CHILD else key


def judgment_command(raw_args: str = "") -> str:
    try:
        words = (raw_args or "").split()
        return core.command(words[0] if words else "status", workspace(), requester())
    except Exception as exc:
        return f"Judgment tracing: the command failed ({type(exc).__name__})."


def register(ctx) -> None:
    _ctx["ctx"] = ctx
    try:
        core.warm()
    except Exception:
        pass
    for hook in HOOKS:
        ctx.register_hook(hook, _callback(hook))
    description = "Judgment tracing for this session: on, off, or status"
    try:
        ctx.register_command("judgment", judgment_command, description=description, args_hint="on|off|status")
    except TypeError:  # Hermes before args_hint
        ctx.register_command("judgment", judgment_command, description=description)
