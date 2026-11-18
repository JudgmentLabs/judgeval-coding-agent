"""Judgment tracing for Hermes Agent: a native plugin of read-only observer hooks.

Each hook runs in Hermes's process, on its thread, so each callback is short, does no I/O, and never
waits on another thread's I/O:

    kwargs -> normalize (the turn's workspace and gate, as Hermes resolves them; files, cached)
           -> tracker.step (records; pure tables)  -> outbox.put (one queue; spooler.py)
    then, off the callback: I/O thread -> helper process (redacts, publishes the inbox, starts the sender)

``_lock`` serializes every tracker access and the put that follows a step, so records reach the
queue in the order the tracker made them. Nothing slow runs under it: the bulky payloads (tool
arguments and results, the model's reply) are snapshotted before it is taken, by one bounded pass each
(snapshot.py), and only for a call the tracker will record.

Callbacks return None, so Hermes never blocks or changes anything because of this plugin, and
any error inside one is swallowed.

Fork: a forked child gets a fresh tracker, lock, and queue before any callback runs (``_after_fork``
here, ``spooler._after_fork`` for the queue's fds and thread). The parent's turns stay the parent's.
"""

from __future__ import annotations

import atexit
import logging
import os
import sys
import threading
import time

from . import core_bridge as core
from .spooler import spool_for
from .tracker import HOOKS, Kind, Tracker, reply_of, text_of
from . import snapshot

log = logging.getLogger("judgment_tracing")
errors = {"count": 0}  # callbacks that failed inside; never raised to Hermes
# Nanoseconds each callback took, per hook, the newest TIMING_KEEP. Kept only when JUDGMENT_HERMES_STATS names a
# file (read once, here), and written to it when a session finalizes. tests/bench_hermes.py reads it.
STATS_PATH = os.environ.get("JUDGMENT_HERMES_STATS") or ""
timings: dict[str, list[int]] = {}
TIMING_KEEP = 10000


def _hermes_version() -> str:
    """Hermes's own version. A git install updates its checkout without reinstalling the package, so the
    package metadata keeps the version it was first installed at; ``hermes_cli.__version__`` follows the checkout."""
    try:
        running = getattr(sys.modules.get("hermes_cli"), "__version__", "")
        if isinstance(running, str) and running not in ("", "0.0.0"):
            return running
    except Exception:
        pass
    try:
        from importlib.metadata import version

        return version("hermes-agent")
    except Exception:
        return ""


def workspace(task_id: str = "") -> str:
    """The folder Hermes works in for this task, resolved the way Hermes resolves it (0.19.0 to 0.23.2).

    1. A cwd registered for the task (TUI, desktop, ACP, batch: tools/terminal_tool.py
       resolve_task_overrides, which tools/file_tools.py reads first too).
    2. The session's pinned cwd, then TERMINAL_CWD, then the launch folder: agent/runtime_cwd.py
       resolve_agent_cwd, which reads a per-context variable a multi-session gateway sets.
    Only modules Hermes already imported are read; outside Hermes, TERMINAL_CWD or the process folder.

    "" when Hermes refuses to resolve the profile's workspace (tools/terminal_scope.py
    TerminalPolicyUnavailable): the turn is then untraced. Falling back to TERMINAL_CWD or the process
    folder would apply the launch profile's opt-in to another profile's conversation.
    """
    terminal = sys.modules.get("tools.terminal_tool")
    if task_id and terminal is not None:
        try:
            cwd = terminal.resolve_task_overrides(task_id).get("cwd")
            if isinstance(cwd, str) and os.path.isabs(cwd) and os.path.isdir(cwd):
                return cwd
        except Exception as exc:
            if _refused(exc):
                return ""
    runtime = sys.modules.get("agent.runtime_cwd")
    if runtime is not None:
        try:
            return str(runtime.resolve_agent_cwd())
        except Exception as exc:
            if _refused(exc):
                return ""
    return os.environ.get("TERMINAL_CWD") or os.getcwd()


def _refused(exc: BaseException) -> bool:
    """Is this Hermes's refusal to resolve a workspace? By name: its module is read only when Hermes imported it."""
    return any(kind.__name__ == "TerminalPolicyUnavailable" for kind in type(exc).__mro__)


def new_tracker() -> Tracker:
    return Tracker(core.resolve, version=_hermes_version(), needs_alias=core.needs_alias)


tracker = new_tracker()
spool = spool_for()
_lock = threading.Lock()


def _after_fork() -> None:
    global tracker, _lock
    _lock = threading.Lock()  # another thread may have held it at fork: it would stay held forever here
    tracker = Tracker(core.resolve, version=tracker.version, needs_alias=core.needs_alias)


os.register_at_fork(after_in_child=_after_fork)


def now_ms() -> int:
    return int(time.time() * 1000)


# Hooks whose payload is snapshotted before the lock: hook -> (kwarg the tracker reads, how it is made).
SNAPSHOTS = {
    "post_tool_call": (("_args", lambda kw: snapshot.take(kw.get("args"))), ("_result", lambda kw: text_of(kw.get("result")))),
    "post_api_request": (("_reply", reply_of),),
}


def normalize(hook: str, kwargs: dict) -> dict:
    """What a step needs that is slow or impure, made before the lock.

    pre_llm_call: the workspace and the gate (files, cached). The gate is resolved for the conversation
    the turn will run under (tracker.key_for), which for a session id that preflight compression just
    rotated is the conversation it was rotated from.
    post_tool_call, post_api_request: the payload's snapshot (SNAPSHOTS), only when the call is traced.
    """
    if hook in SNAPSHOTS:
        with _lock:
            wanted = tracker.traced(kwargs)
        if not wanted:
            return kwargs
        return {**kwargs, **{name: make(kwargs) for name, make in SNAPSHOTS[hook]}}
    if hook != "pre_llm_call" or kwargs.get("platform") == "subagent":
        return kwargs
    folder = workspace(str(kwargs.get("task_id") or ""))
    with _lock:
        key = tracker.key_for(kwargs)
    return {**kwargs, "_workspace": folder, "_gate_key": key, "_gate": core.resolve(key, folder)}


def entries(step) -> list:
    """A step as queue items: its records, a sender start when a turn closed, and the rotated ids' /judgment copies."""
    items = [("rec", gate, record) for gate, record in step.records]
    if step.flush and step.records:
        items.append(("flush", None, (step.records[-1][0].state, core.profile_env())))
    if step.aliases:
        env = core.profile_env()  # read here, in the profile's own context; the writer thread has only the launch profile's
        items += [("alias", None, (key, alias, env)) for key, alias in step.aliases]
    return items


@atexit.register
def _drain_at_exit() -> None:
    """Hand what is queued to the helper, waiting at most spooler.EXIT_WAIT_S. The helper finishes after Hermes exits."""
    try:
        spool.close()
    except Exception:
        pass


def handle(hook: str, kwargs: dict) -> None:
    """One hook: records out of the tracker, into the queue."""
    kwargs = normalize(hook, kwargs)
    with _lock:
        step = tracker.step(hook, kwargs, now_ms())
        items = entries(step)
        if items:
            spool.put(items)


def _dump_timings() -> None:
    import json

    with open(STATS_PATH, "a", encoding="utf-8") as handle_:
        handle_.write(json.dumps({"pid": os.getpid(), "errors": errors["count"], "ns": timings}) + "\n")


def _callback(hook: str):
    def callback(**kwargs):
        start = time.perf_counter_ns()
        try:
            handle(hook, kwargs)
        except Exception as exc:  # fail open: Hermes keeps going, and nothing of the payload is logged
            errors["count"] += 1
            log.debug("judgment tracing: %s failed: %s", hook, type(exc).__name__)
        if not STATS_PATH:
            return None
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


def requester(session_id: str = "") -> tuple:
    """(key, other session ids) of the conversation a /judgment command came from, or ('', ()) when Hermes doesn't say.
    Call it holding ``_lock``.

    Hermes 0.22.3 on passes a command handler the ``session_id`` of the conversation that sent it, on every
    surface (hermes_cli/plugin_command_dispatch.py), and its interactive CLI runs plugins in a gateway
    process, so that id is the only thing that names the requester there. A chat with no session yet gets ''.

    Earlier versions pass only the arguments. Their interactive CLI runs one conversation and its
    PluginManager holds it (``_cli_ref``, cli.py), so its session id names the requester. Their gateway and
    TUI serve many conversations and name none, and any process-wide id (HERMES_SESSION_ID, the newest
    session) could name another chat. Those get ''.

    A compression rotation maps back to the conversation's key.
    """
    sid = session_id if isinstance(session_id, str) else ""
    if not sid:
        try:
            cli = getattr(getattr(_ctx["ctx"], "_manager", None), "_cli_ref", None)
            sid = str(getattr(cli, "session_id", "") or "")
        except Exception:
            return "", ()
    if not sid:
        return "", ()
    key = tracker.key_of(sid) or sid
    if tracker.kind_of(key) is Kind.CHILD:
        return "", ()
    return key, tuple(sorted(tracker.aliases_of(key) - {key}))


def judgment_command(raw_args: str = "", session_id: str = "") -> str:
    try:
        words = (raw_args or "").split()
        folder = workspace()
        # One step with the tracker: a rotation lands before this (its id is among the aliases) or after it (its copy
        # of the mode, queued by that step, then reads what this wrote).
        with _lock:
            key, aliases = requester(session_id)
            return core.command(words[0] if words else "status", folder, key, aliases=aliases)
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
