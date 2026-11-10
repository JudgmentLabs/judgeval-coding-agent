"""Judgment tracing for Hermes Agent: a native plugin of read-only observer hooks.

Each hook runs in Hermes's process, on its thread, so each callback is short: kwargs ->
tracker (records, no I/O) -> core capture policy and redaction -> one inbox file. No network,
no subprocess, except one detached sender at the end of a turn, a subagent, or a session.
Callbacks return None, so Hermes never blocks or changes anything because of this plugin, and
any error inside one is swallowed.
"""

from __future__ import annotations

import logging
import os
import threading
import time

from . import core_bridge as core
from .tracker import HOOKS, Tracker

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


def terminal_cwd() -> str:
    """Where Hermes's tools run: TERMINAL_CWD when Hermes set it, else this process's directory."""
    return os.environ.get("TERMINAL_CWD") or os.getcwd()


def _resolve(session_id: str):
    return core.resolve(session_id, terminal_cwd())


tracker = Tracker(_resolve, version=_hermes_version(), cwd=terminal_cwd)


def now_ms() -> int:
    return int(time.time() * 1000)


# Hermes fires hooks from several threads (a child agent runs on its own). One lock from the
# tracker step to the inbox write keeps inbox files in the order the tracker saw the events.
_order = threading.Lock()


def handle(hook: str, kwargs: dict) -> None:
    """One hook: records out of the tracker, each through the core policy into the inbox."""
    state = None
    with _order:
        step = tracker.step(hook, kwargs, now_ms())
        for gate, record in step.records:
            core.spool(core.prepare(record, gate), gate.state)
            state = gate.state
    if step.flush and state is not None:
        core.launch_sender(state)


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


def _cli_session() -> str:
    """The interactive CLI's session id. It exists before the first message, when no hook has run yet."""
    try:
        cli = getattr(getattr(_ctx["ctx"], "_manager", None), "_cli_ref", None)
        return str(getattr(cli, "session_id", "") or "")
    except Exception:
        return ""


def _session_id() -> str:
    """The session /judgment applies to: the CLI's session, else Hermes's session context, else the newest main session here.

    Building a child agent points the session context at the child, so a child's id is never used.
    """
    cli = _cli_session()
    if cli:
        return cli
    found = ""
    try:
        from gateway.session_context import get_session_env

        found = str(get_session_env("HERMES_SESSION_ID", "") or "")
    except Exception:
        found = os.environ.get("HERMES_SESSION_ID", "")
    if found and found not in tracker.child_ids:
        return found
    return tracker.last_session


def judgment_command(raw_args: str = "") -> str:
    try:
        return core.command((raw_args or "").split()[0] if (raw_args or "").split() else "status", terminal_cwd(), _session_id())
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
