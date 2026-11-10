"""Every call from the Hermes plugin into judgment_core goes through this module.

judgment_core is vendored next to this file (``vendor/judgment-core``, synced by scripts/sync-core.sh).
When the core's opt-in or redaction API changes, this is the one file to update.

    resolve(session_id, cwd)  -> Gate | None   opt-in walk-up, kill switch, session mode, capture,
                                               rules and secret values, export stamp. None = untraced.
    prepare(record, gate)     -> record        the hook layer of the capture policy, as prepare.jq does it:
                                               metadata strips text, sanitized redacts and marks sensitive
                                               tool input, full keeps it. Then the stamps record() adds.
    spool(record, state)                       one inbox file, as core/bin/judgment-hook record() writes it
    launch_sender(state)                       the core flusher, detached, never waited on
    command(action, cwd, session_id) -> str    /judgment on|off|status
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

CORE = Path(__file__).resolve().parent / "vendor" / "judgment-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from judgment_core import __version__ as CORE_VERSION  # noqa: E402
from judgment_core import optin, redact  # noqa: E402
from judgment_core.config import DEFAULT_API_URL, load_credentials  # noqa: E402

__all__ = ["CORE", "CORE_VERSION", "Gate", "warm", "resolve", "prepare", "spool", "launch_sender", "command", "state_dir", "forced_off"]

CAPTURE_TEXT = ("full", "sanitized")


@dataclass(frozen=True)
class Gate:
    """One turn's opt-in decision. Everything prepare needs, resolved once at turn start."""

    root: str
    capture: str  # full | sanitized | metadata, as optin.resolve_capture decides
    environment: str
    rules: object  # redact.Ruleset
    secrets: tuple  # ((name, value), ...) secret env values, longest first
    home: str
    stamp: tuple  # (("export_fp", ...), ...) or ()
    state: Path


def warm() -> None:
    """Compile the redaction rules now, at plugin load, so the first turn does not pay for it. Writes nothing."""
    redact.ruleset({})
    redact.env_secrets({})
    redact.sensitive({"tool_input": {"path": "x"}}, redact.ruleset({}))


def forced_off(env: dict | None = None) -> bool:
    return optin.tracing_forced_off(env)


def state_dir(env: dict | None = None) -> Path:
    return optin.state_dir(env)


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _stamp(env: dict) -> tuple:
    """What the hook records about the destination: the key's fingerprint, never the key."""
    creds = load_credentials(env)
    key, project = creds.get("JUDGMENT_API_KEY", ""), creds.get("JUDGMENT_PROJECT_ID", "")
    if not key or not project:
        return ()
    pairs = [("export_fp", _fingerprint(key)), ("export_project", project), ("export_url", creds.get("JUDGMENT_API_URL") or DEFAULT_API_URL)]
    if creds.get("JUDGMENT_ORG_ID"):
        pairs.append(("export_org", creds["JUDGMENT_ORG_ID"]))
    return tuple(pairs)


def resolve(session_id: str, cwd: str, env: dict | None = None) -> Gate | None:
    """The gate for a turn starting now in ``cwd``, or None when nothing may be recorded."""
    env = dict(os.environ if env is None else env)
    if not session_id or optin.tracing_forced_off(env):
        return None
    found = optin.find_optin(cwd or os.getcwd())
    if found is None:
        return None
    root, file = found
    project = optin.load_project(file)
    if not project["enabled"]:
        return None
    if optin.read_session_mode(root, session_id, project["sessionDefault"], env) != "on":
        return None
    capture = project["capture"]
    rules = redact.ruleset(project)
    api_key = load_credentials(env).get("JUDGMENT_API_KEY", "")
    secrets = tuple(redact.env_secrets(env, api_key)) if capture == "sanitized" else ()
    return Gate(str(root), capture, project["environment"], rules, secrets, str(env.get("HOME") or ""), _stamp(env), optin.state_dir(env))


def prepare(record: dict, gate: Gate) -> dict:
    """Mirror of judgment_prepare in core/bin/prepare.jq, then record()'s export stamp."""
    if gate.capture == "metadata":
        out = {**redact.strip_text(record), "prompt": "", "output": ""}
    elif gate.capture == "sanitized":
        try:
            out = redact.sanitize_event(record, gate.rules, list(gate.secrets), gate.home)
        except Exception:  # a rule that fails drops the event's text, never sends it raw
            out = {**redact.strip_text(record), "redaction_error": True}
    else:
        out = dict(record)
    out["capture"] = gate.capture
    if not isinstance(out.get("ts"), (int, float)):
        out["ts"] = int(time.time() * 1000)
    out["environment"] = gate.environment
    out["project_root"] = gate.root
    if "export_fp" not in out:
        out.update(gate.stamp)
    return out


_seq = itertools.count()
_spool_lock = threading.Lock()


def _private_dir(path: Path) -> None:
    if not path.is_dir():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)


def spool(record: dict, state: Path) -> None:
    """One file per event in ``inbox/``, owner-only, appearing whole (written aside, then linked in).

    The flusher folds files in (mtime, name) order and skips any that sort before the last one it
    folded, so files appear one at a time, in order, under a lock.
    """
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
    inbox = state / "inbox"
    scratch = state / "inbox.tmp"
    pid = os.getpid()
    with _spool_lock:
        _private_dir(state)
        _private_dir(inbox)
        _private_dir(scratch)
        for _ in range(20):
            name = f"{pid}.{next(_seq):010d}.json"
            tmp = scratch / name
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)
            try:
                os.link(tmp, inbox / name)  # never replaces a file a crashed process left
                return
            except FileExistsError:
                continue
            finally:
                os.unlink(tmp)


_senders: list = []


def launch_sender(state: Path, env: dict | None = None) -> bool:
    """Start ``python -m judgment_core.flush`` in its own session, output to flush.log. Never waits on it.

    start_new_session puts the child in a new session before exec, and Popen returns only after
    exec, so killing Hermes's process group right after this cannot reach the sender.
    """
    env = dict(os.environ if env is None else env)
    if env.get("JUDGMENT_FLUSH") == "0" or optin.tracing_forced_off(env):
        return False
    for old in list(_senders):  # reap finished senders so none stays a zombie
        if old.poll() is not None:
            _senders.remove(old)
    _private_dir(state)
    log = os.open(state / "flush.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        env["PYTHONPATH"] = str(CORE) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        env.pop("PYTHONSTARTUP", None)
        proc = subprocess.Popen(
            [sys.executable or "python3", "-m", "judgment_core.flush"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            close_fds=True,
            start_new_session=True,
            cwd="/",
            env=env,
        )
    finally:
        os.close(log)
    _senders.append(proc)
    return True


def command(action: str, cwd: str, session_id: str, env: dict | None = None) -> str:
    """/judgment on|off|status. Same rules and messages as core/bin/judgment-session."""
    env = dict(os.environ if env is None else env)
    action = (action or "status").strip().lower()
    if action not in ("on", "off", "status"):
        return "Usage: /judgment on|off|status"
    if action in ("on", "off"):
        if optin.tracing_forced_off(env):
            return "Judgment tracing is off. JUDGMENT_TRACING=0."
        found = optin.find_optin(cwd)
        if found is None:
            return "Judgment tracing is off. No .judgment/tracing.json was found above this directory."
        root, file = found
        if not optin.load_project(file)["enabled"]:
            return "Judgment tracing is off. .judgment/tracing.json has enabled set to false."
        if not session_id:
            return "Judgment tracing needs a session id to change this conversation."
        _private_dir(optin.state_dir(env))
        optin.write_session_mode(root, session_id, action, env)
    mode, message = optin.session_status(cwd, session_id, env)
    if action == "on" and mode == "on":
        return "Judgment tracing is on for this session. It applies from the next message."
    if action == "off":
        return "Judgment tracing is off for this session. It applies from the next message."
    return message
