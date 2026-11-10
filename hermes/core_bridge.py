"""Every call from the Hermes plugin into judgment_core goes through this module.

judgment_core is vendored next to this file (``vendor/judgment-core``, synced by scripts/sync-core.sh).
When the core's opt-in or redaction API changes, this is the one file to update.

    resolve(key, workspace)  -> Gate | None    opt-in walk-up, kill switch, session mode, capture,
                                               rules and secret values, export stamp. None = untraced.
    prepare(record, gate)     -> record        the hook layer of the capture policy:
                                               metadata strips text, sanitized redacts, full keeps it.
                                               Then the stamps record() adds. See PREPARE below.
    spool(record, state)                       one inbox file, as core/bin/judgment-hook record() writes it
    launch_sender(state, env)                  the core flusher, detached, never waited on
    command(action, cwd, key) -> str           /judgment on|off|status
    copy_session_mode(key, alias)              a rotated session id keeps its conversation's on/off

PREPARE, for one record, as plain stages (each takes and returns a record):

    sanitized:  fit    every text longer than TEXT_CAP is redacted, then cut (never cut, then redacted)
                keys   every dict key in a text field redacted; a key that then collides gets "#2", "#3"
                text   core redact.sanitize_event: sensitive tool calls blank the text, the rest redacted
                meta   the metadata strings Hermes or a model chose (model, provider, tool name and call id,
                       child role, cwd) redacted too; the ids the plugin routes by are Hermes's own
                audit  ``redactions`` counts every placeholder, in text, keys, and metadata
    metadata:   strip, then meta and audit
    full:       cut only
"""

from __future__ import annotations

import functools
import hashlib
import itertools
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CORE = Path(__file__).resolve().parent / "vendor" / "judgment-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from judgment_core import __version__ as CORE_VERSION  # noqa: E402
from judgment_core import optin, redact  # noqa: E402
from judgment_core.config import DEFAULT_API_URL, load_credentials  # noqa: E402

__all__ = ["CORE", "CORE_VERSION", "Gate", "warm", "resolve", "prepare", "spool", "launch_sender", "command", "copy_session_mode",
           "state_dir", "forced_off", "TEXT_CAP", "CUT_MARK"]

# The most text one field keeps. Spans cut at 64 KB anyway.
TEXT_CAP = 64 * 1024
# A secret cut by the redaction window's end sits in the window's last MARGIN characters, which are dropped.
MARGIN = 4 * 1024
CUT_MARK = "… [truncated]"
# Metadata strings that come from Hermes's provider, the model, or Hermes's config, not from the plugin.
UNTRUSTED_META = ("model", "provider", "tool_name", "tool_call_id", "agent_type", "agent_name", "parent_call_id", "cwd", "harness_version")
_NO_PATTERNS = redact.Ruleset((), (), ())


@dataclass(frozen=True)
class Gate:
    """One turn's opt-in decision: plain data, everything prepare needs, resolved once at turn start."""

    root: str
    capture: str  # full | sanitized | metadata, as optin.resolve_capture decides
    environment: str
    patterns: tuple  # the project's redactPatterns
    paths: tuple  # the project's redactPaths
    secrets: tuple  # ((name, value), ...) secret env values, longest first
    home: str
    stamp: tuple  # (("export_fp", ...), ...) or ()
    state: str

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Gate":
        pairs = lambda items: tuple(tuple(item) for item in items)  # noqa: E731
        return cls(data["root"], data["capture"], data["environment"], tuple(data["patterns"]), tuple(data["paths"]),
                   pairs(data["secrets"]), data["home"], pairs(data["stamp"]), data["state"])


@functools.lru_cache(maxsize=64)
def _rules(patterns: tuple, paths: tuple) -> redact.Ruleset:
    """Compiled once per distinct project config, not once per call."""
    return redact.ruleset({"redactPatterns": list(patterns), "redactPaths": list(paths)})


def rules(gate: Gate) -> redact.Ruleset:
    return _rules(gate.patterns, gate.paths)


def warm() -> None:
    """Compile the redaction rules now, at plugin load, so the first turn does not pay for it. Writes nothing."""
    _rules((), ())
    redact.env_secrets({})
    redact.sensitive({"tool_input": {"path": "x"}}, _rules((), ()))


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


# What a turn start reads from files, kept GATE_TTL seconds so a turn start makes no file syscall in
# the steady state (under GIL contention each syscall can wait a whole switch interval). /judgment
# clears it, so its own change applies at the next turn, and so does an edit to tracing.json (one
# stat a turn checks it). A new tracing.json above the workspace, new credentials, or another
# process's session switch applies within GATE_TTL.
GATE_TTL = 2.0
_bases: dict = {}  # (workspace, env fingerprint) -> (expires, base dict or None)
_overrides: dict = {}  # sessions folder -> (expires, file names in it)


def forget_cache() -> None:
    _bases.clear()
    _overrides.clear()


def _base(workspace: str, env: dict) -> dict | None:
    """The per-workspace part of a gate: the project, its rules, the destination stamp, the secret values."""
    found = optin.find_optin(workspace)
    if found is None:
        return None
    root, file = found
    version = _version(file)
    project = optin.load_project(file)
    if not project["enabled"]:
        return {"file": file, "version": version, "off": True}
    capture = project["capture"]
    api_key = load_credentials(env).get("JUDGMENT_API_KEY", "")
    # Metadata capture keeps names a model or provider chose, so it redacts them with the same values.
    secrets = tuple(redact.env_secrets(env, api_key)) if capture != "full" else ()
    return {"root": root, "file": file, "version": version, "project": project, "secrets": secrets, "stamp": _stamp(env), "state": optin.state_dir(env)}


def _version(file) -> tuple:
    try:
        info = os.stat(file)
    except OSError:
        return ()
    return (info.st_mtime_ns, info.st_size, info.st_ino)


def _session_mode(base: dict, key: str, env: dict, now: float) -> str:
    """on|off for the conversation: its override file when it has one, else the project's default."""
    folder = optin.session_override_path(base["root"], key, env).parent
    names = _overrides.get(str(folder))
    if names is None or names[0] < now:
        try:
            names = (now + GATE_TTL, frozenset(os.listdir(folder)))
        except OSError:
            names = (now + GATE_TTL, frozenset())
        _overrides[str(folder)] = names
    default = base["project"]["sessionDefault"]
    if optin.safe_id(key) not in names[1]:
        return "off" if default == "off" else "on"
    return optin.read_session_mode(base["root"], key, default, env)


def resolve(key: str, workspace: str, env: dict | None = None) -> Gate | None:
    """The gate for a turn of conversation ``key`` starting now in ``workspace``, or None when nothing may be recorded."""
    env = dict(os.environ if env is None else env)
    if not key or not workspace or optin.tracing_forced_off(env):
        return None
    now = time.monotonic()
    slot = (workspace, hash(frozenset(env.items())))
    cached = _bases.get(slot)
    # One stat a turn: an edit to tracing.json applies at the next turn, as without the cache.
    if cached is None or cached[0] < now or (cached[1] is not None and _version(cached[1]["file"]) != cached[1]["version"]):
        if len(_bases) > 256:
            _bases.clear()
        cached = _bases[slot] = (now + GATE_TTL, _base(workspace, env))
    base = cached[1]
    if base is None or base.get("off") or _session_mode(base, key, env, now) != "on":
        return None
    project = base["project"]
    return Gate(str(base["root"]), project["capture"], project["environment"], tuple(project["redactPatterns"]), tuple(project["redactPaths"]),
                base["secrets"], str(env.get("HOME") or ""), base["stamp"], str(base["state"]))


# ---- prepare: plain stages -----------------------------------------------------------------------


def _cut(text: str) -> str:
    """At most TEXT_CAP characters, ending before the last partial word. No whitespace there keeps nothing."""
    head = text[:TEXT_CAP]
    space = max(head.rfind(" "), head.rfind("\n"), head.rfind("\t"))
    return (head[:space] if space >= 0 else "") + CUT_MARK


def fit(text: str, rs: redact.Ruleset, secrets: list) -> str:
    """A long text redacted, then cut. Short text is left for the text stage, which redacts it whole.

    Exact secret values go first, over all of it. The patterns then run over the first
    TEXT_CAP + MARGIN characters; whatever a secret cut by that window's end left unmatched is in
    the window's last MARGIN characters, which are dropped before the cut to a word boundary.
    """
    if len(text) <= TEXT_CAP:
        return text
    text = redact.redact_string(text, _NO_PATTERNS, secrets)[0]
    if len(text) <= TEXT_CAP:
        return text
    window = redact.redact_string(text[: TEXT_CAP + MARGIN], rs, [])[0]
    if len(text) > TEXT_CAP + MARGIN:  # the window cut the text, maybe inside a secret
        window = window[: max(len(window) - MARGIN, 0)]
    return _cut(window)


def _keys(value, rs: redact.Ruleset, secrets: list, counts: dict):
    """Every dict key redacted, and every long string fitted. Colliding keys get #2, #3, ... in order."""
    if isinstance(value, str):
        return fit(value, rs, secrets)
    if isinstance(value, list):
        return [_keys(item, rs, secrets, counts) for item in value]
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            new, found = redact.redact_string(fit(str(key), rs, secrets), rs, secrets)
            for rule, count in found.items():
                counts[rule] = counts.get(rule, 0) + count
            name, n = new, 1
            while name in out:
                n += 1
                name = f"{new}#{n}"
            out[name] = _keys(item, rs, secrets, counts)
        return out
    return value


def _meta(out: dict, rs: redact.Ruleset, secrets: list, counts: dict) -> None:
    """The untrusted metadata strings redacted in place, and the id and name of each kept tool call."""
    for name in UNTRUSTED_META:
        if isinstance(out.get(name), str) and out[name]:
            out[name], found = redact.redact_string(fit(out[name], rs, secrets), rs, secrets)
            for rule, count in found.items():
                counts[(rule, name)] = counts.get((rule, name), 0) + count
    calls = out.get("tool_calls")
    if out.get("capture") == "metadata" and isinstance(calls, list):
        for call in calls:
            for name in ("id", "name"):
                if isinstance(call, dict) and isinstance(call.get(name), str):
                    call[name], found = redact.redact_string(call[name], rs, secrets)
                    for rule, count in found.items():
                        counts[(rule, "tool_calls")] = counts.get((rule, "tool_calls"), 0) + count


def _audit(out: dict, extra: dict) -> None:
    """``redactions``: census of placeholders in text fields, plus those in keys and metadata."""
    keep = redact.keep_fields()
    totals: dict = {}
    for item in redact.census(out, [key for key in out if key not in keep]):
        totals[(item["rule"], item["field"])] = item["count"]
    for (rule, field), count in extra.items():
        totals[(rule, field)] = totals.get((rule, field), 0) + count
    if totals:
        out["redactions"] = [{"rule": rule, "field": field, "count": count} for (rule, field), count in sorted(totals.items(), key=lambda x: (x[0][1], x[0][0]))]
    else:
        out.pop("redactions", None)


def prepare(record: dict, gate: Gate) -> dict:
    """The capture policy for one record, then record()'s stamps. Never returns secret text a rule or env value names."""
    rs = rules(gate)
    secrets = list(gate.secrets)
    keep = redact.keep_fields()
    extra: dict = {}
    if gate.capture == "metadata":
        out = {**redact.strip_text(record), "prompt": "", "output": ""}
        out["capture"] = "metadata"
        _meta(out, rs, secrets, extra)
        _audit(out, extra)
    elif gate.capture == "sanitized":
        try:
            if rs.error:
                raise ValueError(rs.error)
            fitted = {}
            for key, value in record.items():
                if key in keep:
                    fitted[key] = value
                    continue
                counts: dict = {}
                fitted[key] = _keys(value, rs, secrets, counts)
                for rule, count in counts.items():
                    extra[(rule, key)] = count
            out = redact.sanitize_event(fitted, rs, secrets, gate.home)
            out["capture"] = "sanitized"
            _meta(out, rs, secrets, extra)
            _audit(out, extra)
        except Exception:  # a rule that fails drops the event's text, never sends it raw
            out = {**redact.strip_text(record), "redaction_error": True}
            for name in UNTRUSTED_META + ("tool_calls",):
                out.pop(name, None)
    else:
        out = {key: (_full(value) if key not in keep else value) for key, value in record.items()}
    out["capture"] = gate.capture
    if not isinstance(out.get("ts"), (int, float)):
        out["ts"] = int(time.time() * 1000)
    out["environment"] = gate.environment
    out["project_root"] = gate.root
    if "export_fp" not in out:
        out.update(gate.stamp)
    return out


def _full(value):
    if isinstance(value, str):
        return _cut(value) if len(value) > TEXT_CAP else value
    if isinstance(value, list):
        return [_full(item) for item in value]
    if isinstance(value, dict):
        return {key: _full(item) for key, item in value.items()}
    return value


# ---- the inbox and the sender --------------------------------------------------------------------

_seq = itertools.count()
_spool_lock = threading.Lock()


def _private_dir(path: Path) -> None:
    if not path.is_dir():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)


def spool(record: dict, state) -> None:
    """One file per event in ``inbox/``, owner-only, appearing whole (written aside, then linked in).

    The flusher folds files in (mtime, name) order and remembers each file it folded by name and
    mtime, so a file that appears late is folded late, never dropped.
    """
    state = Path(state)
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
                os.utime(tmp)  # its order is when it is published, not when it was written
                os.link(tmp, inbox / name)  # never replaces a file a crashed process left
                return
            except FileExistsError:
                continue
            finally:
                os.unlink(tmp)


_senders: list = []


def sender_env(env: dict | None = None) -> dict:
    env = dict(os.environ if env is None else env)
    env["PYTHONPATH"] = str(CORE) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.pop("PYTHONSTARTUP", None)
    return env


def launch_sender(state, env: dict | None = None) -> bool:
    """Start ``python -m judgment_core.flush`` in its own session, output to flush.log. Never waits on it.

    start_new_session puts the child in a new session before exec, and Popen returns only after
    exec, so killing Hermes's process group right after this cannot reach the sender.
    """
    env = dict(os.environ if env is None else env)
    if env.get("JUDGMENT_FLUSH") == "0" or optin.tracing_forced_off(env):
        return False
    state = Path(state)
    for old in list(_senders):  # reap finished senders so none stays a zombie
        if old.poll() is not None:
            _senders.remove(old)
    _private_dir(state)
    log = os.open(state / "flush.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        proc = subprocess.Popen(
            [sys.executable or "python3", "-m", "judgment_core.flush"],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=log,
            close_fds=True,
            start_new_session=True,
            cwd="/",
            env=sender_env(env),
        )
    finally:
        os.close(log)
    _senders.append(proc)
    return True


# ---- /judgment -----------------------------------------------------------------------------------


def command(action: str, cwd: str, key: str, env: dict | None = None) -> str:
    """/judgment on|off|status for conversation ``key``. Same rules and messages as core/bin/judgment-session.

    ``key`` is '' when the plugin cannot tell which conversation asked: status then reports the
    project, and on/off change nothing.
    """
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
        if not key:
            return ("Judgment tracing can't tell which conversation sent this command here, so it changed nothing. "
                    "Use /judgment on or off in the Hermes CLI, or set sessionDefault in .judgment/tracing.json.")
        _private_dir(optin.state_dir(env))
        optin.write_session_mode(root, key, action, env)
        forget_cache()
    mode, message = optin.session_status(cwd, key, env)
    if action == "on" and mode == "on":
        return "Judgment tracing is on for this session. It applies from the next message."
    if action == "off":
        return "Judgment tracing is off for this session. It applies from the next message."
    if not key and action == "status":
        return message + " This command can't tell which conversation asked, so it shows the project setting."
    return message


def copy_session_mode(key: str, alias: str, env: dict | None = None) -> None:
    """Hermes rotated conversation ``key`` to session id ``alias``: a resume under the new id keeps its on/off."""
    env = dict(os.environ if env is None else env)
    source = optin.session_override_path(Path("/"), key, env)
    target = optin.session_override_path(Path("/"), alias, env)
    try:
        mode = source.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if mode in ("on", "off") and not target.exists():
        optin.write_session_mode(Path("/"), alias, mode, env)
        forget_cache()
