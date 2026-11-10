"""Every call from the Hermes plugin into judgment_core goes through this module.

judgment_core is vendored next to this file (``vendor/judgment-core``, synced by scripts/sync-core.sh).
When the core's opt-in or redaction API changes, this is the one file to update. It uses only:

    optin     find_optin, load_project, tracing_forced_off, state_dir, session_override_path,
              read_session_mode, write_session_mode, session_status, safe_id
    redact    ruleset, env_secrets, sanitize_event, scrub_fields, strip_text, field_class, RedactionLimit
    config    load_credentials, DEFAULT_API_URL

They are listed, with the parameters the bridge passes, in ``CONTRACT``; ``contract_problems()``
checks the vendored core against it (tests/test_hermes_contract.py), so a core change that moves one
fails with the name of what moved, not with 50 unrelated test failures. Every redaction of a value
goes through ``_scrub``: when the core's scrub entry point changes, that is the one line to update.

The functions, in the order a record meets them:

    resolve(key, workspace)   -> Gate | None   (in Hermes, at turn start) opt-in walk-up, kill switch, session
                                               mode, capture, rules, secret values, and the stamps, already
                                               checked by redaction. None = untraced.
    needs_alias(gate, id)     -> bool          (in Hermes) would redaction change this tool call id?
    prepare(record, gate)     -> record        (in the helper) the capture policy. See PREPARE.
    stage(record, state, name), publish(state, name)
                                               (in the helper) the inbox file written aside, then linked in
    launch_sender(state, env)                  the core flusher, detached, never waited on
    command(action, cwd, key) -> str           /judgment on|off|status
    copy_session_mode(key, alias)              a rotated session id keeps its conversation's on/off

PREPARE, one record, plain stages. The record arrives with every value whole: the spooler dropped a
value too large to redact whole and listed it in ``omitted``, and never cut one.

    sanitized:  sense    an omitted input (prompt, tool input, tool calls) makes the record sensitive:
                         its sensitivity can't be checked, so its turn is blanked, as for a .env read
                core     redact.sanitize_event on the complete values: sensitive tool calls lose their
                         text, every string and key is redacted, secret-named members go whole
                meta     the metadata the core keeps as is (model, provider, cwd, ...), which in Hermes a
                         model or provider chose, redacted with the same rules
                cut      only now, each text longer than TEXT_CAP cut before its last partial word
                stamp    capture, ts, and the gate's stamps (checked by redaction at resolve)
    metadata:   core (metadata), meta, cut, stamp
    full:       cut, stamp
A stage that fails drops the record's text whole (the core's redaction_error), never sends it raw.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CORE = Path(__file__).resolve().parent / "vendor" / "judgment-core"
if str(CORE) not in sys.path:
    sys.path.insert(0, str(CORE))

from judgment_core import __version__ as CORE_VERSION  # noqa: E402
from judgment_core import optin, redact  # noqa: E402
from judgment_core.config import DEFAULT_API_URL, load_credentials  # noqa: E402

__all__ = ["CORE", "CORE_VERSION", "Gate", "warm", "resolve", "needs_alias", "prepare", "stage", "publish", "launch_sender", "command",
           "copy_session_mode", "state_dir", "forced_off", "TEXT_CAP", "CUT_MARK", "OVERSIZE_RULE"]

# The most text one field keeps, cut only after redaction. Spans cut at 64 KB anyway.
TEXT_CAP = 64 * 1024
CUT_MARK = "… [truncated]"
# The fields sensitivity is read from (redact.tool_texts and the prompt's @mentions).
INPUTS = ("prompt", "tool_input", "tool_calls")
OVERSIZE_RULE = "too_large_to_check"
# Fields the core keeps as is, because other harnesses set them. In Hermes a provider, a model, or
# Hermes's config chose them. tool_call_id is replaced in Hermes when redaction would change it
# (tracker "Tool call id"); it is checked again here.
UNTRUSTED_META = ("model", "provider", "cwd", "harness_version", "tool_call_id", "parent_call_id")
EXPORT = ("export_fp", "export_project", "export_url", "export_org")


@dataclass(frozen=True)
class Gate:
    """One turn's opt-in decision: plain data, everything prepare needs, resolved once at turn start."""

    capture: str  # full | sanitized | metadata, as optin.resolve_capture decides
    patterns: tuple  # the project's redactPatterns
    paths: tuple  # the project's redactPaths
    secrets: tuple  # ((name, value), ...) secret env values, longest first. Crosses only the pipe.
    state: str
    stamps: tuple  # ((field, value), ...) environment, project_root, export_*: each checked by redaction
    withheld: tuple = ()  # stamps redaction changed (environment, project_root: kept redacted; export: dropped), for the helper's log

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict) -> "Gate":
        pairs = lambda items: tuple(tuple(item) for item in items)  # noqa: E731
        return cls(data["capture"], tuple(data["patterns"]), tuple(data["paths"]), pairs(data["secrets"]), data["state"],
                   pairs(data["stamps"]), tuple(data.get("withheld") or ()))


# The core API the bridge depends on: (module, name, the parameters the bridge passes by position, in order).
CONTRACT = (
    ("optin", "find_optin", ("start",)),
    ("optin", "load_project", ("path",)),
    ("optin", "tracing_forced_off", ("env",)),
    ("optin", "state_dir", ("env",)),
    ("optin", "session_override_path", ("root", "session_id", "env")),
    ("optin", "read_session_mode", ("root", "session_id", "default", "env")),
    ("optin", "write_session_mode", ("root", "session_id", "mode", "env")),
    ("optin", "session_status", ("cwd", "session_id", "env")),
    ("optin", "safe_id", ("value",)),
    ("redact", "ruleset", ("project",)),
    ("redact", "env_secrets", ("env", "api_key")),
    ("redact", "sanitize_event", ("event", "rs", "secrets", "capture")),
    ("redact", "scrub_fields", ("fields", "rs", "secrets")),
    ("redact", "strip_text", ("event",)),
    ("redact", "field_class", ("name",)),
    ("redact", "RedactionLimit", None),
    ("redact", "Ruleset", None),
)


def contract_problems() -> list:
    """What in the vendored core no longer matches CONTRACT, as readable lines; [] when it all does.
    Parameter names are compared by position only: a rename the bridge doesn't pass by keyword is fine,
    but a missing function, a new required parameter, or fewer parameters is reported."""
    import inspect

    problems = []
    modules = {"optin": optin, "redact": redact}
    for module, name, params in CONTRACT:
        target = getattr(modules[module], name, None)
        if target is None:
            problems.append(f"judgment_core.{module}.{name} is gone (core_bridge uses it)")
            continue
        if params is None:
            continue
        try:
            sig = inspect.signature(target)
        except (TypeError, ValueError):
            continue
        listed = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        if len(listed) < len(params) and not any(p.kind is p.VAR_POSITIONAL for p in sig.parameters.values()):
            problems.append(f"judgment_core.{module}.{name}{sig} takes fewer than the {len(params)} arguments core_bridge passes {params}")
        required = [p.name for p in listed[len(params):] if p.default is p.empty]
        required += [p.name for p in sig.parameters.values() if p.kind is p.KEYWORD_ONLY and p.default is p.empty]
        if required:
            problems.append(f"judgment_core.{module}.{name}{sig} now requires {required}, which core_bridge does not pass")
    return problems


def _scrub(value, rs: "redact.Ruleset", secrets: list):
    """One value with every secret in its strings and keys replaced. Raises RedactionLimit when it can't be
    checked within the core's limits, or when the rules failed. The one call to the core's scrub entry point."""
    return redact.scrub_fields({"value": value}, rs, secrets)[0]["value"]


@functools.lru_cache(maxsize=64)
def _rules(patterns: tuple, paths: tuple) -> redact.Ruleset:
    """Compiled once per distinct project config, not once per call."""
    return redact.ruleset({"redactPatterns": list(patterns), "redactPaths": list(paths)})


def rules(gate: Gate) -> redact.Ruleset:
    return _rules(gate.patterns, gate.paths)


def warm() -> None:
    """Compile the redaction rules now, at plugin load, so the first turn does not pay for it. Writes nothing."""
    rs = _rules((), ())
    redact.env_secrets({})
    _scrub("warm", rs, [])


def forced_off(env: dict | None = None) -> bool:
    return optin.tracing_forced_off(env)


def state_dir(env: dict | None = None) -> Path:
    return optin.state_dir(env)


def _fingerprint(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]


def _export(env: dict) -> list:
    """What the hook records about the destination: the key's fingerprint, never the key."""
    creds = load_credentials(env)
    key, project = creds.get("JUDGMENT_API_KEY", ""), creds.get("JUDGMENT_PROJECT_ID", "")
    if not key or not project:
        return []
    pairs = [("export_fp", _fingerprint(key)), ("export_project", project), ("export_url", creds.get("JUDGMENT_API_URL") or DEFAULT_API_URL)]
    if creds.get("JUDGMENT_ORG_ID"):
        pairs.append(("export_org", creds["JUDGMENT_ORG_ID"]))
    return pairs


def stamps(environment: str, root: str, export: list, rs: redact.Ruleset, secrets: list) -> tuple[tuple, tuple]:
    """The stamps every record carries, as they may be stored: (kept, withheld names).

    environment and project_root are labels once stored: redacted like text. The flusher reads
    project_root to load the project's own rules for its second pass; a redacted root makes it use the
    built-in rules, and the project's rules already ran here. The export stamp routes the turn, so it
    is never redacted: a stamp redaction would change is withheld whole and the turn is not sent.
    """
    wanted = {"environment": environment, "project_root": root, **dict(export)}
    try:
        if rs.error:
            raise redact.RedactionLimit(rs.error)
        clean = {name: _scrub(value, rs, secrets) for name, value in wanted.items()}
    except redact.RedactionLimit:
        return (), tuple(wanted)
    kept = [(name, clean[name]) for name in ("environment", "project_root")]
    withheld = [name for name in ("environment", "project_root") if clean[name] != wanted[name]]
    if any(clean[name] != value for name, value in export):
        withheld.append("export")
    else:
        kept += export
    return tuple(kept), tuple(withheld)


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
    """The per-workspace part of a gate: the project, its rules, the stamps, the secret values."""
    found = optin.find_optin(workspace)
    if found is None:
        return None
    root, file = found
    version = _version(file)
    project = optin.load_project(file)
    if not project["enabled"]:
        return {"file": file, "version": version, "off": True}
    api_key = load_credentials(env).get("JUDGMENT_API_KEY", "")
    secrets = tuple(redact.env_secrets(env, api_key))
    rs = _rules(tuple(project["redactPatterns"]), tuple(project["redactPaths"]))
    kept, withheld = stamps(project["environment"], str(root), _export(env), rs, list(secrets))
    return {"root": root, "file": file, "version": version, "project": project, "secrets": secrets, "stamps": kept, "withheld": withheld,
            "state": optin.state_dir(env)}


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
    return Gate(project["capture"], tuple(project["redactPatterns"]), tuple(project["redactPaths"]), base["secrets"], str(base["state"]),
                base["stamps"], base["withheld"])


def needs_alias(gate, value: str) -> bool:
    """Would the gate's redaction change this id? Then its records carry a replacement instead (tracker "Tool call id")."""
    if not value or not isinstance(gate, Gate) or gate.capture == "full":
        return False
    rs = rules(gate)
    if rs.error:
        return True
    try:
        return _scrub(value, rs, list(gate.secrets)) != value
    except redact.RedactionLimit:
        return True


# ---- prepare: plain stages -----------------------------------------------------------------------


def cut(text: str) -> str:
    """At most TEXT_CAP characters, ending before the last partial word, which goes whole. No whitespace there keeps nothing."""
    if len(text) <= TEXT_CAP:
        return text
    head = text[:TEXT_CAP]
    space = max(head.rfind(" "), head.rfind("\n"), head.rfind("\t"))
    return (head[:space] if space >= 0 else "") + CUT_MARK


def _cut_all(value):
    if isinstance(value, str):
        return cut(value)
    if isinstance(value, list):
        return [_cut_all(item) for item in value]
    if isinstance(value, dict):
        return {key: _cut_all(item) for key, item in value.items()}
    return value


def _sense(record: dict) -> dict:
    """An input too large to check makes the record sensitive, in the core's own terms (sanitize_event)."""
    if not any(name in INPUTS for name in record.get("omitted") or ()):
        return record
    return {**redact.strip_text(record), "sensitive": True, "sensitive_rule": OVERSIZE_RULE, "sensitive_trigger": ""}


def _unredacted(record: dict) -> dict:
    """What a record keeps when redaction failed: only the fields the core never redacts (its redaction_error shape)."""
    return {key: value for key, value in redact.strip_text(record).items() if not redact.field_class(key)["redact"]}


def _meta(out: dict, rs: redact.Ruleset, secrets: list) -> dict:
    """The untrusted metadata the core keeps as is, redacted with the same rules. Fails closed: dropped."""
    present = {name: out[name] for name in UNTRUSTED_META if isinstance(out.get(name), str) and out[name]}
    if not present:
        return out
    try:
        if rs.error:
            raise redact.RedactionLimit(rs.error)
        clean = {name: _scrub(value, rs, secrets) for name, value in present.items()}
    except redact.RedactionLimit:
        clean = {}
    return {**{key: value for key, value in out.items() if key not in present}, **clean}


def prepare(record: dict, gate: Gate) -> dict:
    """The capture policy for one record, then the stamps. Never returns secret text a rule or env value names."""
    if gate.capture == "full":
        out = _cut_all(record)
    else:
        rs = rules(gate)
        secrets = list(gate.secrets)
        try:
            event = _sense(record) if gate.capture == "sanitized" else record
            out = redact.sanitize_event(event, rs, secrets, gate.capture)
            out = _cut_all(_meta(out, rs, secrets))
        except Exception:  # anything unexpected drops the record's text, never sends it raw
            out = {**_meta(_unredacted(record), rs, secrets), "redaction_error": True}
    out["capture"] = gate.capture
    if not isinstance(out.get("ts"), (int, float)):
        out["ts"] = int(time.time() * 1000)
    out.update(gate.stamps)
    return out


# ---- the inbox and the sender --------------------------------------------------------------------


def _private_dir(path: Path) -> None:
    if not path.is_dir():
        path.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path, 0o700)


def stage(record: dict, state, name: str) -> None:
    """Write the prepared record aside, owner-only, under the name it will be published as. Overwrites a stale copy.

    This write fixes the record's identity: the flusher names an inbox event ``<name>.<mtime_ns>``, and a link
    shares its file's mtime, so the record keeps that id through every later publish of it (see ``publish``).
    """
    state = Path(state)
    scratch = state / "inbox.tmp"
    _private_dir(state)
    _private_dir(state / "inbox")
    _private_dir(scratch)
    line = json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
    fd = os.open(scratch / name, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def publish(state, name: str) -> None:
    """Link a staged record into ``inbox/``, whole, then drop the staged copy. Idempotent, and it never changes
    the record's identity (name, mtime), so a replay can't make an event the flusher already folded look new.

    One staged record, by which of its two links exist (a crash can stop between any two steps):

        STAGED (tmp)  --link-->  LINKED (tmp + inbox)  --unlink tmp-->  PUBLISHED (inbox)  --flusher-->  FOLDED
        LINKED --flusher folds and deletes the inbox link--> FOLDED, tmp left (the helper died before unlink)

    A replay runs ``publish`` again from whatever state is left. From STAGED or LINKED it ends PUBLISHED;
    from FOLDED-with-tmp it links the same inode again, same name, same mtime, so the flusher's id is
    unchanged and its fold (``done`` ids, a turn's stored ``_id``) counts it once; from PUBLISHED or FOLDED it
    finds nothing to do. Nothing here touches the mtime: the stage write set it, once.
    Impossible: one record published under two ids; a staged copy rewritten after its seq was committed
    (Helper.fold re-prepares only above the ledger's seq).
    """
    state = Path(state)
    tmp = state / "inbox.tmp" / name
    try:
        os.link(tmp, state / "inbox" / name)  # never replaces a file
    except FileNotFoundError:
        return  # published and cleaned up already
    except FileExistsError:
        pass
    try:
        os.unlink(tmp)
    except FileNotFoundError:
        pass


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


def log_line(state, text: str) -> None:
    """One line in flush.log. Callers pass counts and names, never text from a record."""
    try:
        _private_dir(Path(state))
        fd = os.open(Path(state) / "flush.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, (time.strftime("%Y-%m-%dT%H:%M:%S ") + text + "\n").encode("utf-8"))
        finally:
            os.close(fd)
    except OSError:
        pass


def _after_fork() -> None:
    _senders.clear()  # the parent's children, not this process's


os.register_at_fork(after_in_child=_after_fork)


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
