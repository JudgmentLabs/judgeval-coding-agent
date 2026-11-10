"""Project opt-in and per-session tracing state.

The shell hook in ``core/bin/judgment-hook`` implements the same rules.
Tests cover both so a session that should be quiet stays quiet.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

OFF_VALUES = {"0", "false", "no", "off"}


def tracing_forced_off(env: dict | None = None) -> bool:
    source = os.environ if env is None else env
    raw = source.get("JUDGMENT_TRACING", "")
    return raw.strip().lower() in OFF_VALUES


def find_optin(start: str | Path) -> tuple[Path, Path] | None:
    """Return ``(project_root, tracing_json)`` walking parents of ``start``."""
    current = Path(start).resolve()
    if current.is_file():
        current = current.parent
    while True:
        candidate = current / ".judgment" / "tracing.json"
        if candidate.is_file():
            return current, candidate
        if current.parent == current:
            return None
        current = current.parent


# Capture modes, strictest last. "sanitized" sends the text with secrets redacted; "full" sends it raw.
CAPTURE_MODES = ("full", "sanitized", "metadata")


def resolve_capture(data: dict) -> str:
    """The one capture table. core/bin/prepare.jq judgment_prepare has the same table; tests check both.

    An explicit ``capture`` wins. Without one, ``"redact": false`` means raw text, and anything
    else, including a missing or unknown value, means sanitized.
    """
    capture = data.get("capture")
    if capture in CAPTURE_MODES:
        return capture
    if data.get("redact") is False:
        return "full"
    return "sanitized"


def _strings(value: object) -> list[str]:
    return [item for item in value if isinstance(item, str) and item] if isinstance(value, list) else []


# The project's redaction settings (judgment_core.redact.ruleset says what each does). Each is a list of strings.
REDACT_KEYS = ("redactPatterns", "redactPaths", "redactCommands", "redactNames", "redactAllow", "redactDisable")
# The ones that add rules: one that is not a list of strings is kept as written, so redact.ruleset fails closed on it.
_STRICT_KEYS = ("redactPatterns", "redactPaths", "redactCommands", "redactNames")


def _setting(key: str, value: object) -> object:
    if key in _STRICT_KEYS and value is not None and not (isinstance(value, list) and all(isinstance(item, str) for item in value)):
        return value
    return _strings(value)


_CLOSED = {"enabled": False, "capture": "sanitized", "sessionDefault": "on", "environment": "production", **{key: [] for key in REDACT_KEYS}}


def load_project(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return dict(_CLOSED)
    if not isinstance(data, dict):
        return dict(_CLOSED)
    enabled = data.get("enabled", True)
    session_default = data.get("sessionDefault", "on")
    environment = data.get("environment") or "production"
    if not isinstance(environment, str) or not environment.strip():
        environment = "production"
    return {
        "enabled": enabled is not False,
        "capture": resolve_capture(data),
        "sessionDefault": "off" if session_default == "off" else "on",
        "environment": environment.strip(),
        **{key: _setting(key, data.get(key)) for key in REDACT_KEYS},
    }


_SAFE_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def safe_id(value: str) -> str:
    """A file name for an id. Ids come from agent payloads, so anything else is hashed."""
    if _SAFE_ID.fullmatch(value) and value not in (".", ".."):
        return value
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:32]


def state_dir(env: dict | None = None) -> Path:
    source = os.environ if env is None else env
    if source.get("JUDGMENT_STATE_DIR"):
        return Path(source["JUDGMENT_STATE_DIR"])
    xdg = source.get("XDG_STATE_HOME")
    base = Path(xdg) if xdg else Path(source.get("HOME", str(Path.home()))) / ".local" / "state"
    if xdg:
        return Path(xdg) / "judgment"
    return base / "judgment"


def config_dir(env: dict | None = None) -> Path:
    source = os.environ if env is None else env
    if source.get("JUDGMENT_CONFIG_DIR"):
        return Path(source["JUDGMENT_CONFIG_DIR"])
    xdg = source.get("XDG_CONFIG_HOME")
    if xdg:
        return Path(xdg) / "judgment"
    return Path(source.get("HOME", str(Path.home()))) / ".config" / "judgment"


def session_override_path(root: Path, session_id: str, env: dict | None = None) -> Path:
    return state_dir(env) / "sessions" / safe_id(session_id)


def read_session_mode(root: Path, session_id: str, default: str, env: dict | None = None) -> str:
    path = session_override_path(root, session_id, env)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError:
        text = ""
    if text in ("on", "off"):
        return text
    return "off" if default == "off" else "on"


def write_session_mode(root: Path, session_id: str, mode: str, env: dict | None = None) -> None:
    if mode not in ("on", "off"):
        raise ValueError(mode)
    path = session_override_path(root, session_id, env)
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(mode + "\n")


def session_status(cwd: str, session_id: str, env: dict | None = None) -> tuple[str, str]:
    """Return ``(mode, message)``. ``mode`` is ``on`` or ``off``."""
    if tracing_forced_off(env):
        return "off", "Judgment tracing is off. JUDGMENT_TRACING=0."
    found = find_optin(cwd)
    if found is None:
        return "off", "Judgment tracing is off. No .judgment/tracing.json was found above this directory."
    root, file = found
    project = load_project(file)
    if not project["enabled"]:
        return "off", "Judgment tracing is off. .judgment/tracing.json has enabled set to false."
    if not session_id:
        mode = project["sessionDefault"]
        if mode == "on":
            return "on", "Judgment tracing is on for new sessions in this project."
        return "off", "Judgment tracing is off for new sessions until you turn a session on."
    mode = read_session_mode(root, session_id, project["sessionDefault"], env)
    if mode == "on":
        return "on", "Judgment tracing is on for this session."
    if project["sessionDefault"] == "off":
        return "off", "Judgment tracing is off for this session until you turn it on."
    return "off", "Judgment tracing is off for this session."
