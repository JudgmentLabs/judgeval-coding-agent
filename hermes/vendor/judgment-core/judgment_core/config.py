"""Credentials live outside the repo. Environment variables override the user file."""

from __future__ import annotations

import os
import stat
from pathlib import Path

from judgment_core.optin import config_dir

DEFAULT_API_URL = "https://api.judgmentlabs.ai"
_KEYS = ("JUDGMENT_API_KEY", "JUDGMENT_PROJECT_ID", "JUDGMENT_ORG_ID", "JUDGMENT_API_URL")


def parse_credentials(text: str) -> dict[str, str]:
    found: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key in _KEYS and value:
            found[key] = value
    return found


def load_credentials(env: dict | None = None) -> dict[str, str]:
    source = os.environ if env is None else env
    path = config_dir(source) / "credentials"
    found: dict[str, str] = {}
    try:
        found.update(parse_credentials(path.read_text(encoding="utf-8")))
    except OSError:
        pass
    for key in _KEYS:
        value = source.get(key, "").strip()
        if value:
            found[key] = value
    found.setdefault("JUDGMENT_API_URL", DEFAULT_API_URL)
    return found


def traces_url(base: str) -> str:
    cleaned = base.strip().rstrip("/")
    suffix = "/otel/v1/traces"
    if cleaned.endswith(suffix):
        return cleaned
    return cleaned + suffix


def credentials_mode_warning(env: dict | None = None) -> str | None:
    path = config_dir(os.environ if env is None else env) / "credentials"
    try:
        mode = path.stat().st_mode
    except OSError:
        return None
    if mode & (stat.S_IRWXG | stat.S_IRWXO):
        return f"{path} is readable by other users. Run chmod 600 on that file."
    return None
