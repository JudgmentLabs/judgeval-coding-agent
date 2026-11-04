"""`judgment-trace` manages the project opt-in file. It does not store credentials."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from judgment_core import __version__
from judgment_core.config import credentials_mode_warning, load_credentials
from judgment_core.optin import find_optin, load_project, session_status, tracing_forced_off


def _git_root(start: Path) -> Path:
    current = start.resolve()
    while True:
        if (current / ".git").exists():
            return current
        if current.parent == current:
            return start.resolve()
        current = current.parent


def _exclude(root: Path) -> None:
    info = root / ".git" / "info"
    if not info.is_dir():
        return
    exclude = info / "exclude"
    line = ".judgment/"
    existing = exclude.read_text(encoding="utf-8") if exclude.exists() else ""
    lines = existing.splitlines()
    if line in lines:
        return
    prefix = "" if existing.endswith("\n") or existing == "" else "\n"
    with exclude.open("a", encoding="utf-8") as handle:
        handle.write(f"{prefix}{line}\n")


def cmd_enable(args: argparse.Namespace) -> int:
    root = _git_root(Path(args.cwd))
    folder = root / ".judgment"
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "enabled": True,
        "capture": args.capture,
        "sessionDefault": args.session_default,
    }
    # Hooks read this file on every event. Replace it in one rename so none sees it half written.
    scratch = folder / "tracing.json.tmp"
    scratch.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(scratch, folder / "tracing.json")
    _exclude(root)
    print(f"Wrote {folder / 'tracing.json'}")
    print("Add .judgment/ to .gitignore. The file stays on this machine and holds no API key.")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    found = find_optin(args.cwd)
    if found is None:
        print("No .judgment/tracing.json was found above this directory.")
        return 0
    root, file = found
    file.unlink()
    try:
        file.parent.rmdir()
    except OSError:
        pass
    print(f"Removed {root / '.judgment' / 'tracing.json'}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    mode, message = session_status(args.cwd, args.session or "", os.environ)
    found = find_optin(args.cwd)
    print(message)
    if found:
        project = load_project(found[1])
        print(f"file: {found[1]}")
        print(f"capture: {project['capture']}")
        print(f"sessionDefault: {project['sessionDefault']}")
    print(f"session: {mode}")
    if tracing_forced_off():
        print("JUDGMENT_TRACING=0")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    ok = True
    jq = shutil.which("jq")
    python = shutil.which("python3")
    print(f"jq: {jq or 'missing'}")
    print(f"python3: {python or 'missing'}")
    if not jq or not python:
        ok = False
    creds = load_credentials()
    print(f"api: {creds.get('JUDGMENT_API_URL')}")
    print(f"JUDGMENT_API_KEY: {'set' if creds.get('JUDGMENT_API_KEY') else 'missing'}")
    print(f"JUDGMENT_PROJECT_ID: {'set' if creds.get('JUDGMENT_PROJECT_ID') else 'missing'}")
    print(f"JUDGMENT_ORG_ID: {'set' if creds.get('JUDGMENT_ORG_ID') else 'unset'}")
    if not creds.get("JUDGMENT_API_KEY") or not creds.get("JUDGMENT_PROJECT_ID"):
        ok = False
    warning = credentials_mode_warning()
    if warning:
        print(warning)
    print(f"plugin: {__version__}")
    cmd_status(args)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="judgment-trace")
    parser.add_argument("--cwd", default=os.getcwd())
    parser.add_argument("--session", default="")
    sub = parser.add_subparsers(dest="command", required=True)

    enable = sub.add_parser("enable")
    enable.add_argument("--capture", choices=("full", "metadata"), default="full")
    enable.add_argument("--session-default", choices=("on", "off"), default="on")
    enable.set_defaults(func=cmd_enable)

    disable = sub.add_parser("disable")
    disable.set_defaults(func=cmd_disable)

    status = sub.add_parser("status")
    status.set_defaults(func=cmd_status)

    doctor = sub.add_parser("doctor")
    doctor.set_defaults(func=cmd_doctor)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
