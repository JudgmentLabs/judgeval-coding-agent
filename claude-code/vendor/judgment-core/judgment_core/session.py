"""In-chat session toggle. Prints one line the agent can show the user."""

from __future__ import annotations

import argparse
import os
import sys

from judgment_core.optin import find_optin, load_project, session_status, tracing_forced_off, write_session_mode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="judgment-session")
    parser.add_argument("action", choices=("on", "off", "status"))
    parser.add_argument("--cwd", required=True)
    parser.add_argument("--session", default="")
    args = parser.parse_args(argv)

    if args.action in ("on", "off"):
        if tracing_forced_off():
            print("Judgment tracing is off. JUDGMENT_TRACING=0.")
            return 0
        found = find_optin(args.cwd)
        if found is None:
            print("Judgment tracing is off. No .judgment/tracing.json was found above this directory.")
            return 0
        root, file = found
        project = load_project(file)
        if not project["enabled"]:
            print("Judgment tracing is off. .judgment/tracing.json has enabled set to false.")
            return 0
        if not args.session:
            print("Judgment tracing needs a session id to change this conversation.")
            return 0
        write_session_mode(root, args.session, args.action)
    mode, message = session_status(args.cwd, args.session, os.environ)
    if args.action == "status":
        print(message)
    elif args.action == "on" and mode == "on":
        print("Judgment tracing is on for this session.")
    elif args.action == "off":
        print("Judgment tracing is off for this session.")
    else:
        print(message)
    return 0


if __name__ == "__main__":
    sys.exit(main())
