#!/usr/bin/env python3
"""Write what the hook loads from core/bin/redact-rules.json, the one source of the redaction rules.

    python3 scripts/gen_redact.py           write core/bin/redact-compiled.json
    python3 scripts/gen_redact.py --check   fail when it is stale

redact-compiled.json is the table with every built-in regex composed (each rule's alternatives behind
the command prefix, path triggers, the name rules split around {names}), the field classes as sets, and
no comments, so jq compiles no composing code and walks no table on the agent's critical path.
judgment_core.redact derives it, and reads the source itself.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "core"))

from judgment_core import redact  # noqa: E402


def outputs() -> dict[Path, str]:
    table = json.loads(redact.RULES_FILE.read_text(encoding="utf-8"))
    return {
        redact.COMPILED_FILE: json.dumps(redact.compiled(table), separators=(",", ":")) + "\n",
    }


def main(argv: list[str]) -> int:
    stale = []
    for path, text in outputs().items():
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == text:
            continue
        if "--check" in argv:
            stale.append(path.relative_to(ROOT))
        else:
            path.write_text(text, encoding="utf-8")
    if stale:
        print("stale: " + ", ".join(map(str, stale)) + ". Run scripts/gen_redact.py.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
