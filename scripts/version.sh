#!/bin/sh
# Every plugin ships the same release version. Agents offer an update only when it changes.
# usage: scripts/version.sh          fail unless every manifest has the same version
#        scripts/version.sh 0.2.0    set it everywhere, then refresh the vendored core
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$root"
python3 - "$@" <<'PY'
import json
import re
import sys
from pathlib import Path

VERSION = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
MANIFESTS = {
    ".claude-plugin/marketplace.json": ("plugins", 0, "version"),
    "claude-code/.claude-plugin/plugin.json": ("version",),
    "codex/.codex-plugin/plugin.json": ("version",),
    "cursor/.cursor-plugin/plugin.json": ("version",),
    "package.json": ("version",),
    "pi/package.json": ("version",),
}
INIT = Path("core/judgment_core/__init__.py")
INIT_LINE = re.compile(r'^__version__ = "([^"]*)"$', re.M)


def lookup(data, path):
    for part in path[:-1]:
        data = data[part]
    return data, path[-1]


if len(sys.argv) > 1:
    new = sys.argv[1]
    if not VERSION.match(new):
        sys.exit(f"not a version: {new}")
    for name, path in MANIFESTS.items():
        data = json.loads(Path(name).read_text())
        parent, key = lookup(data, path)
        parent[key] = new
        Path(name).write_text(json.dumps(data, indent=2) + "\n")
    INIT.write_text(INIT_LINE.sub(f'__version__ = "{new}"', INIT.read_text()))

found = {}
for name, path in MANIFESTS.items():
    try:
        parent, key = lookup(json.loads(Path(name).read_text()), path)
        found[name] = parent[key]
    except (KeyError, IndexError, TypeError, ValueError):
        found[name] = None
match = INIT_LINE.search(INIT.read_text())
found[str(INIT)] = match.group(1) if match else None
for name, value in found.items():
    print(name, value)
if any(value is None or not VERSION.match(str(value)) for value in found.values()) or len(set(found.values())) != 1:
    sys.exit("plugin versions differ or are missing. Run scripts/version.sh <version>.")
PY
if [ $# -gt 0 ]; then
  sh scripts/sync-core.sh
fi
