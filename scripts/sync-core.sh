#!/bin/sh
# Copy core/ into each plugin's vendor tree. --check fails when a copy drifted.
set -eu
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
check=0
if [ "${1:-}" = "--check" ]; then
  check=1
fi
found=0
for plugin in claude-code codex cursor pi hermes; do
  [ -d "$root/$plugin" ] || continue
  found=1
  dest="$root/$plugin/vendor/judgment-core"
  tmp=$(mktemp -d)
  mkdir -p "$tmp/judgment-core"
  cp -a "$root/core/." "$tmp/judgment-core/"
  find "$tmp/judgment-core" -type d -name __pycache__ -exec rm -rf {} +
  find "$tmp/judgment-core" -name '*.pyc' -delete
    if [ "$check" -eq 1 ]; then
      if [ ! -d "$dest" ] || ! diff -rq -x __pycache__ -x '*.pyc' "$tmp/judgment-core" "$dest" >/dev/null; then
      echo "vendor drift in $plugin. Run scripts/sync-core.sh." >&2
      rm -rf "$tmp"
      exit 1
    fi
  else
    rm -rf "$dest"
    mkdir -p "$(dirname "$dest")"
    cp -a "$tmp/judgment-core" "$dest"
  fi
  rm -rf "$tmp"
done
if [ "$found" -eq 0 ]; then
  exit 0
fi
