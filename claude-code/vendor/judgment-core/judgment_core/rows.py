"""Read transcript rows. A turn's rows sit at the end of the file when it is exported, so read from the end."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

# Smaller files are parsed whole. A long session's transcript can be hundreds of MB.
TAIL_MIN_BYTES = 4 * 1024 * 1024
# Rows a little older than the turn's first hook event still belong to it.
MARGIN_MS = 10 * 60 * 1000
_CHUNK = 1024 * 1024
# Stop only after this many rows in a row are older than the cutoff. One old row can be appended late.
_OLD_RUN = 50


def _ms(row: dict) -> int | None:
    value = row.get("timestamp")
    if not isinstance(value, str):
        return None
    try:
        return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)
    except ValueError:
        return None


def _parse(lines: list[bytes]) -> list[dict]:
    rows = []
    for line in lines:
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def records(path: Path | None, since_ms: int | None = None) -> list[dict]:
    """Rows of a JSONL transcript. With ``since_ms``, only the first row and rows from about then on."""
    if path is None:
        return []
    try:
        size = os.path.getsize(path)
        handle = open(path, "rb")
    except OSError:
        return []
    with handle:
        if since_ms is None or size < TAIL_MIN_BYTES:
            return _parse(handle.read().splitlines())
        first = _parse([handle.readline()])
        start = handle.tell()
        kept: list[dict] = []
        old: list[dict] = []
        pos = size
        carry = b""
        cutoff = since_ms - MARGIN_MS
        while pos > start:
            step = min(_CHUNK, pos - start)
            pos -= step
            handle.seek(pos)
            lines = (handle.read(step) + carry).split(b"\n")
            carry = lines.pop(0) if pos > start else b""
            for row in reversed(_parse(lines)):
                stamp = _ms(row)
                if stamp is not None and stamp < cutoff:
                    old.append(row)
                    if len(old) >= _OLD_RUN:
                        return first + list(reversed(kept))
                    continue
                kept.extend(old)
                old = []
                kept.append(row)
        kept.extend(old)
        return first + list(reversed(kept + _parse([carry])))
