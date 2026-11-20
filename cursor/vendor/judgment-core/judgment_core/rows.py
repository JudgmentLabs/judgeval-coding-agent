"""Read transcript rows. A turn's rows sit at the end of the file when it is exported, so read from the end."""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Callable

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


def parse_line(line: bytes) -> dict | None:
    """One JSONL line as a row, or None when it is blank, cut off, or not an object."""
    parsed = _parse([line])
    return parsed[0] if parsed else None


def records(
    path: Path | None,
    since_ms: int | None = None,
    stop: Callable[[dict], bool] | None = None,
    pin: Callable[[list[dict]], Callable[[bytes], dict | None]] | None = None,
) -> list[dict]:
    """Rows of a JSONL transcript. With ``since_ms``, a large file is read from the end: the first row and rows from about then on.

    ``stop`` names a row where the model's history restarts (a compaction). With it, a large file is read back past the
    turn's start to the last such row before the turn, or to the start of the file when there is none, so a reader can
    rebuild what each of the turn's requests saw. A compaction during the turn does not stop the read: the turn's
    earlier requests saw the rows before it.

    ``pin``, given whenever ``stop`` is, keeps rows from before that compaction that a reader still needs, such as the session's tool definitions.
    It is called with the rows kept so far (oldest last) and returns a test for each older raw line, newest first: the
    row to keep, or None. Only a test that matches pays for parsing the line.
    """
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
        pinned: list[dict] = []
        wanted: Callable[[bytes], dict | None] | None = None
        passed = False
        pos = size
        carry = b""
        cutoff = since_ms - MARGIN_MS
        while pos > start:
            step = min(_CHUNK, pos - start)
            pos -= step
            handle.seek(pos)
            lines = (handle.read(step) + carry).split(b"\n")
            carry = lines.pop(0) if pos > start else b""
            for line in reversed(lines):
                if wanted is not None:
                    row = wanted(line)
                    if row is not None:
                        pinned.append(row)
                    continue
                row = parse_line(line)
                if row is None:
                    continue
                stamp = _ms(row)
                if stop is not None:
                    kept.append(row)
                    passed = passed or (stamp is not None and stamp < since_ms)
                    if passed and stop(row):
                        wanted = pin(kept)
                    continue
                if stamp is not None and stamp < cutoff:
                    old.append(row)
                    if len(old) >= _OLD_RUN:
                        return first + list(reversed(kept))
                    continue
                kept.extend(old)
                old = []
                kept.append(row)
        kept.extend(old)
        return first + list(reversed(pinned)) + list(reversed(kept))
