"""Where a Hermes callback's records go: a helper process that redacts and spools them, off Hermes's threads.

Redacting one 56 KB tool output takes the core rules 10-25 ms of Python, and Python regex holds the
GIL, so doing it in a callback (or a thread of Hermes's process) would stall Hermes. Instead each
callback writes its records to a pipe, and one helper process per Hermes process does the rest:

    callback:  tracker records -> frames (length-prefixed JSON) -> pipe            (in Hermes)
    helper:    frame -> core_bridge.prepare (capture policy) -> core_bridge.spool -> inbox
               flush frame -> core_bridge.launch_sender                             (own session)

Plain text crosses only the pipe, which is memory; nothing reaches disk before prepare ran.

Helper lifecycle (``HelperSpool``, explicit states):

    NONE --first frame--> RUNNING (spawned: own session, so a kill of Hermes's group misses it)
    RUNNING --write fails (helper died)--> NONE, and the frame is sent again to a new helper, once
    RUNNING --close() or Hermes exits (pipe EOF)--> DRAINING: the helper folds every frame still
            in the pipe, then exits. A sender a flush frame asked for keeps running after it.
    RUNNING in a forked child (pid changed) --> NONE in the child: a child never shares the pipe
    Impossible: two helpers writing for one Hermes process; a record reaching disk unprepared.

Frames: ``{"t": "gate", "id": n, "gate": {...}}`` defines a gate once; ``{"t": "rec", "g": n, "r":
record}`` is one record; ``{"t": "flush", "state": dir, "env": {...}}`` starts a sender with the
environment Hermes has at that turn end; ``{"t": "forget"}`` drops every gate the helper holds.

When the helper cannot start (``JUDGMENT_HERMES_SPOOL=inline``, or a spawn error), ``InlineSpool``
does the same work in the callback.
"""

from __future__ import annotations

import fcntl
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

if __package__:
    from . import core_bridge as core
else:  # run as the helper: python spooler.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import core_bridge as core  # type: ignore[no-redef]

HELPER = Path(__file__).resolve()
HEAD = struct.Struct("!I")
PIPE_BYTES = 1024 * 1024  # the pipe holds this much while the helper catches up
F_SETPIPE_SZ = 1031
MAX_GATES = 256


def frame(message: dict) -> bytes:
    body = json.dumps(message, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    return HEAD.pack(len(body)) + body


def helper_env(env: dict | None = None) -> dict:
    """What the helper itself runs with. It gets secret values and the sender's environment in frames."""
    source = os.environ if env is None else env
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "PYTHONDONTWRITEBYTECODE")
    return {name: source[name] for name in keep if source.get(name)}


class InlineSpool:
    """The same work, in the callback."""

    mode = "inline"

    def send(self, records: list, flush_state, env: dict | None) -> None:
        for gate, record in records:
            core.spool(core.prepare(record, gate), gate.state)
        if flush_state is not None:
            core.launch_sender(flush_state, env)

    def close(self, timeout: float = 10.0) -> None:
        return None


class HelperSpool:
    """The client end, in Hermes. Callers serialize ``send`` (the plugin's order lock)."""

    mode = "helper"

    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.fd = -1
        self.pid = 0
        self.gates: dict = {}  # Gate -> id, for gates this helper has been sent
        self.fallback = InlineSpool()

    def _start(self, state) -> None:
        read, write = os.pipe()
        try:
            fcntl.fcntl(write, F_SETPIPE_SZ, PIPE_BYTES)
        except OSError:
            pass
        core._private_dir(Path(state))
        log = os.open(Path(state) / "flush.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            self.proc = subprocess.Popen([sys.executable or "python3", str(HELPER)], stdin=read, stdout=log, stderr=log, close_fds=True,
                                         start_new_session=True, cwd="/", env=helper_env())
        except BaseException:
            os.close(write)
            raise
        finally:
            os.close(read)
            os.close(log)
        self.fd, self.pid, self.gates = write, os.getpid(), {}

    def _drop(self) -> None:
        if self.fd >= 0 and self.pid == os.getpid():
            try:
                os.close(self.fd)
            except OSError:
                pass
        self.proc, self.fd, self.pid, self.gates = None, -1, 0, {}

    def _write(self, data: bytes) -> None:
        view = memoryview(data)
        while view:
            view = view[os.write(self.fd, view):]

    def _frames(self, records: list, flush_state, env: dict | None) -> bytes:
        parts = []
        if len(self.gates) > MAX_GATES:
            self.gates = {}
            parts.append(frame({"t": "forget"}))
        for gate, record in records:
            number = self.gates.get(gate)
            if number is None:
                number = self.gates[gate] = len(self.gates) + 1
                parts.append(frame({"t": "gate", "id": number, "gate": gate.to_json()}))
            parts.append(frame({"t": "rec", "g": number, "r": record}))
        if flush_state is not None:
            parts.append(frame({"t": "flush", "state": str(flush_state), "env": dict(os.environ if env is None else env)}))
        return b"".join(parts)

    def send(self, records: list, flush_state, env: dict | None) -> None:
        if not records and flush_state is None:
            return
        if self.pid != os.getpid():
            self._drop()  # a forked child starts its own helper
        for attempt in (1, 2):
            try:
                if self.proc is None:
                    self._start(records[0][0].state if records else flush_state)
                self._write(self._frames(records, flush_state, env))
                return
            except OSError:
                self._drop()
                if attempt == 2:
                    break
        self.fallback.send(records, flush_state, env)  # no helper: do it here rather than lose the turn

    def close(self, timeout: float = 10.0) -> None:
        """Close the pipe and wait for the helper to fold what it holds. For tests and benchmarks."""
        proc = self.proc
        self._drop()
        if proc is not None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass


def spool_for(env: dict | None = None):
    source = os.environ if env is None else env
    return InlineSpool() if source.get("JUDGMENT_HERMES_SPOOL") == "inline" else HelperSpool()


# ---- the helper process --------------------------------------------------------------------------


# Frames the helper holds read but not yet folded. Past this, it stops reading and Hermes's next write waits.
BACKLOG_BYTES = 64 * 1024 * 1024


def _read(fd: int, size: int) -> bytes:
    chunks, got = [], 0
    while got < size:
        chunk = os.read(fd, size - got)
        if not chunk:
            break
        chunks.append(chunk)
        got += len(chunk)
    return b"".join(chunks)


class Backlog:
    """Frames read off the pipe, oldest first. A reader thread empties the pipe into it, so a callback
    in Hermes never waits on redaction, only on the pipe when this holds BACKLOG_BYTES."""

    def __init__(self, limit: int = BACKLOG_BYTES):
        import collections
        import threading

        self.frames: collections.deque = collections.deque()
        self.bytes = 0
        self.limit = limit
        self.closed = False
        self.cond = threading.Condition()

    def put(self, body: bytes | None) -> None:
        with self.cond:
            while body is not None and self.bytes >= self.limit:
                self.cond.wait()
            if body is None:
                self.closed = True
            else:
                self.frames.append(body)
                self.bytes += len(body)
            self.cond.notify_all()

    def get(self) -> bytes | None:
        with self.cond:
            while not self.frames and not self.closed:
                self.cond.wait()
            if not self.frames:
                return None
            body = self.frames.popleft()
            self.bytes -= len(body)
            self.cond.notify_all()
            return body


def read_frames(fd: int, backlog: Backlog) -> None:
    """Empty the pipe in big reads and cut frames out of memory: each read costs this thread one turn
    at the GIL, which the folding thread holds through long regex scans."""
    buffer = bytearray()
    try:
        while True:
            chunk = os.read(fd, PIPE_BYTES)
            if not chunk:
                return
            buffer += chunk
            start = 0
            while len(buffer) - start >= HEAD.size:
                (size,) = HEAD.unpack_from(buffer, start)
                if len(buffer) - start - HEAD.size < size:
                    break
                backlog.put(bytes(buffer[start + HEAD.size : start + HEAD.size + size]))
                start += HEAD.size + size
            del buffer[:start]
    finally:
        backlog.put(None)


def fold(body: bytes, gates: dict) -> None:
    message = json.loads(body)
    kind = message.get("t")
    if kind == "gate":
        gates[message["id"]] = core.Gate.from_json(message["gate"])
    elif kind == "rec":
        gate = gates[message["g"]]
        core.spool(core.prepare(message["r"], gate), gate.state)
    elif kind == "flush":
        core.launch_sender(message["state"], message["env"])
    elif kind == "forget":
        gates.clear()


def serve(fd: int = 0) -> None:
    """Fold frames, in order, until the pipe closes and every frame read is folded.

    A frame that fails is skipped; nothing of it is logged but its error type.
    """
    import threading

    backlog = Backlog()
    reader = threading.Thread(target=read_frames, args=(fd, backlog), daemon=True)
    reader.start()
    gates: dict = {}
    while True:
        body = backlog.get()
        if body is None:
            return
        try:
            fold(body, gates)
        except Exception as exc:  # noqa: BLE001
            print(f"hermes spooler: skip a frame: {type(exc).__name__}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    serve(0)
