"""Where a Hermes callback's records go: one queue in Hermes, one helper process that redacts and publishes them.

Redaction can't run in Hermes's process. A Python regex call holds the GIL for its whole scan, up to
several ms on a secret-dense 20 KB window, and no switch interval preempts it: in-process redaction of
a 56 KB tool output stalled Hermes's own threads 8 ms at p95 (2.7 ms even at a 0.2 ms switch interval;
see the bench notes in hermes/README.md). So the work is split:

    callback (Hermes's thread)   bulky payloads snapshotted (snapshot.py, one bounded pass, before the
                                 plugin's lock) -> tracker records -> bound (sizes only: no walk) ->
                                 Outbox.put: append to one queue, poke the I/O thread. No I/O, no wait,
                                 no lock another thread holds for long.
    I/O thread (Hermes, daemon)  spawn the helper; encode entries as frames; nonblocking writes to its
                                 pipe; read its acks; run alias jobs (a rotated id's /judgment file).
    helper (own session)         frame -> snapshot.unpack -> core_bridge.prepare -> stage -> commit seq
                                 -> publish -> ack;
                                 flush frame -> launch_sender -> ack. Survives Hermes's exit.

Plain text crosses only the pipe, which is memory; nothing reaches disk before prepare ran.

The queue (``Outbox``) holds Entries ``(seq, kind, gate, body, size)`` in two deques, in seq order:
``unsent``, then ``inflight`` (written to the helper, not yet acked). Both count against one budget.

    put:  entry -> unsent, or dropped whole and counted when the budget is full (never waits)
    I/O:  unsent -> inflight once its frame is in the write buffer; inflight -> gone at its ack
    helper died (ack EOF, EPIPE): inflight -> front of unsent, and a new helper gets them again;
          an entry in flight through HELPER_TRIES deaths is dropped whole and counted
    Impossible: an entry in both deques; an acked entry kept; an entry resent to the helper that acked it.

Helper lifecycle (I/O thread):  NONE --work queued--> UP (spawned) --ack EOF / EPIPE--> NONE (respawn
    after a backoff that grows with deaths in a row) ... --close()--> the pipe closed once every entry
    is written: the helper folds what it holds, then exits on EOF.
Outbox lifecycle: OPEN --put first entry--> RUNNING (I/O thread started) --close() at exit-->
    CLOSING (waits at most EXIT_WAIT_S for the I/O thread to write everything) --> CLOSED (put drops).
    A forked child: any state --> OPEN, empty, with its own instance id; the inherited pipe, ack, and
    wake fds closed (so the parent's helper still sees EOF when the parent exits), new locks; the
    parent's entries stay the parent's.

Publication is exactly once per entry, even when a helper dies at any point (``serve``):
    the record is prepared, written to inbox.tmp/<name>, then its seq is written to the ledger (the
    commit point), then linked into inbox/<name>, then acked. <name> is fixed by (instance, seq). A
    replayed entry at or below the ledger's seq is only re-linked if its staged copy is still there;
    above it, it is prepared again and overwrites a stale staged copy.

Frames, length-prefixed marshal (the helper runs on Hermes's own interpreter; a snapshot crosses as the
bytes it already is, so Hermes's I/O thread never re-encodes a large value): ``{"t": "gate", "id": n, "gate": {...}}`` defines a gate for this
helper; ``{"t": "rec", "s": seq, "g": n, "r": record}``; ``{"t": "flush", "s": seq, "state": dir,
"env": {...}}``; ``{"t": "forget"}`` drops the gates. The helper acks each seq with 8 bytes on fd 1.
"""

from __future__ import annotations

import collections
import fcntl
import json
import marshal
import os
import select
import struct
import subprocess
import sys
import threading
import time
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from . import core_bridge as core
    from . import snapshot
else:  # run as the helper: python spooler.py INSTANCE
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import core_bridge as core  # type: ignore[no-redef]
    import snapshot  # type: ignore[no-redef]

HELPER = Path(__file__).resolve()
HEAD = struct.Struct("!I")
ACK = struct.Struct("!Q")
PIPE_BYTES = 1024 * 1024  # the pipe holds this much while the helper catches up
F_SETPIPE_SZ = 1031
MAX_GATES = 256

# A field larger than this can't be redacted whole within the helper's budget: it is dropped whole.
FIELD_CAP = 512 * 1024
RECORD_CAP = 1024 * 1024
OMITTED = "[omitted: too large to redact]"
# What the queue holds, sent and unsent together. Past either, a new entry is dropped and counted.
QUEUE_BYTES = 64 * 1024 * 1024
QUEUE_ENTRIES = 50_000
EXIT_WAIT_S = 0.5  # the most Hermes's exit waits for the I/O thread to hand everything to the helper
HELPER_TRIES = 3  # deaths with an entry in flight before that entry is dropped whole
WRITE_CHUNK = 256 * 1024


def frame(message: dict) -> bytes:
    try:
        body = marshal.dumps(message)
    except ValueError:  # a value marshal can't hold (a type the tracker never makes): as JSON would show it
        body = marshal.dumps(json.loads(json.dumps(message, default=lambda value: value.decode("latin-1") if isinstance(value, bytes) else str(value))))
    return HEAD.pack(len(body)) + body


def helper_env(env: dict | None = None) -> dict:
    """What the helper itself runs with. It gets secret values and the sender's environment in frames."""
    source = os.environ if env is None else env
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "SYSTEMROOT", "PYTHONDONTWRITEBYTECODE")
    return {name: source[name] for name in keep if source.get(name)}


# ---- bound: the one cap before redaction -----------------------------------------------------------


def _weigh(value: Any, limit: int) -> int:
    """What ``value`` weighs (snapshot.weight), counting no further than just past ``limit``. A snapshot or a
    string is weighed by its length, with no walk; only small structures the tracker built (usage, the list
    of tool calls around their snapshotted arguments) are walked."""
    if isinstance(value, (str, bytes, snapshot.Omitted)):
        return snapshot.weight(value)
    total = 0
    stack = [value]
    while stack and total <= limit:
        item = stack.pop()
        if isinstance(item, (str, bytes, snapshot.Omitted)):
            total += snapshot.weight(item)
        elif isinstance(item, dict):
            for key, inner in item.items():
                total += len(key) if isinstance(key, str) else 8
                stack.append(inner)
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        else:
            total += 8
    return total


def bound(record: dict) -> tuple[dict, int]:
    """The record with each field too large to redact whole replaced whole, never cut, and its size.

    A field over FIELD_CAP, or one that would take the record past RECORD_CAP, or one the snapshot already
    found over its budget (snapshot.Omitted, here or inside a tool call), becomes OMITTED and is named in
    ``omitted``. prepare treats an omitted input as sensitive (its turn is blanked). No Omitted leaves here.
    """
    budget = RECORD_CAP
    out: dict = {}
    omitted = []
    for key, value in record.items():
        size = _weigh(value, FIELD_CAP)
        if size > FIELD_CAP or size > budget:
            out[key] = OMITTED
            omitted.append(key)
            size = len(OMITTED)
        else:
            out[key] = value
        budget -= size
    if omitted:
        out["omitted"] = omitted
    return out, RECORD_CAP - budget


# ---- the queue, in Hermes ------------------------------------------------------------------------


@dataclass
class Entry:
    seq: int
    kind: str  # rec | flush | alias
    gate: Any  # rec: the Gate
    body: Any  # rec: record; flush: (state, env); alias: (key, alias)
    size: int
    tries: int = 0  # helper deaths while this entry was in flight


_outboxes: "weakref.WeakSet[Outbox]" = weakref.WeakSet()


def _after_fork() -> None:
    for outbox in list(_outboxes):
        outbox.reset_in_child()


os.register_at_fork(after_in_child=_after_fork)


def _close(fd: int) -> None:
    if fd >= 0:
        try:
            os.close(fd)
        except OSError:
            pass


class Outbox:
    """The one queue of records between Hermes's callbacks and the helper. See the module docstring."""

    def __init__(self):
        self.pipe = self.acks = -1
        self._fresh()
        _outboxes.add(self)

    def _fresh(self) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.unsent: collections.deque = collections.deque()
        self.inflight: collections.deque = collections.deque()
        self.bytes = 0
        self.seq = 0
        self.buffered = 0  # bytes encoded but not yet in the pipe
        self.dropped: dict = {}  # reason -> count, not yet logged
        self.drop_state = ""  # where to log them
        self.phase = "open"  # open | running | closing | closed
        self.thread: threading.Thread | None = None
        self.proc: subprocess.Popen | None = None
        self.pipe = self.acks = -1
        self.gen = 0  # helper incarnation; acks and EOF from an older one are ignored
        self.lost_gen = -1  # the incarnation whose ack reader saw EOF
        self.instance = f"{os.getpid()}-{os.urandom(4).hex()}"
        self.spawned = 0  # helpers started, for tests and the bench
        self.deaths = 0  # helper deaths (or failed spawns) in a row without an ack
        self.retry_at = 0.0

    def reset_in_child(self) -> None:
        """After fork, in the child, before any callback runs: forget the parent's queue, fds, threads, and helper."""
        _close(self.pipe)
        _close(self.acks)
        self._fresh()

    # -- the callback side: no syscall, no wait

    def put(self, items: list) -> None:
        """Queue (kind, gate, body) items, in order. Drops an item whole when the budget is full."""
        with self.lock:
            for kind, gate, body in items:
                size = 64
                if kind == "rec":
                    body, size = bound(body)
                if self.phase in ("closing", "closed"):
                    self._drop("after exit", gate)
                    continue
                if len(self.unsent) + len(self.inflight) >= QUEUE_ENTRIES or self.bytes + size > QUEUE_BYTES:
                    self._drop("queue full", gate)
                    continue
                self.seq += 1
                self.unsent.append(Entry(self.seq, kind, gate, body, size))
                self.bytes += size
            if self.phase == "open" and self.unsent:
                self.phase = "running"
                self.thread = threading.Thread(target=self._run, name="judgment-outbox", daemon=True)
                self.thread.start()  # once per process
            self.cond.notify_all()  # wakes the writer without releasing the GIL

    def _drop(self, reason: str, gate: Any) -> None:
        self.dropped[reason] = self.dropped.get(reason, 0) + 1
        state = getattr(gate, "state", "")
        if state:
            self.drop_state = state

    def handed_over(self) -> bool:
        """Every queued entry is in the helper's pipe (or there is none)."""
        with self.lock:
            return not self.unsent and not self.buffered

    # -- exit

    def close(self, wait: float = EXIT_WAIT_S) -> bool:
        """Stop taking entries; give the writer ``wait`` seconds to write every one to the helper, which then
        finishes on its own. True when everything was handed over. What is left is dropped and counted."""
        deadline = time.monotonic() + wait
        with self.lock:
            if self.phase == "open":
                self.phase = "closed"
            elif self.phase == "running":
                self.phase = "closing"
            self.cond.notify_all()
            while self.phase != "closed" and time.monotonic() < deadline:
                self.cond.wait(max(deadline - time.monotonic(), 0.001))
            done = self.phase == "closed" and not self.unsent
            if not done:
                for entry in self.unsent:
                    self._drop("not handed over at exit", entry.gate)
        self._log_drops()
        return done

    def drain(self, timeout: float = 30.0) -> None:
        """close(), then wait for the helper to publish everything and exit. For tests and the bench."""
        self.close(timeout)
        proc = self.proc
        if proc is not None:
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                pass

    def _log_drops(self) -> None:
        with self.lock:
            dropped, state = dict(self.dropped), self.drop_state
            self.dropped = {}
        if dropped and state:
            core.log_line(state, "hermes: " + ", ".join(f"dropped {n} record(s): {why}" for why, n in sorted(dropped.items())))

    # -- the helper, from the writer thread

    def _spawn(self, state: str) -> None:
        read, write = os.pipe()
        ack_r, ack_w = os.pipe()
        try:
            fcntl.fcntl(write, F_SETPIPE_SZ, PIPE_BYTES)
        except OSError:
            pass
        try:
            core._private_dir(Path(state))
            log = os.open(Path(state) / "flush.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        except OSError:
            log = os.open(os.devnull, os.O_WRONLY)
        try:
            proc = subprocess.Popen([sys.executable or "python3", str(HELPER), self.instance], stdin=read, stdout=ack_w, stderr=log,
                                    close_fds=True, start_new_session=True, cwd="/", env=helper_env())
        except BaseException:
            _close(write)
            _close(ack_r)
            raise
        finally:
            _close(read)
            _close(ack_w)
            _close(log)
        os.set_blocking(write, False)
        with self.lock:
            self.gen += 1
            self.proc, self.pipe, self.acks = proc, write, ack_r
        self.spawned += 1
        threading.Thread(target=self._read_acks, args=(ack_r, self.gen), name="judgment-acks", daemon=True).start()

    def _read_acks(self, fd: int, gen: int) -> None:
        """One per helper: its acks, until EOF (it exited or died). Owns and closes ``fd``."""
        acks = b""
        try:
            while True:
                try:
                    chunk = os.read(fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    with self.lock:
                        if self.gen == gen:
                            self.lost_gen = gen
                            self.cond.notify_all()
                    return
                acks += chunk
                whole = len(acks) // ACK.size * ACK.size
                if whole:
                    upto = ACK.unpack_from(acks, whole - ACK.size)[0]
                    acks = acks[whole:]
                    with self.lock:
                        if self.gen == gen:
                            while self.inflight and self.inflight[0].seq <= upto:
                                self.bytes -= self.inflight.popleft().size
                            self.deaths = 0
                            self.cond.notify_all()
        finally:
            _close(fd)

    def _lost(self, buffer: bytearray, gates: dict) -> None:
        """The helper is gone: its pipe closed, in-flight entries back at the front of the queue."""
        with self.lock:
            proc, pipe = self.proc, self.pipe
            self.proc, self.pipe, self.acks = None, -1, -1
            self.gen += 1  # its ack reader's last words are ignored
            back = []
            for entry in self.inflight:
                entry.tries += 1
                if entry.tries >= HELPER_TRIES:
                    self.bytes -= entry.size
                    self._drop("the helper died on it", entry.gate)
                else:
                    back.append(entry)
            self.inflight.clear()
            self.unsent.extendleft(reversed(back))
            self.buffered = 0
        _close(pipe)
        if proc is not None:
            try:
                proc.kill()
                proc.wait(timeout=1)
            except (OSError, subprocess.TimeoutExpired):
                pass
        buffer.clear()
        gates.clear()
        self._backoff()

    def _backoff(self) -> None:
        self.deaths += 1
        self.retry_at = time.monotonic() + min(0.05 * 2 ** (self.deaths - 1), 5.0)

    def _encode(self, entry: Entry, gates: dict) -> bytes:
        parts = []
        if entry.kind == "rec":
            number = gates.get(entry.gate)
            if number is None:
                if len(gates) >= MAX_GATES:
                    gates.clear()
                    parts.append(frame({"t": "forget"}))
                number = gates[entry.gate] = len(gates) + 1
                parts.append(frame({"t": "gate", "id": number, "gate": entry.gate.to_json()}))
            parts.append(frame({"t": "rec", "s": entry.seq, "g": number, "r": entry.body}))
        else:
            state, env = entry.body
            parts.append(frame({"t": "flush", "s": entry.seq, "state": str(state), "env": env}))
        return b"".join(parts)

    def _closed(self) -> None:
        with self.lock:
            self.phase = "closed"
            self.cond.notify_all()

    def _run(self) -> None:
        """The writer. Waits on the condition when there is nothing to write, so an idle Hermes costs nothing."""
        buffer = bytearray()
        gates: dict = {}
        failures = 0
        while True:
            try:
                self._log_drops()
                with self.lock:
                    while not buffer and not self.unsent and self.phase == "running" and self.lost_gen != self.gen:
                        self.cond.wait()
                    closing = self.phase == "closing"
                    lost = self.proc is not None and self.lost_gen == self.gen
                if lost:
                    self._lost(buffer, gates)
                    continue
                with self.lock:
                    take = []
                    while self.unsent and len(buffer) < WRITE_CHUNK * 4:
                        entry = self.unsent[0]
                        if entry.kind != "alias" and self.proc is None:
                            break
                        take.append(self.unsent.popleft())
                        if entry.kind == "alias":
                            self.bytes -= entry.size
                        else:
                            self.inflight.append(entry)
                    first = self.unsent[0] if self.unsent else None
                for entry in take:
                    if entry.kind == "alias":
                        core.copy_session_mode(*entry.body)
                    else:
                        buffer += self._encode(entry, gates)
                with self.lock:
                    self.buffered = len(buffer)
                if closing and not buffer and first is None:
                    _close(self.pipe)  # EOF: the helper finishes on its own, after Hermes exits
                    self.pipe = -1
                    self._closed()
                    return
                if self.proc is None and first is not None:
                    if closing or time.monotonic() >= self.retry_at:
                        state = first.gate.state if first.kind == "rec" else str(first.body[0])
                        try:
                            self._spawn(state)
                        except OSError:
                            self._backoff()
                            self._drop_dead_end()
                            if closing:  # no helper at exit: close() counts what is left
                                self._closed()
                                return
                    else:
                        with self.lock:
                            self.cond.wait(max(self.retry_at - time.monotonic(), 0.001))
                    continue
                if buffer and self.pipe >= 0:
                    poller = select.poll()
                    poller.register(self.pipe, select.POLLOUT)
                    events = poller.poll(50)  # a stalled helper: look again for closing or its death
                    if events and events[0][1] & (select.POLLERR | select.POLLHUP):
                        self._lost(buffer, gates)
                        continue
                    if events:
                        try:
                            del buffer[: os.write(self.pipe, buffer[:WRITE_CHUNK])]
                        except BlockingIOError:
                            pass
                        except OSError:
                            self._lost(buffer, gates)
                            continue
                        with self.lock:
                            self.buffered = len(buffer)
            except Exception as exc:  # noqa: BLE001  the thread never dies: a bug here must not end tracing
                failures += 1
                if failures <= 3:
                    core.log_line(self.drop_state or core.state_dir(), f"hermes: outbox error {type(exc).__name__}")
                time.sleep(0.05)

    def _drop_dead_end(self) -> None:
        """No helper can start: past HELPER_TRIES failures in a row, the oldest entry goes, so the queue keeps moving."""
        if self.deaths < HELPER_TRIES:
            return
        with self.lock:
            if self.unsent:
                entry = self.unsent.popleft()
                self.bytes -= entry.size
                self._drop("no helper could start", entry.gate)


def spool_for() -> Outbox:
    return Outbox()


# ---- the helper process --------------------------------------------------------------------------


class Ledger:
    """The highest seq this instance committed in a state dir: the commit point of exactly-once publication."""

    def __init__(self, state: str, instance: str):
        folder = Path(state) / "hermes"
        core._private_dir(folder)
        self.path = folder / f"{instance}.seq"
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        raw = os.pread(self.fd, 32, 0)
        self.seq = int(raw) if raw.strip().isdigit() else 0
        _sweep(folder, self.path)

    def commit(self, seq: int) -> None:
        os.pwrite(self.fd, f"{seq:020d}".encode(), 0)  # one write of fixed size: never seen half done
        self.seq = seq

    def remove(self) -> None:
        _close(self.fd)
        try:
            self.path.unlink()
        except OSError:
            pass


def _sweep(folder: Path, keep: Path, age: float = 7 * 86400) -> None:
    """Ledgers a crashed Hermes left, a week on."""
    cutoff = time.time() - age
    for path in folder.glob("*.seq"):
        try:
            if path != keep and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def read_frames(fd: int):
    """Each frame's JSON, in order, until EOF."""
    buffer = bytearray()
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
            yield marshal.loads(bytes(buffer[start + HEAD.size : start + HEAD.size + size]))
            start += HEAD.size + size
        del buffer[:start]


class Helper:
    """One frame at a time: prepare, stage, commit, publish, ack."""

    def __init__(self, instance: str, ack_fd: int = 1):
        self.instance = instance
        self.ack_fd = ack_fd
        self.gates: dict = {}
        self.ledgers: dict = {}
        self.logged: set = set()

    def ledger(self, state: str) -> Ledger:
        if state not in self.ledgers:
            self.ledgers[state] = Ledger(state, self.instance)
        return self.ledgers[state]

    def fold(self, message: dict) -> None:
        kind = message.get("t")
        if kind == "gate":
            self.gates[message["id"]] = core.Gate.from_json(message["gate"])
        elif kind == "forget":
            self.gates.clear()
        elif kind == "rec":
            gate, seq = self.gates[message["g"]], message["s"]
            name = f"hermes.{self.instance}.{seq:012d}.json"
            ledger = self.ledger(gate.state)
            if seq > ledger.seq:
                if gate.withheld and gate not in self.logged:
                    self.logged.add(gate)
                    core.log_line(gate.state, "hermes: redaction changed these stamps: " + ", ".join(gate.withheld))
                core.stage(core.prepare(snapshot.unpack(message["r"]), gate), gate.state, name)
                ledger.commit(seq)
            core.publish(gate.state, name)
        elif kind == "flush":
            core.launch_sender(message["state"], message["env"])

    def ack(self, seq: int) -> None:
        if self.ack_fd < 0:
            return
        try:
            os.write(self.ack_fd, ACK.pack(seq))
        except OSError:  # Hermes is gone: keep publishing what it sent
            self.ack_fd = -1

    def serve(self, fd: int = 0) -> None:
        """Fold frames, in order, until EOF. A frame that fails is skipped whole and acked; only its error type is logged."""
        for message in read_frames(fd):
            try:
                self.fold(message)
            except Exception as exc:  # noqa: BLE001
                print(f"hermes spooler: skip a frame: {type(exc).__name__}", file=sys.stderr, flush=True)
            if message.get("s"):
                self.ack(message["s"])
        for ledger in self.ledgers.values():
            ledger.remove()  # every frame was folded: nothing to replay


if __name__ == "__main__":
    os.umask(0o077)
    Helper(sys.argv[1] if len(sys.argv) > 1 else f"{os.getpid()}-solo").serve(0)
