"""Hermes plugin: callbacks never slow or break Hermes, and every record is published once whatever dies.

    Latency                      callback p95, a callback that fails, a state folder that can't be written
    Sender, Helper               the helper and sender outlive Hermes; a killed helper is replaced
    Backpressure, Fork           a stalled helper, a failed spawn, a forked Hermes
    Replay, Identity             a helper crash at each step, then a replay: each record once, same id
    LatePublication              the core flusher folds a file published late
    BoundedState                 the tracker's tables stay bounded over many sessions
"""

from __future__ import annotations

import json
import os
import re
import signal
import statistics
import subprocess
import sys
import textwrap
import threading
import time
import tracemalloc
import unittest
from pathlib import Path
from unittest import mock

from fake_sink import attrs, roots
from hermes_support import (
    ENV_VALUE,
    LATENCY_GATE_P95_MS,
    LATENCY_TARGET_P95_MS,
    VENDOR_CORE,
    Case,
    Sandbox,
    Turn,
    fire_all,
    latency_trials,
    load_plugin,
    module,
    new_session_id,
    over,
    pct,
    plugin_env,
    simple_turn,
    subagent_turn,
    later,
    wait_until,
)
from support import REPO, wait_for


class Latency(Case):
    """Callback p95 per hook: reported against LATENCY_TARGET_P95_MS, failed at LATENCY_GATE_P95_MS (hermes_support)."""

    REPORT: dict = {}

    @classmethod
    def tearDownClass(cls):
        if cls.REPORT:
            print(f"\nHermes callback latency, in-process, traced path (ms; target p95 < {LATENCY_TARGET_P95_MS}):", file=sys.stderr)
            for hook, (p50, p95, mx, count, tried) in sorted(cls.REPORT.items()):
                flag = "  over target" if p95 >= LATENCY_TARGET_P95_MS else ""
                print(f"  {hook:22s} n={count:5d} p50={p50:.3f} p95={p95:.3f} max={mx:.3f} (trials {tried}){flag}", file=sys.stderr)

    def test_each_callback_p95_under_budget(self):
        """Full traced Hermes sequences, timed per callback as Hermes would see it, until every hook has 1000 calls.

        Each round is a new session: a tool turn whose first answer attempt fails and is retried, a turn
        that starts a background child, then finalize (even rounds) or reset (odd rounds).
        """
        result = json.dumps({"content": "1|" + "hello world line\n" * 200})

        def timed(ctx, plugin, events, times):
            for hook, kwargs, at in events:
                kwargs.setdefault("telemetry_schema_version", "hermes.observer.v1")
                plugin.now_ms = lambda at=at: int(at * 1000)
                (callback,) = ctx.hooks[hook]
                start = time.perf_counter()
                callback(**kwargs)
                times.setdefault(hook, []).append((time.perf_counter() - start) * 1000)

        def rounds(ctx, plugin, times, until: int):
            n = 0
            while min((len(times.get(hook, [])) for hook in plugin.HOOKS), default=0) < until:
                sid = new_session_id()
                turn = Turn(sid, f"prompt {n} with some ordinary text")
                call = turn.api(tool=("read_file", {"path": "notes.txt"}))
                turn.tool(call, "read_file", {"path": "notes.txt"}, result)
                turn.api(error={"status_code": 500, "retry_count": 1})
                turn.api(text=f"answer {n}")
                turn.end(f"answer {n}")
                _p, _c, sub = subagent_turn(sid)
                sub = [event for event in sub if event[0] != "on_session_start"]
                hook = "on_session_finalize" if n % 2 == 0 else "on_session_reset"
                timed(ctx, plugin, turn.events + sub + [(hook, {"session_id": sid, "platform": "cli", "reason": "shutdown"}, later(sub, 0.1))], times)
                n += 1

        with plugin_env(self.sb, JUDGMENT_FLUSH="0") as (plugin, ctx):
            rounds(ctx, plugin, {}, 1)  # warm-up, untimed

            def trial() -> dict:
                times: dict[str, list[float]] = {}
                rounds(ctx, plugin, times, 1000)
                self.assertEqual(plugin.errors["count"], 0)
                return times

            times, tried = latency_trials(trial)
            self.assertGreater(len(list((self.sb.state / "inbox").iterdir())), 1000)
        for hook, values in times.items():
            self.REPORT[hook] = (statistics.median(values), pct(values, 0.95), max(values), len(values), tried)
        self.assertEqual(over(times, LATENCY_GATE_P95_MS), {}, f"p95 over {LATENCY_GATE_P95_MS} ms in each of {tried} trials")

    def test_a_failing_callback_never_raises(self):
        with plugin_env(self.sb) as (plugin, ctx):
            original = plugin.tracker.step

            def broken(*args, **kwargs):
                raise RuntimeError("boom")

            plugin.tracker.step = broken
            try:
                for hook in ctx.hooks:
                    self.assertEqual(ctx.fire(hook, session_id="s", turn_id="t"), [])
            finally:
                plugin.tracker.step = original
            self.assertEqual(plugin.errors["count"], len(ctx.hooks))
            # Garbage kwargs: wrong types everywhere.
            before = plugin.errors["count"]
            for hook in ctx.hooks:
                self.assertEqual(ctx.fire(hook, session_id=object(), turn_id=None, args=object(), usage="x", started_at="soon", result=b"\x00"), [])
            self.assertLessEqual(plugin.errors["count"] - before, len(ctx.hooks))

    def test_unwritable_state_fails_open(self):
        blocker = self.sb.dir / "not-a-dir"
        blocker.write_text("x")
        sid = new_session_id()
        with plugin_env(self.sb, JUDGMENT_STATE_DIR=str(blocker / "state")) as (plugin, ctx):
            self.assertEqual(fire_all(ctx, simple_turn(sid, "hi", path="a", result="b", answer="c").events, plugin), [])
            # Callbacks touch no file, so they never see the error; the writer thread and the helper do, off Hermes's threads.
            self.assertEqual(plugin.errors["count"], 0)


SENDER = textwrap.dedent(
    """
    import json, os, sys, time
    sys.path.insert(0, {tests!r})
    from hermes_support import load_plugin, FakeCtx, simple_turn, fire_all, new_session_id
    plugin = load_plugin()
    ctx = FakeCtx()
    plugin.register(ctx)
    sid = new_session_id()
    turn = simple_turn(sid, "survive", path="a", result="b", answer="lands after the kill")
    start = time.perf_counter()
    fire_all(ctx, turn.events, plugin)
    took = (time.perf_counter() - start) * 1000
    # The hand-off to the helper is asynchronous; a real turn lasts long enough for it. Wait for it (at most 5 s),
    # then die: what Hermes handed over must survive the kill.
    handed = time.perf_counter()
    while not plugin.spool.handed_over() and time.perf_counter() - handed < 5:
        time.sleep(0.001)
    print(json.dumps({{"ms": took, "handoff_ms": (time.perf_counter() - handed) * 1000, "helper": plugin.spool.proc.pid if plugin.spool.proc else 0}}), flush=True)
    time.sleep(60)
    """
)


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


class Sender(Case):
    def spawn(self, sink_url: str) -> subprocess.Popen:
        env = {**self.sb.env, "JUDGMENT_API_URL": sink_url, "TERMINAL_CWD": str(self.sb.project), "JUDGMENT_FLUSH_GRACE_MS": "200"}
        env.pop("JUDGMENT_FLUSH")
        proc = subprocess.Popen([sys.executable, "-c", SENDER.format(tests=str(REPO / "tests"))], cwd=self.sb.project, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True)
        self.addCleanup(proc.stdout.close)
        return proc

    def test_sender_survives_killing_the_process_group(self):
        proc = self.spawn(self.sink.url)
        line = proc.stdout.readline()
        os.killpg(proc.pid, signal.SIGKILL)  # Hermes and its whole group die the moment the turn ends
        proc.wait(timeout=10)
        info = json.loads(line)
        self.assertGreater(info["helper"], 0)
        # The helper runs in its own session, so the kill missed it; it folds the turn and starts the sender.
        self.assertNotEqual(os.getpgid(info["helper"]) if _alive(info["helper"]) else -1, proc.pid)
        started = time.time()
        self.assertTrue(wait_for(lambda: roots(self.sink.spans()), timeout=20), (self.sb.state / "flush.log").read_text() if (self.sb.state / "flush.log").exists() else "no flush.log")
        self.assertEqual(attrs(roots(self.sink.spans())[0])["judgment.output"], "lands after the kill")
        Sender.landed = time.time() - started


class Helper(Case):
    def test_a_killed_helper_is_replaced_and_later_turns_land(self):
        sid = new_session_id()
        one = Turn(sid, "before")
        one.api(text="a")
        one.end("a")
        two = Turn(sid, "after", first=False, start=later(one.events, 1))
        two.api(text="b")
        two.end("b")
        with plugin_env(self.sb) as (plugin, ctx):
            fire_all(ctx, one.events, plugin)
            self.assertTrue(wait_until(lambda: plugin.spool.proc is not None and not plugin.spool.inflight and plugin.spool.handed_over()))
            first = plugin.spool.proc
            first.kill()
            first.wait(timeout=10)
            fire_all(ctx, two.events, plugin)
            self.assertTrue(wait_until(lambda: plugin.spool.proc is not None and plugin.spool.proc.pid != first.pid))
            self.assertEqual(plugin.errors["count"], 0)
        self.assertEqual(sorted(attrs(root)["judgment.input"] for root in roots(self.flushed())), ["after", "before"])

    def test_helper_env_holds_no_secret(self):
        spooler = sys.modules[load_plugin().__name__ + ".spooler"]
        env = spooler.helper_env({**self.sb.env, "OTHER_TOKEN": ENV_VALUE})
        # The allowlist itself holds no secret-shaped name, and only allowlisted names get through.
        self.assertEqual([name for name in spooler.HELPER_ENV if re.search(r"KEY|TOKEN|SECRET|PASS|AUTH|CRED", name)], [])
        self.assertLessEqual(set(env), set(spooler.HELPER_ENV))
        # Of the sandbox env, exactly these: TMPDIR is the suite's private temp folder (support.TMP), which the
        # helper must keep so its temp files land where its parent's do, not in the shared /tmp.
        self.assertEqual(set(env), {"PATH", "HOME", "LANG", "TMPDIR", "PYTHONDONTWRITEBYTECODE"})
        self.assertNotIn(ENV_VALUE, env.values())


class Backpressure(Case):
    def dense_turns(self, count: int) -> list:
        text = ("line " + "z" * 60 + "\n") * 800  # ~52 KB
        events = []
        for n in range(count):
            turn = Turn(new_session_id(), f"p{n}")
            call = turn.api(tool=("read_file", {"path": "f"}))
            turn.tool(call, "read_file", {"path": "f"}, text)
            turn.api(text="ok")
            turn.end("ok")
            events += turn.events
        return events

    def timed(self, ctx, plugin, events) -> list[float]:
        times = []
        for hook, kwargs, at in events:
            plugin.now_ms = lambda at=at: int(at * 1000)
            (callback,) = ctx.hooks[hook]
            start = time.perf_counter()
            callback(**kwargs)
            times.append((time.perf_counter() - start) * 1000)
        return times

    def test_stalled_helper_never_blocks_a_callback_and_memory_stays_bounded(self):
        spooler = module("spooler")
        with mock.patch.object(spooler, "QUEUE_BYTES", 2 * 1024 * 1024):
            with plugin_env(self.sb) as (plugin, ctx):
                self.timed(ctx, plugin, self.dense_turns(1))
                self.assertTrue(wait_until(lambda: plugin.spool.proc is not None and plugin.spool.handed_over()))
                helper = plugin.spool.proc
                os.kill(helper.pid, signal.SIGSTOP)  # the pipe fills (1 MB), then the queue (2 MB)
                try:
                    times = self.timed(ctx, plugin, self.dense_turns(80))
                    self.assertLessEqual(plugin.spool.bytes, 2 * 1024 * 1024)
                    start = time.perf_counter()
                    self.assertFalse(plugin.spool.close(), "a stalled helper can't take everything")
                    self.assertLess(time.perf_counter() - start, 1.0, "exit waits a bounded time")
                finally:
                    os.kill(helper.pid, signal.SIGCONT)
        times.sort()
        self.assertLess(times[int(0.95 * (len(times) - 1))], 3.0)
        self.assertLess(times[-1], 50.0)
        log = (self.sb.state / "flush.log").read_text()
        self.assertIn("dropped", log)
        self.assertIn("queue full", log)
        self.assertEqual(plugin.errors["count"], 0)

    def test_failed_spawn_writes_nothing_from_callbacks_and_exit_is_bounded(self):
        spooler = module("spooler")
        with mock.patch.object(spooler.subprocess, "Popen", side_effect=OSError("no fork")):
            with plugin_env(self.sb) as (plugin, ctx):
                times = self.timed(ctx, plugin, self.dense_turns(5))
                start = time.perf_counter()
                self.assertFalse(plugin.spool.close())
                self.assertLess(time.perf_counter() - start, 1.0)
        self.assertFalse((self.sb.state / "inbox").exists() and any((self.sb.state / "inbox").iterdir()), "no synchronous fallback")
        self.assertLess(max(times), 5.0)
        self.assertIn("dropped", (self.sb.state / "flush.log").read_text())


class Fork(Case):
    def test_forked_child_starts_clean_even_with_locks_held(self):
        sid = new_session_id()
        with plugin_env(self.sb) as (plugin, ctx):
            fire_all(ctx, self.simple(sid, "parent").events, plugin)
            self.assertTrue(wait_until(lambda: plugin.spool.proc is not None and plugin.spool.handed_over()))
            spool = plugin.spool
            inherited = (spool.pipe, spool.instance)
            held = threading.Event()
            release = threading.Event()

            def hold():
                with plugin._lock, spool.lock:
                    held.set()
                    release.wait(10)

            holder = threading.Thread(target=hold)
            holder.start()
            held.wait(5)
            read, write = os.pipe()
            pid = os.fork()
            if pid == 0:  # the child: report, then leave without running Python cleanup
                code = 1
                try:
                    os.close(read)
                    try:
                        os.fstat(inherited[0])
                        pipe_open = True
                    except OSError:
                        pipe_open = False
                    ok_locks = plugin._lock.acquire(timeout=1) and spool.lock.acquire(timeout=1)
                    plugin._lock.release()
                    spool.lock.release()
                    turn = self.simple(new_session_id(), "child")
                    for hook, kwargs, _at in turn.events:  # callbacks run; nothing of the parent's is resent
                        ctx.hooks[hook][0](**kwargs)
                    report = {"pipe_open": pipe_open, "locks": bool(ok_locks), "instance": spool.instance, "proc": spool.proc is not None,
                              "tracker": len(plugin.tracker.convs), "errors": plugin.errors["count"]}
                    os.write(write, json.dumps(report).encode())
                    code = 0
                finally:
                    os._exit(code)
            os.close(write)
            release.set()
            holder.join()
            _pid, status = os.waitpid(pid, 0)
            report = json.loads(os.read(read, 65536) or b"{}")
            os.close(read)
            self.assertEqual(os.waitstatus_to_exitcode(status), 0)
            self.assertFalse(report["pipe_open"], "the child closed its copy of the parent's helper pipe")
            self.assertTrue(report["locks"], "locks held at fork were replaced in the child")
            self.assertNotEqual(report["instance"], inherited[1])
            self.assertEqual(report["errors"], 0)
            parent_helper = spool.proc
        # The parent drained, its helper saw EOF and exited: the child kept no copy of the pipe.
        self.assertIsNotNone(parent_helper.poll())
        self.assertEqual([attrs(root)["judgment.input"] for root in roots(self.flushed())].count("parent"), 1)


class Replay(Case):

    def test_every_crash_point_of_the_helper_publishes_once(self):
        """The helper's steps one at a time: prepare, stage, commit, publish, ack. A crash after any of them, then a new helper
        replaying every unacked frame, publishes each record exactly once."""
        spooler, core = module("spooler"), load_plugin().core
        gate = core.Gate("sanitized", core.settings({}), (), str(self.sb.state), ())
        frames = [{"t": "gate", "id": 1, "gate": gate.to_json()}] + [
            {"t": "rec", "s": n, "g": 1, "r": {"v": 1, "harness": "hermes", "event": "tool", "session_id": "s", "turn_id": "t", "ts": n,
                                               "tool_call_id": f"c{n}", "tool_output": "x"}} for n in range(1, 6)]
        for crash in ("stage", "commit", "publish", "ack"):
            with self.subTest(crash=crash):
                for folder in ("inbox", "inbox.tmp", "hermes"):
                    for path in (self.sb.state / folder).glob("*") if (self.sb.state / folder).is_dir() else []:
                        path.unlink()
                instance = f"t-{crash}"
                acked = []
                dead = spooler.Helper(instance, ack_fd=-1)
                dead.ack = acked.append
                real = {"stage": core.stage, "publish": core.publish}

                class Crash(Exception):
                    pass

                def boom(*args):
                    raise Crash

                for frame in frames:
                    if frame.get("s") == 3:  # die on record 3, right after the step named
                        if crash == "stage":
                            with mock.patch.object(core, "stage", lambda *a: (real["stage"](*a), boom())):
                                self.assertRaises(Crash, dead.fold, frame)
                        elif crash == "commit":
                            with mock.patch.object(spooler.Ledger, "commit", lambda self, seq: (os.pwrite(self.fd, f"{seq:020d}".encode(), 0), boom())):
                                self.assertRaises(Crash, dead.fold, frame)
                        elif crash == "publish":
                            with mock.patch.object(core, "publish", lambda *a: (real["publish"](*a), boom())):
                                self.assertRaises(Crash, dead.fold, frame)
                        else:
                            dead.fold(frame)  # published; the ack never left
                        break
                    dead.fold(frame)
                    if frame.get("s"):
                        dead.ack(frame["s"])
                # The flusher may fold (and delete) what was published before the new helper replays.
                folded = [json.loads(path.read_text()) for path in sorted((self.sb.state / "inbox").iterdir())]
                for path in (self.sb.state / "inbox").iterdir():
                    path.unlink()
                replay = [frame for frame in frames if frame["t"] == "gate" or frame["s"] > max(acked or [0])]
                fresh = spooler.Helper(instance, ack_fd=-1)
                for frame in replay:
                    fresh.fold(frame)
                events = folded + self.sb.inbox()
                self.assertEqual(sorted(event["tool_call_id"] for event in events), [f"c{n}" for n in range(1, 6)], "each exactly once")
                self.assertEqual(list((self.sb.state / "inbox.tmp").iterdir()), [])


class Identity(Case):
    def test_crash_between_link_and_unlink_then_ingest_then_replay_folds_once(self):
        sys.path.insert(0, str(VENDOR_CORE))
        from judgment_core import flush

        spooler, core = module("spooler"), load_plugin().core
        gate = core.Gate("sanitized", core.settings({}), (), str(self.sb.state), ())
        record = {"v": 1, "harness": "hermes", "event": "turn_start", "session_id": "s9", "turn_id": "t9", "ts": 1, "prompt": "hi"}
        frames = [{"t": "gate", "id": 1, "gate": gate.to_json()}, {"t": "rec", "s": 1, "g": 1, "r": record}]

        class Crash(Exception):
            pass

        dead = spooler.Helper("t-link", ack_fd=-1)
        dead.fold(frames[0])
        real_unlink = os.unlink

        def die_on_staged_unlink(path, *args, **kwargs):
            if "inbox.tmp" in str(path):
                raise Crash
            return real_unlink(path, *args, **kwargs)

        with mock.patch.object(core.os, "unlink", die_on_staged_unlink):
            self.assertRaises(Crash, dead.fold, frames[1])
        name = next((self.sb.state / "inbox").iterdir()).name
        self.assertTrue((self.sb.state / "inbox.tmp" / name).exists(), "both links left, as after the crash")
        env = dict(self.sb.env)
        first = flush._inbox(self.sb.state)[0][1]["_id"]
        flush.ingest(self.sb.state, env)  # the flusher folds and deletes the inbox link before the replay
        self.assertEqual(list((self.sb.state / "inbox").iterdir()), [])

        fresh = spooler.Helper("t-link", ack_fd=-1)
        for frame in frames:
            fresh.fold(frame)  # the ledger says seq 1 was committed: publish again, never re-prepare
        replayed = flush._inbox(self.sb.state)
        self.assertEqual(len(replayed), 1)
        self.assertEqual(replayed[0][1]["_id"], first, "a replay changed the event's identity")
        flush.ingest(self.sb.state, env)
        stored = (self.sb.state / "turns" / "hermes" / "s9" / "t9.jsonl").read_text().splitlines()
        self.assertEqual(len(flush._read_jsonl(self.sb.state / "turns" / "hermes" / "s9" / "t9.jsonl")), 1, stored)
        self.assertEqual(list((self.sb.state / "inbox.tmp").iterdir()), [])


class LatePublication(unittest.TestCase):
    """A file published late, with an older mtime than one already folded, is folded, never deleted."""

    def setUp(self):
        sys.path.insert(0, str(VENDOR_CORE))
        from judgment_core import flush

        self.flush = flush
        self.sb = Sandbox()
        self.addCleanup(self.sb.cleanup)
        self.env = dict(self.sb.env)

    def publish(self, name: str, event: dict, mtime_ns: int) -> Path:
        inbox = self.sb.state / "inbox"
        inbox.mkdir(parents=True, exist_ok=True)
        path = inbox / name
        path.write_text(json.dumps(event) + "\n")
        os.utime(path, ns=(mtime_ns, mtime_ns))
        return path

    def event(self, kind: str, turn: str, ts: int) -> dict:
        return {"v": 1, "harness": "hermes", "event": kind, "session_id": "s1", "turn_id": turn, "ts": ts, "capture": "sanitized"}

    def stored(self, turn: str) -> list[str]:
        path = self.sb.state / "turns" / "hermes" / "s1" / f"{turn}.jsonl"
        return [json.loads(line)["event"] for line in path.read_text().splitlines()] if path.exists() else []

    def test_late_older_file_is_folded(self):
        now = time.time_ns()
        self.publish("200.0000000001.json", self.event("turn_start", "t1", 2), now)
        self.flush.ingest(self.sb.state, self.env)
        # Another process wrote this before the file above but linked it into the inbox after the flusher ran.
        late = self.publish("100.0000000001.json", self.event("turn_end", "t1", 3), now - 5_000_000)
        self.flush.ingest(self.sb.state, self.env)
        self.assertFalse(late.exists())
        self.assertEqual(self.stored("t1"), ["turn_start", "turn_end"])

    def test_crash_before_delete_replays_nothing(self):
        now = time.time_ns()
        self.publish("300.0000000001.json", self.event("turn_start", "t2", 2), now)
        with mock.patch.object(Path, "unlink", lambda self, missing_ok=False: None):
            self.flush.ingest(self.sb.state, self.env)  # state saved, then the process died before deleting
        self.flush.ingest(self.sb.state, self.env)
        self.assertEqual(self.stored("t2"), ["turn_start"])
        self.assertFalse(any((self.sb.state / "inbox").iterdir()))

    def test_state_from_the_watermark_build_still_skips_what_it_folded(self):
        now = time.time_ns()
        path = self.publish("400.0000000001.json", self.event("turn_start", "t3", 2), now)
        self.sb.state.mkdir(parents=True, exist_ok=True)
        (self.sb.state / "fold.json").write_text(json.dumps({"inbox": {"last": [now, path.name]}}))
        self.flush.ingest(self.sb.state, self.env)
        self.assertEqual(self.stored("t3"), [])
        self.publish("401.0000000001.json", self.event("turn_end", "t3", 3), now - 1)
        self.flush.ingest(self.sb.state, self.env)
        self.assertEqual(self.stored("t3"), ["turn_end"])
        self.assertIn("done", json.loads((self.sb.state / "fold.json").read_text())["inbox"])


class BoundedState(unittest.TestCase):
    def sessions(self, tracker, count: int, *, finalize_each: bool = True, delegate: bool = True, start: int = 0) -> None:
        for n in range(start, start + count):
            sid = f"s{n}"
            _parent, child, events = subagent_turn(sid)
            if delegate:
                grand = f"g{n}"
                events = events[:-1] + [("subagent_start", {"parent_session_id": child, "parent_turn_id": "", "child_session_id": grand,
                                                            "child_goal": "x"}, events[-1][2]),
                                        ("subagent_stop", {"child_session_id": grand, "child_status": "completed"}, events[-1][2])] + events[-1:]
            if finalize_each:
                hook = "on_session_finalize" if n % 2 else "on_session_reset"
                events = events + [(hook, {"session_id": sid}, events[-1][2] + 1)]
            for hook, kwargs, at in events:
                if hook == "pre_llm_call":
                    kwargs = {**kwargs, "_workspace": "/w"}
                tracker.step(hook, kwargs, int(at * 1000))

    def tables(self, tracker) -> tuple:
        return (len(tracker.ids), len(tracker.convs), len(tracker.turns), len(tracker.owner))

    def test_thousand_finalized_sessions_leave_nothing(self):
        for gate in ("gate", None):  # traced, and opted out
            with self.subTest(gate=gate):
                tracker = load_plugin().Tracker(lambda key, folder, gate=gate: gate)
                self.sessions(tracker, 50)
                tracemalloc.start()
                self.sessions(tracker, 100, start=50)
                after_100, _ = tracemalloc.get_traced_memory()
                self.sessions(tracker, 900, start=150)
                after_1000, _ = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                self.assertEqual(self.tables(tracker), (0, 0, 0, 0))
                self.assertLess(after_1000 - after_100, 64 * 1024, "memory grew with sessions")

    def test_sessions_never_finalized_stay_bounded(self):
        plugin = load_plugin()
        tracker = plugin.Tracker(lambda key, folder: "gate")
        cap = sys.modules[plugin.__name__ + ".tracker"].MAX_CONVS
        self.sessions(tracker, cap + 300, finalize_each=False, delegate=False)
        self.assertLessEqual(len(tracker.convs), cap)
        self.assertLessEqual(len(tracker.ids), cap)


if __name__ == "__main__":
    unittest.main()
