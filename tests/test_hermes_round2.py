"""Hermes plugin: regression tests for review round 2, one class per finding, shaped like Hermes 0.19.0.

Every fake secret is built at runtime (tests/support.py). The sandbox is hermetic (hermes_support.Sandbox).
"""

from __future__ import annotations

import json
import os
import signal
import sys
import threading
import time
import unittest
from unittest import mock

from fake_sink import Sink, attrs, roots
from hermes_support import Sandbox, Turn, finalize, fire_all, flush_vendor, later, load_plugin, new_session_id, plugin_env, subagent_turn
from support import ENV_SECRET_NAME, SECRETS

GH = SECRETS["github_token"]["core"]
OPENAI = SECRETS["openai_key"]["core"]
ENV_VALUE = SECRETS["env_value"]["core"]  # the value of ENV_SECRET_NAME in the sandbox env


def module(name: str):
    return sys.modules[load_plugin().__name__ + "." + name]


def kinds(spans: list[dict], kind: str) -> list[dict]:
    return [span for span in spans if attrs(span).get("judgment.span_kind") == kind]


def latest(spans: list[dict]) -> list[dict]:
    found: dict[str, dict] = {}
    for span in spans:
        found[span["spanId"]] = span
    return list(found.values())


def wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def no_piece(text: str, secret: str, window: int = 6) -> list[str]:
    return [secret[i : i + window] for i in range(len(secret) - window + 1) if secret[i : i + window] in text]


def preflight(turn: Turn, rotated: str) -> list:
    """Hermes's preflight compression: turn_id was built from the old id, then session_id rotated before
    pre_llm_call (agent/turn_context.py:369, :652, :697). Every hook of the turn carries the new id."""
    out = []
    for hook, kwargs, *rest in turn.events:
        kwargs = {key: (rotated if key == "session_id" and value == turn.sid else value) for key, value in kwargs.items()}
        out.append((hook, kwargs, *rest))
    return out


class Case(unittest.TestCase):
    def setUp(self):
        self.sink = Sink().__enter__()
        self.sb = Sandbox(sink_url=self.sink.url)
        self.addCleanup(self.sb.cleanup)
        self.addCleanup(self.sink.__exit__)

    def fire(self, events: list, *, cli: str | None = None, **env):
        with plugin_env(self.sb, **env) as (plugin, ctx):
            if cli is not None:
                ctx.cli(cli)
            self.assertEqual(fire_all(ctx, events, plugin), [])
            self.assertEqual(plugin.errors["count"], 0)
        return plugin

    def flushed(self) -> list[dict]:
        result = flush_vendor(self.sb)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return latest(self.sink.spans())

    def simple(self, sid: str, prompt: str = "hello", answer: str = "done", **kw) -> Turn:
        turn = Turn(sid, prompt, **kw)
        turn.api(text=answer)
        turn.end(answer)
        return turn


# ---- 1. stamped metadata -------------------------------------------------------------------------


class Stamps(Case):
    def test_secret_in_project_path_and_environment_never_stored_or_sent(self):
        """A runtime secret (an env value the gate registers) in the project's folder name and its environment label."""
        project = self.sb.home / ("proj-" + ENV_VALUE)
        (project / ".judgment").mkdir(parents=True)
        (project / ".judgment" / "tracing.json").write_text(json.dumps({"enabled": True, "environment": "env-" + ENV_VALUE}))
        self.fire(self.simple(new_session_id()).events, TERMINAL_CWD=str(project))
        inbox = self.sb.inbox()
        self.assertTrue(inbox)
        for event in inbox:
            self.assertIn("[REDACTED:env:", event["project_root"])
            self.assertIn("[REDACTED:env:", event["environment"])
            if "cwd" in event:
                self.assertIn("[REDACTED:env:", event["cwd"])
        self.assertEqual(no_piece(self.sb.state_text(), ENV_VALUE), [], "inbox")
        spans = self.flushed()
        self.assertTrue(spans, "the export stamp is untouched, so the turn is sent")
        self.assertEqual(no_piece(self.sb.state_text() + self.sink.text(), ENV_VALUE), [], "turn files, flush.log, HTTP body")

    def test_secret_in_export_stamp_withholds_the_stamp_whole(self):
        """The export stamp routes the turn, so it is never redacted: one that holds a secret is dropped, and the turn isn't sent."""
        self.fire(self.simple(new_session_id()).events, JUDGMENT_ORG_ID="org-" + ENV_VALUE)
        self.assertTrue(self.sb.inbox())
        for event in self.sb.inbox():
            self.assertFalse(any(name.startswith("export_") for name in event))
        self.flushed()
        self.assertEqual(self.sink.spans(), [])
        self.assertEqual(no_piece(self.sb.state_text() + self.sink.text(), ENV_VALUE), [])
        self.assertIn("redaction changed these stamps: export", (self.sb.state / "flush.log").read_text())

    def test_a_failing_stage_keeps_no_label_or_metadata_unredacted(self):
        core = load_plugin().core
        gate = core.Gate("sanitized", (), (), (), "/s", ())
        record = {"event": "tool", "tool_name": "t-" + GH, "agent_type": "r-" + GH, "model": "m-" + GH, "tool_output": GH, "session_id": "s"}
        with mock.patch.object(core.redact, "sanitize_event", side_effect=TypeError("boom")):
            out = core.prepare(record, gate)
        self.assertTrue(out["redaction_error"])
        self.assertEqual(no_piece(json.dumps(out), GH[4:]), [])
        self.assertEqual(out["session_id"], "s")

    def test_stamps_are_checked_through_prepare(self):
        core = load_plugin().core
        kept, withheld = core.stamps("prod-" + GH, "/w/" + GH, [("export_fp", "f" * 32), ("export_project", "p1"), ("export_url", "http://x")],
                                     core._rules((), ()), [])
        gate = core.Gate("sanitized", (), (), (), "/s", kept, withheld)
        out = core.prepare({"event": "turn_start", "prompt": "hi", "cwd": "/w/" + GH}, gate)
        self.assertEqual(no_piece(json.dumps(out), GH[4:]), [])
        self.assertEqual(out["export_project"], "p1")
        self.assertEqual(set(withheld), {"environment", "project_root"})


# ---- 2 + 3. truncation and sensitivity on complete values -----------------------------------------


class Truncation(Case):
    def setUp(self):
        super().setUp()
        self.core = load_plugin().core
        self.gate = self.core.Gate("sanitized", (), (), ((ENV_SECRET_NAME, ENV_VALUE),), str(self.sb.state), ())

    def tool_turn(self, args: dict, result: str) -> Turn:
        turn = Turn(new_session_id(), "go")
        call = turn.api(tool=("terminal", args))
        turn.tool(call, "terminal", args, result)
        turn.api(text="ok")
        turn.end("ok")
        return turn

    def test_aggregate_budget_never_cuts_a_short_secret(self):
        """Round 2 probe: the tracker's 256 KB value budget ran out 15 characters into a token."""
        for pad in (256 * 1024 - 15, 256 * 1024 - 4, 300 * 1024):
            with self.subTest(pad=pad):
                args = {"a": "w" * pad, "token": GH}
                self.fire(self.tool_turn(args, "ok").events)
                self.assertEqual(no_piece(self.sb.state_text(), GH[4:]), [])

    def test_secret_at_the_old_raw_cap_after_redaction_shrank_the_text(self):
        """Round 2 probe: a token cut at RAW_CAP survived when earlier redactions shrank the text below TEXT_CAP."""
        dense = (OPENAI + " ") * 1500  # redaction shrinks this by ~60 KB
        for overhang in (5, 15, 30):
            with self.subTest(overhang=overhang):
                text = dense + "y" * (128 * 1024 - len(dense) - overhang) + GH + " tail"
                out = self.core.prepare({"event": "tool", "tool_output": text}, self.gate)["tool_output"]
                self.assertEqual(no_piece(out, GH[4:]), [])
                self.assertEqual(no_piece(out, OPENAI[8:]), [])

    def test_the_cut_after_redaction_drops_the_partial_word_whole(self):
        cap = self.core.TEXT_CAP
        rs_gate = self.core.Gate("sanitized", ("passphrase is [a-z]+ [a-z]+ [a-z]+",), (), (("X_TOKEN", "correct horse " + ENV_VALUE),),
                                 str(self.sb.state), ())
        for at in range(cap - 40, cap + 10, 7):
            with self.subTest(at=at):
                for secret in (GH, "correct horse " + ENV_VALUE, "passphrase is alpha bravo charlie"):
                    text = ("word " * (at // 5 + 1))[:at] + secret + " more" * 3000
                    out = self.core.prepare({"event": "tool", "tool_output": text}, rs_gate)["tool_output"]
                    self.assertTrue(out.endswith(self.core.CUT_MARK))
                    self.assertLessEqual(len(out), cap + len(self.core.CUT_MARK))
                    self.assertEqual(no_piece(out, GH[4:]) + no_piece(out, ENV_VALUE), [])
                    self.assertNotIn("bravo", out)

    def test_a_value_too_large_to_redact_is_dropped_whole(self):
        spooler = module("spooler")
        big = ("q" * 100 + " ") * (spooler.FIELD_CAP // 100) + GH
        record, size = spooler.bound({"event": "tool", "tool_output": big, "tool_input": {"command": "ls"}})
        self.assertEqual(record["tool_output"], spooler.OMITTED)
        self.assertEqual(record["omitted"], ["tool_output"])
        self.assertLessEqual(size, spooler.RECORD_CAP)
        out = self.core.prepare(record, self.gate)
        self.assertEqual(out["tool_output"], spooler.OMITTED)
        self.assertNotIn("sensitive", out, "an output too large to keep is no reason to blank the turn")

    def test_an_input_too_large_to_check_blanks_its_turn(self):
        spooler = module("spooler")
        command = "echo " + "a" * spooler.FIELD_CAP + "; cat .env"
        self.fire(self.tool_turn({"command": command}, "PRIVATE-NOTE-" + "x" * 40).events)
        tool = next(event for event in self.sb.inbox() if event["event"] == "tool")
        self.assertTrue(tool["sensitive"])
        self.assertEqual(tool["sensitive_rule"], self.core.OVERSIZE_RULE)
        spans = self.flushed()
        self.assertEqual(attrs(roots(spans)[0])["judgment.redacted_rule"], self.core.OVERSIZE_RULE)
        self.assertNotIn("PRIVATE-NOTE", self.sb.state_text() + self.sink.text())

    def test_sensitive_file_at_the_end_of_a_long_command_blanks_the_turn(self):
        """Round 2 probe: a ~72 KB command ending in `; cat .env` lost its suffix to fitting before classification."""
        command = "echo " + "word " * (72 * 1024 // 5) + "; cat .env"
        self.fire(self.tool_turn({"command": command}, "PRIVATE-NOTE-" + "x" * 40).events)
        tool = next(event for event in self.sb.inbox() if event["event"] == "tool")
        self.assertTrue(tool["sensitive"])
        self.assertEqual(tool["sensitive_rule"], "dotenv")
        self.assertNotIn("PRIVATE-NOTE", self.sb.state_text())
        root = attrs(roots(self.flushed())[0])
        self.assertEqual((root["judgment.redacted_reason"], root["judgment.redacted_rule"]), ("sensitive_path", "dotenv"))
        self.assertNotIn("PRIVATE-NOTE", self.sink.text())


# ---- 4. backpressure: callbacks never wait, memory and shutdown are bounded ------------------------


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

    def test_puts_after_exit_are_dropped_not_queued(self):
        with plugin_env(self.sb) as (plugin, ctx):
            plugin.spool.close()
            fire_all(ctx, self.simple(new_session_id()).events, plugin)
            self.assertEqual(len(plugin.spool.unsent), 0)
            self.assertGreater(plugin.spool.dropped.get("after exit", 0), 0)


# ---- 5. preflight compression ---------------------------------------------------------------------


class Preflight(Case):
    def test_lineage_reads_the_pre_rotation_id(self):
        lineage = module("tracker").lineage
        self.assertEqual(lineage({"turn_id": "20261109_120000_abcdef:task-1:0a1b2c3d", "task_id": "task-1"}), "20261109_120000_abcdef")
        self.assertEqual(lineage({"turn_id": "gw:chat:42:a:b:0a1b2c3d", "task_id": "a:b"}), "gw:chat:42")
        self.assertEqual(lineage({"turn_id": "session:t:0a1b2c3d", "task_id": "t"}), "")
        for bad in ({"turn_id": "s:t:0a1b2c3", "task_id": "t"}, {"turn_id": "s:u:0a1b2c3d", "task_id": "t"}, {"turn_id": "s:t:0a1b2c3d"}):
            self.assertEqual(lineage(bad), "")

    def test_opted_out_conversation_stays_off_through_preflight_rotation(self):
        sid, rotated = new_session_id(), new_session_id()
        one = self.simple(sid, "traced")
        two = self.simple(sid, "off, rotated before pre_llm_call", first=False, start=later(one.events, 1))
        three = self.simple(rotated, "still off", first=False, start=later(two.events, 1))
        with plugin_env(self.sb) as (plugin, ctx):
            ctx.cli(sid)
            fire_all(ctx, one.events, plugin)
            self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("off"))
            fire_all(ctx, preflight(two, rotated) + three.events, plugin)
            ctx.cli(rotated)
            self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("status"))
        self.assertEqual([attrs(root)["judgment.input"] for root in roots(self.flushed())], ["traced"])
        self.assertEqual((self.sb.state / "sessions" / rotated).read_text().strip(), "off")

    def test_resumed_off_session_rotating_in_its_first_turn_stays_off(self):
        """A new Hermes process: the conversation was turned off earlier; its first turn here rotates before pre_llm_call."""
        sid, rotated = new_session_id(), new_session_id()
        (self.sb.state / "sessions").mkdir(parents=True)
        (self.sb.state / "sessions" / sid).write_text("off\n")
        self.fire(preflight(self.simple(sid, "resumed", first=False), rotated) + [finalize(rotated)])
        self.assertFalse((self.sb.state / "inbox").exists() and any((self.sb.state / "inbox").iterdir()))
        self.assertEqual((self.sb.state / "sessions" / rotated).read_text().strip(), "off")

    def test_traced_conversation_keeps_its_session_through_preflight_rotation(self):
        sid, rotated = new_session_id(), new_session_id()
        one = self.simple(sid, "first")
        two = self.simple(sid, "second", first=False, start=later(one.events, 1))
        plugin = self.fire(one.events + preflight(two, rotated) + [finalize(rotated) + (later(two.events, 1),)])
        self.assertEqual((plugin.tracker.ids, dict(plugin.tracker.convs), plugin.tracker.turns, plugin.tracker.owner), ({}, {}, {}, {}))
        spans = self.flushed()
        self.assertEqual({attrs(span)["judgment.session_id"] for span in spans}, {sid})
        self.assertEqual(sorted(attrs(root)["judgment.input"] for root in roots(spans)), ["first", "second"])

    def test_child_rotating_before_its_first_pre_llm_call_keeps_its_routing(self):
        sid = new_session_id()
        _parent, child_sid, events = subagent_turn(sid)
        rotated = new_session_id() + "r"
        # The child's turn_id was built from child_sid; every child hook from pre_llm_call on carries the new id.
        first = next(i for i, event in enumerate(events) if event[0] == "pre_llm_call" and event[1]["session_id"] == child_sid)
        out = []
        for index, (hook, kwargs, *rest) in enumerate(events):
            if index >= first:
                kwargs = {key: (rotated if key in ("session_id", "child_session_id") and value == child_sid else value) for key, value in kwargs.items()}
            out.append((hook, kwargs, *rest))
        plugin = self.fire(out + [finalize(sid) + (later(out, 0.5),)])
        self.assertEqual((plugin.tracker.ids, dict(plugin.tracker.convs), plugin.tracker.owner), ({}, {}, {}))
        spans = self.flushed()
        agent = [span for span in kinds(spans, "agent") if span.get("parentSpanId")]
        self.assertEqual(len(agent), 1)
        under = [span for span in spans if span.get("parentSpanId") == agent[0]["spanId"]]
        self.assertEqual(sorted(attrs(span)["judgment.span_kind"] for span in under), ["llm", "llm", "tool"])
        self.assertNotIn("judgment.interrupted", attrs(agent[0]))


# ---- 6. fork ------------------------------------------------------------------------------------------


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


# ---- 7. tool call ids ---------------------------------------------------------------------------------


class CallIds(Case):
    def test_secret_bearing_call_ids_get_distinct_consistent_replacements(self):
        other = "gh" + "p_" + "".join(reversed(GH[4:]))  # a second token matching the same rule
        sid = new_session_id()
        turn = Turn(sid, "two calls")
        turn.api(tool=("read_file", {"path": "a"}))
        model = turn.events[-1][1]
        ids = ["call-" + GH, "call-" + other, "call_safe_1"]
        calls = [{"id": cid, "name": "read_file", "arguments": json.dumps({"path": f"f{n}"})} for n, cid in enumerate(ids)]
        model["assistant_message"].tool_calls = [type("C", (), {"id": c["id"], "name": c["name"], "arguments": c["arguments"]})() for c in calls]
        for n, cid in enumerate(ids):
            turn.tool(cid, "read_file", {"path": f"f{n}"}, f"content {n}")
        turn.api(text="done")
        turn.end("done")
        self.fire(turn.events)
        self.assertEqual(no_piece(self.sb.state_text(), GH[4:]) + no_piece(self.sb.state_text(), other[4:]), [])
        inbox = self.sb.inbox()
        tools = [event["tool_call_id"] for event in inbox if event["event"] == "tool"]
        models = [call["id"] for event in inbox if event["event"] == "model" for call in event.get("tool_calls") or []]
        self.assertEqual(len(set(tools)), 3)
        self.assertEqual(tools, models, "the model's calls and the tool events carry the same ids")
        self.assertIn("call_safe_1", tools, "a safe id is kept")
        spans = self.flushed()
        tool_spans = kinds(spans, "tool")
        self.assertEqual(len(tool_spans), 3)
        self.assertEqual(len({span["spanId"] for span in tool_spans}), 3)
        self.assertEqual(no_piece(self.sink.text(), GH[4:]), [])

    def test_a_raw_id_shaped_like_a_replacement_is_replaced_too(self):
        tracker_mod = module("tracker")
        tracker = tracker_mod.Tracker(lambda key, folder: "gate", needs_alias=lambda gate, value: "secret" in value)
        calls = tracker_mod.Calls()
        where = tracker_mod.Where("gate", calls, "s", "t")
        first = tracker._call_id(where, "id-secret")
        lookalike = tracker._call_id(where, first)
        self.assertNotEqual(first, lookalike)
        self.assertEqual(tracker._call_id(where, "id-secret"), first)


# ---- 8. exactly-once publication across helper deaths ------------------------------------------------


class Replay(Case):
    def turns(self, count: int) -> list:
        events = []
        for n in range(count):
            turn = Turn(new_session_id(), f"t{n}")
            call = turn.api(tool=("read_file", {"path": "a"}))
            turn.tool(call, "read_file", {"path": "a"}, "line\n" * 2000)
            turn.api(text=f"a{n}")
            turn.end(f"a{n}")
            events += turn.events
        return events

    def expected(self, events: list) -> int:
        tracker = module("tracker").Tracker(lambda key, folder: "gate")
        return sum(len(tracker.step(hook, {**kwargs, "_workspace": "/w"}, int(at * 1000)).records) for hook, kwargs, at in events)

    def published(self) -> list[str]:
        return sorted(path.name for path in (self.sb.state / "inbox").iterdir())

    def assert_once(self, count: int) -> None:
        names = self.published()
        self.assertEqual(len(names), count, "every record published")
        self.assertEqual(len(set(names)), count, "none twice")
        keys = [(e["event"], e["session_id"], e["turn_id"], e.get("message_id"), e.get("tool_call_id")) for e in self.sb.inbox()]
        self.assertEqual(len(set(keys)), count)
        self.assertEqual(list((self.sb.state / "inbox.tmp").iterdir()), [], "no staged copy left behind")

    def test_helper_killed_before_publishing(self):
        events = self.turns(3)
        with plugin_env(self.sb) as (plugin, ctx):
            fire_all(ctx, events[:2], plugin)
            self.assertTrue(wait_until(lambda: plugin.spool.proc is not None and not plugin.spool.inflight))
            first = plugin.spool.proc
            os.kill(first.pid, signal.SIGSTOP)
            fire_all(ctx, events[2:], plugin)
            self.assertTrue(wait_until(plugin.spool.handed_over))
            self.assertTrue(plugin.spool.inflight, "written to the stopped helper, not acked")
            os.kill(first.pid, signal.SIGKILL)
            self.assertTrue(wait_until(lambda: plugin.spool.spawned >= 2 and not plugin.spool.inflight and not plugin.spool.unsent))
        self.assert_once(self.expected(events))

    def test_helper_killed_midway_through_a_batch(self):
        events = self.turns(40)
        total = self.expected(events)
        with plugin_env(self.sb) as (plugin, ctx):
            fire_all(ctx, events, plugin)
            self.assertTrue(wait_until(lambda: (self.sb.state / "inbox").is_dir() and len(self.published()) >= 20, 20))
            plugin.spool.proc.kill()  # some published and acked, some published not acked, some not read yet
            self.assertTrue(wait_until(lambda: plugin.spool.spawned >= 2 and not plugin.spool.inflight and not plugin.spool.unsent, 20))
        self.assert_once(total)

    def test_every_crash_point_of_the_helper_publishes_once(self):
        """The helper's steps one at a time: prepare, stage, commit, publish, ack. A crash after any of them, then a new helper
        replaying every unacked frame, publishes each record exactly once."""
        spooler, core = module("spooler"), load_plugin().core
        gate = core.Gate("sanitized", (), (), (), str(self.sb.state), ())
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


if __name__ == "__main__":
    unittest.main()
