"""Hermes plugin: regression tests for review round 1 (one class per finding), shaped like Hermes 0.19.0.

Every fake secret is built at runtime (tests/support.py). The plugin runs in-process with only the
sandbox env; the helper process it starts gets a minimal env and secret values only through frames.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import threading
import time
import tracemalloc
import types
import unittest
from pathlib import Path
from unittest import mock

from fake_sink import Sink, attrs, roots
from hermes_support import VENDOR_CORE, Turn, finalize, fire_all, flush_vendor, later, load_plugin, new_session_id, plugin_env, subagent_turn
from hermes_support import LATENCY_GATE_P95_MS, LATENCY_TARGET_P95_MS, Sandbox, latency_trials, over
from support import ENV_SECRET_NAME, SECRETS, leaks

GH = SECRETS["github_token"]["core"]
OPENAI = SECRETS["openai_key"]["core"]
ENV_VALUE = SECRETS["env_value"]["core"]  # the value of ENV_SECRET_NAME in the sandbox env


def kinds(spans: list[dict], kind: str) -> list[dict]:
    return [span for span in spans if attrs(span).get("judgment.span_kind") == kind]


def latest(spans: list[dict]) -> list[dict]:
    found: dict[str, dict] = {}
    for span in spans:
        found[span["spanId"]] = span
    return list(found.values())


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def renamed(events: list, old: str, new: str, start: int, keys: tuple = ("session_id",)) -> list:
    """Hermes's legacy compression: from event ``start`` on, ``old`` is called ``new`` (conversation_compression.py:1247)."""
    out = []
    for index, (hook, kwargs, *rest) in enumerate(events):
        if index >= start:
            kwargs = {key: (new if key in keys and value == old else value) for key, value in kwargs.items()}
        out.append((hook, kwargs, *rest))
    return out


class Case(unittest.TestCase):
    def setUp(self):
        self.sink = Sink().__enter__()
        self.sb = Sandbox(sink_url=self.sink.url)
        self.addCleanup(self.sb.cleanup)
        self.addCleanup(self.sink.__exit__)

    def fire(self, events: list, *, cli: str | None = None, **env):
        """The events through the plugin; the helper folds them all before this returns. Returns the plugin."""
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


# ---- 1. metadata ---------------------------------------------------------------------------------


class MetadataSecrets(Case):
    def events(self, sid: str) -> list:
        """A model name, provider, tool name, tool call id, and child role that hold secrets."""
        turn = Turn(sid, "hello", model="model-" + GH, provider="prov-" + ENV_VALUE)
        call = turn.api(tool=("tool." + OPENAI, {"path": "a"}))
        turn.events[-1][1]["response_model"] = "resp-" + ENV_VALUE
        turn.events[-1][1]["assistant_message"].tool_calls[0].id = "call-" + GH
        turn.tool("call-" + GH, "tool." + OPENAI, {"path": "a"}, "ok")
        turn._add("subagent_start", {"parent_session_id": sid, "parent_turn_id": turn.tid, "child_session_id": sid + "c",
                                     "child_role": "role " + GH, "child_goal": "g"})
        turn._add("subagent_stop", {"parent_session_id": sid, "child_session_id": sid + "c", "child_role": "role " + GH,
                                    "child_summary": "s", "child_status": "completed", "duration_ms": 3})
        turn.api(text="done")
        turn.end("done")
        return turn.events + [finalize(sid)]

    def test_metadata_secrets_never_reach_the_inbox_or_the_sink(self):
        for tracing in ({"enabled": True}, {"enabled": True, "capture": "metadata"}):
            with self.subTest(tracing=tracing):
                self.sb.set_tracing(tracing)
                self.sink.requests.clear()
                sid = new_session_id()
                self.fire(self.events(sid))
                self.assertEqual(leaks(self.sb.state_text(), hashes=False), [], "inbox")
                self.assertIn("[REDACTED:", self.sb.state_text())
                spans = self.flushed()
                self.assertTrue(spans)
                self.assertEqual(leaks(self.sink.text(), hashes=False), [], "sink")


# ---- 2. dict keys --------------------------------------------------------------------------------


class KeySecrets(Case):
    def test_secret_keys_are_redacted(self):
        """Key-aware redaction is the core's (redact.scrub_value): every key, at any depth, as text."""
        sid = new_session_id()
        args = {GH: "first", "nested": {ENV_VALUE: {OPENAI: "deep"}}, "plain": "x"}
        turn = Turn(sid, "hello")
        call = turn.api(tool=("write_file", args))
        turn.tool(call, "write_file", args, "ok")
        turn.api(text="done")
        turn.end("done")
        self.fire(turn.events)
        self.assertEqual(leaks(self.sb.state_text(), hashes=False), [], "inbox")
        tool = next(event for event in self.sb.inbox() if event["event"] == "tool")
        self.assertEqual(tool["tool_input"]["[REDACTED:github_token]"], "first")
        self.assertEqual(tool["tool_input"]["nested"], {f"[REDACTED:env:{ENV_SECRET_NAME}]": {"[REDACTED:openai_key]": "deep"}})
        self.flushed()
        self.assertEqual(leaks(self.sink.text(), hashes=False), [], "sink")


# ---- 3. truncation -------------------------------------------------------------------------------


class Truncation(unittest.TestCase):
    """The cut-point cases are in test_hermes_round2.Truncation; this one runs the whole path."""

    def setUp(self):
        self.core = load_plugin().core

    def test_a_long_tool_result_spooled_without_any_piece(self):
        sb = Sandbox()
        self.addCleanup(sb.cleanup)
        sid = new_session_id()
        blob = ("z" * 70000 + GH + "q" * 100 + OPENAI + "w" * 60000)
        turn = Turn(sid, "read it")
        call = turn.api(tool=("read_file", {"path": "big"}))
        turn.tool(call, "read_file", {"path": "big"}, blob)
        turn.api(text="ok")
        turn.end("ok")
        with plugin_env(sb) as (plugin, ctx):
            fire_all(ctx, turn.events, plugin)
        self.assertEqual(leaks(sb.state_text(), hashes=False), [])
        tool = next(event for event in sb.inbox() if event["event"] == "tool")
        self.assertTrue(tool["tool_output"].endswith(self.core.CUT_MARK))


# ---- 4. workspace --------------------------------------------------------------------------------


class Workspace(Case):
    def setUp(self):
        super().setUp()
        self.elsewhere = self.sb.home / "elsewhere"
        self.elsewhere.mkdir()

    def hermes_modules(self, session_cwd: Path | None, task_cwd: dict | None = None) -> dict:
        """agent.runtime_cwd and tools.terminal_tool as Hermes 0.19.0 has them, already imported."""
        runtime = types.SimpleNamespace(resolve_agent_cwd=lambda: session_cwd if session_cwd else Path(os.environ.get("TERMINAL_CWD") or os.getcwd()))
        terminal = types.SimpleNamespace(resolve_task_overrides=lambda task: (task_cwd or {}).get(task, {}))
        return {"agent.runtime_cwd": runtime, "tools.terminal_tool": terminal}

    def turn(self) -> Turn:
        turn = Turn(new_session_id(), "hello")
        turn.api(text="hi")
        turn.end("hi")
        return turn

    def test_session_cwd_opted_in_while_process_cwd_is_not(self):
        turn = self.turn()
        with mock.patch.dict(sys.modules, self.hermes_modules(self.sb.project)):
            self.fire(turn.events, TERMINAL_CWD=str(self.elsewhere))
        starts = [event for event in self.sb.inbox() if event["event"] == "turn_start"]
        self.assertEqual([event["cwd"] for event in starts], [str(self.sb.project)])

    def test_process_cwd_opted_in_while_session_cwd_is_not(self):
        turn = self.turn()
        with mock.patch.dict(sys.modules, self.hermes_modules(self.elsewhere)):
            self.fire(turn.events, TERMINAL_CWD=str(self.sb.project))
        self.assertFalse((self.sb.state / "inbox").exists() and any((self.sb.state / "inbox").iterdir()))

    def test_task_cwd_wins_and_only_for_its_task(self):
        one, two = self.turn(), self.turn()
        modules = self.hermes_modules(self.elsewhere, {one.task: {"cwd": str(self.sb.project)}})
        with mock.patch.dict(sys.modules, modules):
            self.fire(one.events + two.events, TERMINAL_CWD=str(self.elsewhere))
        self.assertEqual({event["session_id"] for event in self.sb.inbox()}, {one.sid})

    def test_judgment_command_uses_the_session_workspace(self):
        sid = new_session_id()
        with mock.patch.dict(sys.modules, self.hermes_modules(self.sb.project)):
            with plugin_env(self.sb, TERMINAL_CWD=str(self.elsewhere)) as (plugin, ctx):
                ctx.cli(sid)
                self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("off"))
        self.assertEqual((self.sb.state / "sessions" / sid).read_text().strip(), "off")


# ---- 5. /judgment identity -----------------------------------------------------------------------


class CommandIdentity(Case):
    def test_gateway_command_changes_no_conversation(self):
        """Gateway: the command is dispatched before the message's session is bound (gateway/run.py:11194 vs 12128)."""
        chat_a = new_session_id()
        turn = Turn(chat_a, "hi from chat A", platform="telegram")
        turn.api(text="hi")
        turn.end("hi")
        with plugin_env(self.sb) as (plugin, ctx):
            ctx.cli(None)
            fire_all(ctx, turn.events, plugin)
            os.environ["HERMES_SESSION_ID"] = chat_a  # process-wide leftovers name chat A
            handler = ctx.commands["judgment"]["handler"]
            for action in ("off", "on"):
                self.assertIn("changed nothing", handler(action))
            status = handler("status")
            self.assertIn("project setting", status)
        self.assertFalse((self.sb.state / "sessions").exists())

    def test_cli_command_targets_only_its_conversation(self):
        mine, other = new_session_id(), new_session_id()
        events = []
        for sid in (other, mine):
            turn = Turn(sid, "hi")
            turn.api(text="hi")
            turn.end("hi")
            events += turn.events
        with plugin_env(self.sb) as (plugin, ctx):
            fire_all(ctx, events, plugin)
            ctx.cli(mine)
            self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("off"))
        self.assertEqual(sorted(path.name for path in (self.sb.state / "sessions").iterdir()), [mine])


# ---- 6. compression rotation ---------------------------------------------------------------------


class Rotation(Case):
    def test_main_session_rotates_mid_turn(self):
        sid, rotated = new_session_id(), new_session_id()
        one = Turn(sid, "first")
        call = one.api(tool=("read_file", {"path": "a"}))
        rotate_at = len(one.events)  # Hermes compresses before the next request
        one.tool(call, "read_file", {"path": "a"}, "contents")
        one.api(text="done one")
        one.end("done one")
        two = Turn(rotated, "second", first=False, start=later(one.events, 1))
        two.api(text="done two")
        two.end("done two")
        events = renamed(one.events, sid, rotated, rotate_at) + two.events + [finalize(rotated) + (later(two.events, 1),)]
        plugin = self.fire(events)
        self.assertEqual((plugin.tracker.ids, dict(plugin.tracker.convs), plugin.tracker.turns, plugin.tracker.owner), ({}, {}, {}, {}))
        spans = self.flushed()
        self.assertEqual({attrs(span)["judgment.session_id"] for span in spans}, {sid})
        self.assertEqual(sorted(attrs(root)["judgment.input"] for root in roots(spans)), ["first", "second"])
        first = next(root for root in roots(spans) if attrs(root)["judgment.input"] == "first")
        under = [span for span in spans if span.get("parentSpanId") == first["spanId"]]
        self.assertEqual(sorted(attrs(span)["judgment.span_kind"] for span in under), ["llm", "llm", "tool"])

    def test_override_survives_rotation(self):
        """/judgment off, then a turn in which Hermes rotates the id: the next turn, under the new id, stays off."""
        sid, rotated = new_session_id(), new_session_id()
        one = Turn(sid, "traced")
        one.api(text="a")
        one.end("a")
        two = Turn(sid, "off, rotates", first=False, start=later(one.events, 1))
        two.api(text="b")
        rotate_at = len(two.events)
        two.api(text="b2")
        two.end("b2")
        three = Turn(rotated, "still off", first=False, start=later(two.events, 1))
        three.api(text="c")
        three.end("c")
        with plugin_env(self.sb) as (plugin, ctx):
            ctx.cli(sid)
            fire_all(ctx, one.events, plugin)
            self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("off"))
            fire_all(ctx, renamed(two.events, sid, rotated, rotate_at) + three.events, plugin)
            ctx.cli(rotated)  # Hermes's CLI follows the new id (cli.py:12285)
            self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("status"))
        self.assertEqual([attrs(root)["judgment.input"] for root in roots(self.flushed())], ["traced"])
        # A resume under the new id in another process keeps it too.
        self.assertEqual((self.sb.state / "sessions" / rotated).read_text().strip(), "off")

    def test_child_rotates_during_a_delegated_turn(self):
        sid = new_session_id()
        _parent, child_sid, events = subagent_turn(sid)
        rotated = new_session_id() + "r"
        # The child compresses after its first request: its later calls and its stop carry the new id.
        first_child_api = next(i for i, event in enumerate(events) if event[0] == "post_api_request" and event[1]["session_id"] == child_sid)
        events = renamed(events, child_sid, rotated, first_child_api + 1, keys=("session_id", "child_session_id"))
        plugin = self.fire(events + [finalize(sid) + (later(events, 0.5),)])
        self.assertEqual((plugin.tracker.ids, dict(plugin.tracker.convs), plugin.tracker.owner), ({}, {}, {}))
        spans = self.flushed()
        agent = [span for span in kinds(spans, "agent") if span.get("parentSpanId")]
        self.assertEqual(len(agent), 1)
        self.assertEqual(attrs(agent[0])["judgment.output"], "one two three")
        self.assertNotIn("judgment.interrupted", attrs(agent[0]))
        under = [span for span in spans if span.get("parentSpanId") == agent[0]["spanId"]]
        self.assertEqual(sorted(attrs(span)["judgment.span_kind"] for span in under), ["llm", "llm", "tool"])


# ---- 7. retries ----------------------------------------------------------------------------------


class Retries(Case):
    def test_attempts_of_one_request_are_separate_spans_with_their_own_start(self):
        """Hermes reuses api_request_id and started_at for every attempt, and a credential-pool retry keeps retry_count."""
        sid = new_session_id()
        turn = Turn(sid, "hello")
        turn.api(error={"status_code": 429, "retry_count": 0, "message": "rate limited"})
        turn.api(error={"status_code": 429, "retry_count": 0, "message": "rate limited"}, retry=True)
        turn.api(text="finally", retry=True)
        turn.end("finally")
        pres = [kwargs for hook, kwargs, _at in turn.events if hook == "pre_api_request"]
        self.assertEqual(len({kwargs["api_request_id"] for kwargs in pres}), 1)
        self.assertEqual(len({kwargs["started_at"] for kwargs in pres}), 1)
        self.fire(turn.events)
        self.flushed()
        llms = kinds(self.sink.spans(), "llm")
        self.assertEqual(len({span["spanId"] for span in llms}), 3, "one span per attempt")
        ordered = sorted(latest(llms), key=lambda span: int(span["startTimeUnixNano"]))
        self.assertEqual([span.get("status", {}).get("code") for span in ordered], [2, 2, None])
        for before, after in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(int(after["startTimeUnixNano"]), int(before["endTimeUnixNano"]))

    def test_attempt_ids_from_the_tracker(self):
        plugin = load_plugin()
        tracker = plugin.Tracker(lambda key, folder: "gate", version="t")
        sid = new_session_id()
        turn = Turn(sid, "hello")
        turn.api(error={"retry_count": 0})
        turn.api(error={"retry_count": 0}, retry=True)
        turn.api(text="ok", retry=True)
        turn.end("ok")
        records = []
        for hook, kwargs, at in turn.events:
            if hook == "pre_llm_call":
                kwargs = {**kwargs, "_workspace": "/w"}
            records += [record for _gate, record in tracker.step(hook, kwargs, int(at * 1000)).records]
        models = [record for record in records if record["event"] == "model"]
        rid = next(kwargs["api_request_id"] for hook, kwargs, _at in turn.events if hook == "pre_api_request")
        self.assertEqual([record["message_id"] for record in models], [rid, rid + "#2", rid + "#3"])
        pre_times = [int(at * 1000) for hook, _kwargs, at in turn.events if hook == "pre_api_request"]
        self.assertEqual([record["start_ts"] for record in models], pre_times)


# ---- 8. spool publication (core flusher) ---------------------------------------------------------


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


# ---- 9. background grandchildren -----------------------------------------------------------------


class Grandchild(Case):
    def test_grandchild_started_after_the_next_turn_opened_stays_on_its_turn(self):
        sid = new_session_id()
        one = Turn(sid, "turn one delegates")
        call = one.api(tool=("delegate_task", {"goal": "g"}))
        child = sid + "c"
        child_tid = f"{child}:task:1"
        one._add("subagent_start", {"parent_session_id": sid, "parent_turn_id": one.tid, "child_session_id": child, "child_role": "orchestrator",
                                    "child_goal": "plan"})
        one._add("pre_llm_call", {"session_id": child, "turn_id": child_tid, "user_message": "plan", "platform": "subagent", "model": "m"})
        one.tool(call, "delegate_task", {"goal": "g"}, '{"status": "started"}')
        one.api(text="started in background")
        one.end("started in background")
        two = Turn(sid, "turn two", first=False, start=later(one.events, 1))
        two.api(text="two")
        grand = sid + "g"
        # The child, still running from turn one, delegates after turn two opened.
        two._add("subagent_start", {"parent_session_id": child, "parent_turn_id": child_tid, "child_session_id": grand, "child_role": "leaf",
                                    "child_goal": "leaf work"})
        two._add("post_tool_call", {"session_id": grand, "turn_id": f"{grand}:task:1", "tool_name": "terminal", "args": {"command": "true"},
                                    "tool_call_id": "gc1", "result": "{}", "duration_ms": 5, "status": "ok"})
        two._add("subagent_stop", {"parent_session_id": child, "child_session_id": grand, "child_summary": "leaf done", "child_status": "completed"})
        two._add("subagent_stop", {"parent_session_id": sid, "child_session_id": child, "child_summary": "plan done", "child_status": "completed"})
        two.end("two")
        self.fire(one.events + two.events + [finalize(sid) + (later(two.events, 1),)])
        spans = self.flushed()
        trace_of = {attrs(root)["judgment.input"]: root["traceId"] for root in roots(spans)}
        agents = {attrs(span)["judgment.agent_id"]: span for span in kinds(spans, "agent") if span.get("parentSpanId")}
        self.assertEqual(agents[grand]["traceId"], trace_of["turn one delegates"])
        self.assertEqual(agents[child]["traceId"], trace_of["turn one delegates"])
        tool = next(span for span in kinds(spans, "tool") if attrs(span).get("gen_ai.tool.name") == "terminal")
        self.assertEqual(tool["traceId"], trace_of["turn one delegates"])


# ---- 10. latency ---------------------------------------------------------------------------------


def dense_payload(count: int = 1000) -> str:
    """56 KB holding ``count`` fake credentials, built at runtime."""
    import random
    import string

    rng = random.Random(5)
    token = lambda n: "".join(rng.choice(string.ascii_letters + string.digits) for _ in range(n - 1)) + "7"  # noqa: E731
    makers = (lambda: "gh" + "p_" + token(36), lambda: "sk-" + "proj-" + token(40), lambda: "AK" + "IA" + token(16).upper(),
              lambda: "xo" + "xb-" + "1234567890-" + token(20), lambda: "DB_PASS" + "WORD=" + token(16), lambda: "Authorization: Bearer " + token(24))
    text = "\n".join(f"k{i} {makers[i % len(makers)]()}" for i in range(count))
    return text[: 56 * 1024]


class Latency(Case):
    """Callback p95 per hook: reported against LATENCY_TARGET_P95_MS, failed at LATENCY_GATE_P95_MS (hermes_support)."""

    REPORT: dict = {}

    @classmethod
    def tearDownClass(cls):
        for case, (p50, p95, mx, count, tried, slow) in sorted(cls.REPORT.items()):
            flag = f"  over {LATENCY_TARGET_P95_MS} ms target: {slow}" if slow else ""
            print(f"  {case:34s} n={count:5d} p50={p50:.3f} p95={p95:.3f} max={mx:.3f} ms (trials {tried}){flag}", file=sys.stderr)

    def timed(self, ctx, plugin, events, times: dict, only: set | None = None, lock: threading.Lock | None = None) -> None:
        for hook, kwargs, at in events:
            kwargs.setdefault("telemetry_schema_version", "hermes.observer.v1")
            (callback,) = ctx.hooks[hook]
            start = time.perf_counter()
            callback(**kwargs)
            took = (time.perf_counter() - start) * 1000
            if only is None or hook in only:
                with lock or threading.Lock():
                    times.setdefault(hook, []).append(took)

    def check(self, case: str, times: dict, tried: int) -> None:
        values = [value for series in times.values() for value in series]
        self.REPORT[case] = (statistics.median(values), pct(values, 0.95), max(values), len(values), tried, over(times, LATENCY_TARGET_P95_MS))
        self.assertEqual(over(times, LATENCY_GATE_P95_MS), {}, f"{case}: p95 over {LATENCY_GATE_P95_MS} ms in each of {tried} trials")

    def tool_turn(self, sid: str, result: str, answer: str) -> Turn:
        turn = Turn(sid, "read it")
        call = turn.api(tool=("read_file", {"path": "notes.txt"}))
        turn.tool(call, "read_file", {"path": "notes.txt"}, result)
        turn.api(text=answer)
        turn.end(answer)
        return turn

    def test_secret_dense_payloads(self):
        dense = dense_payload()
        with plugin_env(self.sb, JUDGMENT_FLUSH="0") as (plugin, ctx):
            self.timed(ctx, plugin, self.tool_turn(new_session_id(), "warm", "warm").events, {})

            def trial() -> dict:
                times: dict = {}
                for _ in range(60):
                    self.timed(ctx, plugin, self.tool_turn(new_session_id(), dense, dense).events, times,
                               only={"post_tool_call", "post_api_request", "on_session_end"})
                return times

            times, tried = latency_trials(trial)
        self.check("dense 56 KB, 1000 credentials", times, tried)
        self.assertEqual(leaks(self.sb.state_text(), hashes=False), [])

    def test_eight_threads_at_once(self):
        plain = ("line of ordinary tool output text\n" * 128)[:4096]
        lock = threading.Lock()
        with plugin_env(self.sb, JUDGMENT_FLUSH="0") as (plugin, ctx):
            self.timed(ctx, plugin, self.tool_turn(new_session_id(), "warm", "warm").events, {})

            def trial() -> dict:
                times: dict = {}

                def worker(n: int) -> None:
                    for i in range(40):
                        self.timed(ctx, plugin, self.tool_turn(new_session_id() + f"t{n}", plain, f"a{i}").events, times, lock=lock)

                threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                self.assertEqual(plugin.errors["count"], 0)
                return times

            times, tried = latency_trials(trial)
        self.check("8 threads, 4 KB tool output", times, tried)
        self.assertEqual(len(list((self.sb.state / "inbox").iterdir())), tried * 8 * 40 * 5 + 5)  # every trial's turns, and the warm-up turn

    def test_turn_end_with_the_real_sender_launch(self):
        with plugin_env(self.sb, JUDGMENT_FLUSH=None, JUDGMENT_API_URL="http://127.0.0.1:9") as (plugin, ctx):
            self.timed(ctx, plugin, self.tool_turn(new_session_id(), "warm", "warm").events, {})

            def trial() -> dict:
                times: dict = {}
                for i in range(40):
                    self.timed(ctx, plugin, self.tool_turn(new_session_id(), "ok", f"a{i}").events, times, only={"on_session_end"})
                    time.sleep(0.02)
                return times

            times, tried = latency_trials(trial)
            helper = plugin.spool.proc
        self.check("turn end, sender launched", times, tried)
        self.assertIsNotNone(helper)
        log = self.sb.state / "flush.log"
        self.assertTrue(log.exists() and "post attempt" in log.read_text() or wait_log(log))


def wait_until(predicate, timeout: float = 10.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return False


def wait_log(log: Path) -> bool:
    deadline = time.time() + 20
    while time.time() < deadline:
        if log.exists() and "post attempt" in log.read_text():
            return True
        time.sleep(0.1)
    return False


# ---- 11. bounded state ---------------------------------------------------------------------------


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


# ---- the helper process ---------------------------------------------------------------------------


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
        self.assertEqual(set(env) - {"PATH", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE"}, set())


if __name__ == "__main__":
    unittest.main()
