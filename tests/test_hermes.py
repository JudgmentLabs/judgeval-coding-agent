"""Hermes plugin: hook kwargs -> inbox -> flusher -> fake sink. Spans, redaction, opt-out, latency, the sender.

Hook kwargs come from tests/hermes_support.py, shaped like what Hermes 0.19.0 passes. The plugin runs
in-process, as in Hermes, with only the sandbox env. tests/e2e_hermes.py runs real Hermes.
"""

from __future__ import annotations

import json
import os
import signal
import statistics
import subprocess
import sys
import textwrap
import time
import unittest

from fake_sink import Sink, attrs, roots
from hermes_support import (
    PLUGIN_DIR,
    VENDOR_CORE,
    Turn,
    finalize,
    fire_all,
    flush_vendor,
    later,
    load_plugin,
    new_session_id,
    plugin_env,
    simple_turn,
    state_files,
    subagent_turn,
)
from hermes_support import Sandbox
from support import ENV_SECRET_NAME, REPO, SECRETS, leaks, wait_for, whole_leaks

ENV_FILE = "DATABASE_URL=postgres://app:" + "Lq" + "9xT" * 4 + "@db/app\nFEATURE_FLAG=on\n"
GIT_SHA = "4f1e2d3c9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"
UUID = "123e4567-e89b-12d3-a456-426614174000"
CODE = "response = client.create(model=m, max_tokens=4096, temperature=0.2)"
BUDGET_MS = float(os.environ.get("JUDGMENT_HERMES_P95_MS", "2"))


def by_kind(spans: list[dict], kind: str) -> list[dict]:
    return [span for span in spans if attrs(span).get("judgment.span_kind") == kind]


def latest(spans: list[dict]) -> list[dict]:
    """The newest version of each span: a resend repeats span ids with a higher update id."""
    found: dict[str, dict] = {}
    for span in spans:
        found[span["spanId"]] = span
    return list(found.values())


class HermesCase(unittest.TestCase):
    def setUp(self):
        self.sink = Sink().__enter__()
        self.sb = Sandbox(sink_url=self.sink.url)
        self.addCleanup(self.sb.cleanup)
        self.addCleanup(self.sink.__exit__)

    def run_events(self, events: list, *, flush: bool = True, **env) -> tuple:
        with plugin_env(self.sb, **env) as (plugin, ctx):
            results = fire_all(ctx, events, plugin)
            errors = plugin.errors["count"]
        self.assertEqual(results, [], "a callback returned a value to Hermes")
        if flush:
            result = flush_vendor(self.sb)
            self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        return errors, latest(self.sink.spans())


class SpanTree(HermesCase):
    def test_tool_turn_tree_ids_tokens_and_text(self):
        sid = new_session_id()
        prompt = "Read notes.txt and say what it holds. Ünïcode ✓ \"quotes\" and\nnew lines."
        result = json.dumps({"content": "1|hello notes\n2|", "total_lines": 1, "file_size": 12, "truncated": False})
        turn = simple_turn(sid, prompt, path=str(self.sb.project / "notes.txt"), result=result, answer="The notes say hello. ✓")
        errors, spans = self.run_events(turn.events + [finalize(sid) + (later(turn.events, 0.5),)])
        self.assertEqual(errors, 0)
        root = roots(spans)
        self.assertEqual(len(root), 1)
        root = root[0]
        ra = attrs(root)
        self.assertEqual(root["name"], "hermes turn")
        self.assertEqual(ra["judgment.span_kind"], "agent")
        self.assertTrue(ra["judgment.is_agent_entry_point"])
        for span in spans:
            a = attrs(span)
            self.assertEqual(a["judgment.session_id"], sid)
            self.assertEqual(a["gen_ai.conversation.id"], sid)
            self.assertEqual(span["traceId"], root["traceId"])
            if span is not root:
                self.assertEqual(span["parentSpanId"], root["spanId"])
                self.assertGreaterEqual(int(span["startTimeUnixNano"]), int(root["startTimeUnixNano"]))
                self.assertLessEqual(int(span["endTimeUnixNano"]), int(root["endTimeUnixNano"]))
        # Normal text arrives byte for byte.
        self.assertEqual(ra["judgment.input"], prompt)
        self.assertEqual(ra["judgment.output"], "The notes say hello. ✓")
        llms = sorted(by_kind(spans, "llm"), key=lambda s: int(s["startTimeUnixNano"]))
        self.assertEqual(len(llms), 2)
        first, second = (attrs(s) for s in llms)
        for a, (inp, out, cache) in zip((first, second), ((1001, 21, 0), (1002, 22, 512))):
            self.assertEqual(a["judgment.llm.model"], "mock-model-1")
            self.assertEqual(a["judgment.llm.provider"], "custom")
            self.assertEqual(int(a["judgment.usage.non_cached_input_tokens"]), inp)
            self.assertEqual(int(a["judgment.usage.output_tokens"]), out)
            self.assertEqual(int(a["judgment.usage.cache_read_input_tokens"]), cache)
        self.assertEqual(json.loads(first["judgment.llm.tool_calls"])[0]["name"], "read_file")
        messages = json.loads(second["judgment.input"])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[0]["parts"][0]["content"], prompt)
        self.assertEqual(messages[2]["parts"][0]["response"], result)
        self.assertEqual(json.loads(second["judgment.output"])[0]["parts"][0]["content"], "The notes say hello. ✓")
        tool = attrs(by_kind(spans, "tool")[0])
        self.assertEqual(tool["gen_ai.tool.name"], "read_file")
        self.assertEqual(json.loads(tool["judgment.input"]), {"path": str(self.sb.project / "notes.txt")})
        self.assertEqual(tool["judgment.output"], result)
        resource = {item["key"]: next(iter(item["value"].values())) for item in json.loads(self.sink.bodies()[-1])["resourceSpans"][0]["resource"]["attributes"]}
        self.assertEqual(resource["agent.harness"], "hermes")
        self.assertEqual(resource["agent.harness.version"], "0.19.0")

    def test_inbox_events_and_permissions(self):
        sid = new_session_id()
        turn = simple_turn(sid, "hi", path="a.txt", result="ok", answer="done")
        self.run_events(turn.events, flush=False)
        events = self.sb.inbox()
        self.assertEqual([e["event"] for e in events], ["turn_start", "model", "tool", "model", "turn_end"])
        for event in events:
            self.assertEqual(event["harness"], "hermes")
            self.assertEqual(event["session_id"], sid)
            self.assertEqual(event["turn_id"], turn.tid)
            self.assertEqual(event["capture"], "sanitized")
            self.assertEqual(event["export_project"], "proj-test-1")
            self.assertEqual(len(event["export_fp"]), 32)
            self.assertNotIn("JUDGMENT_API_KEY", json.dumps(event))
        self.assertEqual(oct(self.sb.state.stat().st_mode & 0o777), "0o700")
        self.assertEqual(oct((self.sb.state / "inbox").stat().st_mode & 0o777), "0o700")
        for path in state_files(self.sb.state):
            self.assertEqual(oct(path.stat().st_mode & 0o777), "0o600", path)
        self.assertEqual(list((self.sb.state / "inbox.tmp").iterdir()), [])

    def test_two_turns_two_traces_one_session(self):
        sid = new_session_id()
        one = simple_turn(sid, "first", path="a", result="A", answer="one")
        two = simple_turn(sid, "second", path="b", result="B", answer="two", first=False, start=later(one.events, 2))
        _errors, spans = self.run_events(one.events + two.events)
        rs = roots(spans)
        self.assertEqual(len({s["traceId"] for s in rs}), 2)
        self.assertEqual({attrs(s)["judgment.session_id"] for s in spans}, {sid})
        second = next(s for s in rs if attrs(s)["judgment.input"] == "second")
        llm = next(s for s in by_kind(spans, "llm") if s["traceId"] == second["traceId"])
        texts = [p.get("content") for m in json.loads(attrs(llm)["judgment.input"]) for p in m["parts"]]
        self.assertIn("first", texts)  # the earlier turn rides along as context
        self.assertIn("second", texts)

    def test_background_subagent_nests_under_its_turn(self):
        sid = new_session_id()
        _parent, child_sid, events = subagent_turn(sid)
        errors, spans = self.run_events(events + [finalize(sid) + (later(events, 0.5),)])
        self.assertEqual(errors, 0)
        root = roots(spans)
        self.assertEqual(len(root), 1)
        root = root[0]
        agents = [s for s in by_kind(spans, "agent") if s is not root]
        self.assertEqual(len(agents), 1)
        agent = agents[0]
        aa = attrs(agent)
        self.assertEqual(agent["parentSpanId"], root["spanId"])
        self.assertEqual(aa["judgment.agent_id"], child_sid)
        self.assertEqual(aa["judgment.agent_class_name"], "leaf")
        self.assertEqual(aa["judgment.input"], "count to three")
        self.assertEqual(aa["judgment.output"], "one two three")
        children = [s for s in spans if s.get("parentSpanId") == agent["spanId"]]
        self.assertEqual(sorted(attrs(s)["judgment.span_kind"] for s in children), ["llm", "llm", "tool"])
        self.assertEqual(sum(int(attrs(s)["judgment.usage.output_tokens"]) for s in children if attrs(s)["judgment.span_kind"] == "llm"), 21 + 5)
        self.assertEqual({attrs(s)["judgment.session_id"] for s in spans}, {sid})
        self.assertLessEqual(int(agent["endTimeUnixNano"]), int(root["endTimeUnixNano"]))
        # The child's own turn never became a trace.
        self.assertFalse(any(attrs(s)["judgment.input"] == "count to three" for s in roots(spans)))

    def test_subagent_still_running_at_exit_is_closed(self):
        sid = new_session_id()
        _parent, child_sid, events = subagent_turn(sid)
        events = [event for event in events if event[0] != "subagent_stop"]
        _errors, spans = self.run_events(events + [finalize(sid) + (later(events, 0.5),)])
        agent = [s for s in by_kind(spans, "agent") if s.get("parentSpanId")]
        self.assertEqual(len(agent), 1)
        self.assertEqual(attrs(agent[0])["judgment.agent_id"], child_sid)

    def test_failed_attempt_and_retry(self):
        sid = new_session_id()
        turn = Turn(sid, "hello")
        turn.api(error={"status_code": 503, "message": "overloaded", "retry_count": 1})
        turn.api(text="hi there")
        turn.end("hi there")
        _errors, spans = self.run_events(turn.events)
        llms = sorted(by_kind(spans, "llm"), key=lambda s: int(s["startTimeUnixNano"]))
        self.assertEqual(len(llms), 2)
        failed, ok = llms
        self.assertEqual(failed.get("status"), {"code": 2})
        self.assertIn("503", attrs(failed)["judgment.output"])
        self.assertIn("overloaded", attrs(failed)["judgment.output"])
        self.assertNotIn("status", ok)
        # The error never reached the model, so the retry's input does not hold it.
        self.assertNotIn("overloaded", attrs(ok)["judgment.input"])

    def test_interrupted_turns(self):
        sid = new_session_id()
        one = Turn(sid, "long task")
        call = one.api(tool=("terminal", {"command": "sleep 100"}))
        one.tool(call, "terminal", {"command": "sleep 100"}, "", post=False)  # pre_tool_call, then the user hits Ctrl-C
        one.end(None, interrupted=True)
        # A turn Hermes never closed: the next prompt closes it.
        two = Turn(sid, "never closed", first=False, start=later(one.events, 1))
        two.api(text="partial")
        three = Turn(sid, "after", first=False, start=later(two.events, 1))
        three.api(text="ok")
        three.end("ok")
        _errors, spans = self.run_events(one.events + two.events + three.events)
        rs = {attrs(s)["judgment.input"]: s for s in roots(spans)}
        self.assertEqual(set(rs), {"long task", "never closed", "after"})
        tool = [s for s in by_kind(spans, "tool") if s["traceId"] == rs["long task"]["traceId"]]
        self.assertEqual(len(tool), 1)
        self.assertEqual(tool[0].get("status"), {"code": 2})
        self.assertIn("no result", attrs(tool[0])["judgment.output"])
        events = [json.loads(line) for path in (self.sb.state / "turns").rglob("*.jsonl") for line in path.read_text().splitlines()]
        ends = [e for e in events if e["event"] == "turn_end"]
        self.assertEqual(sorted(bool(e.get("interrupted")) for e in ends), [False, True, True])

    def test_blocked_and_cancelled_tools_are_errors(self):
        sid = new_session_id()
        turn = Turn(sid, "do things")
        call = turn.api(tool=("terminal", {"command": "rm -rf /tmp/x"}))
        turn.tool(call, "terminal", {"command": "rm -rf /tmp/x"}, json.dumps({"error": "BLOCKED: denied"}), status="blocked", pre=False)
        call2 = turn.api(tool=("terminal", {"command": "sleep 5"}))
        turn.tool(call2, "terminal", {"command": "sleep 5"}, json.dumps({"error": "Tool execution cancelled by user interrupt", "status": "cancelled"}), status="cancelled")
        turn.api(text="stopped")
        turn.end("stopped")
        _errors, spans = self.run_events(turn.events)
        tools = by_kind(spans, "tool")
        self.assertEqual(len(tools), 2)
        self.assertTrue(all(s.get("status") == {"code": 2} for s in tools))
        self.assertIn("BLOCKED", " ".join(attrs(s)["judgment.output"] for s in tools))

    def test_capture_metadata_and_full(self):
        secret = SECRETS["github_token"]["core"]
        for tracing, expect_text in (({"enabled": True, "capture": "metadata"}, False), ({"enabled": True, "redact": False}, True)):
            with self.subTest(tracing=tracing):
                self.sink.requests.clear()
                self.sb.set_tracing(tracing)
                sid = new_session_id()
                turn = simple_turn(sid, "show " + secret, path="a", result="token " + secret, answer="ok " + secret)
                _errors, spans = self.run_events(turn.events)
                body = self.sink.text()
                llm = attrs(by_kind([s for s in spans if attrs(s)["judgment.session_id"] == sid], "llm")[0])
                self.assertEqual(int(llm["judgment.usage.non_cached_input_tokens"]), 1001)
                if expect_text:
                    self.assertIn(secret, body)  # redact: false is raw text, as the project asked
                else:
                    self.assertEqual(whole_leaks(body), [])
                    self.assertEqual(attrs(roots([s for s in spans if attrs(s)["judgment.session_id"] == sid])[0])["judgment.input"], "")


class Redaction(HermesCase):
    def test_fake_secrets_everywhere_are_redacted(self):
        sid = new_session_id()
        gh, oa, aws, slack = (SECRETS[k]["core"] for k in ("github_token", "openai_key", "aws_access_key_id", "slack_token"))
        env_value = SECRETS["env_value"]["core"]
        turn = Turn(sid, f"deploy with key {oa}, keep it safe")
        call = turn.api(tool=("terminal", {"command": f"curl -H 'Authorization: Bearer {SECRETS['bearer_token']['core']}' https://x"}), usage=(1001, 21, 0))
        turn.tool(call, "terminal", {"command": f"echo {aws}"}, json.dumps({"output": f"token={gh}\nservice {env_value}\n{SECRETS['private_key']['text']}", "exit_code": 0}))
        turn.api(error={"status_code": 401, "message": f"invalid key {SECRETS['anthropic_key']['core']}"})
        turn.api(text=f"Deployed. The slack token {slack} still works. Build {GIT_SHA}, id {UUID}.\n{CODE}")
        turn.end(f"Deployed. The slack token {slack} still works. Build {GIT_SHA}, id {UUID}.\n{CODE}")
        _errors, spans = self.run_events(turn.events + [finalize(sid) + (later(turn.events, 0.5),)])
        body = self.sink.text()
        self.assertEqual(leaks(body, hashes=False), [], "a fake secret reached the sink")
        self.assertEqual(leaks(self.sb.state_text(), hashes=False), [], "a fake secret stayed under the state dir")
        for rule in ("openai_key", "github_token", "aws_access_key_id", "slack_token", "env:" + ENV_SECRET_NAME, "anthropic_key", "private_key", "authorization_header"):
            self.assertIn(f"[REDACTED:{rule}]", body, rule)
        # False positives stay byte for byte.
        for text in (GIT_SHA, UUID, CODE):
            self.assertIn(json.dumps(text)[1:-1], body)
        root = attrs(roots(spans)[0])
        self.assertTrue(root["judgment.redacted"])
        self.assertGreaterEqual(int(root["judgment.redaction_count"]), 8)
        # Structure and tokens survive redaction.
        self.assertEqual(sorted(attrs(s)["judgment.span_kind"] for s in spans), ["agent", "llm", "llm", "llm", "tool"])
        self.assertEqual(sum(int(attrs(s).get("judgment.usage.output_tokens", 0)) for s in by_kind(spans, "llm")), 21 + 21)

    def test_dotenv_read_blanks_the_turn_and_not_the_next(self):
        sid = new_session_id()
        env_path = str(self.sb.project / ".env")
        one = simple_turn(sid, "what is in the env file", path=env_path, result=json.dumps({"content": ENV_FILE}), answer="It sets DATABASE_URL.")
        two = simple_turn(sid, "now run the tests", path="tests.log", result="42 passed", answer="All 42 tests pass.", first=False,
                          tool_name="terminal", args={"command": "make test"}, start=later(one.events, 2))
        _errors, spans = self.run_events(one.events + two.events)
        rs = {s["traceId"]: s for s in roots(spans)}
        blanked = next(s for s in rs.values() if attrs(s).get("judgment.redacted_reason"))
        ba = attrs(blanked)
        self.assertEqual(ba["judgment.redacted_reason"], "sensitive_path")
        self.assertEqual(ba["judgment.redacted_rule"], "dotenv")
        self.assertEqual(ba["judgment.redacted_trigger"], "~/proj/.env")
        same = [s for s in spans if s["traceId"] == blanked["traceId"]]
        self.assertEqual(len(same), 4)
        for span in same:
            self.assertEqual(attrs(span).get("judgment.input", ""), "")
            self.assertEqual(attrs(span).get("judgment.output", ""), "")
        self.assertEqual(sum(int(attrs(s).get("judgment.usage.output_tokens", 0)) for s in by_kind(same, "llm")), 43)
        other = next(s for s in rs.values() if s is not blanked)
        self.assertEqual(attrs(other)["judgment.input"], "now run the tests")
        self.assertEqual(attrs(other)["judgment.output"], "All 42 tests pass.")
        self.assertNotIn("judgment.redacted_reason", attrs(other))
        everything = self.sink.text() + self.sb.state_text()
        self.assertNotIn("Lq9xT", everything)
        self.assertNotIn("FEATURE_FLAG", everything)

    def test_env_dump_command_blanks_the_turn(self):
        sid = new_session_id()
        turn = simple_turn(sid, "show the env", path="", result="PATH=/bin\n" + ENV_SECRET_NAME + "=" + SECRETS["env_value"]["core"],
                           answer="Here it is.", tool_name="terminal", args={"command": "printenv"})
        _errors, spans = self.run_events(turn.events)
        root = attrs(roots(spans)[0])
        self.assertEqual(root["judgment.redacted_reason"], "sensitive_path")
        self.assertEqual(leaks(self.sink.text() + self.sb.state_text(), hashes=False), [])

    def test_inbox_holds_no_secret_before_the_flusher_runs(self):
        sid = new_session_id()
        turn = simple_turn(sid, "key " + SECRETS["openai_key"]["core"], path="n.txt", result="gh " + SECRETS["github_token"]["core"], answer="ok")
        self.run_events(turn.events, flush=False)
        self.assertEqual(leaks(self.sb.state_text(), hashes=False), [])
        self.assertIn("[REDACTED:openai_key]", self.sb.state_text())

    def test_a_cut_long_result_drops_the_partial_token(self):
        """The cap comes after redaction: a token across the cap is redacted whole, then the cut drops the partial word."""
        core = load_plugin().core
        token = SECRETS["github_token"]["core"]
        text = "x " * (core.TEXT_CAP // 2 - 10) + token + " tail"
        gate = core.Gate("sanitized", (), (), (), str(self.sb.state), ())
        out = core.prepare({"event": "tool", "tool_output": text}, gate)["tool_output"]
        self.assertTrue(out.endswith(core.CUT_MARK))
        self.assertNotIn(token[4:12], out)


class OptOut(HermesCase):
    def assert_nothing(self, events: list, **env) -> None:
        """No inbox file from the plugin, and no span once the flusher has run."""
        errors, _ = self.run_events(events, flush=False, **env)
        self.assertEqual(errors, 0)
        inbox = self.sb.state / "inbox"
        self.assertFalse(inbox.exists() and any(inbox.iterdir()), "an inbox file was written")
        self.files_before_flush = state_files(self.sb.state)
        flush_vendor(self.sb)
        self.assertEqual(self.sink.requests, [])

    def test_no_optin_file_writes_nothing(self):
        self.sb.cleanup()
        self.sb = Sandbox(tracing=None, sink_url=self.sink.url)
        sid = new_session_id()
        _p, _c, events = subagent_turn(sid)
        self.assert_nothing(events + [finalize(sid)])
        self.assertEqual(self.files_before_flush, [])

    def test_enabled_false_writes_nothing(self):
        self.sb.set_tracing({"enabled": False})
        sid = new_session_id()
        self.assert_nothing(simple_turn(sid, "hi", path="a", result="b", answer="c").events + [finalize(sid)])
        self.assertEqual(self.files_before_flush, [])

    def test_kill_switch(self):
        sid = new_session_id()
        self.assert_nothing(simple_turn(sid, "hi", path="a", result="b", answer="c").events + [finalize(sid)], JUDGMENT_TRACING="0")
        self.assertEqual(self.files_before_flush, [])

    def test_judgment_off_command(self):
        sid = new_session_id()
        turn = simple_turn(sid, "hi", path="a", result="b", answer="c")
        start, rest = turn.events[0], turn.events[1:]
        with plugin_env(self.sb) as (plugin, ctx):
            ctx.cli(sid)  # the interactive CLI on this session
            fire_all(ctx, [start], plugin)
            handler = ctx.commands["judgment"]["handler"]
            self.assertIn("on for this session", handler("status"))
            self.assertIn("off for this session", handler("off"))
            self.assertIn("off for this session", handler("status"))
            fire_all(ctx, rest + [finalize(sid)], plugin)
            self.assertIn("on for this session", handler("on"))
        self.assertEqual((self.sb.state / "sessions" / sid).read_text().strip(), "on")
        inbox = self.sb.state / "inbox"
        self.assertFalse(inbox.exists() and any(inbox.iterdir()))
        flush_vendor(self.sb)
        self.assertEqual(self.sink.requests, [])

    def test_judgment_command_ignores_a_child_session_in_context(self):
        sid = new_session_id()
        _p, child_sid, events = subagent_turn(sid)
        with plugin_env(self.sb) as (plugin, ctx):
            ctx.cli(sid)
            fire_all(ctx, events, plugin)
            os.environ["HERMES_SESSION_ID"] = child_sid  # Hermes points its session context at a child it builds
            self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("off"))
        self.assertEqual((self.sb.state / "sessions" / sid).read_text().strip(), "off")
        self.assertFalse((self.sb.state / "sessions" / child_sid).exists())

    def test_session_default_off_until_turned_on(self):
        self.sb.set_tracing({"enabled": True, "sessionDefault": "off"})
        sid = new_session_id()
        one = simple_turn(sid, "quiet", path="a", result="b", answer="c")
        two = simple_turn(sid, "loud", path="a", result="b", answer="d", first=False, start=later(one.events, 2))
        with plugin_env(self.sb) as (plugin, ctx):
            ctx.cli(sid)
            fire_all(ctx, one.events, plugin)
            self.assertIn("on for this session", ctx.commands["judgment"]["handler"]("on"))
            fire_all(ctx, two.events, plugin)
        flush_vendor(self.sb)
        self.assertEqual([attrs(s)["judgment.input"] for s in roots(self.sink.spans())], ["loud"])

    def test_gateway_session_without_a_project_writes_nothing(self):
        sid = new_session_id()
        turn = Turn(sid, "hello from telegram", platform="telegram")
        turn.api(text="hi")
        turn.end("hi")
        outside = self.sb.home / "gateway-cwd"
        outside.mkdir()
        self.assert_nothing(turn.events + [finalize(sid)], TERMINAL_CWD=str(outside))
        self.assertEqual(self.files_before_flush, [])

    def test_untraced_turn_child_never_opens_a_turn(self):
        self.sb.set_tracing({"enabled": True, "sessionDefault": "off"})
        sid = new_session_id()
        _p, _c, events = subagent_turn(sid)
        self.assert_nothing(events + [finalize(sid)])


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class Latency(HermesCase):
    REPORT: dict = {}

    @classmethod
    def tearDownClass(cls):
        if cls.REPORT:
            print("\nHermes callback latency, in-process, traced path (ms):", file=sys.stderr)
            for hook, (p50, p95, mx, count) in sorted(cls.REPORT.items()):
                print(f"  {hook:22s} n={count:5d} p50={p50:.3f} p95={p95:.3f} max={mx:.3f}", file=sys.stderr)

    def test_each_callback_p95_under_budget(self):
        """Full traced Hermes sequences, timed per callback as Hermes would see it, until every hook has 1000 calls.

        Each round is a new session: a tool turn whose first answer attempt fails and is retried, a turn
        that starts a background child, then finalize (even rounds) or reset (odd rounds).
        """
        times: dict[str, list[float]] = {}
        result = json.dumps({"content": "1|" + "hello world line\n" * 200})

        def timed(ctx, plugin, events):
            for hook, kwargs, at in events:
                kwargs.setdefault("telemetry_schema_version", "hermes.observer.v1")
                plugin.now_ms = lambda at=at: int(at * 1000)
                (callback,) = ctx.hooks[hook]
                start = time.perf_counter()
                callback(**kwargs)
                times.setdefault(hook, []).append((time.perf_counter() - start) * 1000)

        with plugin_env(self.sb, JUDGMENT_FLUSH="0") as (plugin, ctx):
            n = 0
            while min((len(times.get(hook, [])) for hook in plugin.HOOKS), default=0) < 1000:
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
                timed(ctx, plugin, turn.events + sub + [(hook, {"session_id": sid, "platform": "cli", "reason": "shutdown"}, later(sub, 0.1))])
                n += 1
            self.assertEqual(plugin.errors["count"], 0)
            self.assertGreater(len(list((self.sb.state / "inbox").iterdir())), 1000)
        for hook, values in times.items():
            self.REPORT[hook] = (statistics.median(values), pct(values, 0.95), max(values), len(values))
        slow = {hook: round(v[1], 3) for hook, v in self.REPORT.items() if v[1] >= BUDGET_MS}
        self.assertEqual(slow, {}, f"p95 over {BUDGET_MS} ms")

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


class Sender(HermesCase):
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

    def test_sink_down_callbacks_stay_fast(self):
        proc = self.spawn("http://127.0.0.1:9")
        line = proc.stdout.readline()
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        info = json.loads(line)
        self.assertLess(info["ms"], 250, "a whole turn's callbacks, sender launch included")
        log = self.sb.state / "flush.log"
        self.assertTrue(wait_for(lambda: log.exists() and "post attempt" in log.read_text(), timeout=30))
        self.assertEqual(leaks(log.read_text(), hashes=False), [])


class Install(unittest.TestCase):
    def test_manifest_and_vendored_core(self):
        manifest = (PLUGIN_DIR / "plugin.yaml").read_text()
        self.assertIn("name: judgment-tracing", manifest)
        version = next(line.split(":", 1)[1].strip() for line in manifest.splitlines() if line.startswith("version:"))
        sys.path.insert(0, str(REPO / "core"))
        from judgment_core import __version__

        self.assertEqual(version, __version__)
        self.assertTrue((VENDOR_CORE / "judgment_core" / "flush.py").is_file())
        plugin = load_plugin()
        self.assertEqual(set(plugin.HOOKS), {line.strip()[2:] for line in manifest.splitlines() if line.strip().startswith("- ")})

    def test_register_only_observes(self):
        from hermes_support import FakeCtx

        plugin = load_plugin()
        ctx = FakeCtx()
        plugin.register(ctx)
        self.assertEqual(set(ctx.hooks), set(plugin.HOOKS))
        self.assertTrue(set(ctx.hooks).isdisjoint({"transform_tool_result", "transform_llm_output", "transform_terminal_output", "pre_gateway_dispatch", "pre_verify"}))
        self.assertEqual(list(ctx.commands), ["judgment"])

    def test_importing_and_registering_writes_nothing(self):
        sb = Sandbox()
        try:
            code = f"import sys; sys.path.insert(0, {str(REPO / 'tests')!r}); from hermes_support import load_plugin, FakeCtx; load_plugin().register(FakeCtx())"
            result = subprocess.run([sys.executable, "-c", code], env={**sb.env, "TERMINAL_CWD": str(sb.project)}, cwd=sb.project, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(sb.state.exists())
        finally:
            sb.cleanup()


if __name__ == "__main__":
    unittest.main()
