"""Hermes plugin: what a traced session looks like, when nothing is traced, and which conversation an event belongs to.

Hook kwargs come from tests/hermes_support.py, shaped like what Hermes 0.19.0 passes. The plugin runs
in-process, as in Hermes, with only the sandbox env. tests/e2e_hermes.py runs real Hermes.

    SpanTree, Retries, Incomplete, Grandchild, Late    hook kwargs -> inbox -> flusher -> spans in the fake sink
    OptOut, Workspace, CommandIdentity                 opt-in, the kill switch, /judgment on|off, which folder decides
    Rotation, Rotated, Preflight                       Hermes renaming a session id mid-conversation
    Install, Contract                                  the manifest, the hooks registered, the vendored core's API
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import subprocess
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

from fake_sink import attrs, roots
from hermes_support import (
    PLUGIN_DIR,
    VENDOR_CORE,
    Case,
    Sandbox,
    Turn,
    finalize,
    fire_all,
    flush_vendor,
    gh_token,
    kinds,
    later,
    latest,
    load_plugin,
    new_session_id,
    plugin_env,
    preflight,
    renamed,
    simple_turn,
    state_files,
    subagent_turn,
)
from support import REPO, SECRETS, whole_leaks


class SpanTree(Case):
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
        llms = sorted(kinds(spans, "llm"), key=lambda s: int(s["startTimeUnixNano"]))
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
        tool = attrs(kinds(spans, "tool")[0])
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
        llm = next(s for s in kinds(spans, "llm") if s["traceId"] == second["traceId"])
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
        agents = [s for s in kinds(spans, "agent") if s is not root]
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
        tool = [s for s in kinds(spans, "tool") if s["traceId"] == rs["long task"]["traceId"]]
        self.assertEqual(len(tool), 1)
        self.assertEqual(tool[0].get("status"), {"code": 2})
        self.assertIn("no result", attrs(tool[0])["judgment.output"])
        events = [json.loads(line) for path in (self.sb.state / "turns").rglob("*.jsonl") for line in path.read_text().splitlines()]
        ends = [e for e in events if e["event"] == "turn_end"]
        self.assertEqual(sorted(bool(e.get("interrupted")) for e in ends), [False, True, True])

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
                llm = attrs(kinds([s for s in spans if attrs(s)["judgment.session_id"] == sid], "llm")[0])
                self.assertEqual(int(llm["judgment.usage.non_cached_input_tokens"]), 1001)
                if expect_text:
                    self.assertIn(secret, body)  # redact: false is raw text, as the project asked
                else:
                    self.assertEqual(whole_leaks(body), [])
                    self.assertEqual(attrs(roots([s for s in spans if attrs(s)["judgment.session_id"] == sid])[0])["judgment.input"], "")


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
        self.assertIn("rate limited", attrs(ordered[0])["judgment.output"])
        # The error never reached the model, so the attempt that worked does not hold it.
        self.assertNotIn("rate limited", attrs(ordered[2])["judgment.input"])
        for before, after in zip(ordered, ordered[1:]):
            self.assertGreaterEqual(int(after["startTimeUnixNano"]), int(before["endTimeUnixNano"]))


class Incomplete(Case):
    def test_an_answered_turn_hermes_reports_incomplete_is_interrupted(self):
        """Hermes can answer with an error explanation and then end the turn with completed=False."""
        turn = Turn(new_session_id(), "go")
        turn.api(text="the provider failed")
        turn.end("the provider failed")
        hook, kwargs, at = turn.events[-1]
        turn.events[-1] = (hook, {**kwargs, "completed": False, "failed": True}, at)
        self.fire(turn.events)
        [end] = [event for event in self.sb.inbox() if event["event"] == "turn_end"]
        self.assertTrue(end["interrupted"])


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


class Late(Case):
    def test_a_closed_turns_late_tool_result_joins_no_turn(self):
        """Turn one is off and ends with a tool still running. Its result arrives during turn two, which is traced."""
        sid = new_session_id()
        secret = "late-" + gh_token()[4:20]
        one = Turn(sid, "untraced")
        call = one.api(tool=("terminal", {"command": "sleep 60"}))
        one.tool(call, "terminal", {"command": "sleep 60"}, secret, pre=True, post=False)
        one.end("gave up")
        two = Turn(sid, "traced", first=False, start=later(one.events, 1))
        two.api(text="ok")
        late = Turn(sid, "x", first=False, start=later(two.events, 0.5))
        late.task, late.tid, late.events = one.task, one.tid, []
        late.tool(call, "terminal", {"command": "sleep 60"}, secret, pre=False, post=True)
        with plugin_env(self.sb) as (plugin, ctx):
            ctx.cli(sid)
            fire_all(ctx, one.events[:1], plugin)
            ctx.commands["judgment"]["handler"]("off")
            fire_all(ctx, one.events[1:], plugin)
            ctx.commands["judgment"]["handler"]("on")
            fire_all(ctx, two.events + late.events, plugin)
            self.assertEqual(plugin.errors["count"], 0)
            two.events = []
            two.end("ok")
            fire_all(ctx, two.events, plugin)
        inbox = self.sb.inbox()
        self.assertEqual({event["turn_id"] for event in inbox if event.get("turn_id")}, {two.tid})
        self.assertEqual([event["event"] for event in inbox if event["event"] == "tool"], [])
        self.assertNotIn(secret, self.sb.state_text())


class OptOut(Case):
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


class Rotated(Case):
    def test_off_after_a_rotation_reaches_the_new_id(self):
        """Hermes rotates the id mid-turn, then the user turns tracing off: a later Hermes process resumes under the
        new id and must read off."""
        sid, rotated = new_session_id(), new_session_id()
        one = Turn(sid, "rotates")
        one.api(text="a")
        rotate_at = len(one.events)
        one.api(text="a2")
        one.end("a2")
        with plugin_env(self.sb) as (plugin, ctx):
            fire_all(ctx, renamed(one.events, sid, rotated, rotate_at), plugin)
            ctx.cli(rotated)
            self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("off"))
            modes = {name: (self.sb.state / "sessions" / name).read_text().strip() for name in (sid, rotated)}
            self.assertEqual(modes, {sid: "off", rotated: "off"})
            self.assertIn("on for this session", ctx.commands["judgment"]["handler"]("on"))
        modes = {name: (self.sb.state / "sessions" / name).read_text().strip() for name in (sid, rotated)}
        self.assertEqual(modes, {sid: "on", rotated: "on"})


class Preflight(Case):

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


class Contract(unittest.TestCase):
    def test_the_vendored_core_matches_what_the_bridge_calls(self):
        core = load_plugin().core
        self.assertEqual(core.contract_problems(), [], "judgment_core changed: update hermes/core_bridge.py (see CONTRACT)")

    def test_every_core_call_in_the_bridge_is_declared(self):
        core = load_plugin().core
        tree = ast.parse(inspect.getsource(core))
        declared = {(module_, name) for module_, name, _ in core.CONTRACT}
        used = {(node.value.id, node.attr) for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("optin", "redact")}
        self.assertEqual(used - declared, set(), "core_bridge calls a core function CONTRACT does not list")


if __name__ == "__main__":
    unittest.main()
