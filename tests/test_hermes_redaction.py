"""Hermes plugin: no secret reaches the state folder or the sink, whatever field or size it arrives in.

Every fake secret is built at runtime (tests/support.py). The sandbox is hermetic (hermes_support.Sandbox).

    Redaction, MetadataSecrets, KeySecrets, Collisions, CallIds    text, names, dict keys, and tool call ids
    Settings, Malformed                                            tracing.json's redact* keys, valid and not
    Stamps                                                         secrets in the project path, environment label, export stamp
    Truncation, Structured, Snapshot                               values cut or omitted for size never keep part of a secret
"""

from __future__ import annotations

import json
import marshal
import unittest
from unittest import mock

from fake_sink import attrs, roots
from hermes_support import ENV_VALUE, GH, OPENAI, Case, Turn, finalize, gh_token, kinds, later, load_plugin, module, new_session_id, no_piece, rows, simple_turn
from support import ENV_SECRET_NAME, SECRETS, leaks


ENV_FILE = "DATABASE_URL=postgres://app:" + "Lq" + "9xT" * 4 + "@db/app\nFEATURE_FLAG=on\n"


GIT_SHA = "4f1e2d3c9a8b7c6d5e4f3a2b1c0d9e8f7a6b5c4d"


UUID = "123e4567-e89b-12d3-a456-426614174000"


CODE = "response = client.create(model=m, max_tokens=4096, temperature=0.2)"


class Redaction(Case):
    def test_fake_secrets_everywhere_are_redacted(self):
        sid = new_session_id()
        gh, oa, aws, slack = (SECRETS[k]["core"] for k in ("github_token", "openai_key", "aws_access_key_id", "slack_token"))
        env_value = SECRETS["env_value"]["core"]
        turn = Turn(sid, f"deploy with key {oa}, keep it safe")
        call = turn.api(tool=("terminal", {"command": f"curl -H 'Authorization: Bearer {SECRETS['bearer_token']['core']}' https://x"}), usage=(1001, 21, 0))
        turn.tool(call, "terminal", {"command": f"echo {aws}"}, json.dumps({"output": f"token={gh}\npushed with {gh}\nservice {env_value}\n{SECRETS['private_key']['text']}", "exit_code": 0}))
        turn.api(error={"status_code": 401, "message": f"invalid key {SECRETS['anthropic_key']['core']}"})
        turn.api(text=f"Deployed. The slack token {slack} still works. Build {GIT_SHA}, id {UUID}.\n{CODE}")
        turn.end(f"Deployed. The slack token {slack} still works. Build {GIT_SHA}, id {UUID}.\n{CODE}")
        _errors, spans = self.run_events(turn.events + [finalize(sid) + (later(turn.events, 0.5),)])
        body = self.sink.text()
        self.assertEqual(leaks(body, hashes=False), [], "a fake secret reached the sink")
        self.assertEqual(leaks(self.sb.state_text(), hashes=False), [], "a fake secret stayed under the state dir")
        # PR1's rules take `token=<value>` whole as secret_assignment (the longer pair applies first); the bare
        # token on the next line is github_token's.
        for rule in ("openai_key", "github_token", "secret_assignment", "aws_access_key_id", "slack_token", "env:" + ENV_SECRET_NAME, "anthropic_key", "private_key", "authorization_header"):
            self.assertIn(f"[REDACTED:{rule}]", body, rule)
        # False positives stay byte for byte.
        for text in (GIT_SHA, UUID, CODE):
            self.assertIn(json.dumps(text)[1:-1], body)
        root = attrs(roots(spans)[0])
        self.assertTrue(root["judgment.redacted"])
        self.assertGreaterEqual(int(root["judgment.redaction_count"]), 8)
        # Structure and tokens survive redaction.
        self.assertEqual(sorted(attrs(s)["judgment.span_kind"] for s in spans), ["agent", "llm", "llm", "llm", "tool"])
        self.assertEqual(sum(int(attrs(s).get("judgment.usage.output_tokens", 0)) for s in kinds(spans, "llm")), 21 + 21)

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
        self.assertEqual(sum(int(attrs(s).get("judgment.usage.output_tokens", 0)) for s in kinds(same, "llm")), 43)
        other = next(s for s in rs.values() if s is not blanked)
        self.assertEqual(attrs(other)["judgment.input"], "now run the tests")
        self.assertEqual(attrs(other)["judgment.output"], "All 42 tests pass.")
        self.assertNotIn("judgment.redacted_reason", attrs(other))
        everything = self.sink.text() + self.sb.state_text()
        self.assertNotIn("Lq9xT", everything)
        self.assertNotIn("FEATURE_FLAG", everything)


class MetadataSecrets(Case):
    def events(self, sid: str) -> list:
        """A model name, provider, tool name, tool call id, and child role that hold secrets."""
        turn = Turn(sid, "hello", model="model-" + GH, provider="prov-" + ENV_VALUE)
        turn.api(tool=("tool." + OPENAI, {"path": "a"}))
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


class Collisions(unittest.TestCase):
    def test_two_secret_keys_matching_one_rule_keep_both_entries(self):
        core = load_plugin().core
        gate = core.Gate("sanitized", core.settings({}), (), "/nonexistent", ())
        one, two = gh_token(), gh_token()
        record = {"v": 1, "harness": "hermes", "event": "tool", "session_id": "s", "turn_id": "t", "ts": 1,
                  "tool_input": {one: "first", two: "second", "nested": {one: "deep"}}, "tool_output": "ok"}
        out = core.prepare(record, gate)
        text = json.dumps(out)
        self.assertNotIn(one, text)
        self.assertNotIn(two, text)
        values = out["tool_input"]
        self.assertEqual(sorted(value for key, value in values.items() if key != "nested"), ["first", "second"])
        self.assertTrue(all(key.startswith("[REDACTED:github_token]") for key in values if key != "nested"))
        self.assertEqual(list(values["nested"].values()), ["deep"])
        self.assertEqual(sum(item["count"] for item in out["redactions"] if item["rule"] == "github_token"), 3)


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


class Settings(Case):
    def test_every_redaction_key_in_tracing_json_applies(self):
        """Each assertion differs from what the built-in rules alone would do, so a key the bridge dropped fails here."""
        pin = "pw" + gh_token()[4:14]
        token = gh_token()
        project = self.sb.home / "settings"
        (project / ".judgment").mkdir(parents=True)
        (project / ".judgment" / "tracing.json").write_text(json.dumps({
            "enabled": True, "redactCommands": ["mycli[^\\S\\n]+show-secret"], "redactNames": ["vault_pin"],
            "redactAllow": ["changeme"], "redactDisable": ["github_token"]}))
        turn = Turn(new_session_id(), "go")
        calls = [("mycli show-secret db", "x"), ("ls", "vault_pin = " + pin), ("ls", "password = changeme"), ("ls", "gh " + token)]
        for command, result in calls:
            call = turn.api(tool=("terminal", {"command": command}))
            turn.tool(call, "terminal", {"command": command}, result)
        turn.api(text="ok")
        turn.end("ok")
        self.fire(turn.events, TERMINAL_CWD=str(project))
        tools = [event for event in self.sb.inbox() if event.get("event") == "tool"]
        self.assertEqual([event.get("sensitive_rule") for event in tools], ["custom_command", None, None, None])
        self.assertEqual([event["tool_output"] for event in tools[1:]],
                         ["vault_pin = [REDACTED:secret_assignment]", "password = changeme", "gh " + token])


class Malformed(Case):
    BAD = ({"redactNames": "pin"}, {"redactPatterns": {"INTERNAL-[0-9]+": 1}}, {"redactCommands": "vaultctl"}, {"redactPaths": ""},
           {"redactPatterns": 5}, {"redactNames": True}, {"redactPaths": [["x"]]}, {"redactCommands": {}})

    def test_a_rule_setting_that_is_not_a_list_of_strings_drops_the_text(self):
        for number, bad in enumerate(self.BAD):
            with self.subTest(bad=bad):
                value = "pw" + gh_token()[4:16]
                project = self.sb.home / f"bad-{number}"
                (project / ".judgment").mkdir(parents=True)
                (project / ".judgment" / "tracing.json").write_text(json.dumps({"enabled": True, **bad}))
                sid = new_session_id()
                turn = Turn(sid, "pin=" + value)
                call = turn.api(tool=("terminal", {"command": "ls"}))
                turn.tool(call, "terminal", {"command": "ls"}, "pin=" + value)
                turn.api(text="ok")
                turn.end("ok")
                self.fire(turn.events, TERMINAL_CWD=str(project))
                events = [event for event in self.sb.inbox() if event.get("session_id") == sid]
                self.assertTrue(events)
                self.assertTrue(all(event.get("redaction_error") for event in events))
                self.assertNotIn(value, self.sb.state_text())


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
        gate = core.Gate("sanitized", core.settings({}), (), "/s", ())
        record = {"event": "tool", "tool_name": "t-" + GH, "agent_type": "r-" + GH, "model": "m-" + GH, "tool_output": GH, "session_id": "s"}
        with mock.patch.object(core.redact, "sanitize_event", side_effect=TypeError("boom")):
            out = core.prepare(record, gate)
        self.assertTrue(out["redaction_error"])
        self.assertEqual(no_piece(json.dumps(out), GH[4:]), [])
        self.assertEqual(out["session_id"], "s")


class Truncation(Case):
    def setUp(self):
        super().setUp()
        self.core = load_plugin().core
        self.gate = self.core.Gate("sanitized", self.core.settings({}), ((ENV_SECRET_NAME, ENV_VALUE),), str(self.sb.state), ())

    def tool_turn(self, args: dict, result: str) -> Turn:
        turn = Turn(new_session_id(), "go")
        call = turn.api(tool=("terminal", args))
        turn.tool(call, "terminal", args, result)
        turn.api(text="ok")
        turn.end("ok")
        return turn

    def test_aggregate_budget_never_cuts_a_short_secret(self):
        """The tracker's 256 KB value budget ran out 15 characters into a token."""
        for pad in (256 * 1024 - 15, 256 * 1024 - 4, 300 * 1024):
            with self.subTest(pad=pad):
                args = {"a": "w" * pad, "token": GH}
                self.fire(self.tool_turn(args, "ok").events)
                self.assertEqual(no_piece(self.sb.state_text(), GH[4:]), [])

    def test_the_cut_after_redaction_drops_the_partial_word_whole(self):
        cap = self.core.TEXT_CAP
        rs_gate = self.core.Gate("sanitized", self.core.settings({"redactPatterns": ["passphrase is [a-z]+ [a-z]+ [a-z]+"]}), (("X_TOKEN", "correct horse " + ENV_VALUE),),
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
        """A ~72 KB command ending in `; cat .env` lost its suffix to fitting before classification."""
        command = "echo " + "word " * (72 * 1024 // 5) + "; cat .env"
        self.fire(self.tool_turn({"command": command}, "PRIVATE-NOTE-" + "x" * 40).events)
        tool = next(event for event in self.sb.inbox() if event["event"] == "tool")
        self.assertTrue(tool["sensitive"])
        self.assertEqual(tool["sensitive_rule"], "dotenv")
        self.assertNotIn("PRIVATE-NOTE", self.sb.state_text())
        root = attrs(roots(self.flushed())[0])
        self.assertEqual((root["judgment.redacted_reason"], root["judgment.redacted_rule"]), ("sensitive_path", "dotenv"))
        self.assertNotIn("PRIVATE-NOTE", self.sink.text())


class Structured(Case):
    def tool_turn(self, args, result: str = "ok") -> Turn:
        turn = Turn(new_session_id(), "load the rows")
        call = turn.api(tool=("load_rows", args))
        turn.tool(call, "load_rows", args, result)
        turn.api(text="done")
        turn.end("done")
        return turn

    def test_oversized_structured_arguments_blank_the_turn(self):
        big = rows(10_000)
        self.fire(self.tool_turn(big, "PRIVATE-NOTE-" + "x" * 40).events)
        events = self.sb.inbox()
        tool = next(event for event in events if event["event"] == "tool")
        self.assertTrue(tool["sensitive"])
        self.assertEqual(tool["sensitive_rule"], load_plugin().core.OVERSIZE_RULE)
        self.assertNotIn("tool_input", tool, "a record blanked as too large to check keeps no input text")
        spans = self.flushed()
        self.assertEqual(attrs(roots(spans)[0])["judgment.redacted_rule"], load_plugin().core.OVERSIZE_RULE)
        self.assertNotIn("PRIVATE-NOTE", self.sink.text())
        self.assertNotIn("r0000001", self.sink.text(), "the blanked turn sends none of its arguments")

    def test_deeply_nested_sensitive_path_blanks_the_turn(self):
        """Depth past the budget used to become "…", dropping the .env path that made the read
        sensitive while keeping its private output. Now it is an omission: the turn is blanked, with a reason."""
        args: dict = {"path": ".env"}
        for _ in range(20):
            args = {"nested": args}
        self.fire(self.tool_turn(args, "PRIVATE-NOTE-" + "x" * 40).events)
        tool = next(event for event in self.sb.inbox() if event["event"] == "tool")
        self.assertTrue(tool["sensitive"])
        self.assertEqual(tool["sensitive_rule"], load_plugin().core.OVERSIZE_RULE)
        self.assertNotIn("tool_input", tool)
        # The model's JSON-text copy of the arguments is parsed whole in the helper, so the turn names .env itself.
        self.assertIn(attrs(roots(self.flushed())[0])["judgment.redacted_rule"], {"dotenv", load_plugin().core.OVERSIZE_RULE})
        self.assertNotIn("PRIVATE-NOTE", self.sink.text())

    def test_structured_arguments_within_budget_arrive_whole(self):
        args = rows(1_000)
        self.fire(self.tool_turn(args).events)
        events = self.sb.inbox()
        self.assertEqual(next(event for event in events if event["event"] == "tool")["tool_input"], args)
        model = next(event for event in events if event["event"] == "model" and event.get("tool_calls"))
        self.assertEqual(model["tool_calls"][0]["arguments"], args, "JSON-text arguments are parsed in the helper")

    def test_snapshot_runs_outside_the_lock_and_only_when_traced(self):
        plugin = load_plugin()
        snapshot = module("snapshot")
        real = snapshot.take
        held = []

        def spy(value, *args):
            held.append(plugin._lock.locked())
            return real(value, *args)

        turn = self.tool_turn(rows(500))
        with mock.patch.object(snapshot, "take", spy):
            self.fire(turn.events)
        self.assertTrue(held)
        self.assertEqual(held, [False] * len(held), "a snapshot taken while the plugin's lock was held")
        held.clear()
        with mock.patch.object(snapshot, "take", spy):
            self.fire(self.tool_turn(rows(500)).events, JUDGMENT_TRACING="0")
        self.assertEqual(held, [], "an untraced call was snapshotted")


class Snapshot(unittest.TestCase):
    def setUp(self):
        self.snap = module("snapshot")

    def test_within_budget_is_a_copy_not_a_reference(self):
        args = rows(100)
        taken = self.snap.take(args)
        self.assertIsInstance(taken, bytes)
        args["rows"][0]["v"] = "changed after the callback"
        args["extra"] = 1
        self.assertEqual(marshal.loads(taken), rows(100))

    def test_budget_is_checked_while_walking_not_after(self):
        """Many small lists, each under budget, add up: the walk stops at the first member past it."""
        budget = self.snap.Budget(nodes=100, chars=10_000, depth=12)
        wide = [[n] for n in range(1000)]
        self.assertIsInstance(self.snap.take(wide, budget), self.snap.Omitted)
        self.assertIsInstance(self.snap.take({"a": "x" * 10_001}, budget), self.snap.Omitted)
        self.assertEqual(marshal.loads(self.snap.take({"a": "x" * 9_000}, budget)), {"a": "x" * 9_000})
        self.assertEqual(self.snap.take("y" * 10_001, budget).reason, "chars")

    def test_depth_is_a_budget_like_the_others(self):
        """At the limit: kept whole, by either path. One level past it: omitted whole, never cut to a stub."""
        budget = self.snap.Budget(nodes=100, chars=10_000, depth=3)

        def nest(levels: int, leaf: object) -> dict:
            value: object = leaf
            for _ in range(levels):
                value = {"d": value}
            return value

        for leaf in ({}, (1,)):  # (1,) takes the copy path
            self.assertIsInstance(self.snap.take(nest(3, leaf), budget), bytes)
            over = self.snap.take(nest(4, leaf), budget)
            self.assertIsInstance(over, self.snap.Omitted)
            self.assertEqual(over.reason, "depth")


if __name__ == "__main__":
    unittest.main()
