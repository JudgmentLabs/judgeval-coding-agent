"""judgment.customer_user_id is resolved when the hook records an event and sent on every span of the session."""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HOOK = ROOT / "core" / "bin" / "judgment-hook"
CLAUDE_MAP = ROOT / "claude-code" / "hooks" / "map.jq"
os.environ["JUDGMENT_FLUSH_GRACE_MS"] = "0"
sys.path.insert(0, str(ROOT / "core"))

from fake_sink import attrs  # noqa: E402
from hermes_support import Case, load_plugin, new_session_id  # noqa: E402
from judgment_core import flush, redact  # noqa: E402
from unittest import mock  # noqa: E402

KEY = "judgment.customer_user_id"


class Collector(BaseHTTPRequestHandler):
    """Answers each OTLP post with the next queued status, 200 once the queue is empty."""

    statuses: list[int] = []
    bodies: list[dict] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        status = Collector.statuses.pop(0) if Collector.statuses else 200
        if status == 200:
            Collector.bodies.append(json.loads(gzip.decompress(body)))
        self.send_response(status)
        self.end_headers()

    def log_message(self, *args) -> None:
        pass


class CustomerUserIdTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = HTTPServer(("127.0.0.1", 0), Collector)
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()

    def setUp(self) -> None:
        Collector.statuses, Collector.bodies = [], []
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.home = self.tmp / "home"
        self.project = self.tmp / "project"
        self.config = self.tmp / "config"
        for path in (self.home, self.project / ".judgment", self.config):
            path.mkdir(parents=True)
        self.optin({"enabled": True})
        self.env = {
            "PATH": os.environ["PATH"],
            "HOME": str(self.home),
            "JUDGMENT_STATE_DIR": str(self.tmp / "state"),
            "JUDGMENT_CONFIG_DIR": str(self.config),
            "JUDGMENT_API_KEY": "test-key",
            "JUDGMENT_PROJECT_ID": "test-project",
            "JUDGMENT_ORG_ID": "test-org",
            "JUDGMENT_API_URL": f"http://127.0.0.1:{self.server.server_port}",
            "JUDGMENT_FLUSH": "0",
        }

    def optin(self, config: dict) -> None:
        (self.project / ".judgment" / "tracing.json").write_text(json.dumps(config))

    def hook(self, record: dict, env: dict | None = None) -> None:
        subprocess.run([str(HOOK), "--raw"], input=json.dumps(record), text=True, env=env or self.env, check=True)

    def pi_turn(self, session: str, turn: str, start: int, env: dict | None = None) -> None:
        """A Pi turn with a model call, a tool call, a thought, and a subagent with its own model and tool."""
        base = {"v": 1, "harness": "pi", "session_id": session, "turn_id": turn, "cwd": str(self.project)}
        records = [
            {"event": "turn_start", "prompt": "fix the bug"},
            {"event": "model", "model": "claude-sonnet-4-5", "output": "", "tool_calls": [{"id": "c1", "name": "read", "arguments": {}}]},
            {"event": "tool", "tool_name": "read", "tool_call_id": "c1", "tool_input": {}, "tool_output": "ok"},
            {"event": "thought", "output": "the bug is in main"},
            {"event": "model", "agent_id": "child", "model": "claude-sonnet-4-5", "prompt": "look", "output": "found it"},
            {"event": "tool", "agent_id": "child", "tool_name": "grep", "tool_call_id": "c2", "tool_input": {}, "tool_output": "main.py"},
            {"event": "subagent_end", "agent_id": "child", "agent_type": "scout", "prompt": "look", "output": "found it"},
            {"event": "turn_end", "output": "fixed"},
        ]
        for offset, record in enumerate(records):
            ts = start + offset * 10
            self.hook({**base, **record, "start_ts": ts, "end_ts": ts + 5, "ts": ts + 5}, env)

    def export(self) -> int:
        return flush.export_once(Path(self.env["JUDGMENT_STATE_DIR"]), self.env)

    def spans(self) -> dict[str, list[dict]]:
        """Every exported span's attributes, by session id."""
        found: dict[str, list[dict]] = {}
        for body in Collector.bodies:
            for span in body["resourceSpans"][0]["scopeSpans"][0]["spans"]:
                attrs = {item["key"]: next(iter(item["value"].values())) for item in span["attributes"]}
                attrs["name"] = span["name"]
                found.setdefault(attrs["judgment.session_id"], []).append(attrs)
        return found

    def assert_person(self, spans: list[dict], person: str | None) -> None:
        self.assertTrue(spans)
        for span in spans:
            self.assertEqual(span.get(KEY), person, span["name"])

    def test_every_span_of_the_session_carries_the_person(self) -> None:
        self.env["JUDGMENT_CUSTOMER_USER_ID"] = "alice@example.com"
        self.pi_turn("s1", "t1", 1_000)
        self.assertEqual(self.export(), 1)
        spans = self.spans()["s1"]
        kinds = {span.get("judgment.span_kind") for span in spans}
        self.assertTrue({"agent", "llm", "tool"} <= kinds, kinds)
        self.assertTrue(any(span.get("judgment.agent_id") == "child" for span in spans), "no subagent span")
        self.assertTrue(any("thought" in span["name"].lower() for span in spans), [span["name"] for span in spans])
        self.assert_person(spans, "alice@example.com")

    def test_concurrent_sessions_keep_their_own_person(self) -> None:
        runs = [
            threading.Thread(target=self.pi_turn, args=(f"s-{person}", "t1", 1_000, {**self.env, "JUDGMENT_CUSTOMER_USER_ID": person}))
            for person in ("alice", "bob")
        ]
        for run in runs:
            run.start()
        for run in runs:
            run.join()
        self.assertEqual(self.export(), 2)
        spans = self.spans()
        self.assert_person(spans["s-alice"], "alice")
        self.assert_person(spans["s-bob"], "bob")

    def test_a_queued_turn_keeps_the_person_it_was_recorded_with(self) -> None:
        credentials = self.config / "credentials"
        credentials.write_text("JUDGMENT_CUSTOMER_USER_ID=alice\n")
        self.pi_turn("s1", "t1", 1_000)
        Collector.statuses = [503] * flush.ATTEMPTS
        self.assertEqual(self.export(), -1)  # the send fails and the turn stays queued
        credentials.write_text("JUDGMENT_CUSTOMER_USER_ID=bob\n")
        self.pi_turn("s1", "t2", 2_000)  # the running session keeps its person
        self.pi_turn("s2", "t1", 3_000)  # a new session reads the new one
        self.assertEqual(self.export(), 3)
        spans = self.spans()
        self.assert_person(spans["s1"], "alice")
        self.assert_person(spans["s2"], "bob")

    def test_the_env_wins_over_the_credentials_file(self) -> None:
        (self.config / "credentials").write_text("JUDGMENT_CUSTOMER_USER_ID=from-file\n")
        self.env["JUDGMENT_CUSTOMER_USER_ID"] = "from-env"
        self.pi_turn("s1", "t1", 1_000)
        self.export()
        self.assert_person(self.spans()["s1"], "from-env")

    def test_git_email_is_the_default_and_metadata_capture_keeps_it(self) -> None:
        subprocess.run(["git", "config", "--global", "user.email", "carol@example.com"], env=self.env, check=True)
        self.optin({"enabled": True, "capture": "metadata"})
        payload = {"session_id": "s1", "prompt_id": "t1", "cwd": str(self.project), "transcript_path": ""}
        for event, extra in (
            ("UserPromptSubmit", {"prompt": "hi"}),
            ("PostToolUse", {"tool_name": "Bash", "tool_use_id": "c1", "tool_input": {"command": "ls"}, "tool_response": "a"}),
            ("Stop", {"last_assistant_message": "done"}),
        ):
            subprocess.run(
                [str(HOOK), "--map", str(CLAUDE_MAP), "--event", event, "--block", "claude"],
                input=json.dumps({**payload, **extra}),
                text=True,
                env=self.env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
        self.assertEqual(self.export(), 1)
        self.assert_person(self.spans()["s1"], "carol@example.com")

    def test_a_project_rule_that_matches_it_redacts_it(self) -> None:
        self.optin({"enabled": True, "redactPatterns": ["[a-z]+@example[.]com"]})
        self.env["JUDGMENT_CUSTOMER_USER_ID"] = "alice@example.com"
        self.pi_turn("s1", "t1", 1_000)
        self.export()
        self.assert_person(self.spans()["s1"], "[REDACTED:custom]")

    def inbox(self) -> list[dict]:
        return [json.loads(path.read_text()) for path in sorted((self.tmp / "state" / "inbox").glob("*.json"))]

    def state_text(self) -> str:
        return "\n".join(path.read_text(errors="replace") for path in (self.tmp / "state").rglob("*") if path.is_file())

    def test_an_event_whose_redaction_failed_carries_no_person(self) -> None:
        self.optin({"enabled": True, "redactPatterns": ["("]})  # an invalid regex: every event fails closed
        self.env["JUDGMENT_CUSTOMER_USER_ID"] = "alice@example.com"
        self.pi_turn("s1", "t1", 1_000)
        events = self.inbox()
        self.assertTrue(events)
        for event in events:
            self.assertTrue(event["redaction_error"])
            self.assertNotIn("customer_user_id", event)
            self.assertEqual(redact.outcome(event), "error")
        self.assertNotIn("alice@example.com", self.state_text())

    def test_pi_without_jq_carries_no_person(self) -> None:
        tools = self.tmp / "tools"
        tools.mkdir()
        for name in ("sh", "cat", "tr", "mkdir", "sha256sum", "python3", "rm", "git"):
            found = shutil.which(name)
            if found:
                os.symlink(found, tools / name)
        self.pi_turn("s1", "t1", 1_000, {**self.env, "PATH": str(tools), "JUDGMENT_CUSTOMER_USER_ID": "alice@example.com"})
        events = self.inbox()
        self.assertTrue(events)
        for event in events:
            self.assertTrue(event["redaction_error"])
            self.assertNotIn("customer_user_id", event)
            self.assertEqual(redact.outcome(event), "error")
        self.assertNotIn("alice@example.com", self.state_text())

    def test_a_redacted_person_never_reaches_the_disk(self) -> None:
        self.env["JUDGMENT_CUSTOMER_USER_ID"] = "alice@example.com"
        for capture in ("sanitized", "metadata", "full"):
            with self.subTest(capture=capture):
                self.optin({"enabled": True, "capture": capture, "redactPatterns": ["[a-z]+@example[.]com"]})
                session = f"s-{capture}"
                self.pi_turn(session, "t1", 1_000)
                inbox = self.inbox()
                self.assertTrue(inbox)
                self.assertEqual({event.get("customer_user_id") for event in inbox}, {"[REDACTED:custom]"})
                Collector.statuses = [503] * flush.ATTEMPTS
                self.assertEqual(self.export(), -1)  # the send fails, so the turn stays on disk under turns/
                self.assertTrue(list((self.tmp / "state" / "turns").rglob("*.jsonl")))
                self.assertNotIn("alice@example.com", self.state_text())
                Collector.statuses = []
                self.assertEqual(self.export(), 1)
                self.assert_person(self.spans()[session], "[REDACTED:custom]")

    def test_whitespace_around_the_env_value_is_trimmed_on_every_event(self) -> None:
        self.env["JUDGMENT_CUSTOMER_USER_ID"] = "  alice@example.com \t"
        self.pi_turn("s1", "t1", 1_000)
        self.assertEqual({event["customer_user_id"] for event in self.inbox()}, {"alice@example.com"})
        self.export()
        self.assert_person(self.spans()["s1"], "alice@example.com")

    def test_a_blank_env_value_falls_back_on_every_event(self) -> None:
        subprocess.run(["git", "config", "--global", "user.email", "carol@example.com"], env=self.env, check=True)
        self.env["JUDGMENT_CUSTOMER_USER_ID"] = "   "
        self.pi_turn("s1", "t1", 1_000)
        self.assertEqual({event["customer_user_id"] for event in self.inbox()}, {"carol@example.com"})
        self.export()
        self.assert_person(self.spans()["s1"], "carol@example.com")

    def test_a_blank_env_value_falls_back_to_the_credentials_file(self) -> None:
        (self.config / "credentials").write_text("JUDGMENT_CUSTOMER_USER_ID=from-file\n")
        self.env["JUDGMENT_CUSTOMER_USER_ID"] = "  "
        self.pi_turn("s1", "t1", 1_000)
        self.export()
        self.assert_person(self.spans()["s1"], "from-file")

    def test_no_person_sends_no_attribute(self) -> None:
        self.pi_turn("s1", "t1", 1_000)
        self.export()
        self.assert_person(self.spans()["s1"], None)


class HermesTest(Case):
    def assert_person(self, person: str) -> None:
        spans = self.flushed()
        self.assertTrue(spans)
        for span in spans:
            self.assertEqual(attrs(span).get(KEY), person, span["name"])

    def test_the_env_names_the_person(self) -> None:
        (self.sb.home / ".gitconfig").write_text("[user]\n\temail = carol@example.com\n")
        self.fire(self.simple(new_session_id()).events, JUDGMENT_CUSTOMER_USER_ID="alice@example.com")
        self.assert_person("alice@example.com")

    def test_git_email_is_the_default(self) -> None:
        (self.sb.home / ".gitconfig").write_text("[user]\n\temail = carol@example.com\n")
        self.fire(self.simple(new_session_id()).events)
        self.assert_person("carol@example.com")

    def test_git_config_global_comes_from_the_profile_env(self) -> None:
        core = load_plugin().core
        core._git_email.cache_clear()
        self.addCleanup(core._git_email.cache_clear)
        config = self.sb.home / "profile.gitconfig"
        config.write_text("[user]\n\temail = dave@example.com\n")
        (self.sb.home / ".gitconfig").write_text("[user]\n\temail = carol@example.com\n")
        env = {"HOME": str(self.sb.home), "JUDGMENT_CONFIG_DIR": str(self.sb.home / "no-config")}
        self.assertEqual(core.customer_user({**env, "GIT_CONFIG_GLOBAL": str(config)}), "dave@example.com")
        # A GIT_CONFIG_GLOBAL inherited by the process is not the profile's: it is not used.
        with mock.patch.dict(os.environ, {"GIT_CONFIG_GLOBAL": str(config)}):
            self.assertEqual(core.customer_user(env), "carol@example.com")

    def test_a_record_whose_redaction_failed_carries_no_person(self) -> None:
        core = load_plugin().core
        stamps = (("environment", "production"), ("project_root", "/p"), ("customer_user_id", "alice@example.com"))
        gate = core.Gate("sanitized", core.settings({}), (), "/nonexistent", stamps)
        record = {"v": 1, "harness": "hermes", "event": "turn_start", "session_id": "s", "turn_id": "t", "ts": 1, "prompt": "hi"}
        self.assertEqual(core.prepare(record, gate)["customer_user_id"], "alice@example.com")
        with mock.patch.object(core.redact, "sanitize_event", side_effect=RuntimeError("boom")):
            out = core.prepare(record, gate)
        self.assertTrue(out["redaction_error"])
        self.assertNotIn("customer_user_id", out)
        self.assertEqual(out["environment"], "production")
        self.assertEqual(core.redact.outcome(out), "error")


if __name__ == "__main__":
    unittest.main()
