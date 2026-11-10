"""The real hook, with each agent's map, on realistic payloads. Nothing secret may reach the state dir."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from payloads import adapter_turns, pi_turn
from support import SECRETS, Sandbox, leaks, whole_leaks


def run(sb: Sandbox, adapter: str, steps, env: dict | None = None):
    results = []
    for event, payload in steps:
        result = sb.hook(adapter, event, payload, env=env)
        results.append(result)
    return results


class HookTests(unittest.TestCase):
    def setUp(self):
        self.sb = Sandbox()

    def tearDown(self):
        self.sb.cleanup()

    def test_every_adapter_spools_no_secret(self):
        for adapter, steps in adapter_turns(str(self.sb.project)).items():
            with self.subTest(adapter=adapter):
                for result in run(self.sb, adapter, steps):
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertEqual(result.stderr, "")
        events = self.sb.inbox()
        self.assertEqual({event["harness"] for event in events}, {"claude-code", "codex", "cursor", "pi"})
        text = self.sb.state_text()
        self.assertEqual(whole_leaks(text), [])
        self.assertEqual(leaks(text, window=12, hashes=False), [])
        self.assertIn("[REDACTED:", text)
        for event in events:
            self.assertEqual(event["capture"], "sanitized")
            self.assertEqual(event["export_fp"] and len(event["export_fp"]), 32)
        rules = {item["rule"] for event in events for item in event.get("redactions", [])}
        for rule in ("openai_key", "github_token", "aws_access_key_id", "private_key", "jwt", "url_password", "secret_assignment", "env:JUDGMENT_API_KEY", "env:MY_CUSTOM_SERVICE_TOKEN"):
            self.assertIn(rule, rules)

    def test_ids_tokens_and_times_survive(self):
        steps = adapter_turns(str(self.sb.project))["pi"]
        run(self.sb, "pi", steps)
        events = self.sb.inbox()
        self.assertEqual([event["event"] for event in events], [payload["event"] for _, payload in steps])
        for (_, payload), event in zip(steps, events):
            for field in ("session_id", "turn_id", "ts", "start_ts", "end_ts", "usage", "tool_call_id", "model"):
                self.assertEqual(event.get(field), payload.get(field), field)

    def test_cursor_permission_answer_is_unchanged(self):
        steps = dict(adapter_turns(str(self.sb.project))["cursor"])
        for tracing in ({"enabled": True}, {"enabled": True, "redactPatterns": ["(bad"]}, None):
            with self.subTest(tracing=tracing):
                if tracing is None:
                    (self.sb.project / ".judgment" / "tracing.json").write_text("{not json")
                else:
                    self.sb.set_tracing(tracing)
                result = self.sb.hook("cursor", "preToolUse", steps["preToolUse"])
                self.assertEqual((result.returncode, result.stdout), (0, '{"permission":"allow"}\n'))

    def test_invalid_user_regex_drops_text_not_the_event(self):
        self.sb.set_tracing({"enabled": True, "redactPatterns": ["(unclosed"]})
        for adapter, steps in adapter_turns(str(self.sb.project)).items():
            for result in run(self.sb, adapter, steps):
                self.assertEqual(result.returncode, 0)
        events = self.sb.inbox()
        self.assertTrue(events)
        self.assertTrue(all(event.get("redaction_error") for event in events))
        self.assertEqual(whole_leaks(self.sb.state_text()), [])

    def test_sensitive_tool_call_is_marked_in_the_inbox(self):
        steps = adapter_turns(str(self.sb.project))["claude"]
        event, payload = steps[1]
        payload = {**payload, "tool_name": "Read", "tool_input": {"file_path": str(self.sb.project / ".env")}, "tool_response": {"file": {"content": "A=1"}}}
        self.assertEqual(self.sb.hook("claude", event, payload).returncode, 0)
        [stored] = self.sb.inbox()
        self.assertEqual((stored["sensitive"], stored["sensitive_rule"], stored["sensitive_trigger"]), (True, "dotenv", "~/proj/.env"))
        self.assertNotIn("tool_output", stored)

    def test_pi_raw_full_capture_keeps_raw_text(self):
        self.sb.set_tracing({"enabled": True, "capture": "full"})
        steps = pi_turn(str(self.sb.project), "pi-raw", "pi-raw:1", int(time.time() * 1000))
        run(self.sb, "pi", steps)
        events = self.sb.inbox()
        self.assertTrue(all(event["capture"] == "full" for event in events))
        self.assertIn(SECRETS["aws_access_key_id"]["core"], events[0]["prompt"])

    def test_pi_raw_without_jq_records_no_text(self):
        tools = Path(tempfile.mkdtemp(prefix="jt-nojq-"))
        try:
            for name in ("sh", "cat", "tr", "mkdir", "sha256sum", "python3", "setsid", "rm"):
                found = shutil.which(name)
                if found:
                    os.symlink(found, tools / name)
            steps = pi_turn(str(self.sb.project), "pi-nojq", "pi-nojq:1", 1794000000000)
            for _, payload in steps:
                result = self.sb.hook("pi", "", payload, env={"PATH": str(tools)})
                self.assertEqual(result.returncode, 0, result.stderr)
        finally:
            shutil.rmtree(tools)
        events = self.sb.inbox()
        self.assertEqual([event["event"] for event in events], [payload["event"] for _, payload in steps])
        for (_, payload), event in zip(steps, events):
            self.assertTrue(event["redaction_error"])
            self.assertEqual((event["session_id"], event["turn_id"], event["ts"]), (payload["session_id"], payload["turn_id"], payload["ts"]))
            for field in ("prompt", "output", "tool_input", "tool_output", "tool_calls"):
                self.assertNotIn(field, event)
        self.assertEqual(whole_leaks(self.sb.state_text()), [])

    def test_hook_never_sees_the_key_on_argv(self):
        text = Path(__file__).resolve().parent.parent.joinpath("core", "bin", "judgment-hook").read_text()
        self.assertNotIn("--arg key", text)
        self.assertNotIn('--arg api', text)


if __name__ == "__main__":
    unittest.main()
