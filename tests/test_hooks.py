"""The real hook, with each agent's map, on realistic payloads. Nothing secret may reach the state dir."""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from payloads import adapter_turns, pi_turn
from support import HOOK, SECRETS, Sandbox, jq_sanitize, leaks, whole_leaks


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

    def test_contained_assignment_finds_fit_the_hook_budget(self):
        text = "\n".join(f"SERVICE_{i}_API_KEY={SECRETS['stripe_key']['core']}{i}" for i in range(100))
        payload = {"v": 1, "harness": "pi", "event": "tool", "session_id": "dense-assignments", "turn_id": "t",
                   "cwd": str(self.sb.project), "tool_name": "bash", "tool_call_id": "c",
                   "tool_input": {"command": "make"}, "tool_output": text}
        result = self.sb.hook("pi", "", payload)
        self.assertEqual(result.returncode, 0, result.stderr)
        [event] = self.sb.inbox()
        self.assertNotIn("redaction_error", event)
        self.assertEqual(event["tool_output"], "\n".join(f"SERVICE_{i}_API_KEY=[REDACTED:secret_assignment]" for i in range(100)))
        self.assertEqual(event["redactions"], [{"rule": "secret_assignment", "field": "tool_output", "count": 100}])

    def test_hook_matches_single_pass_redaction(self):
        cfg = {"enabled": True, "redactPatterns": ["(?<=ab)aba"]}
        self.sb.set_tracing(cfg)
        for value in ("ordinary build output", "", SECRETS["github_token"]["core"], "ababa",
                      {"password": SECRETS["github_token"]["core"], "pending": [True, "ordinary"]}):
            with self.subTest(value=value):
                payload = {"v": 1, "harness": "pi", "event": "tool", "session_id": "prepare-parity", "turn_id": "t",
                           "cwd": str(self.sb.project), "tool_name": "bash", "tool_call_id": "c", "ts": 12345,
                           "tool_input": {"command": "make"}, "tool_output": value}
                expected = jq_sanitize([payload], cfg)[0]
                before = len(self.sb.inbox())
                result = self.sb.hook("pi", "", payload)
                self.assertEqual(result.returncode, 0, result.stderr)
                [stored] = self.sb.inbox()[before:]
                self.assertEqual(stored["capture"], "sanitized")
                for field in ("tool_output", "redactions", "redaction_error", "ts"):
                    self.assertEqual({field: stored[field]} if field in stored else {},
                                     {field: expected[field]} if field in expected else {})

    def test_non_string_secret_members_are_masked_by_every_hook(self):
        marker = "[REDACTED:secret_field]"
        for adapter, event in (("claude", "PostToolUse"), ("codex", "PostToolUse"), ("cursor", "postToolUse"), ("pi", "")):
            with self.subTest(adapter=adapter):
                payload = {"v": 1, "harness": "pi", "event": "tool", "session_id": "typed", "conversation_id": "typed",
                           "turn_id": "t", "cwd": str(self.sb.project), "tool_name": "custom", "tool_use_id": "c", "tool_call_id": "c",
                           "tool_input": {"password": 12345678, "token": [SECRETS["github_token"]["core"]], "secret": {"value": "private"}},
                           "tool_response": "ok", "tool_output": "ok"}
                if adapter == "claude":
                    payload.pop("conversation_id")
                before = len(self.sb.inbox())
                result = self.sb.hook(adapter, event, payload)
                self.assertEqual(result.returncode, 0, result.stderr)
                [stored] = self.sb.inbox()[before:]
                self.assertNotIn("redaction_error", stored)
                self.assertEqual(stored["tool_input"], {"password": marker, "token": marker, "secret": marker})
                self.assertEqual(stored["redactions"], [{"rule": "secret_field", "field": "tool_input", "count": 3}])

    def test_non_string_member_budget_failure_drops_text(self):
        for value, error in (({"password": [], "token": {}}, False),
                             ({"items": [{"password": n} for n in range(129)]}, True)):
            with self.subTest(error=error):
                payload = {"v": 1, "harness": "pi", "event": "tool", "session_id": "typed-budget", "turn_id": "t",
                           "cwd": str(self.sb.project), "tool_name": "custom", "tool_input": value, "tool_output": "private output"}
                before = len(self.sb.inbox())
                result = self.sb.hook("pi", "", payload)
                self.assertEqual(result.returncode, 0, result.stderr)
                [stored] = self.sb.inbox()[before:]
                self.assertEqual(bool(stored.get("redaction_error")), error)
                if error:
                    self.assertNotIn("tool_input", stored)
                    self.assertNotIn("tool_output", stored)
                else:
                    self.assertEqual(stored["tool_input"], {"password": "[REDACTED:secret_field]", "token": "[REDACTED:secret_field]"})

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
        # The hook leaves the audit to the sender, which counts the placeholders.
        from judgment_core import redact

        rules = {item["rule"] for event in events for item in redact.census(event)}
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

    def test_invalid_custom_rules_fail_closed_without_a_command(self):
        for field in ("redactPatterns", "redactNames"):
            for prompt in ("private prompt marker", ""):
                with self.subTest(field=field, prompt=prompt):
                    self.sb.set_tracing({"enabled": True, field: ["("]})
                    before = len(self.sb.inbox())
                    result = self.sb.hook("codex", "UserPromptSubmit", {
                        "session_id": "invalid-custom", "turn_id": "t", "cwd": str(self.sb.project), "prompt": prompt})
                    self.assertEqual(result.returncode, 0, result.stderr)
                    [stored] = self.sb.inbox()[before:]
                    self.assertTrue(stored.get("redaction_error"))
                    self.assertNotIn("prompt", stored)

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


class NoJqFieldTests(unittest.TestCase):
    def test_pi_without_jq_writes_only_fields_never_redacted(self):
        """The hook's no-jq list of fields copied from a Pi payload: ids only, never a label the rules would redact."""
        import re

        from judgment_core import redact

        text = HOOK.read_text()
        fields = re.search(r"for field in ([a-z_ ]+); do\n    judgment_field \"\$payload\"", text).group(1).split()
        self.assertIn("tool_call_id", fields)
        for field in fields:
            cls = redact.field_class(field)
            self.assertTrue(cls["strip"] and not cls["redact"], field)


if __name__ == "__main__":
    unittest.main()
