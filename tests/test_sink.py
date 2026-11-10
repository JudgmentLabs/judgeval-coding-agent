"""Hook -> inbox -> flusher -> a fake OTLP sink. What reaches the sink, and what stays on disk."""

from __future__ import annotations

import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import unittest

from fake_sink import Sink, attrs, roots
from payloads import claude_turn, pi_turn
from support import HOOK, SECRETS, Sandbox, claude_transcript, leaks, secret_blob, wait_for, whole_leaks

ENV_FILE = "DATABASE_URL=postgres://app:" + "Lq" + "9xT" * 4 + "@db/app\nFEATURE_FLAG=on\n"
TIME_KEYS = ("startTimeUnixNano", "endTimeUnixNano")
REDACTION_KEYS = ("judgment.redacted", "judgment.redaction_count", "judgment.redactions", "judgment.redacted_reason",
                  "judgment.redacted_rule", "judgment.redacted_trigger", "judgment.redaction_error")
TOKEN_KEYS = ("judgment.usage.non_cached_input_tokens", "judgment.usage.output_tokens", "judgment.usage.cache_read_input_tokens",
              "judgment.usage.total_cost_usd", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens")


def claude_session(sb: Sandbox, start_ms: int) -> None:
    """Three turns in one session: one with secrets, one that reads .env, one plain."""
    transcript = sb.home / "session.jsonl"
    cwd = str(sb.project)
    blob = secret_blob()
    answer_a = "Deployed. The token " + SECRETS["github_token"]["core"] + " still works."
    claude_transcript(transcript, "cc-1", "pa", "deploy with key " + SECRETS["openai_key"]["core"], start_ms, "pa-tool", "./deploy.sh", blob, answer_a)
    env_path = str(sb.project / ".env")
    claude_transcript(transcript, "cc-1", "pb", "what is in the env file", start_ms + 10_000, "pb-tool", "cat .env", ENV_FILE, "It sets DATABASE_URL and FEATURE_FLAG.")
    claude_transcript(transcript, "cc-1", "pc", "now run the tests", start_ms + 20_000, "pc-tool", "make test", "42 passed", "All 42 tests pass.")
    for event, payload in claude_turn(cwd, "cc-1", "pa", transcript=str(transcript), command="./deploy.sh", output=blob, answer=answer_a):
        assert sb.hook("claude", event, payload).returncode == 0
    steps = claude_turn(cwd, "cc-1", "pb", transcript=str(transcript), prompt="what is in the env file", answer="It sets DATABASE_URL and FEATURE_FLAG.")
    for event, payload in steps:
        if event == "PostToolUse":
            payload = {**payload, "tool_name": "Read", "tool_use_id": "pb-tool", "tool_input": {"file_path": env_path}, "tool_response": {"file": {"content": ENV_FILE}}}
        if event == "PostToolUseFailure":
            continue
        assert sb.hook("claude", event, payload).returncode == 0
    steps = claude_turn(cwd, "cc-1", "pc", transcript=str(transcript), prompt="now run the tests", command="make test", output="42 passed", answer="All 42 tests pass.")
    for event, payload in steps:
        if event != "PostToolUseFailure":
            assert sb.hook("claude", event, payload).returncode == 0


def by_turn(spans: list[dict]) -> dict[str, list[dict]]:
    found: dict[str, list[dict]] = {}
    for span in spans:
        found.setdefault(span["traceId"], []).append(span)
    return {next(attrs(s)["judgment.input"] for s in group if not s.get("parentSpanId")) or trace: group for trace, group in found.items()}


def structure(spans: list[dict], *, times: bool = False) -> list:
    """Span ids, parents, names, and token attributes. Text and redaction attributes are left out."""
    out = []
    for span in sorted(spans, key=lambda item: item["spanId"]):
        a = attrs(span)
        out.append((span["traceId"], span["spanId"], span.get("parentSpanId"), span["name"], tuple((key, a.get(key)) for key in TOKEN_KEYS),
                    tuple(span[key] for key in TIME_KEYS) if times else ()))
    return out


def bodies_without_redaction(spans: list[dict]) -> list[dict]:
    out = []
    for span in sorted(spans, key=lambda item: item["spanId"]):
        span = json.loads(json.dumps(span))
        span["attributes"] = [item for item in span["attributes"] if item["key"] not in REDACTION_KEYS]
        out.append(span)
    return out


class SinkTests(unittest.TestCase):
    def run_session(self, tracing: dict) -> tuple[Sandbox, list[dict], str]:
        with Sink() as sink:
            sb = Sandbox(tracing, sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            claude_session(sb, int(time.time() * 1000) - 60_000)
            result = sb.flush()
            self.assertEqual(result.returncode, 0, result.stderr)
            return sb, sink.spans(), sink.text()

    def test_sanitized_session(self):
        sb, spans, text = self.run_session({"enabled": True})
        self.assertEqual(whole_leaks(text), [])
        self.assertEqual(leaks(text, window=12, hashes=False), [])
        self.assertNotIn("Lq9xT", text)
        self.assertEqual(whole_leaks(sb.state_text()), [])
        self.assertNotIn("Lq9xT", sb.state_text())
        turns = by_turn(spans)
        self.assertEqual(len(turns), 3)
        a = next(group for key, group in turns.items() if key.startswith("deploy with key"))
        root_a = attrs(roots(a)[0])
        self.assertIn("[REDACTED:openai_key]", root_a["judgment.input"])
        self.assertTrue(root_a["judgment.redacted"])
        self.assertGreater(int(root_a["judgment.redaction_count"]), 15)
        audit = json.loads(root_a["judgment.redactions"])
        self.assertIn(("openai_key", "prompt"), {(item["rule"], item["field"]) for item in audit})
        self.assertEqual(leaks(json.dumps(audit)), [])
        self.assertTrue(any("judgment.redaction_count" in attrs(span) for span in a if span.get("parentSpanId")))

        blanked = [group for group in turns.values() if attrs(roots(group)[0]).get("judgment.redacted_reason")]
        self.assertEqual(len(blanked), 1)
        root_b = attrs(roots(blanked[0])[0])
        self.assertEqual((root_b["judgment.redacted_reason"], root_b["judgment.redacted_rule"], root_b["judgment.redacted_trigger"]), ("sensitive_path", "dotenv", "~/proj/.env"))
        self.assertTrue(root_b["judgment.redacted"])
        for span in blanked[0]:
            a_ = attrs(span)
            self.assertEqual((a_.get("judgment.input", ""), a_.get("judgment.output", "")), ("", ""), span["name"])
        self.assertTrue(any(attrs(span).get("judgment.usage.output_tokens") for span in blanked[0]))

        c = turns["now run the tests"]
        root_c = attrs(roots(c)[0])
        self.assertNotIn("judgment.redacted_reason", root_c)
        self.assertEqual((root_c["judgment.redacted"], root_c["judgment.redaction_count"]), (False, "0"))
        self.assertEqual(root_c["judgment.output"], "All 42 tests pass.")
        history = next(attrs(span)["judgment.input"] for span in c if attrs(span).get("judgment.span_kind") == "llm")
        self.assertIn("[REDACTED:openai_key]", history)  # turn A rides along, redacted
        self.assertNotIn("what is in the env file", history)  # the blanked turn does not
        self.assertNotIn("DATABASE_URL", history)

        index = json.loads((sb.state / "index.json").read_text())
        flagged = [item for item in index.values() if item.get("blanked")]
        self.assertEqual(len(flagged), 1)
        self.assertIsNone(flagged[0]["summary"])
        stored = (sb.state / "turns" / "claude-code" / "cc-1" / "pb.jsonl").read_text()
        self.assertNotIn("what is in the env file", stored)
        self.assertNotIn("DATABASE_URL", stored)

    def test_structure_matches_raw(self):
        _, sanitized, _ = self.run_session({"enabled": True})
        _, raw, raw_text = self.run_session({"enabled": True, "redact": False})
        self.assertIn(SECRETS["openai_key"]["core"], raw_text)  # raw really is raw
        self.assertEqual(len(sanitized), len(raw))
        self.assertEqual(structure(sanitized), structure(raw))

    def test_secret_free_turn_is_identical_to_raw(self):
        def run(tracing):
            with Sink() as sink:
                sb = Sandbox(tracing, sink_url=sink.url)
                self.addCleanup(sb.cleanup)
                steps = pi_turn(str(sb.project), "pi-same", "pi-same:1", 1794000000000, prompt="add a test for the parser",
                                command="pytest -q tests/test_parser.py", output="3 passed in 0.12s", answer="Added test_parser_handles_empty_input.")
                for _, payload in steps:
                    self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
                self.assertEqual(sb.flush().returncode, 0)
                return sink.spans()

        sanitized, raw = run({"enabled": True}), run({"enabled": True, "capture": "full"})
        self.assertTrue(sanitized)
        self.assertEqual(bodies_without_redaction(sanitized), bodies_without_redaction(raw))
        self.assertEqual(attrs(roots(sanitized)[0])["judgment.redaction_count"], "0")
        self.assertNotIn("judgment.redacted", attrs(roots(raw)[0]))

    def test_inbox_from_an_older_build_is_redacted(self):
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            stamp = {"export_fp": hashlib.sha256(SECRETS["judgment_key"]["core"].encode()).hexdigest()[:32], "export_project": "proj-test-1",
                     "export_url": sink.url, "export_org": "org-test-1", "project_root": str(sb.project), "environment": "production"}
            (sb.state / "inbox").mkdir(parents=True)
            for number, (_, payload) in enumerate(pi_turn(str(sb.project), "pi-old", "pi-old:2", int(time.time() * 1000) - 5000)):
                (sb.state / "inbox" / f"1.{number}.json").write_text(json.dumps({**payload, **stamp}))
                os.utime(sb.state / "inbox" / f"1.{number}.json", ns=(time.time_ns() + number * 1000, time.time_ns() + number * 1000))
            # An older index holds an earlier turn's summary in plain text.
            dest = f"{sink.url} org-test-1 proj-test-1"
            (sb.state / "index.json").write_text(json.dumps({"pi\tpi-old\tpi-old:1": {
                "first": 1, "last": 1, "update_id": 0, "ready": False, "dest": dest,
                "summary": {"turn_id": "pi-old:1", "start": 1, "prompt": "use " + SECRETS["stripe_key"]["core"], "answer": "ok"}}}))
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            self.assertEqual(whole_leaks(sink.text()), [])
            self.assertIn("[REDACTED:stripe_key]", sink.text())
            turns = (sb.state / "turns").rglob("*.jsonl")
            self.assertEqual(whole_leaks("".join(path.read_text() for path in turns)), [])


class OptOutTests(unittest.TestCase):
    def assert_quiet(self, sb: Sandbox, sink: Sink, env: dict | None = None):
        env = {"JUDGMENT_FLUSH": "", **(env or {})}
        for event, payload in claude_turn(str(sb.project), "q-1", "q1"):
            result = sb.hook("claude", event, payload, env=env)
            self.assertEqual(result.returncode, 0)
        time.sleep(1.5)
        sb.flush(env)
        self.assertEqual(sink.requests, [])
        self.assertEqual(sb.inbox(), [])
        self.assertFalse((sb.state / "turns").exists() and any((sb.state / "turns").rglob("*.jsonl")))

    def test_no_opt_in_file(self):
        with Sink() as sink:
            sb = Sandbox(None, sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            self.assert_quiet(sb, sink)

    def test_tracing_forced_off(self):
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            self.assert_quiet(sb, sink, {"JUDGMENT_TRACING": "0"})

    def test_session_turned_off(self):
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            command = {"session_id": "q-1", "cwd": str(sb.project), "prompt": "/judgment:off", "prompt_id": "q0"}
            result = sb.hook("claude", "UserPromptSubmit", command)
            self.assertIn('"decision":"block"', result.stdout)
            self.assertIn("off for this session", result.stdout)
            self.assert_quiet(sb, sink)


class BackgroundTests(unittest.TestCase):
    def test_sender_outlives_the_agent(self):
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            steps = pi_turn(str(sb.project), "bg-1", "bg-1:1", int(time.time() * 1000) - 5000)
            for _, payload in steps[:-1]:
                sb.hook("pi", "", payload)
            stop = sb.dir / "stop.json"
            stop.write_text(json.dumps(steps[-1][1]))
            env = {**sb.env, "JUDGMENT_FLUSH": "", "PYTHONPATH": ""}
            # The agent runs the hook, then its whole process group is killed.
            parent = subprocess.Popen(["sh", "-c", f'"{HOOK}" --raw < "{stop}"; kill -9 0'], env=env, start_new_session=True)
            parent.wait(timeout=10)
            self.assertEqual(parent.returncode, -signal.SIGKILL)
            self.assertTrue(wait_for(lambda: sink.requests, timeout=20), "the detached sender never posted")
            self.assertEqual(whole_leaks(sink.text()), [])

    def test_hook_returns_quickly_with_the_sink_down(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        sb = Sandbox(sink_url=f"http://127.0.0.1:{port}")
        self.addCleanup(sb.cleanup)
        steps = pi_turn(str(sb.project), "down-1", "down-1:1", int(time.time() * 1000) - 5000)
        started = time.monotonic()
        for _, payload in steps:
            result = sb.hook("pi", "", payload, env={"JUDGMENT_FLUSH": ""})
            self.assertEqual(result.returncode, 0)
        self.assertLess(time.monotonic() - started, 3.0)
        self.assertTrue(wait_for(lambda: (sb.state / "flush.log").is_file() and "post attempt" in (sb.state / "flush.log").read_text(), timeout=20))
        self.assertEqual(whole_leaks((sb.state / "flush.log").read_text()), [])


if __name__ == "__main__":
    unittest.main()
