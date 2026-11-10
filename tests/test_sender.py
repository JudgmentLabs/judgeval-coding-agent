"""The sender and the hook as a system: what the regression table's rows cannot say.

Rows in tests/regressions.json cover what each rule replaces or blanks. These cover what happens around the rules: a
turn made sensitive after it was sent, the sender surviving the agent, earlier turns' text as history, text read back
from a transcript when the hook and the sender hold different environments, and labels as span names.

Every fake secret is built at runtime from parts, so no file holds one.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path

import random
import string

from fake_sink import Sink, attrs, roots
from payloads import pi_turn
from support import (CORE, ENV_SECRET_NAME, HOOK, SECRETS, Sandbox, assert_parity, both, claude_transcript, clean_env, filler, inject,
                     leaked, norm, pieces, stamp, tool, wait_for, whole_leaks)

from judgment_core import flush, redact

_rng = random.Random(2020)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


class LabelTests(unittest.TestCase):
    """Finding 3: agent_name was exempt as structure, though Codex fills it from transcript task names."""

    def test_labels_are_redacted_in_both_layers(self):
        secret = SECRETS["github_token"]["core"]
        event = {"v": 1, "event": "subagent_start", "session_id": "s", "turn_id": "t", "agent_id": "a1", "agent_name": "task " + secret,
                 "agent_type": "worker " + secret, "tool_name": "mcp__x__" + secret}
        py, js = both([event])
        self.assertEqual(norm(py), norm(js))
        self.assertEqual(whole_leaks(json.dumps(py)), [])
        self.assertEqual(py[0]["agent_type"], "worker [REDACTED:github_token]")

    def test_label_never_becomes_a_span_name(self):
        secret = SECRETS["github_token"]["core"]
        for blanked in (False, True):
            with self.subTest(blanked=blanked), Sink() as sink:
                sb = Sandbox(sink_url=sink.url)
                self.addCleanup(sb.cleanup)
                t0 = int(time.time() * 1000) - 5000
                base = {"v": 1, "harness": "pi", "session_id": "lb-1", "turn_id": "lb-1:1", "cwd": str(sb.project), **stamp(sb, sink.url)}
                tool_input = {"file_path": str(sb.project / ".env")} if blanked else {"command": "make"}
                inject(sb, [
                    {**base, "event": "turn_start", "prompt": "split the work", "ts": t0},
                    {**base, "event": "subagent_start", "agent_id": "a1", "agent_type": "worker", "agent_name": "Rotate " + secret, "ts": t0 + 10},
                    {**base, "event": "tool", "tool_name": "read", "tool_call_id": "c1", "tool_input": tool_input, "tool_output": "A=1", "ts": t0 + 20},
                    {**base, "event": "subagent_end", "agent_id": "a1", "agent_type": "worker", "agent_name": "Rotate " + secret, "output": "done", "ts": t0 + 30},
                    {**base, "event": "turn_end", "output": "ok", "ts": t0 + 40},
                ])
                self.assertEqual(sb.flush().returncode, 0)
                self.assertTrue(sink.requests)
                self.assertEqual(whole_leaks(sink.text() + sb.state_text()), [])
                names = [span["name"] for span in sink.spans()]
                if blanked:
                    self.assertFalse([name for name in names if "Rotate" in name], names)
                else:
                    self.assertIn("Rotate [REDACTED:github_token]", names)


class DetectBeforeRedactTests(unittest.TestCase):
    """Finding 6: the sender redacted first, so .env.<token> became .env.[REDACTED:...] and was not seen as sensitive."""

    def test_transcript_only_read_blanks_the_turn(self):
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            token = SECRETS["github_token"]["core"]
            leftover = "Lv" + _rand(18)  # matches no rule: only blanking keeps it out
            transcript = sb.home / "t.jsonl"
            start = int(time.time() * 1000) - 60_000
            claude_transcript(transcript, "dr-1", "p1", "show me the file", start, "p1-tool", "cat .env." + token, "VALUE=" + leftover, "It holds VALUE.")
            base = {"session_id": "dr-1", "cwd": str(sb.project), "transcript_path": str(transcript), "prompt_id": "p1"}
            for event, payload in (("UserPromptSubmit", {**base, "prompt": "show me the file"}), ("Stop", {**base, "last_assistant_message": "It holds VALUE."})):
                self.assertEqual(sb.hook("claude", event, payload).returncode, 0)
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            text = sink.text()
            self.assertNotIn(leftover, text)
            self.assertNotIn(token, text)
            root = attrs(next(span for span in sink.spans() if not span.get("parentSpanId")))
            self.assertEqual((root["judgment.redacted_reason"], root["judgment.redacted_rule"]), ("sensitive_path", "dotenv"))
            self.assertEqual(root["judgment.redacted_trigger"], ".env.[REDACTED:github_token]")

    def test_placeholder_in_a_dotenv_name_is_still_sensitive(self):
        hit = redact.sensitive(tool({"command": "cat .env.[REDACTED:github_token]"}), redact.ruleset())
        self.assertEqual(hit and hit["rule"], "dotenv")


class RuleErrorHistoryTests(unittest.TestCase):
    """Finding 7: an invalid custom rule dropped the event's text but sent earlier turns' summaries."""

    def test_history_is_dropped_when_rules_fail(self):
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            marker = "Hm" + _rand(18)
            t0 = int(time.time() * 1000) - 60_000
            first = pi_turn(str(sb.project), "he-1", "he-1:1", t0, prompt="remember " + marker, answer="noted " + marker)
            for _, payload in first:
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            self.assertEqual(sb.flush().returncode, 0)
            sent = len(sink.requests)
            self.assertIn(marker, sink.text())  # sanitized: the marker is no secret, so the first turn holds it
            sb.set_tracing({"enabled": True, "redactPatterns": ["(unclosed"]})
            for _, payload in pi_turn(str(sb.project), "he-1", "he-1:2", t0 + 10_000, prompt="next step", answer="done"):
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            self.assertEqual(sb.flush().returncode, 0)
            later = "".join(request["body"] for request in sink.requests[sent:])
            self.assertTrue(later)
            self.assertNotIn(marker, later)
            self.assertIn("judgment.redaction_error", later)

    def test_redact_text_is_empty_on_error(self):
        broken = redact.ruleset({"redactPatterns": ["(unclosed"]})
        self.assertTrue(broken.error)
        self.assertEqual(redact.redact_text("plain words", broken, []), ("", []))


class LateSensitivityTests(unittest.TestCase):
    """Finding 5: a sent turn that a late child made sensitive was blanked only when resent, after waiting for the child.

    Meanwhile the next turn carried its old summary as context, and a failed resend left that summary in index.json.
    """

    def run_turns(self, sink_status: int = 200):
        sink = Sink(status=200)
        sink.__enter__()
        self.addCleanup(sink.__exit__, None, None, None)
        sb = Sandbox(sink_url=sink.url)
        self.addCleanup(sb.cleanup)
        marker = "Codeword" + _rand(12)
        t0 = int(time.time() * 1000) - 60_000
        for _, payload in pi_turn(str(sb.project), "lc-1", "lc-1:1", t0, prompt="remember " + marker, answer="noted " + marker):
            self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
        self.assertEqual(sb.flush().returncode, 0)
        self.assertIn(marker, sink.text())  # sent once, with its summary
        base = {"v": 1, "harness": "pi", "session_id": "lc-1", "turn_id": "lc-1:1", "cwd": str(sb.project)}
        # A child of the first turn starts late, and reads .env. It is still running.
        for payload in ({**base, "event": "subagent_start", "agent_id": "a1", "agent_type": "worker", "ts": t0 + 5000},
                        {**base, "event": "tool", "agent_id": "a1", "tool_name": "read", "tool_call_id": "c9",
                         "tool_input": {"file_path": str(sb.project / ".env")}, "tool_output": "A=1", "ts": t0 + 5100}):
            self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
        return sink, sb, marker, t0

    def test_next_turn_never_carries_the_old_summary(self):
        sink, sb, marker, t0 = self.run_turns()
        sent = len(sink.requests)
        for _, payload in pi_turn(str(sb.project), "lc-1", "lc-1:2", t0 + 20_000, prompt="next step", answer="done"):
            self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
        self.assertEqual(sb.flush().returncode, 0)
        later = "".join(request["body"] for request in sink.requests[sent:])
        self.assertIn("next step", later)  # the second turn went out while the child still runs
        self.assertNotIn(marker, later)
        index = json.loads((sb.state / "index.json").read_text())
        first = index["pi\tlc-1\tlc-1:1"]
        self.assertEqual((first["blanked"]["rule"], first["summary"]), ("dotenv", None))
        from judgment_core.flush import _turn_file

        self.assertNotIn(marker, _turn_file(sb.state, "pi\tlc-1\tlc-1:1").read_text())

    def test_failed_resend_leaves_no_old_summary(self):
        sink, sb, marker, _ = self.run_turns()
        sink.status = 500
        base = {"v": 1, "harness": "pi", "session_id": "lc-1", "turn_id": "lc-1:1", "cwd": str(sb.project)}
        self.assertEqual(sb.hook("pi", "", {**base, "event": "subagent_end", "agent_id": "a1", "output": "ok", "ts": int(time.time() * 1000)}).returncode, 0)
        sb.flush()
        index_text = (sb.state / "index.json").read_text()
        self.assertNotIn(marker, index_text)
        first = json.loads(index_text)["pi\tlc-1\tlc-1:1"]
        self.assertTrue(first["blanked"])
        self.assertIsNone(first["summary"])
        self.assertNotIn(marker, sb.state_text())

    def test_blank_found_only_in_the_transcript_is_settled_before_the_post(self):
        """The sender finds the .env read in the transcript at resend. The post fails: the summary is gone anyway."""
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            marker = "Codeword" + _rand(12)
            transcript = sb.home / "t.jsonl"
            start = int(time.time() * 1000) - 120_000
            claude_transcript(transcript, "tr-1", "p1", "remember " + marker, start, "p1-tool", "make test", "ok", "noted " + marker)
            base = {"session_id": "tr-1", "cwd": str(sb.project), "transcript_path": str(transcript), "prompt_id": "p1"}
            for event, payload in (("UserPromptSubmit", {**base, "prompt": "remember " + marker}), ("Stop", {**base, "last_assistant_message": "noted " + marker})):
                self.assertEqual(sb.hook("claude", event, payload).returncode, 0)
            self.assertEqual(sb.flush().returncode, 0)
            self.assertIn(marker, sink.text())
            # The same turn reads .env later, seen only in its transcript; a late hook event sends it again.
            claude_transcript(transcript, "tr-1", "p1", "remember " + marker, start + 5000, "p1-env", "cat .env", "A=1", "noted " + marker)
            sink.status = 500
            late = {**base, "tool_name": "Bash", "tool_use_id": "p1-late", "tool_input": {"command": "true"}, "tool_response": {"stdout": ""}}
            self.assertEqual(sb.hook("claude", "PostToolUse", late).returncode, 0)
            sb.flush()
            item = json.loads((sb.state / "index.json").read_text())["claude-code\ttr-1\tp1"]
            self.assertEqual((item["blanked"]["rule"], item["summary"]), ("dotenv", None))


class DetachTests(unittest.TestCase):
    """Finding 8: the hook returned without knowing the sender had left its process group.

    The agent is played by sh: run the hook, then kill the whole process group the moment it returns.
    """

    def tools(self, *, setsid: Path | None = None, bash: bool = True) -> str:
        folder = Path(tempfile.mkdtemp(prefix="jt-tools-"))
        self.addCleanup(shutil.rmtree, folder, True)
        names = ["sh", "dash", "cat", "tr", "mkdir", "rm", "sha256sum", "jq", "python3", "kill", "sleep"] + (["bash"] if bash else [])
        for name in names:
            if shutil.which(name):
                os.symlink(shutil.which(name), folder / name)
        if setsid is not None:
            os.symlink(setsid, folder / "setsid")
        return str(folder)

    def slow_setsid(self) -> Path:
        """setsid(1) that takes 200 ms to start: the review's delayed startup."""
        folder = Path(tempfile.mkdtemp(prefix="jt-slow-"))
        self.addCleanup(shutil.rmtree, folder, True)
        script = folder / "setsid"
        script.write_text(f"#!{shutil.which('dash') or '/bin/sh'}\nsleep 0.2\nexec {shutil.which('setsid')} \"$@\"\n")
        script.chmod(0o755)
        return script

    def run_and_kill(self, shell: str, path: str) -> float:
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            steps = pi_turn(str(sb.project), "dt-1", "dt-1:1", int(time.time() * 1000) - 5000)
            for _, payload in steps[:-1]:
                sb.hook("pi", "", payload)
            stop = sb.dir / "stop.json"
            stop.write_text(json.dumps(steps[-1][1]))
            env = {**sb.env, "JUDGMENT_FLUSH": "", "PYTHONPATH": "", "PATH": path}
            started = time.monotonic()
            parent = subprocess.Popen([shutil.which("sh"), "-c", f'{shell} "{HOOK}" --raw < "{stop}"; kill -9 0'], env=env, start_new_session=True)
            parent.wait(timeout=10)
            elapsed = time.monotonic() - started
            self.assertEqual(parent.returncode, -signal.SIGKILL)
            self.assertTrue(wait_for(lambda: sink.requests, timeout=20), "the sender died with the agent's process group")
            return elapsed

    def require(self, *names: str) -> None:
        for name in names:
            if not shutil.which(name):
                self.skipTest(f"no {name} here")

    def test_dash_with_setsid(self):
        self.require("dash", "setsid")
        self.run_and_kill(shutil.which("dash"), self.tools(setsid=Path(shutil.which("setsid"))))

    def test_dash_with_a_slow_setsid_falls_back_to_bash(self):
        self.require("dash", "setsid", "bash")
        elapsed = self.run_and_kill(shutil.which("dash"), self.tools(setsid=self.slow_setsid()))
        self.assertLess(elapsed, 0.2 + 1.0, "the hook waited for the slow setsid instead of its bound")

    def test_dash_without_setsid(self):
        self.require("dash")
        self.run_and_kill(shutil.which("dash"), self.tools(bash=False))

    def test_bash_without_setsid(self):
        self.require("bash")
        self.run_and_kill(shutil.which("bash"), self.tools(bash=True))

    def test_hook_waits_no_longer_than_its_bound(self):
        """With a slow setsid and no bash, the hook returns after its bound (about 40 ms), not after the launcher."""
        self.require("dash", "setsid")
        sb = Sandbox()
        self.addCleanup(sb.cleanup)
        steps = pi_turn(str(sb.project), "bd-1", "bd-1:1", int(time.time() * 1000) - 5000)
        env = {"JUDGMENT_FLUSH": "", "PYTHONPATH": "", "PATH": self.tools(setsid=self.slow_setsid(), bash=False)}
        best = None
        for _ in range(3):
            started = time.monotonic()
            subprocess.run([shutil.which("dash"), str(HOOK), "--raw"], input=json.dumps(steps[-1][1]), text=True, env={**sb.env, **env}, timeout=30)
            elapsed = time.monotonic() - started
            best = elapsed if best is None else min(best, elapsed)
        self.assertLess(best, 0.15)


class HistoryAuditTests(unittest.TestCase):
    """Finding 9: what was redacted in earlier turns' text while building a request was missing from the root's audit."""

    def test_history_redactions_are_on_the_root(self):
        token = SECRETS["github_token"]["core"]
        with Sink() as sink:
            sb = Sandbox({"enabled": True, "capture": "full"}, sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            t0 = int(time.time() * 1000) - 60_000
            for _, payload in pi_turn(str(sb.project), "ha-1", "ha-1:1", t0, prompt="use " + token, answer="ok", output="plain", command="ls"):
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            self.assertEqual(sb.flush().returncode, 0)
            sent = len(sink.requests)
            sb.set_tracing({"enabled": True})
            for _, payload in pi_turn(str(sb.project), "ha-1", "ha-1:2", t0 + 10_000, prompt="next", answer="done", output="plain", command="ls"):
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            self.assertEqual(sb.flush().returncode, 0)
            later = "".join(request["body"] for request in sink.requests[sent:])
            self.assertNotIn(token, later)
            self.assertIn("[REDACTED:github_token]", later)
            spans = [span for request in sink.requests[sent:] for resource in json.loads(request["body"])["resourceSpans"]
                     for scope in resource["scopeSpans"] for span in scope["spans"]]
            root = attrs(roots(spans)[0])
            self.assertTrue(root["judgment.redacted"])
            self.assertEqual(root["judgment.redaction_count"], "1")
            self.assertEqual(json.loads(root["judgment.redactions"]), [{"rule": "github_token", "field": "history", "count": 1}])


class EnrichmentProvenanceTests(unittest.TestCase):
    """Finding 2: the hook removed an env value, then a sender without that variable replaced the field with transcript text."""

    def turn(self, sb: Sandbox, value: str) -> None:
        transcript = sb.home / "t.jsonl"
        start = int(time.time() * 1000) - 60_000
        claude_transcript(transcript, "ep-1", "p1", "check the service", start, "p1-tool", "cat svc.txt", "svc " + value, "The service is up.")
        base = {"session_id": "ep-1", "cwd": str(sb.project), "transcript_path": str(transcript), "prompt_id": "p1"}
        tool_event = {**base, "tool_name": "Bash", "tool_use_id": "p1-tool", "tool_input": {"command": "cat svc.txt"}, "tool_response": {"stdout": "svc " + value}}
        for event, payload in (("UserPromptSubmit", {**base, "prompt": "check the service"}), ("PostToolUse", tool_event),
                               ("Stop", {**base, "last_assistant_message": "The service is up."})):
            self.assertEqual(sb.hook("claude", event, payload).returncode, 0)

    def other_sender(self, sb: Sandbox) -> subprocess.CompletedProcess:
        """The sender another agent started: same key, without the variable the hook redacted."""
        env = {key: value for key, value in sb.env.items() if key != ENV_SECRET_NAME}
        return subprocess.run([sys.executable, "-m", "judgment_core.flush"], capture_output=True, text=True, env={**env, "PYTHONPATH": str(CORE)}, timeout=60)

    def test_sender_without_the_variable(self):
        value = SECRETS["env_value"]["core"]
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            self.turn(sb, value)
            inbox = json.dumps(sb.inbox())
            self.assertIn(f"[REDACTED:env:{ENV_SECRET_NAME}]", inbox)
            self.assertEqual(self.other_sender(sb).returncode, 0)
            self.assertTrue(sink.requests)
            for where in (inbox, sb.state_text(), sink.text()):
                self.assertEqual([piece for piece in pieces(value) if piece in where], [])
            # The hook's sanitized output is what the tool span shows.
            self.assertIn(f"svc [REDACTED:env:{ENV_SECRET_NAME}]", sink.text())

    def test_sender_with_the_variable_still_redacts(self):
        value = SECRETS["env_value"]["core"]
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            self.turn(sb, value)
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            self.assertNotIn(value, sink.text() + sb.state_text())
            self.assertIn("The service is up.", sink.text())

    def test_carry_records_rules(self):
        record = {"rule": f"env:{ENV_SECRET_NAME}", "field": "tool_output", "count": 1}
        stored = [{"_id": "a", "event": "tool", "tool_output": "svc [REDACTED]", "redactions": [record]},
                  {"_id": "b", "event": "turn_start", "prompt": "hi"}]
        prepared = [{"_id": "a", "event": "tool", "tool_output": "svc RAW"}, {"_id": "b", "event": "turn_start", "prompt": "hi there", "extra": "x"},
                    {"event": "model", "output": "RAW again", "model": "m"}]
        # The hook's field always stays. With the variable, transcript text elsewhere is the sender's to redact.
        held = flush.carry_records(stored, prepared, {ENV_SECRET_NAME})
        self.assertEqual(held[0]["tool_output"], "svc [REDACTED]")
        self.assertEqual(held[0]["redactions"], [record])
        self.assertEqual((held[1]["prompt"], held[2]["output"]), ("hi there", "RAW again"))
        # Without it, no transcript text stays: changed fields go back to the hook's, added ones and new events lose it.
        lacking = flush.carry_records(stored, prepared, set())
        self.assertEqual(lacking[0]["tool_output"], "svc [REDACTED]")
        self.assertEqual(lacking[1]["prompt"], "hi")
        self.assertNotIn("extra", lacking[1])
        self.assertNotIn("output", lacking[2])
        self.assertEqual(lacking[2]["model"], "m")


class EnvProvenanceTests(unittest.TestCase):
    """Finding 1: when a member or an assignment rule swallowed an env value whole, no env record said the hook's text
    held it, so a sender without that variable restored transcript text that held it."""

    NAME = "SERVICE" + "_TOKEN"

    def setUp(self):
        self.value = "Sv" + _rand(30)
        self.env = clean_env(**{self.NAME: self.value})

    def test_both_layers_record_the_env_value_whatever_rule_wins(self):
        events = [tool({"command": "x"}, {"password": self.value}), tool({"command": "x"}, "password=" + self.value),
                  tool({"command": "x"}, "svc " + self.value)]
        py, _ = assert_parity(self, events, env=self.env)
        want = [{"name": self.NAME, "field": "tool_output"}]
        for out in py:
            self.assertEqual(out.get("env_found"), want)
            self.assertEqual(leaked(self.value, json.dumps(out)), [])
        # The first two were replaced by another rule: no env record, which is why provenance is separate from the records.
        self.assertFalse(any(item["rule"].startswith("env:") for item in py[0].get("redactions", []) + py[1].get("redactions", [])))

    def test_recorded_when_the_rules_fail(self):
        py, _ = assert_parity(self, [tool({"command": "x"}, "svc " + self.value)], cfg={"redactPatterns": ["("]}, env=self.env)
        self.assertTrue(py[0]["redaction_error"])
        self.assertEqual(py[0]["env_found"], [{"name": self.NAME, "field": "tool_output"}])

    def test_a_payload_cannot_forge_it(self):
        forged = tool({"command": "x"}, "nothing here", env_found=[{"name": "OTHER", "field": "prompt"}])
        py, _ = assert_parity(self, [forged], env=self.env)
        self.assertNotIn("env_found", py[0])

    def test_carry_records_reads_provenance(self):
        stored = [{"_id": "a", "event": "tool", "tool_output": {"password": "[REDACTED:secret_field]"},
                   "redactions": [{"rule": "secret_field", "field": "tool_output", "count": 1}],
                   "env_found": [{"name": self.NAME, "field": "tool_output"}]}]
        prepared = [{"_id": "a", "event": "tool", "tool_output": "svc RAW"}, {"event": "model", "output": "RAW again", "model": "m"}]
        lacking = flush.carry_records(stored, prepared, set())
        self.assertEqual(lacking[0]["tool_output"], {"password": "[REDACTED:secret_field]"})
        self.assertNotIn("output", lacking[1])
        holding = flush.carry_records(stored, prepared, {self.NAME})
        self.assertEqual(holding[0]["tool_output"], {"password": "[REDACTED:secret_field]"})  # the hook's field stays
        self.assertEqual(holding[1]["output"], "RAW again")  # elsewhere the sender redacts with the variable it holds

    def end_to_end(self, hook_response) -> None:
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            sb.env[self.NAME] = self.value
            transcript = sb.home / "t.jsonl"
            start = int(time.time() * 1000) - 60_000
            # The transcript holds the value bare, where only the env rule could find it.
            claude_transcript(transcript, "pv-1", "p1", "check", start, "p1-tool", "cat svc.txt", "svc " + self.value, "Up.")
            base = {"session_id": "pv-1", "cwd": str(sb.project), "transcript_path": str(transcript), "prompt_id": "p1"}
            tool_event = {**base, "tool_name": "Bash", "tool_use_id": "p1-tool", "tool_input": {"command": "cat svc.txt"}, "tool_response": hook_response}
            for event, payload in (("UserPromptSubmit", {**base, "prompt": "check"}), ("PostToolUse", tool_event), ("Stop", {**base, "last_assistant_message": "Up."})):
                self.assertEqual(sb.hook("claude", event, payload).returncode, 0)
            inbox = json.dumps(sb.inbox())
            env = {key: value for key, value in sb.env.items() if key != self.NAME}
            sender = subprocess.run([sys.executable, "-m", "judgment_core.flush"], capture_output=True, text=True, env={**env, "PYTHONPATH": str(CORE)}, timeout=60)
            self.assertEqual(sender.returncode, 0, sender.stderr)
            self.assertTrue(sink.requests)
            self.assertEqual(leaked(self.value, inbox, sb.state_text(), sink.text()), [])

    def test_end_to_end_member(self):
        self.end_to_end({"password": self.value})

    def test_end_to_end_assignment(self):
        self.end_to_end({"stdout": "password=" + self.value})


class DenseSecretTests(unittest.TestCase):
    """The sender on text dense with genuine secrets: 1,000 distinct keys in 56 KB are all redacted, none dropped, in
    linear time (one finditer per rule and window, no rescans). The hook keeps its smaller budget and fails closed fast."""

    def text(self) -> tuple[str, list[str]]:
        keys = []
        for n in range(1000):
            kind = n % 4
            keys.append(["sk-" + "proj-" + _rand(40), "gh" + "p_" + _rand(36), "AK" + "IA" + _rand(16, string.ascii_uppercase + string.digits),
                         _rand(30)][kind])
        lines = []
        for n, key in enumerate(keys):
            lines.append(f"entry {n:04d} ok: " + (("api_key=" + key) if n % 4 == 3 else key) + " done")
        text = "\n".join(lines)
        text += " " + filler(56_000 - len(text) - 1) if len(text) < 56_000 else ""
        return text, keys

    def test_sender_redacts_all_quickly(self):
        text, keys = self.text()
        self.assertGreaterEqual(len(text), 56_000)
        rs, event = redact.ruleset(), tool({"command": "x"}, text)
        best = None
        for _ in range(3):
            started = time.perf_counter()
            out = redact.redact_event(event, rs, [])
            elapsed = (time.perf_counter() - started) * 1000
            best = elapsed if best is None else min(best, elapsed)
        self.assertNotIn("redaction_error", out)
        dumped = json.dumps(out)
        self.assertEqual([key for key in keys if key[-12:] in dumped], [])
        self.assertEqual(sum(item["count"] for item in out["redactions"]), 1000)
        self.assertLess(best, 150, f"best of 3: {best:.0f} ms")

    def test_hook_budget_fails_closed_fast(self):
        text, keys = self.text()
        started = time.perf_counter()
        py, js = both([tool({"command": "x"}, text)])
        self.assertLess(time.perf_counter() - started, 10)
        for out in py + js:
            self.assertTrue(out["redaction_error"])
            self.assertEqual([key for key in keys if key[-12:] in json.dumps(out)], [])


if __name__ == "__main__":
    unittest.main()
