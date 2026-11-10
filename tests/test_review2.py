"""Regressions for the second review of capture: sanitized. One class per finding, each reproducing the review's case.

Every fake secret is built at runtime from parts (support.SECRETS, or _rand below), so no file holds one.
"""

from __future__ import annotations

import json
import os
import random
import shutil
import signal
import string
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from fake_sink import Sink, attrs, roots
from payloads import pi_turn
from support import HOOK, SECRETS, Sandbox, claude_transcript, clean_env, hermetic_dir, jq_sanitize, norm, py_sanitize, wait_for, whole_leaks
from test_review_fixes import both, inject, stamp, tool

from judgment_core import redact

_rng = random.Random(9090)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def pieces(secret: str, size: int = 8) -> list[str]:
    return [secret[i : i + size] for i in range(0, len(secret) - size + 1)]


def filler(n: int) -> str:
    """Ordinary words, no rule's literal in them."""
    words = ("alpha", "beta", "gamma", "delta", "omega", "build", "step", "done")
    out, i = [], 0
    while sum(len(word) + 1 for word in out) < n:
        out.append(words[i % len(words)])
        i += 1
    return " ".join(out)[:n]


class WindowBoundaryTests(unittest.TestCase):
    """Finding 1: a token crossing a window boundary was matched in part, and its suffix leaked."""

    def test_every_offset_across_the_window_end(self):
        limits = redact.rules()["limits"]
        key = SECRETS["openai_key"]["core"]
        events = []
        for boundary in (limits["window"], limits["step"] + limits["window"]):
            for offset in range(boundary - len(key) - 2, boundary + 3):
                text = filler(offset - 1) + " " + key + " " + filler(limits["window"] + limits["step"])
                events.append(tool({"command": "make"}, text))
        py, js = both(events)
        self.assertEqual(len(py), len(js))
        for n, (a, b) in enumerate(zip(py, js)):
            with self.subTest(case=n):
                self.assertEqual(a, b)
                for out in (a, b):
                    self.assertNotIn("redaction_error", out)
                    text = json.dumps(out)
                    self.assertEqual([piece for piece in pieces(SECRETS["openai_key"]["random"]) if piece in text], [])
                    self.assertEqual(out["redactions"], [{"rule": "openai_key", "field": "tool_output", "count": 1}])

    def test_oversized_candidate_fails_closed(self):
        limits = redact.rules()["limits"]
        # It starts before the second window does and runs past the end of the first: no window holds it whole.
        giant = "sk-" + "proj-" + _rand(limits["window"] - limits["step"] + 400)
        text = filler(limits["step"] - 200) + " " + giant + " " + filler(limits["window"])
        py, js = both([tool({"command": "make"}, text)])
        for out in (py[0], js[0]):
            self.assertTrue(out.get("redaction_error"))
            self.assertNotIn("tool_output", out)
            self.assertNotIn(giant[8:40], json.dumps(out))

    def test_hook_inbox_and_request(self):
        """The review's reproduction, end to end: the key crosses the boundary in a Pi tool output."""
        limits = redact.rules()["limits"]
        key = SECRETS["openai_key"]["core"]
        output = filler(limits["window"] - 20) + " " + key + " " + filler(limits["step"])
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            for _, payload in pi_turn(str(sb.project), "wb-1", "wb-1:1", int(time.time() * 1000) - 5000, output=output):
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            inbox = json.dumps(sb.inbox())
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            for where in (inbox, sb.state_text(), sink.text()):
                self.assertEqual([piece for piece in pieces(SECRETS["openai_key"]["random"]) if piece in where], [])


class EnvValueShapeTests(unittest.TestCase):
    """Finding 2: short, lowercase, and /-leading values of secret-named variables were exempt by shape."""

    def test_values_go_whatever_their_shape(self):
        go = {
            "PIN_PASSWORD": "".join(_rng.choice(string.digits) for _ in range(8)),
            "LOWER_SECRET": "".join(_rng.choice(string.ascii_lowercase) for _ in range(9)),
            "UPPER_TOKEN": "".join(_rng.choice(string.ascii_uppercase) for _ in range(10)),
            "HELPER_CREDENTIAL": "/opt/" + _rand(6),
        }
        stay = {"SHORT_PASSWORD": "4821", "BLANK_SECRET": " " * 12, "KEY_PATH": "/home/tester/.ssh/id_x", "SSH_AUTH_SOCK": "/tmp/agent." + _rand(6)}
        env = clean_env(**go, **stay)
        text = " | ".join(list(go.values()) + list(stay.values()))
        py, js = both([tool({"command": "x"}, text)], env=env)
        self.assertEqual(norm(py), norm(js))
        out = py[0]["tool_output"]
        for name, value in go.items():
            self.assertNotIn(value, out)
            self.assertIn(f"[REDACTED:env:{name}]", out)
        for value in stay.values():
            self.assertIn(value, out)  # a value under 8 characters is a known limit: README, Secrets are redacted by default

    def test_eight_digit_pin_never_reaches_storage_or_the_request(self):
        pin = "".join(_rng.choice(string.digits) for _ in range(8))
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            sb.env["PIN_PASSWORD"] = pin
            for _, payload in pi_turn(str(sb.project), "pin-1", "pin-1:1", int(time.time() * 1000) - 5000, output="pin is " + pin):
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            inbox = json.dumps(sb.inbox())
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            for where in (inbox, sb.state_text(), sink.text()):
                self.assertNotIn(pin, where)


class StructuredLiteralTests(unittest.TestCase):
    """Finding 3: a member under a secret key kept any value that did not look like a generated credential."""

    PASSPHRASE = "correct horse battery staple"

    def test_any_literal_under_a_secret_key(self):
        short = "pw" + _rand(3).lower()
        cases = {"password": self.PASSPHRASE, "pwd": short, "X-Api-Key": "k" + _rand(5), "dbPassword": "hunter" + _rand(2),
                 "client_secret": "abc def", "Authorization": "Basic " + _rand(10), "smtp_password": "x"}
        py, js = both([tool({"command": "x"}, {"config": cases})])
        self.assertEqual(norm(py), norm(js))
        self.assertEqual(py[0]["tool_output"]["config"], {key: "[REDACTED:secret_field]" for key in cases})
        self.assertEqual(py[0]["redactions"], [{"rule": "secret_field", "field": "tool_output", "count": len(cases)}])

    def test_references_placeholders_and_non_strings_stay(self):
        keep = {"password": "${DB_PASSWORD}", "api_key": "<your-api-key>", "secret": "changeme", "access_token": "{{ token }}",
                "client_secret": "os.environ['CLIENT_SECRET']", "refresh_token": "", "pwd": True, "passphrase": None, "auth_token": 1234,
                "private_key": "[REDACTED:private_key]", "max_tokens": "4096", "token_count": "12", "tokenizer": "bert-base", "sort_key": "name"}
        py, js = both([tool({"command": "x"}, {"config": keep})])
        for out in (py[0], js[0]):
            self.assertEqual(out["tool_output"], {"config": keep})
            self.assertNotIn("redactions", out)

    def test_quoted_literal_in_text(self):
        texts = {f'password = "{self.PASSPHRASE}"': self.PASSPHRASE, "DB_PASSWORD: 'abc'": "abc",
                 '{"apiKey": "lower case words"}': "lower case words", 'export SMTP_PASSWORD="x y"': "x y"}
        keep = ['password = "${DB_PASSWORD}"', "password: '<placeholder>'", 'token_count = "12"', 'max_tokens: "4096"']
        events = [tool({"command": "x"}, text) for text in list(texts) + keep]
        py, js = both(events)
        self.assertEqual(norm(py), norm(js))
        for out, secret in zip(py, texts.values()):
            self.assertNotIn(secret, out["tool_output"])
        for out, text in zip(py[len(texts):], keep):
            self.assertEqual(out["tool_output"], text)

    def test_passphrase_reaches_no_storage_or_request(self):
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            for _, payload in pi_turn(str(sb.project), "pp-1", "pp-1:1", int(time.time() * 1000) - 5000):
                if payload["event"] == "tool":
                    payload = {**payload, "tool_output": {"password": self.PASSPHRASE, "user": "pat"}}
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            inbox = json.dumps(sb.inbox())
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            for where in (inbox, sb.state_text(), sink.text()):
                self.assertNotIn(self.PASSPHRASE, where)


class FailureLabelTests(unittest.TestCase):
    """Finding 4: when the rules failed, tool_calls kept their names, and Pi without jq kept tool_name and agent_type."""

    def test_rule_failure_keeps_ids_only(self):
        token = SECRETS["github_token"]["core"]
        event = {"v": 1, "event": "model", "session_id": "s", "turn_id": "t", "agent_type": "w " + token, "tool_name": "t " + token,
                 "tool_calls": [{"id": "c1", "name": "mcp__" + token, "arguments": {"command": "ls"}}]}
        cfg = {"redactPatterns": ["(unclosed"]}
        py, js = py_sanitize([event], cfg), jq_sanitize([event], cfg)
        for out in (py[0], js[0]):
            self.assertTrue(out["redaction_error"])
            self.assertEqual(out["tool_calls"], [{"id": "c1"}])
            self.assertEqual(whole_leaks(json.dumps(out)), [])

    def test_rule_failure_end_to_end(self):
        token = SECRETS["github_token"]["core"]
        with Sink() as sink:
            sb = Sandbox({"enabled": True, "redactPatterns": ["(unclosed"]}, sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            for _, payload in pi_turn(str(sb.project), "fl-1", "fl-1:1", int(time.time() * 1000) - 5000):
                if payload["event"] == "model" and payload.get("tool_calls"):
                    payload = {**payload, "tool_calls": [{**payload["tool_calls"][0], "name": "mcp__" + token}]}
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            inbox = json.dumps(sb.inbox())
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            for where in (inbox, sb.state_text(), sink.text()):
                self.assertNotIn(token, where)

    def test_pi_without_jq_writes_no_labels(self):
        token = SECRETS["github_token"]["core"]
        tools = Path(tempfile.mkdtemp(prefix="jt-nojq-"))
        self.addCleanup(shutil.rmtree, tools, True)
        for name in ("sh", "cat", "tr", "mkdir", "rm", "sha256sum"):
            if shutil.which(name):
                os.symlink(shutil.which(name), tools / name)
        sb = Sandbox()
        self.addCleanup(sb.cleanup)
        payload = {"v": 1, "harness": "pi", "session_id": "nj-1", "turn_id": "nj-1:1", "cwd": str(sb.project), "event": "tool",
                   "tool_name": "mcp__" + token, "agent_type": "w" + token, "agent_id": "a1", "tool_call_id": "c1", "ts": 5}
        self.assertEqual(sb.hook("pi", "", payload, env={"PATH": str(tools)}).returncode, 0)
        [stored] = sb.inbox()
        self.assertTrue(stored["redaction_error"])
        self.assertNotIn("tool_name", stored)
        self.assertNotIn("agent_type", stored)
        self.assertNotIn(token, json.dumps(stored))
        self.assertEqual((stored["agent_id"], stored["tool_call_id"]), ("a1", "c1"))


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


class KeyCollisionTests(unittest.TestCase):
    """Finding 6: two different secret keys became the same placeholder key, and one member was lost."""

    def test_every_member_stays(self):
        first, second = "gh" + "p_" + _rand(36), "gh" + "p_" + _rand(36)
        output = {"results": {first: {"ok": True, "rows": 3}, second: {"ok": False, "rows": 5}, "[REDACTED:github_token]": "taken"}, "note": "plain"}
        py, js = both([tool({"command": "x"}, output)])
        self.assertEqual(norm(py), norm(js))
        results = py[0]["tool_output"]["results"]
        self.assertEqual(sorted(results), ["[REDACTED:github_token]", "[REDACTED:github_token]#1", "[REDACTED:github_token]#2"])
        self.assertEqual(results["[REDACTED:github_token]"], "taken")
        self.assertEqual(sorted((results["[REDACTED:github_token]#1"], results["[REDACTED:github_token]#2"]), key=lambda v: v["rows"]),
                         [{"ok": True, "rows": 3}, {"ok": False, "rows": 5}])
        self.assertEqual(py[0]["redactions"], [{"rule": "github_token", "field": "tool_output", "count": 2}])
        self.assertEqual(whole_leaks(json.dumps(py)), [])

    def test_distinct_keys_without_collision_keep_plain_names(self):
        token = "gh" + "p_" + _rand(36)
        py, js = both([tool({"command": "x"}, {token: 1, "other": 2})])
        self.assertEqual(norm(py), norm(js))
        self.assertEqual(py[0]["tool_output"], {"[REDACTED:github_token]": 1, "other": 2})

    def test_a_collision_that_numbering_cannot_fix_fails_closed(self):
        first, second = "gh" + "p_" + _rand(36), "gh" + "p_" + _rand(36)
        output = {first: 1, second: 2, "[REDACTED:github_token]": 3, "[REDACTED:github_token]#1": 4, "[REDACTED:github_token]#2": 5}
        py, js = both([tool({"command": "x"}, output)])
        for out in (py[0], js[0]):
            self.assertTrue(out["redaction_error"])
            self.assertNotIn("tool_output", out)


class ConstantReferenceTests(unittest.TestCase):
    """Finding 7: a constant identifier as a value (API_KEY_2024, DEFAULT_PASSWORD2) was redacted as a credential."""

    def test_constants_stay_and_literals_go(self):
        keep = ["api_key = API_KEY_2024", "password = DEFAULT_PASSWORD2", "token: GITHUB_TOKEN_V2", "secret = PROD_SECRET_KEY",
                "DB_PASSWORD=PASSWORD123"]
        literal = "X7K2" + _rand(10).upper()
        go = {"api_key = " + literal: literal, 'password = "DEFAULT_PASSWORD2"': "DEFAULT_PASSWORD2", "token = Zq" + literal.lower(): "Zq" + literal.lower()}
        py, js = both([tool({"command": "x"}, text) for text in keep + list(go)])
        self.assertEqual(norm(py), norm(js))
        for text, out in zip(keep, py):
            self.assertEqual(out["tool_output"], text)
        for out, secret in zip(py[len(keep):], go.values()):
            self.assertNotIn(secret, out["tool_output"])


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


class AuditRecordTests(unittest.TestCase):
    """Finding 10: the audit was read back from placeholders, so some environment names were cut short or not counted."""

    def test_names_of_any_shape_are_counted_exactly(self):
        names = ["ODD]NAME_TOKEN", "TOKEN_" + "X" * 300, "x.y@z TOKEN", "npm_config_//r.example.com/:_authToken"]
        env = clean_env(**{name: "Vv" + _rand(20) for name in names})
        event = tool({"command": "x"}, " ".join(env[name] for name in names) + " and a literal [REDACTED:not_a_rule] in the text")
        py, js = both([event], env=env)
        self.assertEqual(norm(py), norm(js))
        for out in (py[0], js[0]):
            self.assertEqual(sorted(redact.census(out), key=lambda item: item["rule"]),
                             sorted(({"rule": f"env:{name}", "field": "tool_output", "count": 1} for name in names), key=lambda item: item["rule"]))

    def test_sender_counts_once_across_layers_and_enrichment(self):
        """The hook's records travel with the event; the sender adds only what it replaces itself, and drops the records of
        a field that enrichment replaced."""
        from judgment_core.flush import carry_records

        stored = [{"_id": "1", "event": "tool", "tool_output": "[REDACTED:github_token]", "prompt": "[REDACTED:jwt]",
                   "redactions": [{"rule": "github_token", "field": "tool_output", "count": 1}, {"rule": "jwt", "field": "prompt", "count": 1}]}]
        prepared = [{**stored[0], "tool_output": "fresh " + SECRETS["stripe_key"]["core"]}]
        carried = carry_records(stored, prepared)
        self.assertEqual(carried[0]["redactions"], [{"rule": "jwt", "field": "prompt", "count": 1}])
        out = redact.redact_event(carried[0], redact.ruleset(), [])
        self.assertEqual(out["redactions"], [{"rule": "jwt", "field": "prompt", "count": 1}, {"rule": "stripe_key", "field": "tool_output", "count": 1}])
        self.assertEqual(redact.redact_event(out, redact.ruleset(), [])["redactions"], out["redactions"])


class HermeticTests(unittest.TestCase):
    """Finding 11: a .judgment or .git above the temp dir changed what the tests saw."""

    def test_temp_roots_skip_folders_under_an_opt_in(self):
        decoy = hermetic_dir("jt-decoy-")
        self.addCleanup(shutil.rmtree, decoy, True)
        (decoy / ".judgment").mkdir()
        (decoy / ".judgment" / "tracing.json").write_text('{"enabled": true}')
        (decoy / ".git").mkdir()
        old = tempfile.tempdir
        tempfile.tempdir = str(decoy)
        try:
            folder = hermetic_dir("jt-inner-")
        finally:
            tempfile.tempdir = old
        self.addCleanup(shutil.rmtree, folder, True)
        self.assertNotIn(decoy, folder.parents)

    def test_no_opt_in_is_quiet_even_under_a_stray_opt_in(self):
        """A sandbox without an opt-in file stays quiet, whatever /tmp holds: its temp root has no opt-in above it."""
        sb = Sandbox(None)
        self.addCleanup(sb.cleanup)
        from judgment_core.optin import find_optin

        self.assertIsNone(find_optin(sb.project))


if __name__ == "__main__":
    unittest.main()
