"""Regressions for the first review of capture: sanitized. One class per finding.

Every fake secret is built at runtime from parts (support.SECRETS, or _rand below), so no file holds one.
"""

from __future__ import annotations

import hashlib
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

from fake_sink import Sink, attrs
from payloads import pi_turn
from support import HOOK, SECRETS, Sandbox, claude_transcript, clean_env, jq_sanitize, norm, py_sanitize, wait_for, whole_leaks

from judgment_core import redact

_rng = random.Random(4242)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def tool(tool_input, tool_output="", **extra) -> dict:
    return {"v": 1, "harness": "pi", "event": "tool", "session_id": "s", "turn_id": "t", "tool_name": "bash", "tool_call_id": "c1",
            "tool_input": tool_input, "tool_output": tool_output, **extra}


def both(events: list[dict], cfg: dict | None = None, env: dict | None = None) -> tuple[list[dict], list[dict]]:
    """Layer 1 (jq) and its Python mirror, which must agree."""
    py, js = py_sanitize(events, cfg, env), jq_sanitize(events, cfg, env)
    return py, js


def stamp(sb: Sandbox, url: str) -> dict:
    return {"export_fp": hashlib.sha256(SECRETS["judgment_key"]["core"].encode()).hexdigest()[:32], "export_project": "proj-test-1",
            "export_url": url, "export_org": "org-test-1", "project_root": str(sb.project), "environment": "production", "capture": "sanitized"}


def inject(sb: Sandbox, events: list[dict]) -> None:
    """Events straight into the inbox, as text the sender reads that the hook never saw (transcript enrichment)."""
    folder = sb.state / "inbox"
    folder.mkdir(parents=True, exist_ok=True)
    base = time.time_ns()
    for number, event in enumerate(events):
        path = folder / f"9.{number}.json"
        path.write_text(json.dumps(event))
        os.utime(path, ns=(base + number * 1000, base + number * 1000))


class StructuredSecretTests(unittest.TestCase):
    """Finding 1: values were walked without their key, and keys were never redacted."""

    def setUp(self):
        self.api = "Zq" + _rand(26)  # no pattern knows it: only its key says it is secret
        self.password = "pw" + _rand(14)
        self.client = "cs" + _rand(30)
        self.argument = "tk" + _rand(22)
        self.key_secret = SECRETS["github_token"]["core"]

    def event(self) -> dict:
        return {
            **tool({"api_key": self.api, "options": [{"password": self.password}]},
                   {"result": {"client_secret": self.client}, self.key_secret: True, "count": 3}),
            "tool_calls": [{"id": "c1", "name": "http", "arguments": {"X-Api-Key": self.argument}}],
        }

    def test_members_and_keys_in_both_layers(self):
        py, js = both([self.event()])
        self.assertEqual(norm(py), norm(js))
        out = py[0]
        text = json.dumps(out)
        for secret in (self.api, self.password, self.client, self.argument, self.key_secret):
            self.assertNotIn(secret, text)
        self.assertEqual(out["tool_input"], {"api_key": "[REDACTED:secret_field]", "options": [{"password": "[REDACTED:secret_field]"}]})
        self.assertEqual(out["tool_output"], {"result": {"client_secret": "[REDACTED:secret_field]"}, "[REDACTED:github_token]": True, "count": 3})
        audit = {(item["rule"], item["field"]) for item in redact.census(out)}
        self.assertTrue({("secret_field", "tool_input"), ("secret_field", "tool_output"), ("github_token", "tool_output"), ("secret_field", "tool_calls")} <= audit)

    def test_ordinary_members_stay(self):
        keep = {"max_tokens": 4096, "key": "user_id_2024_v1", "password_hint": "ask Pat", "secret": False, "credentials": None,
                "author": "Pat Q", "token_count": "12", "tokenizer": "bert-base"}
        py, js = both([tool(keep, {"rows": [keep]})])
        for out in (py[0], js[0]):
            self.assertEqual(out["tool_input"], keep)
            self.assertEqual(out["tool_output"], {"rows": [keep]})

    def test_known_limit_any_string_under_a_name_goes(self):
        """Known limit (reference-value): a word or a reference under a secret name goes like a credential."""
        refs = {"token": "the", "api_key": "${API_KEY}"}
        py, js = both([tool(refs)])
        self.assertEqual(norm(py), norm(js))
        self.assertEqual(py[0]["tool_input"], {key: "[REDACTED:secret_field]" for key in refs})

    def test_structured_secret_reaches_no_request(self):
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            steps = pi_turn(str(sb.project), "st-1", "st-1:1", int(time.time() * 1000) - 5000)
            for _, payload in steps:
                if payload["event"] == "tool":
                    payload = {**payload, "tool_input": self.event()["tool_input"], "tool_output": self.event()["tool_output"]}
                self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
            inbox = json.dumps(sb.inbox())
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            for where in (inbox, sink.text(), sb.state_text()):
                for secret in (self.api, self.password, self.client, self.key_secret):
                    self.assertNotIn(secret, where)
            self.assertIn("secret_field", sink.text())


class EnvValueTests(unittest.TestCase):
    """Finding 2: only a value's first line was replaced, and short, numeric, or lowercase values were skipped.

    Threshold since review 2: a value (or one line of it) of 8 or more characters, of any shape, that is not only
    whitespace. Shorter values stay: replacing every short run such as 2024 in all text would destroy ordinary text.
    """

    def test_every_line_of_a_multiline_value(self):
        lines = ["-----BEGIN " + "OPENSSH PRIV" + "ATE KEY-----", _rand(64, string.ascii_letters + "+/"), _rand(64, string.ascii_letters + "+/"), "tail" + _rand(9)]
        value = "\r\n".join(lines[:2]) + "\n" + "\n".join(lines[2:])
        env = clean_env(DEPLOY_PRIVATE_KEY=value)
        events = [tool({"command": "ssh deploy"}, "whole:\n" + value), tool({"command": "x"}, "one line: " + lines[2] + " and " + lines[3])]
        py, js = both(events, env=env)
        self.assertEqual(norm(py), norm(js))
        for out in py:
            for line in lines[1:]:
                self.assertNotIn(line, out["tool_output"])
            self.assertIn("[REDACTED:env:DEPLOY_PRIVATE_KEY]", out["tool_output"])

    def test_lowercase_numeric_and_short_values(self):
        values = {
            "LOWER_SECRET": "".join(_rng.choice(string.ascii_lowercase) for _ in range(16)),  # one class, long enough
            "PIN_PASSWORD": "".join(_rng.choice(string.digits) for _ in range(12)),
            "MIXED_TOKEN": "Ab" + _rand(7),  # 9 characters, two classes
        }
        values.update({"AUTH_PROVIDER": "keycloak", "DIGIT_TOKEN": "20240101", "CREDENTIAL_HELPER": "/usr/bin/helper-x"})
        kept = {"SHORT_TOKEN": "Ab1", "PIN_CODE_TOKEN": "4821"}
        env = clean_env(**values, **kept)
        event = tool({"command": "x"}, " | ".join(list(values.values()) + list(kept.values())))
        py, js = both([event], env=env)
        self.assertEqual(norm(py), norm(js))
        for name, value in values.items():
            self.assertNotIn(value, py[0]["tool_output"])
            self.assertIn(f"[REDACTED:env:{name}]", py[0]["tool_output"])
        for value in kept.values():
            self.assertIn(value, py[0]["tool_output"])
        self.assertEqual(sorted({name for name, _ in redact.env_secrets(env)}), sorted(values))


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


class StringToolInputTests(unittest.TestCase):
    """Finding 4: jq used the fromjson error message as the command when tool_input was a plain string."""

    def test_plain_string_commands(self):
        events = [tool("env", "SECRET_THING=" + _rand(20)), tool("printenv", "x"), tool("cat .env", "A=1"), tool("make build", "ok"), tool('{"command": "env"}', "x")]
        py, js = both(events)
        self.assertEqual(norm(py), norm(js))
        self.assertEqual([out.get("sensitive_rule") for out in py], ["env_dump", "env_dump", "dotenv", None, "env_dump"])
        for out in py[:3]:
            self.assertNotIn("tool_output", out)

    def test_plain_string_env_never_reaches_the_inbox(self):
        sb = Sandbox()
        self.addCleanup(sb.cleanup)
        marker = "Mk" + _rand(18)
        steps = pi_turn(str(sb.project), "ps-1", "ps-1:1", int(time.time() * 1000))
        payload = {**steps[2][1], "tool_input": "env", "tool_output": "UNRELATED_VALUE=" + marker}
        self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
        [stored] = sb.inbox()
        self.assertEqual(stored["sensitive_rule"], "env_dump")
        self.assertNotIn(marker, sb.state_text())


class SensitiveAccessTests(unittest.TestCase):
    """Finding 5: cat credentials, /usr/bin/env, and env -0 were missed. Known limits: env with options or by its path is
    not seen (wrapped-command), and a search for the word credentials blanks the turn (mention-blanks)."""

    SENSITIVE = [
        ({"command": "cat credentials"}, "credentials_file"),
        ({"command": "cd ~/.aws && head -n 5 credentials"}, "credentials_file"),
        ({"command": "less ./credentials"}, "credentials_file"),
        ({"file_path": "credentials"}, "credentials_file"),
        ({"command": ["bash", "-lc", "env"]}, "env_dump"),
        ({"command": "grep -rn credentials src/"}, "credentials_file"),
        ({"command": "git log --grep credentials"}, "credentials_file"),
    ]
    NOT_SENSITIVE = [
        {"command": "env FOO=1 make test"},
        {"command": "/usr/bin/env python3 script.py"},
        {"command": "env -u HOME bash -lc make"},
        {"command": "env -i PATH=/usr/bin make"},
        {"command": "cat credentials.md.txt"},
        {"command": "/usr/bin/env"},
        {"command": "env -0"},
        {"command": "env --null | sort -z"},
        {"command": "env -u HOME"},
        {"command": "sudo /usr/bin/env -0 > /tmp/e"},
    ]

    def test_detected_in_both_layers(self):
        inputs = [item for item, _ in self.SENSITIVE] + self.NOT_SENSITIVE
        py, js = both([tool(item, "x") for item in inputs])
        self.assertEqual(norm(py), norm(js))
        expected = [rule for _, rule in self.SENSITIVE] + [None] * len(self.NOT_SENSITIVE)
        for item, out, rule in zip(inputs, py, expected):
            with self.subTest(item=item):
                self.assertEqual(out.get("sensitive_rule"), rule)


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


class CodeReferenceTests(unittest.TestCase):
    """Finding 9 asked that member expressions and versioned identifiers stay. Known limit (reference-value): after a secret
    name they go like any value now."""

    KEEP = [
        "api_key = config2.api_key",
        "token = response_v2.token",
        "secret = settings.SECRET_KEY_V2",
        "password = form2.cleaned_data",
        "access_token = self.client_v3.auth.token",
        "api_key = api_key_v2",
        "token: tokenV2",
    ]

    def test_known_limit_code_references_go(self):
        py, js = both([tool({"command": "x"}, text) for text in self.KEEP])
        self.assertEqual(norm(py), norm(js))
        for text, out in zip(self.KEEP, py):
            self.assertEqual(out["tool_output"], text.split(" ")[0] + " " + ("= " if " = " in text else "") + "[REDACTED:secret_assignment]")

    def test_literal_values_still_go(self):
        value = "Q7" + _rand(20)
        for text in ("api_key = " + value, 'token: "' + value + '"', "password=" + value + "=="):
            py, js = both([tool({"command": "x"}, text)])
            self.assertEqual(norm(py), norm(js))
            self.assertNotIn(value, py[0]["tool_output"])


class AuditCompletenessTests(unittest.TestCase):
    """Finding 10: a placeholder whose env name had /, :, or other characters was not counted."""

    def test_every_env_name_is_audited(self):
        names = ["npm_config_//registry.example.com/:_authToken", "weird name TOKEN", "A.B@C+TOKEN", "GITHUB_TOKEN"]
        env = clean_env(**{name: "Vv" + _rand(20) for name in names})
        event = tool({"command": "x"}, " ".join(env[name] for name in names))
        py, js = both([event], env=env)
        self.assertEqual(norm(py), norm(js))
        for out in py + js:
            audit = {item["rule"]: item["count"] for item in redact.census(out)}
            self.assertEqual(audit, {f"env:{name}": 1 for name in names})


class SenderDetachTests(unittest.TestCase):
    """Finding 11: without setsid(1) the sender stayed in the agent's process group and died with it."""

    def run_and_kill(self, shell: list[str], path: str) -> None:
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            steps = pi_turn(str(sb.project), "dt-1", "dt-1:1", int(time.time() * 1000) - 5000)
            for _, payload in steps[:-1]:
                sb.hook("pi", "", payload)
            stop = sb.dir / "stop.json"
            stop.write_text(json.dumps(steps[-1][1]))
            env = {**sb.env, "JUDGMENT_FLUSH": "", "PYTHONPATH": "", "PATH": path}
            # The agent runs the hook, then its whole process group is killed the moment the hook returns.
            command = " ".join(shell) + f' "{HOOK}" --raw < "{stop}"; kill -9 0'
            parent = subprocess.Popen([shutil.which("sh"), "-c", command], env=env, start_new_session=True)
            parent.wait(timeout=10)
            self.assertEqual(parent.returncode, -signal.SIGKILL)
            self.assertTrue(wait_for(lambda: sink.requests, timeout=20), "the sender died with the agent's process group")

    def tools_without_setsid(self) -> str:
        tools = Path(tempfile.mkdtemp(prefix="jt-nosetsid-"))
        self.addCleanup(shutil.rmtree, tools, True)
        for name in ("sh", "bash", "cat", "tr", "mkdir", "rm", "sha256sum", "jq", "python3", "kill"):
            found = shutil.which(name)
            if found:
                os.symlink(found, tools / name)
        self.assertIsNone(shutil.which("setsid", path=str(tools)))
        return str(tools)

    def test_with_setsid(self):
        if not shutil.which("setsid"):
            self.skipTest("no setsid(1) here")
        self.run_and_kill([], os.environ.get("PATH", "/usr/bin:/bin"))

    def test_without_setsid_under_bash(self):
        # macOS has no setsid(1), and runs /bin/sh as bash: job control gives the sender its own process group.
        if not shutil.which("bash"):
            self.skipTest("no bash here")
        self.run_and_kill([shutil.which("bash")], self.tools_without_setsid())

    def test_sender_leaves_the_hook_session(self):
        from judgment_core import flush

        child = subprocess.run(
            ["python3", "-c", "import os, sys; from judgment_core import flush; before = os.getsid(0); flush._detach(); print(before != os.getsid(0))"],
            capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(Path(flush.__file__).resolve().parent.parent)}, timeout=30,
        )
        self.assertEqual(child.stdout.strip(), "True", child.stderr)


if __name__ == "__main__":
    unittest.main()
