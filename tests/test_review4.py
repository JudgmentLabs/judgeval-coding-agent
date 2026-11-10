"""Regressions for the fourth review of capture: sanitized. One class per finding, each reproducing the review's case in
both layers (jq and its Python mirror, which must agree) and, where it applies, end to end: hook, inbox, stored turn, and
the HTTP body the sender posts. Then the sender's speed on text dense with secrets, and the suite's private temp folder.

Every fake secret is built at runtime from parts (_rand below), so no file holds one.
"""

from __future__ import annotations

import json
import os
import random
import re
import string
import subprocess
import sys
import tempfile
import time
import unittest

import support
from fake_sink import Sink
from support import CORE, ENV_SECRET_NAME, Sandbox, claude_transcript, clean_env, norm
from test_review3 import everywhere, filler, pi_run, pieces
from test_review_fixes import both, tool

from judgment_core import flush, redact

_rng = random.Random(4444)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def leaked(secret: str, *texts: str, size: int = 8) -> list[str]:
    """Pieces of the secret found in any of the texts."""
    return [piece for piece in pieces(secret, size) for text in texts if piece in text]


def assert_parity(test: unittest.TestCase, events: list[dict], cfg: dict | None = None, env: dict | None = None) -> tuple[list[dict], list[dict]]:
    py, js = both(events, cfg, env)
    test.assertEqual(norm(py), norm(js))
    for out in py + js:
        redact.outcome(out)  # raises on an impossible state
    return py, js


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


class ColonPathTests(unittest.TestCase):
    """Finding 2: path rules could start after every colon and scan to the end of the word, so a word of many colons took
    seconds. Each rule now scans one component (between / and :), so the work is linear."""

    CASES = {"a:*50000 + .keyX": "cat " + "a:" * 50000 + ".keyX", "secret:*20000 + .py": "cat " + "secret:" * 20000 + ".py",
             "a:*50000 + .key": "cat " + "a:" * 50000 + ".key", ":*100000": "cat " + ":" * 100000 + "secret.json"}

    def test_through_the_hook(self):
        for name, command in self.CASES.items():
            with self.subTest(name):
                sb = Sandbox()
                self.addCleanup(sb.cleanup)
                payload = {"v": 1, "harness": "pi", "event": "tool", "session_id": "colon", "turn_id": "t", "tool_name": "bash", "tool_call_id": "c",
                           "cwd": str(sb.project), "tool_input": {"command": command}, "tool_output": "ok"}
                best = min(self._timed(sb, payload) for _ in range(2))
                self.assertLess(best, 3000, name)

    def _timed(self, sb: Sandbox, payload: dict) -> float:
        started = time.perf_counter()
        result = sb.hook("pi", "", payload)  # 30 s timeout in the helper
        self.assertEqual(result.returncode, 0, result.stderr)
        return (time.perf_counter() - started) * 1000

    def test_python_is_linear(self):
        rs = redact.ruleset()
        for name, command in self.CASES.items():
            started = time.perf_counter()
            redact.sensitive(tool({"command": command}), rs)
            self.assertLess((time.perf_counter() - started) * 1000, 1500, name)

    def test_rules_still_match_at_a_revision_and_after_a_colon(self):
        events = [tool({"command": c}) for c in ("git show HEAD:deploy.key", "cat a:b:server.pem", "git show v1:secrets.json", "cat x:y:z")]
        py, _ = assert_parity(self, events)
        self.assertEqual([out.get("sensitive_rule") for out in py], ["key_file", "key_file", "secret_file", None])


class LongUnquotedTests(unittest.TestCase):
    """Finding 3: an unquoted credential longer than 256 characters matched nothing. It is now matched whole; one too long
    for any window fails closed."""

    def test_302_characters(self):
        token = _rand(302)
        events = [tool({"command": "x"}, "refresh_token=" + token), tool({"command": "x"}, "export API_KEY=" + token + "\nnext"),
                  tool({"command": "x"}, {"token": token})]
        py, _ = assert_parity(self, events)
        for out in py:
            self.assertNotIn("redaction_error", out)
            self.assertEqual(leaked(token, json.dumps(out)), [])

    def test_too_long_for_a_window_fails_closed(self):
        limits = redact.rules()["limits"]
        token = _rand(limits["window"] + 100)
        py, _ = assert_parity(self, [tool({"command": "x"}, filler(100) + " refresh_token=" + token + " " + filler(100))])
        self.assertTrue(py[0]["redaction_error"])
        self.assertEqual(leaked(token, json.dumps(py)), [])

    def test_end_to_end(self):
        token = _rand(302)
        sb, sink, inbox = pi_run(self, output="refresh_token=" + token)
        self.assertEqual(leaked(token, *everywhere(sb, sink, inbox)), [])


class OversizedMemberTests(unittest.TestCase):
    """Finding 4: a structured secret member too long for a window had no closing quote in it, so it matched nothing.
    A value that runs to the end of a window now matches there, and one no window holds whole fails closed."""

    def test_20000_character_password(self):
        big = _rand(20000)
        events = [tool({"command": "x"}, {"password": big}), tool({"command": "x"}, {"nested": [{"api_key": big}]}), tool({"password": big})]
        py, _ = assert_parity(self, events)
        for out in py:
            self.assertTrue(out.get("redaction_error"))
            self.assertEqual(leaked(big, json.dumps(out)), [])

    def test_long_but_inside_a_window_is_redacted(self):
        value = _rand(6000)
        py, _ = assert_parity(self, [tool({"command": "x"}, {"note": filler(9000), "password": value})])
        self.assertNotIn("redaction_error", py[0])
        self.assertEqual(py[0]["tool_output"]["password"], "[REDACTED:secret_field]")

    def test_end_to_end(self):
        big = _rand(20000)
        sb, sink, inbox = pi_run(self, output={"password": big})
        self.assertEqual(leaked(big, *everywhere(sb, sink, inbox)), [])


class EnvDumpWrapperTests(unittest.TestCase):
    """Finding 5: shell wrappers, command prefixes, and leading assignments hid an environment dump. Commands are now read
    by a shell tokenizer (redact-rules.json defs sh_*): a command position is the start, after a separator, inside a shell's
    -c string or eval's, or inside a "...$(...)"; assignments and wrappers before the command word are skipped."""

    DUMPS = ['bash -c "env"', "command env", "FOO=1 printenv", "sh -c 'cd /tmp; env'", 'bash -lc "ls && env"', 'echo "$(env)"',
             'echo "$(date)"; env; echo "done"', "sudo -u root env", "env FOO=1 env", "nohup env &", "timeout 5 printenv",
             "if true; then env; fi", "eval 'env'", "FOO='a b' env | sort", "x=`printenv`", "/usr/bin/env -0 > /tmp/e",
             "cat <<'EOF' > f\nhello\nEOF\nenv"]

    def test_both_layers(self):
        py, _ = assert_parity(self, [tool({"command": c}) for c in self.DUMPS])
        self.assertEqual([c for c, out in zip(self.DUMPS, py) if out.get("sensitive_rule") not in ("env_dump", "printenv")], [])

    def test_end_to_end(self):
        opaque = _rand(28)
        sb, sink, inbox = pi_run(self, command='bash -c "env"', output="SOMETHING_ELSE=" + opaque)
        for where in everywhere(sb, sink, inbox):
            self.assertEqual(leaked(opaque, where), [])
        self.assertIn('"sensitive_rule": "env_dump"', inbox)


class NonExecutingMentionTests(unittest.TestCase):
    """Finding 6: informational options and quoted text counted as dumps. --help and --version do not dump; a quoted
    string, a quoted heredoc's body, and a lookup (which, command -v) run nothing."""

    MENTIONS = ["env --help", "env --version", "printenv --help", "sudo env --version", 'echo "foo; printenv"', "echo 'a | env'",
                "git commit -m 'run env; then set'", "grep -n 'printenv' src", "cat > f <<'EOF'\nenv\nprintenv\ncat credentials\nEOF",
                "set -euo pipefail", "export PATH=/x:$PATH", "python -m venv .venv", 'bash -c "echo \'env\'"', "command -v env",
                'git commit -m "rotate secrets"', "rg secrets", "echo credentials"]

    def test_both_layers(self):
        py, _ = assert_parity(self, [tool({"command": c}) for c in self.MENTIONS])
        self.assertEqual([(c, out.get("sensitive_rule")) for c, out in zip(self.MENTIONS, py) if out.get("sensitive")], [])

    def test_end_to_end_keeps_the_output(self):
        marker = "usage " + _rand(10)
        sb, sink, inbox = pi_run(self, command="env --help", output=marker)
        self.assertIn(marker, sink.text())
        sb, sink, inbox = pi_run(self, command='echo "foo; printenv"', output=marker)
        self.assertIn(marker, sink.text())

    def test_tokens_never_read_text_two_ways(self):
        """The grammar's runs exclude every character another token starts with: the bug that made backslashes exponential."""
        d = redact.rules()["defs"]
        run = redact._compile(redact.compose(d["sh_run"]))
        for char in "\\'\"`$;&|()<> \t\n":
            self.assertIsNone(run.fullmatch(char), repr(char))

    def test_adversarial_commands_are_linear_in_python(self):
        rs = redact.ruleset()
        for unit in ["\\", '"$(', "sudo -u x ", "env -u ", "bash -c \"", "'", "FOO=1 ", ";a", "cat secret:"]:
            text = (unit * (100_000 // len(unit) + 1))[:100_000]
            started = time.perf_counter()
            redact.sensitive(tool({"command": text}), rs)
            self.assertLess((time.perf_counter() - started) * 1000, 2000, repr(unit))


class MultilineQuotedTests(unittest.TestCase):
    """Finding 7: a quoted value was read to the end of its line. A value whose first line has no blank and ends the line
    now runs to its closing quote, across lines."""

    def test_both_layers(self):
        first, second = _rand(20), _rand(24)
        events = [tool({"command": "x"}, 'password = "' + first + "\n" + second + '"\nnext line'),
                  tool({"command": "x"}, "DB_PASSWORD='" + first + "\n  " + second + "'"),
                  tool({"command": "x"}, 'api_key: "' + first + "\\\n" + second + '"')]
        py, _ = assert_parity(self, events)
        for out in py:
            self.assertNotIn("redaction_error", out)
            self.assertEqual(leaked(first, json.dumps(out)) + leaked(second, json.dumps(out)), [])
        self.assertIn("next line", py[0]["tool_output"])

    def test_ordinary_code_is_not_swallowed(self):
        text = 'print("password: " + pw)\nlog("done")\nx = "ok"'
        py, _ = assert_parity(self, [tool({"command": "x"}, text)])
        self.assertIn('log("done")', py[0]["tool_output"])

    def test_end_to_end(self):
        second = _rand(24)
        sb, sink, inbox = pi_run(self, output='password = "' + _rand(20) + "\n" + second + '"')
        self.assertEqual(leaked(second, *everywhere(sb, sink, inbox)), [])


def nest(value: dict, depth: int) -> str:
    """The value serialized ``depth`` times, each inside a message field, as a JSON log writes it."""
    text = json.dumps(value)
    for level in range(depth - 1):
        text = json.dumps({"level": level, "message": text})
    return text


class NestedJsonTests(unittest.TestCase):
    """Finding 8: the rules read a quote and one escaped quote. JSON nested deeper in a string is now decoded, up to
    limits.depth layers, and what each layer holds is replaced as encoded where it sits; deeper fails closed."""

    def test_codex_case_and_deeper(self):
        for depth in (3, 4, 5):
            with self.subTest(depth=depth):
                value = _rand(24)
                py, _ = assert_parity(self, [tool({"command": "x"}, nest({"password": value, "user": "ann"}, depth))])
                self.assertNotIn("redaction_error", py[0])
                self.assertEqual(leaked(value, json.dumps(py)), [])
                self.assertIn("ann", py[0]["tool_output"])
                json.loads(py[0]["tool_output"])  # still valid JSON

    def test_jsonl_and_log_prefix(self):
        value = _rand(24)
        line = nest({"password": value}, 3)
        text = "\n".join([line, "INFO payload=" + line, json.dumps({"ok": 1})])
        py, _ = assert_parity(self, [tool({"command": "x"}, text)])
        self.assertEqual(leaked(value, json.dumps(py)), [])

    def test_value_with_quotes_and_backslashes(self):
        value = 'Pa"ss\\w' + _rand(20)
        py, _ = assert_parity(self, [tool({"command": "x"}, nest({"password": value}, 3))])
        self.assertEqual(leaked(value[8:], json.dumps(py)), [])

    def test_too_deep_fails_closed(self):
        value = _rand(24)
        depth = redact.rules()["limits"]["depth"] + 3
        py, _ = assert_parity(self, [tool({"command": "x"}, nest({"password": value}, depth))])
        self.assertTrue(py[0]["redaction_error"])
        self.assertEqual(leaked(value, json.dumps(py)), [])

    def test_written_another_way_fails_closed(self):
        # \u escapes where the layer's encoding has the character itself: the find cannot be replaced where it sits.
        value = "p\u00e9" + _rand(22)
        inner = json.dumps({"password": value}, ensure_ascii=True)
        text = json.dumps({"m": json.dumps({"body": inner}, ensure_ascii=True)}, ensure_ascii=True)
        py, _ = assert_parity(self, [tool({"command": "x"}, text)])
        self.assertEqual(leaked(value[2:], json.dumps(py, ensure_ascii=False)), [])

    def test_end_to_end(self):
        value = _rand(24)
        sb, sink, inbox = pi_run(self, output=nest({"password": value}, 3))
        self.assertEqual(leaked(value, *everywhere(sb, sink, inbox)), [])


class SecretFilenameTests(unittest.TestCase):
    """Finding 9: dotenv backups and extensionless secret files were missed."""

    SENSITIVE = {"cat .env.local~": "dotenv", "cat .env~": "dotenv", "cp .env.production.bak /tmp": "dotenv", "cat secrets": "secret_file",
                 "cat config/secrets": "secret_file", "git show HEAD:secrets": "secret_file", "less ./secrets": "secret_file",
                 "head -n 3 SECRETS": "secret_file"}
    KEPT = ["cat .env.example~", "cat .env.sample", "cat secrets.py", "cat src/secret_manager.ts", 'git commit -m "rotate secrets"',
            "grep -r secrets .", "echo secrets", "ls secret_manager/"]

    def test_both_layers(self):
        commands = list(self.SENSITIVE) + self.KEPT
        py, _ = assert_parity(self, [tool({"command": c}) for c in commands])
        got = {c: out.get("sensitive_rule") for c, out in zip(commands, py)}
        self.assertEqual({c: got[c] for c in self.SENSITIVE}, self.SENSITIVE)
        self.assertEqual({c: got[c] for c in self.KEPT}, {c: None for c in self.KEPT})

    def test_end_to_end(self):
        for command in ("cat .env.local~", "cat secrets"):
            opaque = _rand(28)
            sb, sink, inbox = pi_run(self, command=command, output="X=" + opaque)
            self.assertEqual(leaked(opaque, *everywhere(sb, sink, inbox)), [], command)


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


class HermeticTempTests(unittest.TestCase):
    """The suite runs in its own temp folder, for this process and every process a test starts."""

    def test_private_tmpdir(self):
        self.assertEqual(tempfile.gettempdir(), str(support.TMP))
        self.assertEqual(os.environ["TMPDIR"], str(support.TMP))
        self.assertEqual(clean_env()["TMPDIR"], str(support.TMP))
        sb = Sandbox()
        self.addCleanup(sb.cleanup)
        self.assertEqual(sb.env["TMPDIR"], str(support.TMP))
        self.assertEqual(support._ancestor_markers(support.TMP), [])
        self.assertTrue(support.TMP.name.startswith("jt-suite-"))
        self.assertNotEqual(support.TMP, re.sub("/+$", "", "/tmp"))


if __name__ == "__main__":
    unittest.main()
