"""Regressions for the tenth review of capture: sanitized. Each case runs in both layers (jq and its Python mirror, which
must agree), and the sender's own pass where the finding was a sender leak.

Every fake secret is built at runtime from parts (_rand below), so no file holds one.
"""

from __future__ import annotations

import json
import random
import string
import tempfile
import time
import unittest
from pathlib import Path

from support import Sandbox
from test_review9 import outputs, rules_of
from test_review_fixes import tool

from judgment_core import optin, redact

_rng = random.Random(1010)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


class UnreadableProjectTests(unittest.TestCase):
    """Finding 1: the sender read a tracing.json it could not parse as a project with no rules of its own, so transcript
    text went out under the built-in rules alone. The hook's jq had read the same file."""

    def load(self, text: str | None) -> dict:
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracing.json"
            if text is not None:
                path.write_bytes(text.encode("utf-8"))
            return optin.load_project(path)

    def test_read_as_jq_reads_it(self):
        written = json.dumps({"enabled": True, "redactPatterns": ["ticket=[0-9]+"]})
        for text in (written, "\ufeff" + written, written + "\n{}", "\n  " + written):
            with self.subTest(text=text[:12]):
                project = self.load(text)
                self.assertEqual((project["enabled"], project["redactPatterns"]), (True, ["ticket=[0-9]+"]))
                self.assertEqual(redact.ruleset(project).error, "")

    def test_unreadable_fails_closed(self):
        for text in (None, "", "{", '{"enabled": true, "redactPatterns": ["ticket', "[1]", "null"):
            with self.subTest(text=text):
                project = self.load(text)
                self.assertFalse(project["enabled"])
                rs = redact.ruleset(project)
                self.assertTrue(rs.error)
                event = redact.redact_event(tool({"command": "x"}, "ticket=12345 and more"), rs, [])
                self.assertTrue(event["redaction_error"])
                self.assertNotIn("tool_output", event)


class EscapeBoundaryTests(unittest.TestCase):
    """Finding 2: a secret right after \\uXXXX, \\b, \\f, or an ANSI color code failed every identifier boundary."""

    def test_secrets_after_more_escapes_go(self):
        token = "gh" + "p_" + _rand(36)
        password = "Pw" + _rand(14)
        texts = ['{"Args":["--dsn=x?sslmode=disable\\u0026password=' + password + '"]}', '{"log": "a\\u0000' + token + '"}',
                 '{"log": "a\\u00a0' + token + '"}', '{"log": "a\\b' + token + '"}', '{"log": "a\\f' + token + '"}',
                 "\x1b[32m" + token + "\x1b[0m", "\x1b[1mpassword=" + password, "\x1b[1;31m" + token]
        py = outputs(self, texts)
        self.assertEqual([text for text, out in zip(texts, py) if token in json.dumps(out) or password in json.dumps(out)], [])
        sender = [redact.redact_event(tool({"command": "x"}, text), redact.ruleset(), []) for text in texts]
        self.assertEqual([text for text, out in zip(texts, sender) if token in json.dumps(out) or password in json.dumps(out)], [])

    def test_ordinary_text_stays(self):
        texts = ["took 30ms total", "C:\\build\\ui\\form.py", "rgb 3;4mode", "100max_tokens=5"]
        self.assertEqual([out["tool_output"] for out in outputs(self, texts)], texts)


class AllowOrderTests(unittest.TestCase):
    """Finding 3: the hook kept one pair per found text before it dropped allowed finds, so a built-in find the allow list
    names took a project's redactPatterns find of the same text with it."""

    def test_a_custom_find_survives_an_allowed_one(self):
        cfg = {"redactPatterns": ["[a-z_]*token=[a-z]+"]}
        py = outputs(self, ["x token=true y", "x token=None y"], cfg)
        self.assertEqual(py[0]["tool_output"], "x [REDACTED:custom] y")
        self.assertEqual(py[1]["tool_output"], "x token=None y")


class CommandTests(unittest.TestCase):
    def test_help_check_is_linear(self):
        """Finding 4: ( and { start a command but did not stop the help check, so each start read to the end of the text."""
        rs = redact.ruleset()
        times = []
        for count in (4_000, 16_000):
            event = tool({"command": "$(gh auth token " * count + " --help"}, "out")
            start = time.perf_counter()
            self.assertEqual(redact.sensitive(event, rs)["rule"], "secret_cli")
            times.append(time.perf_counter() - start)
        self.assertLess(times[1], times[0] * 10 + 0.05)

    def test_env_dump_with_a_redirect(self):
        """Finding 5: a file descriptor or an input redirect after env hid the dump."""
        commands = ["env 2>&1", "env 2>/dev/null", "env 2>&1 | grep -i key", "set 2>&1 | head", "export 2>&1", "env </dev/null",
                    "env 1>out.txt", "env > x", "env &>out"]
        self.assertEqual(rules_of(self, commands), {command: "env_dump" for command in commands})
        harmless = ["env FOO=1 make", "env 2", "set -e", "export FOO=1"]
        self.assertEqual(rules_of(self, harmless), {command: None for command in harmless})

    def test_shell_by_its_path(self):
        """Finding 7: sh -c counted, and /usr/bin/env counted, but not /bin/sh -c env."""
        commands = ["/bin/sh -c 'env'", "/bin/bash -c env", "/usr/bin/bash -lc \"printenv\"", "/usr/local/bin/zsh -c 'gh auth token'"]
        py = rules_of(self, commands)
        self.assertEqual([command for command, rule in py.items() if rule is None], [])


class EngineParityTests(unittest.TestCase):
    def test_information_separators_are_no_blank(self):
        """Finding 8: Python's \\s takes U+001C to U+001F, Oniguruma's does not, so a value ended at one in the sender only."""
        password = "Pw" + _rand(10)
        tail = "Zz" + _rand(10)
        texts = ["password=" + password + sep + tail for sep in ("\x1c", "\x1d", "\x1e", "\x1f")]
        py = outputs(self, texts)
        self.assertEqual([text for text, out in zip(texts, py) if tail in json.dumps(out)], [])
        self.assertEqual(rules_of(self, ["env\x1f", "gh\x1fauth\x1ftoken"]), {"env\x1f": None, "gh\x1fauth\x1ftoken": None})

    def test_blank_classes_match_oniguruma(self):
        blank = redact._compile(r"\s")
        inner = redact._compile(r"[^\S\n]")
        outer = redact._compile(r"[^\s,]")
        for code in range(1, 0x3100):
            char = chr(code)
            expected = char in redact._BLANKS
            self.assertEqual(bool(blank.fullmatch(char)), expected, hex(code))
            self.assertEqual(bool(inner.fullmatch(char)), expected and char != "\n", hex(code))
            self.assertEqual(bool(outer.fullmatch(char)), not expected and char != ",", hex(code))
        for regex in (r"[\s\S]", r"(?<![\s\S])a", r"[\S\n]", r"[^\S\n\x20]", r"[^\S a-c]"):
            redact._compile(regex)

    def test_posix_class_fails_closed(self):
        """Finding 9: Oniguruma reads [[:alpha:]] as a class, Python as a set of characters, with a warning."""
        cfg = {"redactPatterns": ["[[:alpha:]]+=12345"]}
        self.assertTrue(redact.ruleset(cfg).error)
        event = tool({"command": "x"}, "hello ticket=12345")
        self.assertTrue(redact.redact_event(event, redact.ruleset(cfg), [])["redaction_error"])

    def test_kept_group_is_the_first_group(self):
        """Finding 9: the layers kept different text when p was not at the start of the match, or not the first group."""
        text = "hello ticket=12345"
        for regex, expected in (("(?<p>ticket=)[0-9]+", "hello ticket=[REDACTED:custom]"), ("ticket=(?<p>[0-9])[0-9]+", "hello 1[REDACTED:custom]"),
                                ("(ticket=)(?<p>[0-9]+)", "hello [REDACTED:custom]"), ("(?<q>ticket=)(?<p>[0-9]+)", "hello [REDACTED:custom]")):
            with self.subTest(regex=regex):
                self.assertEqual(outputs(self, [text], {"redactPatterns": [regex]})[0]["tool_output"], expected)


class NameClosingTests(unittest.TestCase):
    """Finding 10: a name closed by a backtick or by Markdown bold was no name, and a value after a bold colon was **."""

    def test_markdown_names_lose_their_value(self):
        password = "Pw" + _rand(12)
        texts = ["`password`: " + password, "**password**: " + password, "**Password:** " + password]
        py = outputs(self, texts)
        self.assertEqual([out["tool_output"] for out in py],
                         ["`password`: [REDACTED:secret_assignment]", "**password**: [REDACTED:secret_assignment]", "**Password:** [REDACTED:secret_assignment]"])


class SendTests(unittest.TestCase):
    """Finding 1, end to end: a tracing.json cut off after the turn was recorded sends the turn without its text."""

    def test_sender_drops_text_when_the_project_file_broke(self):
        from fake_sink import Sink
        from payloads import pi_turn

        for broken in (False, True):
            with self.subTest(broken=broken), Sink() as sink:
                sb = Sandbox({"enabled": True, "redactPatterns": ["ticket=[0-9]+"]}, sink_url=sink.url)
                self.addCleanup(sb.cleanup)
                for _, payload in pi_turn(str(sb.project), "pi-broken", "pi-broken:1", int(time.time() * 1000) - 5000):
                    if payload["event"] == "turn_start":
                        payload = {**payload, "prompt": "marmalade ticket=424242"}
                    self.assertEqual(sb.hook("pi", "", payload).returncode, 0)
                if broken:
                    (sb.project / ".judgment" / "tracing.json").write_text('{"enabled": true, "redactPatterns": ["ticket')
                self.assertEqual(sb.flush().returncode, 0)
                self.assertTrue(sink.requests)
                self.assertNotIn("424242", sink.text())
                self.assertEqual("marmalade" in sink.text(), not broken)


if __name__ == "__main__":
    unittest.main()
