"""Regressions for the ninth review of capture: sanitized. Each case runs in both layers (jq and its Python mirror, which
must agree), and the sender's own pass where the finding was a sender leak.

Every fake secret is built at runtime from parts (_rand below), so no file holds one.
"""

from __future__ import annotations

import json
import random
import string
import tempfile
import unittest
from pathlib import Path

from support import clean_env
from test_review4 import assert_parity
from test_review_fixes import tool

from judgment_core import optin, redact

_rng = random.Random(8080)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def outputs(test: unittest.TestCase, texts: list, cfg: dict | None = None, env: dict | None = None) -> list[dict]:
    return assert_parity(test, [tool({"command": "x"}, text) for text in texts], cfg, env)[0]


def rules_of(test: unittest.TestCase, commands: list[str], cfg: dict | None = None) -> dict:
    py = assert_parity(test, [tool({"command": command}, "out") for command in commands], cfg)[0]
    return {command: out.get("sensitive_rule") for command, out in zip(commands, py)}


class EscapeBoundaryTests(unittest.TestCase):
    """Finding 1: a secret right after a JSON escape (\\n, \\t, \\r) in JSON nested in a string failed every identifier
    boundary, in both layers and the sender."""

    def test_secrets_after_an_escape_go(self):
        token = "gh" + "p_" + _rand(36)
        password = "Pw" + _rand(14)
        texts = ['{"log": "started\\n' + token + '"}', '{"log": "a\\t' + token + '"}', '{"log": "a\\r' + token + '"}',
                 '{"log": "started\\npassword=' + password + '"}']
        py = outputs(self, texts)
        secrets = [token, token, token, password]
        self.assertEqual([text for text, out, s in zip(texts, py, secrets) if s in json.dumps(out)], [])
        sender = [redact.redact_event(tool({"command": "x"}, text), redact.ruleset(), []) for text in texts]
        self.assertEqual([text for text, out, s in zip(texts, sender, secrets) if s in json.dumps(out)], [])

    def test_escapes_that_are_no_boundary_stay(self):
        texts = ["C:\\new\\tokenizer.py", "a\\nmax_tokens=5"]
        py = outputs(self, texts)
        self.assertEqual([out["tool_output"] for out in py], texts)


class EnvDumpThenCommandTests(unittest.TestCase):
    """Finding 2: env_dump consumed its separator, so the help check read the next command's --version."""

    def test_dump_before_another_command(self):
        commands = ["env; node --version", "set; git --version", "env\ngit --version", "env | grep -h FOO"]
        self.assertEqual(rules_of(self, commands), {command: "env_dump" for command in commands})


class NestedParityTests(unittest.TestCase):
    """Finding 3: the sender looked for nested JSON at an escaped backslash before a quote, the hook at an escaped quote."""

    def test_env_value_in_nested_json(self):
        value = "Ab" + "c9\"d\\e/f<g"
        env = clean_env(MY_API_TOKEN=value)
        text = json.dumps({"a": "v " + value})
        py = outputs(self, [text], env=env)
        self.assertEqual(py[0]["tool_output"], '{"a": "v [REDACTED:env:MY_API_TOKEN]"}')
        sender = redact.redact_event(tool({"command": "x"}, text), redact.ruleset(), redact.env_secrets(env))
        self.assertEqual(sender["tool_output"], py[0]["tool_output"])


class AssignmentFormTests(unittest.TestCase):
    """Findings 5, 6, 7, 11: aligned config, key prefixes with spaces or colons, template literals, and type annotations."""

    def test_aligned_values_go(self):
        value = "hu" + _rand(12)
        texts = ["password" + " " * 20 + "= " + value, "api_key:" + " " * 24 + value]
        self.assertEqual([out["tool_output"] for out in outputs(self, texts) if value in out["tool_output"]], [])

    def test_keys_with_spaces_colons_and_slashes(self):
        value = "Zq" + _rand(20)
        values = [{"DB Password": value}, {"spring:datasource:password": value}, {"secrets/api_token": value}]
        py = assert_parity(self, [tool({"command": "x"}, item) for item in values])[0]
        self.assertEqual([out["tool_output"] for out in py if value in json.dumps(out)], [])

    def test_template_literal(self):
        value = "hu" + _rand(12)
        py = outputs(self, ["const password = `" + value + "`;"])
        self.assertEqual(py[0]["tool_output"], "const password = `[REDACTED:secret_assignment]`;")

    def test_types_and_constants_stay(self):
        texts = ["password: str", "def f(api_key: Optional[str] = None):", "token = None", "secret: boolean;", "let token: string = x",
                 "password = null", "PASSWORD = True", "credentials: bool"]
        py = outputs(self, texts)
        self.assertEqual([out["tool_output"] for out in py], texts)


class WrappedCommandTests(unittest.TestCase):
    """Finding 8: a command position now reads through shell keywords, assignments, sudo with options, sh -c, and a path."""

    CAUGHT = ['bash -c "env"', "bash -lc 'printenv'", "FOO=1 printenv", "sudo -u root env", "sudo -E env", "/usr/bin/env", "if x; then env; fi",
              "for a in b; do printenv; done", "{ env; }", "nohup env", "command env"]
    KEPT = ["which env", "x=$(which env)", "command -v env", "env --help", "bash -c 'make env'", "conda env list", "env FOO=1 make"]

    def test_caught(self):
        got = rules_of(self, self.CAUGHT)
        self.assertEqual([command for command, rule in got.items() if rule not in ("env_dump", "printenv")], [])

    def test_kept(self):
        self.assertEqual({command: rule for command, rule in rules_of(self, self.KEPT).items() if rule}, {})


class PathGapTests(unittest.TestCase):
    """Finding 9: dotenv files named another way, and secret-named source files."""

    def test_paths(self):
        blanked = ["cat prod.env", "cat .env-prod", "cat config/.envrc.local"]
        kept = ["cat example.env", "cat src/secrets.sh", "vim secret.c", "cat templates/secret.html", "cat db/secrets.sql"]
        got = rules_of(self, blanked + kept)
        self.assertEqual({c: got[c] for c in blanked}, {c: "dotenv" for c in blanked})
        self.assertEqual({c: got[c] for c in kept if got[c]}, {})


class ConfigTypeTests(unittest.TestCase):
    """Finding 10: a setting that is not a list of strings was skipped, so a rule the project asked for silently did nothing."""

    BAD = [{"redactPatterns": "hunter[0-9]+"}, {"redactPaths": {"a": 1}}, {"redactCommands": ["ok", 5]}, {"redactNames": "pin"}]

    def test_both_layers_fail_closed(self):
        for cfg in self.BAD:
            py = assert_parity(self, [tool({"command": "x"}, "hello")], cfg)[0]
            self.assertTrue(py[0].get("redaction_error"), cfg)

    def test_load_project_keeps_the_bad_setting(self):
        for cfg in self.BAD:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "tracing.json"
                path.write_text(json.dumps({"enabled": True, **cfg}))
                self.assertTrue(redact.ruleset(optin.load_project(path)).error, cfg)

    def test_trims_stay_lenient(self):
        py = assert_parity(self, [tool({"command": "x"}, "hello")], {"redactAllow": "x", "redactDisable": 3})[0]
        self.assertEqual(py[0]["tool_output"], "hello")


class TrimBuiltinsTests(unittest.TestCase):
    """Finding 12: !name drops a built-in name, !glob excepts a path from blanking."""

    def test_drop_a_name(self):
        value = "hu" + _rand(12)
        py = outputs(self, ["token=" + value, "password=" + value], {"redactNames": ["!token", "pin"]})
        self.assertEqual([out["tool_output"] for out in py], ["token=" + value, "password=[REDACTED:secret_assignment]"])

    def test_drop_an_unknown_name_fails_closed(self):
        py = outputs(self, ["hello"], {"redactNames": ["!pin"]})
        self.assertTrue(py[0].get("redaction_error"))

    def test_except_a_path(self):
        got = rules_of(self, ["cat .env.test", "cat app/.env.test", "cat .env"], {"redactPaths": ["!.env.test"]})
        self.assertEqual(got, {"cat .env.test": None, "cat app/.env.test": None, "cat .env": "dotenv"})


if __name__ == "__main__":
    unittest.main()
