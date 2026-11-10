"""Regressions for the fifth review of capture: sanitized. One class per blocking finding, each in both layers (jq and its
Python mirror, which must agree) and end to end: hook, inbox, stored turn, and the HTTP body the sender posts.

Every fake secret is built at runtime from parts (_rand below), so no file holds one.
"""

from __future__ import annotations

import json
import random
import string
import time
import unittest

from test_review3 import everywhere, pi_run
from test_review4 import assert_parity, leaked
from test_review_fixes import tool

from judgment_core import redact

_rng = random.Random(5555)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def outputs(test: unittest.TestCase, texts: list[str]) -> list[dict]:
    py, _ = assert_parity(test, [tool({"command": "x"}, text) for text in texts])
    return py


def commands(test: unittest.TestCase, cmds: list[str]) -> list[dict]:
    py, _ = assert_parity(test, [tool({"command": cmd}) for cmd in cmds])
    return py


CODEX_JS = 'console.log("password: "+password);\nrunTests();\nconsole.log("all tests passed");'

# Ordinary code that mentions a secret word in a string, across languages: the quote after the colon (or =) closes the
# string, so nothing here opens a value. Each must come out byte for byte.
PRESERVED = [
    CODEX_JS,
    "console.log('password: '+password);\nrunTests();\nconsole.log('all tests passed');",
    'console.log("Enter password: "+pw);\nnext();\nlog("x");',
    "console.log(`token: ${token}`);\nrun();\nconsole.log(`done`);",
    'console.log(`password: ` + password);\nrun();\nconsole.log("ok");',
    'console.log("token=" + token);\nrun();\nlog("done");',
    'throw new Error("invalid secret: "+s);\n}\nreturn "ok";',
    'assert.equal(x, "password: "+y);\nnext();\nconst z = "q";',
    '  "password: "+password);\nrunTests();\nconsole.log("all");',
    'logger.info("secret: %s", redacted)\nrun()\nprint("ok")',
    'print(f"password: {pw}")\nrun()\nprint("ok")',
    "print(f'api_key: {k}')\nrun()\nprint('ok')",
    'print("password:", pw)\nrun()\nprint("ok")',
    "print('password: '+pw)\nrun()\nprint('ok')",
    'msg = "api_key: " + key\nrun()\nprint("ok")',
    'raise ValueError("secret: " + s)\nrun()\nx = "y"',
    'fmt.Println("password: "+pw)\nrun()\nfmt.Println("ok")',
    'fmt.Printf("password=%s\\n", pw)\nrun()\nfmt.Println("ok")',
    'log.Printf("secret: "+s)\nrun()\nlog.Printf("ok")',
    'return fmt.Errorf("bad password: "+err.Error())\n}\nfmt.Println("ok")',
    'echo "password: "$PASSWORD\nrun_tests\necho "all tests passed"',
    "echo 'secret: '$S\nmake\necho 'done'",
    'printf "password: %s\\n" "$P"\nmake\necho "done"',
    'read -p "Password: "$x\nmake\necho "done"',
]


class AssignmentContextTests(unittest.TestCase):
    """Finding 1: the closing quote of a string such as "password: " was read as a value's opening quote, and the value ran
    across lines, deleting the code after it. A value now runs across lines only in an assignment's context."""

    def test_codex_case(self):
        out = outputs(self, [CODEX_JS])[0]
        self.assertEqual(out["tool_output"], CODEX_JS)
        self.assertNotIn("redactions", out)

    def test_preservation_corpus_byte_for_byte(self):
        changed = [(text, out["tool_output"]) for text, out in zip(PRESERVED, outputs(self, PRESERVED)) if out["tool_output"] != text]
        self.assertEqual(changed, [])

    def test_multiline_secrets_in_context_are_still_redacted(self):
        first, second = _rand(20), _rand(24)
        forms = ['password = "{a}\n{b}"\nnext line', "DB_PASSWORD='{a}\n  {b}'", 'api_key: "{a}\\\n{b}"', '    password = "{a}\n{b}"',
                 'export DB_PASSWORD="{a}\n{b}"', '{{"password": "{a}\n{b}"}}', '  "client_secret": "{a}\n{b}",', "  'password': '{a}\n{b}'",
                 'self.password = "{a}\n{b}"', '- password: "{a}\n{b}"', '$password = "{a}\n{b}"', 'password := "{a}\n{b}"',
                 'run --db-password="{a}\n{b}"', "docker run -e PASSWORD='{a}\n{b}' img", 'x = 1\nconst password = "{a}\n{b}";']
        texts = [form.format(a=first, b=second) for form in forms]
        for form, out in zip(forms, outputs(self, texts)):
            self.assertNotIn("redaction_error", out, form)
            self.assertEqual(leaked(first, json.dumps(out)) + leaked(second, json.dumps(out)), [], form)
        self.assertIn("next line", outputs(self, [texts[0]])[0]["tool_output"])

    def test_a_secret_beside_logging_code_is_still_redacted(self):
        value = _rand(22)
        text = CODEX_JS + '\npassword = "' + value + '"\nrunTests();'
        out = outputs(self, [text])[0]
        self.assertEqual(leaked(value, json.dumps(out)), [])
        self.assertTrue(out["tool_output"].startswith(CODEX_JS))
        self.assertTrue(out["tool_output"].endswith('"\nrunTests();'))

    def test_single_line_spans_are_unchanged(self):
        """Context is read only for a value that spans lines: a one-line value matches from its key, as it did."""
        value = _rand(22)
        out = outputs(self, ['{"api_key": "' + value + '"}', '  password = "' + value + ' x"'])
        self.assertEqual(out[0]["tool_output"], '{"api_key": "[REDACTED:secret_assignment]"}')
        self.assertEqual(out[1]["tool_output"], '  password = "[REDACTED:quoted_secret]"')

    def test_end_to_end(self):
        sb, sink, inbox = pi_run(self, output=CODEX_JS)
        for where in everywhere(sb, sink, inbox):
            for piece in ("+password);", "runTests();", "all tests passed"):
                self.assertIn(piece, where)
        second = _rand(24)
        sb, sink, inbox = pi_run(self, output=CODEX_JS + '\npassword = "' + _rand(20) + "\n" + second + '"\nrunTests();')
        self.assertEqual(leaked(second, *everywhere(sb, sink, inbox)), [])
        self.assertIn("all tests passed", sink.text())


class EnvLongOptionTests(unittest.TestCase):
    """Finding 2: an option's separate argument (env --unset DEBUG) was read as the command env runs, so the dump was
    missed. env's option grammar is now data (shell_options.env): options that take an argument consume it, in either
    form, before the rule decides whether a command follows."""

    DUMPS = ["env --unset DEBUG", "env --chdir /tmp", "env --unset DEBUG printenv", "env -u X", "env -C dir", "env -i",
             "env --ignore-environment", "env --null", "env -0", "env --unset=DEBUG", "env --chdir=/tmp", "env -uX", "env -iu X",
             "env -u X -C /tmp", "env --uns DEBUG", "env --unset DEBUG --chdir=/tmp -0", "env -", "env --", "env -- printenv",
             "env -S printenv", "env -S 'printenv'", "env --split-string=printenv", "env -u DEBUG env", "sudo env --unset X",
             "FOO=1 env --chdir /tmp", "bash -c 'env --unset X'", "env --block-signal=INT", "env -P /bin printenv", "env --argv0 x",
             "env -i -u HOME", "env --unset X printenv HOME", "env -v -u X", "env -u X | sort", "/usr/bin/env --unset=A --unset B",
             'echo "$(env --unset X)"', "env --unset DEBUG; ls"]
    RUNS = ["env --unset DEBUG ls", "env -C /tmp ls", "env -u X make test", "env --chdir=/tmp make", "env -S 'ls -l'",
            "env --split-string='ls -l'", "env --help", "env --version", "env -iu X ls", "env --block-signal INT", "env -- ls",
            "env --unset printenv ls", "env -i PATH=/bin ls", "env --argv0 x ls", "env -P /bin ls", "sudo env --help"]

    def test_dumps(self):
        py = commands(self, self.DUMPS)
        self.assertEqual([c for c, out in zip(self.DUMPS, py) if out.get("sensitive_rule") not in ("env_dump", "printenv")], [])

    def test_commands_that_run_something_else(self):
        py = commands(self, self.RUNS)
        self.assertEqual([(c, out.get("sensitive_rule")) for c, out in zip(self.RUNS, py) if out.get("sensitive")], [])

    def test_abbreviations_only_as_far_as_getopt_accepts(self):
        frag = redact.generated()
        long_arg = redact._compile("(?:" + frag["env_long_arg"] + ")")
        for word in ("u", "uns", "unset", "c", "chdir", "a", "argv0"):
            self.assertIsNotNone(long_arg.fullmatch(word), word)
        for word in ("unsetx", "ch-dir", "null", "i"):
            self.assertIsNone(long_arg.fullmatch(word), word)
        self.assertNotIn("u", frag["env_short_flag"])
        self.assertNotIn("S", frag["env_short_flag"])

    def test_end_to_end(self):
        opaque = _rand(28)
        sb, sink, inbox = pi_run(self, command="env --unset DEBUG", output="SOMETHING_ELSE=" + opaque)
        self.assertEqual(leaked(opaque, *everywhere(sb, sink, inbox)), [])
        self.assertIn('"sensitive_rule": "env_dump"', inbox)


class SecretCliPositionTests(unittest.TestCase):
    """Finding 3: secret_cli matched its words anywhere, so a commit message or a search pattern naming one blanked the turn.
    It now reads the command with the shell tokenizer, as the env rules do."""

    REAL = ["gh auth token", "aws configure get aws_secret_access_key", "aws configure export-credentials --profile p",
            "git credential fill", "gcloud auth print-access-token", "gcloud auth application-default print-access-token",
            "security find-generic-password -s x -w", "op read op://v/i/f", "vault read secret/x", "vault kv get secret/x",
            "kubectl get secret -o yaml", "kubectl get secrets -o json", "kubectl describe secret db", "kubectl get secret/db -o json",
            "kubectl -n prod get secret db -o yaml", "aws --profile p configure get aws_secret_access_key", "sudo gh auth token", "bash -c 'gh auth token'",
            "FOO=1 gh auth token", "env -u X gh auth token", "x=$(gh auth token)", 'echo "$(gh auth token)"', "cd repo && gh auth token",
            "ls; kubectl get secret -o json", "TOKEN=`gh auth token`", "/usr/local/bin/gh auth token", "cat <<'EOF'\nx\nEOF\ngh auth token",
            "gh auth token | pbcopy", "timeout 5 vault kv get secret/x", "sudo -u root kubectl get secret -o yaml", "cat <<EOF\ngh auth token\nEOF"]
    MENTIONS = ['git commit -m "fix gh auth token documentation"', "rg 'aws configure get' .", "echo gh auth token",
                "grep -e 'gh auth token' x", "cat <<'EOF'\ngh auth token\nEOF", "git log --grep='kubectl get secret'",
                "printf '%s\\n' 'vault read x'", "rg -g '*.md' 'op read'", "git commit -m 'docs: kubectl get secrets'",
                'echo "run: gcloud auth print-access-token"', "gh auth status", "aws configure list", "kubectl get pods", "man gh",
                'grep -rn "git credential fill" docs/', "sed -i 's/gh auth token/gh auth status/' README.md",
                "python -c 'print(\"op read\")'", "git grep 'security find-generic-password'", "rg aws configure get"]

    def test_real_invocations(self):
        py = commands(self, self.REAL)
        self.assertEqual([c for c, out in zip(self.REAL, py) if out.get("sensitive_rule") != "secret_cli"], [])

    def test_mentions_do_not_blank_the_turn(self):
        py = commands(self, self.MENTIONS)
        self.assertEqual([(c, out.get("sensitive_rule")) for c, out in zip(self.MENTIONS, py) if out.get("sensitive")], [])

    def test_end_to_end(self):
        marker = "committed " + _rand(10)
        sb, sink, inbox = pi_run(self, command='git commit -m "fix gh auth token documentation"', output=marker)
        for where in everywhere(sb, sink, inbox):
            self.assertIn(marker, where)
        value = "tok" + _rand(30)
        sb, sink, inbox = pi_run(self, command="gh auth token", output=value)
        self.assertEqual(leaked(value, *everywhere(sb, sink, inbox)), [])
        self.assertIn('"sensitive_rule": "secret_cli"', inbox)


class LinearGrammarTests(unittest.TestCase):
    """The new option grammar and the assignment context cost linear work on adversarial input (Python; the hook's
    windows and budgets bound jq, tests/test_latency.py)."""

    def test_commands(self):
        rs = redact.ruleset()
        for unit in ["env --unset ", "env -iu ", "env -S ", "env --split-string=", "gh -x ", "kubectl -n a ", "env --uns=x ", "- -- "]:
            text = (unit * (100_000 // len(unit) + 1))[:100_000]
            started = time.perf_counter()
            redact.sensitive(tool({"command": text}), rs)
            self.assertLess((time.perf_counter() - started) * 1000, 2000, repr(unit))

    def test_text(self):
        rs = redact.ruleset()
        for unit in ['password = "', '"password": "', ' --password="', '\n  password: "a', 'password="ab\n', "\nexport -e password='"]:
            text = (unit * (100_000 // len(unit) + 1))[:100_000]
            started = time.perf_counter()
            redact.sanitize_event(tool({"command": "x"}, text), rs, [])
            self.assertLess((time.perf_counter() - started) * 1000, 3000, repr(unit))


if __name__ == "__main__":
    unittest.main()
