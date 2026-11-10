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
# Known limit (reference-value): a format placeholder or a $reference right after the name is replaced; nothing else changes.
REFERENCES = {PRESERVED[i] for i in (3, 9, 17, 20, 21, 22, 23)}


class AssignmentContextTests(unittest.TestCase):
    """Finding 1: the closing quote of a string such as "password: " was read as a value's opening quote, and the value ran
    across lines, deleting the code after it. A value never crosses a line now, and a quoted value cannot start with a
    blank or + ) ; ] }, which follow a string that just closed."""

    def test_codex_case(self):
        out = outputs(self, [CODEX_JS])[0]
        self.assertEqual(out["tool_output"], CODEX_JS)
        self.assertNotIn("redactions", out)

    def test_preservation_corpus_byte_for_byte(self):
        kept = [text for text in PRESERVED if text not in REFERENCES]
        changed = [(text, out["tool_output"]) for text, out in zip(kept, outputs(self, kept)) if out["tool_output"] != text]
        self.assertEqual(changed, [])

    def test_known_limit_references_after_the_name_go(self):
        refs = sorted(REFERENCES)
        for text, out in zip(refs, outputs(self, refs)):
            result = out["tool_output"]
            self.assertIn("[REDACTED:secret_assignment]", result, text)
            self.assertEqual(result.split("\n")[1:], text.split("\n")[1:], text)

    def test_one_line_values_in_every_form(self):
        value = _rand(20)
        forms = ['password = "{a}"\nnext line', "DB_PASSWORD='{a}'", '    password = "{a}"', 'export DB_PASSWORD="{a}"', '{{"password": "{a}"}}',
                 '  "client_secret": "{a}",', "  'password': '{a}'", 'self.password = "{a}"', '- password: "{a}"', '$password = "{a}"',
                 'password := "{a}"', 'run --db-password="{a}"', "docker run -e PASSWORD='{a}' img", 'x = 1\nconst password = "{a}";']
        texts = [form.format(a=value) for form in forms]
        for form, out in zip(forms, outputs(self, texts)):
            self.assertNotIn("redaction_error", out, form)
            self.assertEqual(leaked(value, json.dumps(out)), [], form)
        self.assertIn("next line", outputs(self, [texts[0]])[0]["tool_output"])

    def test_a_secret_beside_logging_code_is_still_redacted(self):
        value = _rand(22)
        text = CODEX_JS + '\npassword = "' + value + '"\nrunTests();'
        out = outputs(self, [text])[0]
        self.assertEqual(out["tool_output"], CODEX_JS + '\npassword = "[REDACTED:secret_assignment]"\nrunTests();')

    def test_end_to_end(self):
        sb, sink, inbox = pi_run(self, output=CODEX_JS)
        for where in everywhere(sb, sink, inbox):
            for piece in ("+password);", "runTests();", "all tests passed"):
                self.assertIn(piece, where)


class EnvLongOptionTests(unittest.TestCase):
    """Finding 2: env with options (env --unset DEBUG) dumps the environment. Known limit (wrapped-command): only a bare env
    counts now, so none of these blanks the turn; secret-named environment values in the output are still redacted. env
    that runs another command never blanks it."""

    DUMPS = ["env --unset DEBUG", "env --chdir /tmp", "env -u X", "env -i", "env --null", "env -0", "env -u X | sort", 'echo "$(env --unset X)"']
    RUNS = ["env --unset DEBUG ls", "env -C /tmp ls", "env -u X make test", "env --chdir=/tmp make", "env -S 'ls -l'",
            "env --help", "env --version", "env -iu X ls", "env -- ls", "env -i PATH=/bin ls", "sudo env --help"]

    def test_known_limit_dumps_with_options_are_not_seen(self):
        py = commands(self, self.DUMPS)
        self.assertEqual([(c, out.get("sensitive_rule")) for c, out in zip(self.DUMPS, py) if out.get("sensitive")], [])

    def test_commands_that_run_something_else(self):
        py = commands(self, self.RUNS)
        self.assertEqual([(c, out.get("sensitive_rule")) for c, out in zip(self.RUNS, py) if out.get("sensitive")], [])

    def test_env_values_in_the_output_still_go(self):
        opaque = _rand(28)
        sb, sink, inbox = pi_run(self, command="env --unset DEBUG", output="SERVICE_TOKEN=" + opaque, env={"SERVICE_TOKEN": opaque})
        self.assertEqual(leaked(opaque, *everywhere(sb, sink, inbox)), [])


class SecretCliPositionTests(unittest.TestCase):
    """Finding 3: secret_cli matched its words anywhere, so a commit message or a search pattern naming one blanked the turn.
    It now counts only at a command position (sensitive.command.prefix). Known limits: behind
    another wrapper or with options between its words it is not seen (WRAPPED); a heredoc line and a bare secret or
    secrets word still blank the turn (BLANKED, mention-blanks)."""

    REAL = ["gh auth token", "aws configure get aws_secret_access_key", "aws configure export-credentials --profile p",
            "git credential fill", "gcloud auth print-access-token", "gcloud auth application-default print-access-token",
            "security find-generic-password -s x -w", "op read op://v/i/f", "vault read secret/x", "vault kv get secret/x",
            "kubectl get secret -o yaml", "kubectl get secrets -o json", "kubectl describe secret db", "kubectl get secret/db -o json",
            "sudo gh auth token", "x=$(gh auth token)", 'echo "$(gh auth token)"', "cd repo && gh auth token",
            "ls; kubectl get secret -o json", "TOKEN=`gh auth token`", "cat <<'EOF'\nx\nEOF\ngh auth token",
            "gh auth token | pbcopy", "cat <<EOF\ngh auth token\nEOF", "bash -c 'gh auth token'", "FOO=1 gh auth token",
            "/usr/local/bin/gh auth token"]
    WRAPPED = ["aws --profile p configure get aws_secret_access_key", "env -u X gh auth token", "timeout 5 vault kv get secret/x"]
    BLANKED = {"cat <<'EOF'\ngh auth token\nEOF": "secret_cli", "git log --grep='kubectl get secret'": "secret_file",
               "git commit -m 'docs: kubectl get secrets'": "secret_file", "kubectl -n prod get secret db -o yaml": "secret_file",
               "sudo -u root kubectl get secret -o yaml": "secret_cli"}
    MENTIONS = ['git commit -m "fix gh auth token documentation"', "rg 'aws configure get' .", "echo gh auth token",
                "grep -e 'gh auth token' x", "printf '%s\\n' 'vault read x'", "rg -g '*.md' 'op read'",
                'echo "run: gcloud auth print-access-token"', "gh auth status", "aws configure list", "kubectl get pods", "man gh",
                'grep -rn "git credential fill" docs/', "sed -i 's/gh auth token/gh auth status/' README.md",
                "python -c 'print(\"op read\")'", "git grep 'security find-generic-password'", "rg aws configure get"]

    def test_real_invocations(self):
        py = commands(self, self.REAL)
        self.assertEqual([c for c, out in zip(self.REAL, py) if out.get("sensitive_rule") != "secret_cli"], [])

    def test_mentions_do_not_blank_the_turn(self):
        py = commands(self, self.MENTIONS)
        self.assertEqual([(c, out.get("sensitive_rule")) for c, out in zip(self.MENTIONS, py) if out.get("sensitive")], [])

    def test_known_limits(self):
        py = commands(self, self.WRAPPED + list(self.BLANKED))
        self.assertEqual([(c, out.get("sensitive_rule")) for c, out in zip(self.WRAPPED, py) if out.get("sensitive")], [])
        self.assertEqual({c: out.get("sensitive_rule") for c, out in zip(self.BLANKED, py[len(self.WRAPPED):])}, self.BLANKED)

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
    """Command and name rules cost linear work on adversarial input (Python; the hook's windows and budgets bound jq,
    tests/test_latency.py)."""

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
