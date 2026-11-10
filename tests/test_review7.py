"""Regressions for the seventh review of capture: sanitized. One class per blocking finding, each in both layers (jq and its
Python mirror, which must agree) and end to end: hook, inbox, stored turn, and the HTTP body the sender posts. The rows
behind each fix are also crossed with their forms in tests/test_matrix.py, and every case of every review is a row of
tests/regressions.json (tests/test_regressions.py).

Every fake secret is built at runtime from parts (_rand below), so no file holds one.
"""

from __future__ import annotations

import json
import random
import string
import time
import unittest

from test_matrix import codex_patch, model_call, unified
from test_review3 import everywhere, pi_run
from test_review4 import assert_parity, leaked
from test_review6 import blanked_rule, codex_run
from test_review_fixes import tool

from judgment_core import redact

_rng = random.Random(7070)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def _alpha(n: int) -> str:
    return "".join(_rng.choice(string.ascii_lowercase) for _ in range(n))


def _phrase() -> str:
    return " ".join(_alpha(_rng.randint(5, 7)) for _ in range(4))


class LineLocationTests(unittest.TestCase):
    """Finding 1: rg -n, grep -n, and nl put a location before each line, so a quoted alphabetic password or passphrase on a
    printed line had no assignment context. Locations (assignment.locations) now come before the line context."""

    FORMS = ['12:  password: "{v}"', 'config.yml:12:password="{v}"', "config/app.yml:7:  db_password: '{v}'", 'config.yml-13-  secret: "{v}"',
             './deploy/env.yml:  passphrase: "{v}"', '    12\tpassword: "{v}"', '     3\texport DB_PASSWORD="{v}"', '7:- api_key: "{v}"']

    def test_both_layers(self):
        cases = [(form, value) for form in self.FORMS for value in (_alpha(16), _phrase())]
        py = assert_parity(self, [tool({"command": "rg -n password ."}, form.format(v=value)) for form, value in cases])[0]
        self.assertEqual([(form, out["tool_output"]) for (form, value), out in zip(cases, py) if value in out["tool_output"]], [])
        # Only the value goes: the location, the key, and the quotes stay.
        self.assertEqual(py[0]["tool_output"], '12:  password: "[REDACTED:quoted_secret]"')

    def test_located_code_is_kept(self):
        texts = ['app.py:12:    print("password:", password, "done")', 'app.py-13-    log("Enter secret: ", s, "now")',
                 '   40\tconsole.log("password: "+password);', "notes.md:3:The password is in the vault."]
        py = assert_parity(self, [tool({"command": "x"}, text) for text in texts])[0]
        self.assertEqual([(text, out["tool_output"]) for text, out in zip(texts, py) if out["tool_output"] != text], [])

    def test_end_to_end(self):
        value = _phrase()
        sb, sink, inbox = pi_run(self, command="rg -n passphrase config.yml", output='12:  passphrase: "' + value + '"')
        for where in everywhere(sb, sink, inbox):
            self.assertNotIn(value, where)
            self.assertIn("12:  passphrase:", where)


class CliInvocationTests(unittest.TestCase):
    """Finding 2: kubectl get -o jsonpath=... secret session: the output flag came before the resource operand, and with was
    looked for only after it. with and unless are now looked for over the whole invocation, from the command word."""

    PRINTS = ['kubectl get -o jsonpath="{.data.cookie}" secret session | base64 -d', "kubectl get -o yaml secret db", "kubectl -o json get secret/db",
              "kubectl get secret db -o yaml", "kubectl -n prod get -ojson secrets", "gh auth -t status", "gh auth status --show-token",
              "security find-generic-password -w -s x", "aws ssm get-parameter --with-decryption --name /db/pw", "op item get --reveal db --fields password",
              # A help flag of another command does not count: arguments stop at the separator.
              "kubectl get -o yaml secret db; kubectl --help"]
    HARMLESS = ["kubectl get -o yaml pods", "kubectl get --help secret db -o yaml", "gh --help auth token", "gh auth --help status -t",
                "kubectl get secret db", "aws ssm get-parameter --name /app/region", "kubectl get pods -o yaml && kubectl get secret db"]

    def test_both_layers(self):
        commands = self.PRINTS + self.HARMLESS
        py = assert_parity(self, [tool({"command": c}, "out") for c in commands])[0]
        got = {c: out.get("sensitive_rule") for c, out in zip(commands, py)}
        self.assertEqual({c: rule for c, rule in got.items() if c in self.PRINTS and rule != "secret_cli"}, {})
        self.assertEqual({c: rule for c, rule in got.items() if c in self.HARMLESS and rule}, {})

    def test_end_to_end(self):
        cookie = "Ck" + _rand(30)
        sb, sink, inbox = pi_run(self, command=self.PRINTS[0], output=cookie)
        self.assertIn('"sensitive_rule": "secret_cli"', inbox)
        self.assertEqual(leaked(cookie, *everywhere(sb, sink, inbox)), [])
        self.assertEqual(blanked_rule(sink), ("sensitive_path", "secret_cli"))
        marker = "pods " + _rand(8)
        sb, sink, inbox = pi_run(self, command="kubectl get -o yaml pods", output=marker)
        for where in everywhere(sb, sink, inbox):
            self.assertIn(marker, where)
        self.assertIsNone(blanked_rule(sink))


class DefinitePathTests(unittest.TestCase):
    """Finding 3: a patch target named credentials or secrets (no / or : before it) escaped the path rules, which take such a
    bare name only after a separator. A patch target, or a whole path value, is a definite path: it is also seen as ./name."""

    def setUp(self):
        self.cookie = "Ck" + _rand(26)

    def patches(self, name: str) -> list[dict]:
        body = f"@@\n-SESSION_COOKIE={self.cookie}\n+SESSION_COOKIE="
        return [tool({"input": codex_patch(("Update File", name), body=body)}), model_call({"input": codex_patch(("Delete File", name), body=body)}),
                tool({"patch": unified(name) + f"-SESSION_COOKIE={self.cookie}\n"}), tool({"input": codex_patch(("Update File", "a.py"), ("Move to", name))}),
                tool({"file_path": name}, self.cookie), tool({"diff": f'--- "a/{name}"\n+++ "b/{name}"\n-x={self.cookie}\n'})]

    def test_bare_names_blank_from_every_patch_source(self):
        for name, rule in (("credentials", "credentials_file"), ("secrets", "secret_file"), ("secret", "secret_file"), ("Secrets", "secret_file")):
            py = assert_parity(self, self.patches(name))[0]
            self.assertEqual([out.get("sensitive_rule") for out in py], [rule] * 6, name)
            self.assertEqual(leaked(self.cookie, json.dumps(py)), [], name)

    def test_lookalikes_and_mentions_keep_the_turn(self):
        events = [tool({"input": codex_patch(("Update File", "credentials.md"))}), tool({"input": codex_patch(("Update File", "secrets.py"))}),
                  tool({"input": codex_patch(("Update File", "src/app.py"), body="@@\n-load('credentials')\n+load(path)")}),
                  tool({"file_path": "src/secrets_test.go"}), tool({"input": "explain what credentials and secrets are for"})]
        py = assert_parity(self, events)[0]
        self.assertEqual([out.get("sensitive_rule") for out in py], [None] * len(events))

    def test_targets_are_definite_words(self):
        values = redact.tool_values(tool({"input": codex_patch(("Update File", "credentials")), "file_path": "/srv/.env", "command": "ls"}))
        self.assertEqual([(v.kind, v.text[:3]) for v in values], [("patch", "***"), ("path", "/sr"), ("command", "ls")])
        self.assertEqual([redact.definite(p) for p in ("credentials", "/srv/.env", "~/.aws/credentials", "")], ["./credentials", "", "", ""])

    def test_hook_route_end_to_end(self):
        marker = "rotate it " + _rand(8)
        patch = codex_patch(("Update File", "credentials"), body=f"@@\n-SESSION_COOKIE={self.cookie}\n+SESSION_COOKIE=")
        sb, sink, inbox = codex_run(self, prompt=marker, hook_input={"input": patch})
        self.assertIn('"sensitive_rule": "credentials_file"', inbox)
        for where in everywhere(sb, sink, inbox):
            self.assertEqual(leaked(self.cookie, where), [])
        self.assertNotIn(marker, sink.text())
        self.assertEqual(blanked_rule(sink), ("sensitive_path", "credentials_file"))

    def test_transcript_only_route_end_to_end(self):
        marker = "rotate it " + _rand(8)
        patch = codex_patch(("Update File", "secrets"), body=f"@@\n-SESSION_COOKIE={self.cookie}\n+SESSION_COOKIE=")
        sb, sink, inbox = codex_run(self, prompt=marker, transcript_patch=patch)
        self.assertNotIn(self.cookie, inbox)  # the hook never saw the patch
        for where in everywhere(sb, sink, inbox)[1:]:
            self.assertEqual(leaked(self.cookie, where), [])
        self.assertNotIn(marker, sink.text())
        self.assertEqual(blanked_rule(sink), ("sensitive_path", "secret_file"))


# Logging calls whose strings hold code punctuation (, ( ; [ {) before a key: byte for byte in both layers.
PUNCTUATED = [
    'print("username, password:", username, password, "done")', 'print("user (password:", user, password, "ok")',
    'logger.info("a; secret:", a, secret, "b"); run()', 'console.log("[token] {api_key:", k, "}")', "print('user, password:', u, pw, 'x')",
    'log.debug("x, y, passwd=", p, "z", extra={"a": 1})', 'fmt.Printf("host, port, password: %s", h, p, pw, "end")',
    'throw new Error("bad (api_key: " + k + ")!")', 'print("a, password:", a, password, "b, c", d, "e")', 'f("x,{k}=".format(k=1), y, "z")',
    'warn("it\'s wrong, password:", pw, "done")', 'printf("%s, password: %s\\n", u, p); exit(1);',
]


class QuotedPunctuationTests(unittest.TestCase):
    """Finding 4: print("username, password:", username, password, "done") lost its middle arguments: the comma inside the
    string was taken for code. A code context's value must now leave the rest of its line made of whole strings."""

    def test_preservation_corpus_byte_for_byte(self):
        texts = PUNCTUATED + [text + "\nrun()\nprint('ok')" for text in PUNCTUATED] + ["app.py:3:    " + text for text in PUNCTUATED]
        py = assert_parity(self, [tool({"command": "x"}, text) for text in texts])[0]
        self.assertEqual([(text, out["tool_output"]) for text, out in zip(texts, py) if out["tool_output"] != text or "redactions" in out], [])

    def test_real_values_after_punctuation_still_go(self):
        forms = ['connect(host="db", password="{v}", port=5432)', 'f(a, password="{v}")', "# don't commit, password = '{v}'",
                 'cfg = {{"user": "a, b", "password": "{v}"}}', 'x = g(1, 2); secret = "{v}"', 'f(a, password="{v}", api_token="{w}")',
                 'run(password="{v}")  # it\'s fine']
        cases = [(form, _alpha(14), _phrase()) for form in forms]
        py = assert_parity(self, [tool({"command": "x"}, form.format(v=v, w=w)) for form, v, w in cases])[0]
        self.assertEqual([(form, out["tool_output"]) for (form, v, w), out in zip(cases, py) if v in out["tool_output"] or w in out["tool_output"]], [])
        self.assertTrue(py[0]["tool_output"].endswith('", port=5432)'))

    def test_the_check_stays_bounded(self):
        """A line of thousands of code-context values: the check reads at most after_bound characters for each."""
        text = ', password="ab"' * 7000
        started = time.perf_counter()
        py = assert_parity(self, [tool({"command": "x"}, text)])[0][0]
        self.assertLess(time.perf_counter() - started, 5)
        self.assertNotIn('"ab"', json.dumps(py))

    def test_end_to_end(self):
        text = "\n".join(PUNCTUATED)
        sb, sink, inbox = pi_run(self, output=text)
        for where in everywhere(sb, sink, inbox):
            self.assertIn(json.dumps(text)[1:-1], where)


class ShapedKeyTests(unittest.TestCase):
    """Finding 5: plain key and auth never got the credential-shape check that token gets. They are now shaped words
    (secret_keys.shaped): a credential-shaped value under them goes, a name stays, and only the whole identifier counts."""

    def setUp(self):
        self.cred = "Ck" + _rand(22)

    def test_both_layers(self):
        c = self.cred
        go = ['{"key":"' + c + '"}', '{"auth": "' + c + '"}', "key = " + c, "AUTH=" + c, "client --key=" + c, 'cfg.key = "' + c + '"', "Key: " + c]
        stay = ['{"key": "name", "auth": "oauth2"}', "sort_key=" + c, "monkey=" + c, "oauth=" + c, "keyboard=" + c, '"key": "user_id_2024_v1"']
        py = assert_parity(self, [tool({"command": "x"}, text) for text in go + stay] + [tool({"command": "x"}, {"key": c, "auth": c, "id": 7}),
                                                                                       tool({"command": "x"}, {"key": "name", "auth": "oauth2"})])[0]
        self.assertEqual([text for text, out in zip(go, py) if c in out["tool_output"]], [])
        self.assertEqual([(text, out["tool_output"]) for text, out in zip(stay, py[len(go):]) if out["tool_output"] != text], [])
        self.assertEqual(py[-2]["tool_output"], {"key": "[REDACTED:secret_field]", "auth": "[REDACTED:secret_field]", "id": 7})
        self.assertEqual(py[-1]["tool_output"], {"key": "name", "auth": "oauth2"})

    def test_end_to_end(self):
        sb, sink, inbox = pi_run(self, output={"key": self.cred, "region": "us-east-1"})
        self.assertEqual(leaked(self.cred, *everywhere(sb, sink, inbox)), [])
        self.assertIn("us-east-1", sink.text())
        other = "Dm" + _rand(22)
        sb, sink, inbox = pi_run(self, output='{"auth":"' + other + '"}\n{"key": "name"}')
        self.assertEqual(leaked(other, *everywhere(sb, sink, inbox)), [])
        self.assertIn('\\"key\\": \\"name\\"', sink.text())


if __name__ == "__main__":
    unittest.main()
