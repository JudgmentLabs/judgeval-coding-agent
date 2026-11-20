"""The test matrix, generated from the rules table (core/bin/redact-rules.json): every secret name crossed with the
forms it must be found in and the forms it must not, in both layers (jq and its Python mirror, which must agree). A name
added to the table adds its cases here; nothing below names a rule.

    secret names        secretNames.names  x  ASSIGNMENTS (line, quoted key, escaped key, flag, code) and LOCATIONS,
                        and x code that mentions them (MENTIONS, byte for byte)
    key and auth        not secret names: only a format rule takes a value under them

Known limits (docs/trace-format.md) are asserted as such: OVER_REDACTED mentions lose text after the name
(code-after-name), and an unformatted value under key or auth stays (key-auth-unformatted).

Every fake value is built at runtime (_rand).
"""

from __future__ import annotations

import json
import random
import re
import string
import unittest


from support import assert_parity, tool

from judgment_core import redact

_rng = random.Random(6006)
TABLE = redact.rules()


def _rand(n: int, alphabet: str = string.ascii_letters) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n))


def run(test: unittest.TestCase, events: list[dict]) -> list[dict]:
    """Both layers on every event, in one jq run; they must agree. Returns the Python layer's events."""
    py, _ = assert_parity(test, events)
    return py


# ---------------------------------------------------------------- secret names: forms x names, and mentions


def names() -> list[str]:
    return list(TABLE["secretNames"]["names"])


# Forms of a value after a name. {k} the name, {K} it in capitals, {v} the value.
ASSIGNMENTS = {
    "line": ['{k} = "{v}"', '  {k}: "{v}"', "{K}='{v}'", 'export {K}="{v}"', "- {k}: '{v}'", 'const {k} = "{v}";', '# {k} = "{v}"', "{k}={v}"],
    "quoted_key": ['"{k}": "{v}"', "x = {{'{k}': '{v}', 'user': 'bob'}}", '{{"{k}":"{v}","n":1}}'],
    "escaped_key": ['{{\\"{k}\\":\\"{v}\\"}}', '{{\\\\\\"{k}\\\\\\":\\\\\\"{v}\\\\\\"}}'],
    "flag": ['run --{k}="{v}" --verbose', "docker run -e {K}='{v}' img", "run --{k}={v} -q"],
    "code": ['f(user, {k}="{v}")', '{{{k}: "{v}"}}', 'x = 1; self.{k} = "{v}"', 'FOO=1 {K}="{v}" ./run',
             'log("a, b"); f(x, {k}="{v}")', 'connect(host="db", {k}="{v}", port=1)', "# it's set, {k} = '{v}'",
             'connect({k}="{v}") # users\' {k}'],
}
# What grep -n, rg --vimgrep, nl, and cat -n put before a line, crossed with every line form.
LOCATIONS = ["12:", "src/app.py:12:", "src/app.py:12:3: ", "src/app.py:12:3:", "config.yml-13-", "    12\t", "./deploy/env.yml:  "]
# Code that names a key inside a string or beside one: the quote after the operator closes a string. Byte for byte.
MENTIONS = [
    'print("{k}:", {k}, "done")', 'print("{k}: ", {k}, "done")', 'print("{k}:",{k},"done")', 'print("{k} =", {k}, "done")',
    'console.log("{k}=", {k}, "x")', 'print(f"{k}: {{{k}}}", "done")', 'log("Enter {k}: ", {k}, "now")', 'print("{K}:", {k}, "done")',
    'assert x == "{k}:", "done"', 'print("user:", u, "{k}:", {k}, "done")', 'fmt.Println("{k}:", {k}, "end")',
    'print("user, {k}:", user, {k}, "done")', 'log("x ({k}:", {k}, "y")', 'logger.info("a; {k}:", {k}, "b"); run()',
    'console.log("[{k}:", {k}, "]")', "print('a, {k}:', a, {k}, 'b')", 'log("{{{k}:", {k}, "}}")', 'f("x,{k}=", {k}, "y", z)',
    'msg = "{k}: " + {k} + " done"', "msg = '{k}:' + {k} + 'done'", 'print("{k}:"); print({k}); print("done")',
    'throw new Error("bad {k}: " + {k} + "!")', 'console.log("{k}: "+{k});\nrun();',
]
# Known limit (code-after-name): a format placeholder right after "name:" in a string is taken for the value and
# redacted. Only that placeholder changes.
OVER_REDACTED = ['logger.info("{k}: %s", {k}); run()', 'log.Printf("{k}=%q", {k}, "x")', 'echo "{k}: "${K}']
PLACEHOLDER = re.compile(r"\[REDACTED:[A-Za-z0-9_:.-]+\]")


class AssignmentMatrix(unittest.TestCase):
    maxDiff = None

    def test_values_lose_only_the_value(self):
        """A value only secret_assignment can find (no format rule takes it), under every name in every form."""
        cases = []
        for context, forms in ASSIGNMENTS.items():
            for name in names():
                for form in forms:
                    value = "pw-" + _rand(10)
                    cases.append((context, form, value, form.format(k=name, K=name.upper(), v=value)))
        out = run(self, [tool({"command": "x"}, text) for *_, text in cases])
        bad = []
        for (context, form, value, text), got in zip(cases, out):
            result = got["tool_output"]
            if value in result or PLACEHOLDER.sub(value, result) != text:
                bad.append((context, text, result))
        self.assertEqual(bad, [])

    def test_located_lines_lose_only_their_value(self):
        """A location before a line, such as rg --vimgrep prints, changes nothing."""
        cases = []
        for where in LOCATIONS:
            for name in names():
                for form in ASSIGNMENTS["line"]:
                    value = "pw " + _rand(10) if '"' in form or "'" in form else "pw" + _rand(10)
                    cases.append((where + form.format(k=name, K=name.upper(), v=value), value))
        out = run(self, [tool({"command": "x"}, text) for text, _ in cases])
        self.assertEqual([(text, got["tool_output"]) for (text, value), got in zip(cases, out)
                          if value in got["tool_output"] or PLACEHOLDER.sub(value, got["tool_output"]) != text], [])

    def test_mentions_come_out_byte_for_byte(self):
        texts = [where + form.format(k=name, K=name.upper()) + tail for name in names() for form in MENTIONS
                 for where, tail in (("", ""), ("", "\nrun()\nprint('ok')"), ("app.py:3:    ", ""))]
        out = run(self, [tool({"command": "x"}, text) for text in texts])
        self.assertEqual([(text, got["tool_output"]) for text, got in zip(texts, out) if got["tool_output"] != text], [])

    def test_known_limit_code_after_a_name_is_redacted(self):
        texts = [form.format(k=name, K=name.upper()) for name in names() for form in OVER_REDACTED]
        out = run(self, [tool({"command": "x"}, text) for text in texts])
        for text, got in zip(texts, out):
            result = got["tool_output"]
            self.assertNotEqual(result, text)
            self.assertEqual({item["rule"] for item in got["redactions"]}, {"secret_assignment"}, text)


# ---------------------------------------------------------------- key and auth: not secret names


def plain_words() -> list[str]:
    return [case for word in ("key", "auth") for case in dict.fromkeys([word, word.upper(), word.capitalize()])]


# Text forms of key or auth with a value; {k} the key, {v} the value. And the same in structured members.
PLAIN_TEXT = ['{{"{k}":"{v}"}}', '{{"{k}": "{v}"}}', '{k} = {v}', '{k}: {v}', '{k}={v}', 'cli --{k}={v} -q', 'cfg.{k} = "{v}"', "x = {{'{k}': '{v}'}}"]
# Values that are names, not credentials: S3 object keys, uuids, hashes.
PLAIN_NAMES = ["name", "oauth2", "user_id_2024_v1", "reports/2026/summary.csv", "3f2b8c1e-9a4d-4e7b-8c2a-1d5e6f7a8b9c",
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"]
# The environment form, KEY=value, is a variable assignment: key_assignment takes whatever the value is.
ENV_FORMS = {'{k}={v}', 'cli --{k}={v} -q'}


def plain_forms(k: str) -> list[str]:
    return [form for form in PLAIN_TEXT if not (k == "KEY" and form in ENV_FORMS)]


class KeyAuthMatrix(unittest.TestCase):
    maxDiff = None

    def test_names_stay(self):
        texts = [form.format(k=k, v=v) for k in plain_words() for form in plain_forms(k) for v in PLAIN_NAMES]
        members = [{k: v, "Bucket": "b"} for k in plain_words() for v in PLAIN_NAMES]
        out = run(self, [tool({"command": "x"}, item) for item in texts + members] + [tool(item) for item in members])
        items = texts + members
        self.assertEqual([(item, got["tool_output"]) for item, got in zip(items, out) if got["tool_output"] != item], [])
        self.assertEqual([(item, got["tool_input"]) for item, got in zip(members, out[len(items):]) if got["tool_input"] != item], [])

    def test_formatted_credentials_still_go(self):
        token = "gh" + "p_" + _rand(36, string.ascii_letters + string.digits)
        cases = [form.format(k=k, v=token) for k in plain_words() for form in PLAIN_TEXT] + [{k: token} for k in plain_words()]
        out = run(self, [tool({"command": "x"}, item) for item in cases])
        self.assertEqual([item for item, got in zip(cases, out) if token in json.dumps(got)], [])

    def test_known_limit_unformatted_values_stay(self):
        cred = "Ck" + _rand(14) + "7" + _rand(5)
        cases = [form.format(k=k, v=cred) for k in plain_words() for form in plain_forms(k)] + [{k: cred} for k in plain_words()]
        out = run(self, [tool({"command": "x"}, item) for item in cases])
        self.assertEqual([(item, got["tool_output"]) for item, got in zip(cases, out) if got["tool_output"] != item], [])

    def test_env_form_takes_any_value(self):
        cred = "Ck" + _rand(14) + "7" + _rand(5)
        cases = [(form.format(k="KEY", v=v), v) for form in sorted(ENV_FORMS) for v in PLAIN_NAMES + [cred]]
        out = run(self, [tool({"command": "x"}, text) for text, _ in cases])
        self.assertEqual([text for (text, value), got in zip(cases, out)
                          if value in got["tool_output"] or {item["rule"] for item in got["redactions"]} != {"key_assignment"}], [])


def matrix_size() -> dict[str, int]:
    """How many cases the matrix generates, by family (each runs in both layers)."""
    b = (len(names()) * sum(len(forms) for forms in ASSIGNMENTS.values()) + len(LOCATIONS) * len(names()) * len(ASSIGNMENTS["line"])
         + len(names()) * (3 * len(MENTIONS) + len(OVER_REDACTED)))
    d = len(plain_words()) * (len(PLAIN_TEXT) * (len(PLAIN_NAMES) + 2) + 3 * len(PLAIN_NAMES) + 2)
    return {"names": b, "key_auth": d}


if __name__ == "__main__":
    print(json.dumps(matrix_size()))
    unittest.main()
