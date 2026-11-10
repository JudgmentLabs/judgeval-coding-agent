"""The test matrix, generated from the rules table (core/bin/redact-rules.json): every row of the three rule families
crossed with the forms it must be found in and the forms it must not, in both layers (jq and its Python mirror, which
must agree). A row added to the table adds its cases here; nothing below names a rule.

    (a) sensitive paths     sensitive.paths[].examples / not  x  PATH_SOURCES (tool args, shell, patch headers, transcript)
    (b) secret names        secretNames.names  x  ASSIGNMENTS (line, quoted key, escaped key, flag, code) and LOCATIONS,
                            and x code that mentions them (MENTIONS, byte for byte)
    (c) sensitive commands  sensitive.commands[].examples / not  x  WRAPS (command positions) and QUOTED
    (d) key and auth        not secret names: only a format rule takes a value under them

Known limits (docs/trace-format.md) are asserted as such: PATH_MENTIONS blank the turn (mention-blanks), LIMIT_WRAPS
hide a command (wrapped-command), OVER_REDACTED mentions lose text after the name (code-after-name), and an unformatted
value under key or auth stays (key-auth-unformatted).

Every fake value is built at runtime (_rand).
"""

from __future__ import annotations

import json
import random
import re
import string
import unittest

from test_review4 import assert_parity
from test_review_fixes import tool

from judgment_core import redact

_rng = random.Random(6006)
TABLE = redact.rules()


def _rand(n: int, alphabet: str = string.ascii_letters) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n))


def run(test: unittest.TestCase, events: list[dict]) -> list[dict]:
    """Both layers on every event, in one jq run; they must agree. Returns the Python layer's events."""
    py, _ = assert_parity(test, events)
    return py


# ---------------------------------------------------------------- (a) sensitive paths, from every source


def codex_patch(*ops: tuple[str, str], body: str = "@@\n-a = 1\n+a = 2") -> str:
    """A Codex apply_patch text: each op is (header, path)."""
    return "*** Begin Patch\n" + "".join(f"*** {head}: {path}\n{body}\n" for head, path in ops) + "*** End Patch"


def unified(path: str) -> str:
    return f"--- a/{path}\t2026-01-01 00:00:00\n+++ b/{path}\n@@ -1 +1 @@\n-a\n+b\n"


def model_call(arguments) -> dict:
    """A model event as transcript enrichment writes it: the call's arguments, as codex_transcript wraps free-form input."""
    return {"v": 1, "harness": "codex", "event": "model", "session_id": "s", "turn_id": "t", "output": "ok",
            "tool_calls": [{"id": "c1", "name": "apply_patch", "arguments": arguments}]}


PATH_SOURCES = {
    "file_path": lambda p: tool({"file_path": p}),
    "command": lambda p: tool({"command": "cat " + p}),
    "update": lambda p: tool({"input": codex_patch(("Update File", p))}),
    "add": lambda p: tool({"input": codex_patch(("Add File", p))}),
    "delete": lambda p: tool({"input": codex_patch(("Delete File", p))}),
    "move": lambda p: tool({"input": codex_patch(("Update File", "src/app.py"), ("Move to", p))}),
    "multi_file": lambda p: tool({"input": codex_patch(("Update File", "src/app.py"), ("Update File", p), ("Add File", "README.md"))}),
    "unified": lambda p: tool({"input": unified(p)}),
    "diff_git": lambda p: tool({"patch": f"diff --git a/{p} b/{p}\nindex 1..2 100644\n"}),
    "quoted_name": lambda p: tool({"diff": f'--- "a/{p}"\n+++ "b/{p}"\n'}),
    "string_input": lambda p: tool(codex_patch(("Update File", p))),
    "shell_heredoc": lambda p: tool({"command": "apply_patch <<'EOF'\n" + codex_patch(("Update File", p)) + "\nEOF"}),
    "transcript": lambda p: model_call({"input": codex_patch(("Update File", p))}),
    "transcript_diff": lambda p: model_call({"input": unified(p)}),
}
# A sensitive name that a patch only mentions, or that sits in a value no header names. Known limit (mention-blanks): each
# blanks the turn, since a mention is not told apart from a read or a write.
PATH_MENTIONS = {
    "patch_body": lambda p: tool({"input": codex_patch(("Update File", "src/app.py"), body=f"@@\n-load('{p}')\n+load(path)")}),
    "unified_body": lambda p: tool({"input": f"--- a/src/app.py\n+++ b/src/app.py\n@@\n+# reads {p}\n-x\n"}),
    "plain_input": lambda p: tool({"input": f"explain what {p} is for"}),
    "transcript_body": lambda p: model_call({"input": codex_patch(("Add File", "docs/setup.md"), body=f"+Copy {p} from the vault.")}),
}


class PathMatrix(unittest.TestCase):
    maxDiff = None
    rows = TABLE["sensitive"]["paths"]

    def test_every_row_has_examples(self):
        self.assertEqual([row["id"] for row in self.rows if not row.get("examples") or not row.get("not")], [])

    def test_examples_blank_from_every_source(self):
        cases = [(row["id"], path, form) for row in self.rows for path in row["examples"] for form in PATH_SOURCES]
        out = run(self, [PATH_SOURCES[form](path) for _, path, form in cases])
        self.assertEqual([(rule, path, form, got.get("sensitive_rule")) for (rule, path, form), got in zip(cases, out) if got.get("sensitive_rule") != rule], [])

    def test_lookalikes_from_every_source_keep_the_turn(self):
        cases = [(path, form) for row in self.rows for path in row["not"] for form in PATH_SOURCES]
        out = run(self, [PATH_SOURCES[form](path) for path, form in cases])
        self.assertEqual([(path, form, got.get("sensitive_rule")) for (path, form), got in zip(cases, out) if got.get("sensitive")], [])

    def test_known_limit_mentions_blank_the_turn(self):
        cases = [(row["id"], path, form) for row in self.rows for path in row["examples"] for form in PATH_MENTIONS]
        out = run(self, [PATH_MENTIONS[form](path) for _, path, form in cases])
        self.assertEqual([(rule, path, form, got.get("sensitive_rule")) for (rule, path, form), got in zip(cases, out) if got.get("sensitive_rule") != rule], [])


# ---------------------------------------------------------------- (b) secret names: forms x names, and mentions


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
        """Findings a (rg --vimgrep) and review 7: a location before a line changes nothing."""
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


# ---------------------------------------------------------------- (c) sensitive commands


# Command positions: the start, after ; & | ( ` or a newline, behind sudo.
WRAPS = ["{}", "sudo {}", "cd repo && {}", "x=$({})", "{} | cat", 'echo "$({})"', "ls; {}", "cat <<'EOF'\nx\nEOF\n{}", "a || {}", "x=`{}`"]
# Known limit (wrapped-command): behind any other wrapper the command is not seen. Secret-named environment values in the
# output are still redacted (env_value).
LIMIT_WRAPS = ["FOO=1 {}", "timeout 5 {}", "env -u X {}", "bash -c '{}'", "command {}", "nohup {}"]
# Not a command position: quoted text, a pattern, a commit message.
QUOTED = ["echo '{}'", 'git commit -m "{}"', "rg '{}' .", "grep -e '{}' x", "git log --grep='{}'"]


def command_rows() -> list[dict]:
    return TABLE["sensitive"]["commands"]


COMMAND_IDS = {row["id"] for row in command_rows()}


def _fits(form: str, command: str) -> bool:
    """A form that quotes its command with ' or " cannot hold that quote."""
    return not any(q in command and q in form.replace("{}", "") for q in "'\"")


def _examples(wraps: list[str], key: str = "examples") -> list[tuple[str, str]]:
    return [(row["id"], wrap.format(cmd)) for row in command_rows() for cmd in row[key] for wrap in wraps if _fits(wrap, cmd)]


class CommandMatrix(unittest.TestCase):
    maxDiff = None

    def test_every_row_has_examples(self):
        self.assertEqual([row["id"] for row in command_rows() if not row.get("examples") or not row.get("not")], [])

    def test_examples_at_every_command_position(self):
        cases = _examples(WRAPS)
        out = run(self, [tool({"command": cmd}) for _, cmd in cases])
        self.assertEqual([(rule, cmd, got.get("sensitive_rule")) for (rule, cmd), got in zip(cases, out) if got.get("sensitive_rule") != rule], [])

    def test_harmless_invocations_keep_the_turn(self):
        cases = _examples(WRAPS, "not")
        out = run(self, [tool({"command": cmd}, "ordinary output") for _, cmd in cases])
        self.assertEqual([(cmd, got.get("sensitive_rule")) for (_, cmd), got in zip(cases, out) if got.get("sensitive") or got["tool_output"] != "ordinary output"], [])

    def test_quoted_mentions_are_no_command(self):
        """No command rule fires. A path rule still may: kubectl get secret names a token secret (mention-blanks)."""
        cases = _examples(QUOTED)
        out = run(self, [tool({"command": cmd}) for _, cmd in cases])
        self.assertEqual([(cmd, got.get("sensitive_rule")) for (_, cmd), got in zip(cases, out) if got.get("sensitive_rule") in COMMAND_IDS], [])

    def test_known_limit_wrapped_commands_are_not_seen(self):
        cases = _examples(LIMIT_WRAPS)
        out = run(self, [tool({"command": cmd}) for _, cmd in cases])
        self.assertEqual([(cmd, got.get("sensitive_rule")) for (_, cmd), got in zip(cases, out) if got.get("sensitive_rule") in COMMAND_IDS], [])


# ---------------------------------------------------------------- (d) key and auth: not secret names


def plain_words() -> list[str]:
    return [case for word in ("key", "auth") for case in dict.fromkeys([word, word.upper(), word.capitalize()])]


# Text forms of key or auth with a value; {k} the key, {v} the value. And the same in structured members.
PLAIN_TEXT = ['{{"{k}":"{v}"}}', '{{"{k}": "{v}"}}', '{k} = {v}', '{k}: {v}', '{k}={v}', 'cli --{k}={v} -q', 'cfg.{k} = "{v}"', "x = {{'{k}': '{v}'}}"]
# Values that are names, not credentials: S3 object keys, uuids, hashes (finding c).
PLAIN_NAMES = ["name", "oauth2", "user_id_2024_v1", "reports/2026/summary.csv", "3f2b8c1e-9a4d-4e7b-8c2a-1d5e6f7a8b9c",
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"]


class KeyAuthMatrix(unittest.TestCase):
    maxDiff = None

    def test_names_stay(self):
        texts = [form.format(k=k, v=v) for k in plain_words() for form in PLAIN_TEXT for v in PLAIN_NAMES]
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
        cases = [form.format(k=k, v=cred) for k in plain_words() for form in PLAIN_TEXT] + [{k: cred} for k in plain_words()]
        out = run(self, [tool({"command": "x"}, item) for item in cases])
        self.assertEqual([(item, got["tool_output"]) for item, got in zip(cases, out) if got["tool_output"] != item], [])


def matrix_size() -> dict[str, int]:
    """How many cases the matrix generates, by family (each runs in both layers)."""
    paths = TABLE["sensitive"]["paths"]
    a = sum(len(r["examples"]) + len(r["not"]) for r in paths) * len(PATH_SOURCES) + sum(len(r["examples"]) for r in paths) * len(PATH_MENTIONS)
    b = (len(names()) * sum(len(forms) for forms in ASSIGNMENTS.values()) + len(LOCATIONS) * len(names()) * len(ASSIGNMENTS["line"])
         + len(names()) * (3 * len(MENTIONS) + len(OVER_REDACTED)))
    c = len(_examples(WRAPS)) + len(_examples(WRAPS, "not")) + len(_examples(QUOTED)) + len(_examples(LIMIT_WRAPS))
    d = len(plain_words()) * (len(PLAIN_TEXT) * (len(PLAIN_NAMES) + 2) + 3 * len(PLAIN_NAMES) + 2)
    return {"paths": a, "names": b, "commands": c, "key_auth": d}


if __name__ == "__main__":
    print(json.dumps(matrix_size()))
    unittest.main()
