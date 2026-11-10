"""The test matrix, generated from the rules table (core/bin/redact-rules.json): every row of the three rule families
crossed with the forms it must be found in and the forms it must not, in both layers (jq and its Python mirror, which
must agree). A row added to the table adds its cases here; nothing below names a rule.

    (a) sensitive paths     sensitive.paths[].examples / not  x  PATH_SOURCES (tool args, shell, patch headers, transcript)
    (b) assignments         secret_keys names  x  assignment.contexts' forms (ASSIGNMENTS)  and  x  code that mentions them
                            (MENTIONS, byte for byte), and each line form behind each assignment.locations example
    (c) secret-printing CLI sensitive.commands[secret_cli].rows[].examples / not  x  WRAPS (command positions) and QUOTED
    (d) shaped keys         secret_keys.shaped (key, auth)  x  text and member forms, with credentials, names, and as suffixes

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
# A sensitive name that a patch only mentions, or that sits in a value no header names: none blanks the turn.
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

    def test_mentions_keep_the_turn(self):
        cases = [(path, form) for row in self.rows for path in row["examples"] for form in PATH_MENTIONS]
        out = run(self, [PATH_MENTIONS[form](path) for path, form in cases])
        self.assertEqual([(path, form, got.get("sensitive_rule")) for (path, form), got in zip(cases, out) if got.get("sensitive")], [])

    def test_every_header_row_is_a_source(self):
        """Each header row of the patch table, alone, names a target the path rules see."""
        texts = [row["prefix"] + ("a/.env b/.env" if row.get("words") else ".env") for row in TABLE["sensitive"]["patch"]["headers"]]
        out = run(self, [tool({"input": "x\n" + text + "\ny"}) for text in texts])
        self.assertEqual([text for text, got in zip(texts, out) if got.get("sensitive_rule") != "dotenv"], [])


# ---------------------------------------------------------------- (b) assignments: contexts x key names, and mentions


def names() -> list[str]:
    keys = TABLE["secret_keys"]
    return list(keys["names"]) + list(keys["whole"])


# One form per assignment context, and its common spellings. {k} the key, {K} it in capitals, {v} the value.
ASSIGNMENTS = {
    "line": ['{k} = "{v}"', '  {k}: "{v}"', "{K}='{v}'", 'export {K}="{v}"', "- {k}: '{v}'", 'const {k} = "{v}";', '# {k} = "{v}"'],
    "quoted_key": ['"{k}": "{v}"', "x = {{'{k}': '{v}', 'user': 'bob'}}", '{{"{k}":"{v}","n":1}}'],
    "escaped_key": ['{{\\"{k}\\":\\"{v}\\"}}'],
    "flag": ['run --{k} "{v}"', 'run --{k}="{v}" --verbose', "docker run -e {K}='{v}' img"],
    "code": ['f(user, {k}="{v}")', '{{{k}: "{v}"}}', 'x = 1; self.{k} = "{v}"', 'FOO=1 {K}="{v}" ./run',
             'log("a, b"); f(x, {k}="{v}")', 'connect(host="db", {k}="{v}", port=1)', "# it's set, {k} = '{v}'"],
}
# What grep, rg, nl, and cat -n put before a line (assignment.locations[].examples), crossed with every line form.
LOCATIONS = [example for row in TABLE["assignment"]["locations"] for example in row["examples"]]
# Code that names a key inside a string or beside one: the quote after the operator closes a string. Byte for byte.
MENTIONS = [
    'print("{k}:", {k}, "done")', 'print("{k}: ", {k}, "done")', 'print("{k}:",{k},"done")', 'print("{k} =", {k}, "done")',
    'console.log("{k}=", {k}, "x")', 'logger.info("{k}: %s", {k}); run()', 'printf("{k}: %s\\n", {k}); x = "y"',
    'print(f"{k}: {{{k}}}", "done")', 'console.log(`{k}: ${{{k}}}`, "done")', 'msg = "{k}: " + {k} + " done"',
    "msg = '{k}:' + {k} + 'done'", 'log("Enter {k}: ", {k}, "now")', 'print("{k}:"); print({k}); print("done")',
    'throw new Error("bad {k}: " + {k} + "!")', 'echo "{k}:" "${K}" "done"', 'print("user:", u, "{k}:", {k}, "done")',
    'System.out.println("{k}: " + {k} + ", ok");', 'fmt.Println("{k}:", {k}, "end")', "puts '{k}: ' + {k} + ' end'",
    'log.Printf("{k}=%q", {k}, "x")', 'print("{K}:", {k}, "done")', 'assert x == "{k}:", "done"',
    # Code punctuation inside the string (review 7, finding 4): it is not code.
    'print("user, {k}:", user, {k}, "done")', 'log("x ({k}:", {k}, "y")', 'logger.info("a; {k}:", {k}, "b"); run()',
    'console.log("[{k}:", {k}, "]")', "print('a, {k}:', a, {k}, 'b')", 'log("{{{k}:", {k}, "}}")', 'f("x,{k}=", {k}, "y", z)',
]
PLACEHOLDER = re.compile(r"\[REDACTED:[A-Za-z0-9_:.-]+\]")


class AssignmentMatrix(unittest.TestCase):
    maxDiff = None
    def test_every_context_has_forms(self):
        self.assertEqual(sorted(row["id"] for row in TABLE["assignment"]["contexts"]), sorted(ASSIGNMENTS))

    def test_assignments_lose_only_their_value(self):
        """A value only quoted_secret can find (no digit: no shape rule takes it), under every key in every context."""
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

    def test_every_location_has_examples(self):
        self.assertEqual([row["id"] for row in TABLE["assignment"]["locations"] if not row.get("examples")], [])

    def test_located_lines_lose_only_their_value(self):
        """Review 7, finding 1: a line as grep -n, rg -n, nl, or cat -n prints it is still a line: every location example
        before every line form, under every key, with a value only quoted_secret finds."""
        cases = []
        for where in LOCATIONS:
            for name in names():
                for form in ASSIGNMENTS["line"]:
                    value = "pw " + _rand(10)
                    cases.append((where + form.format(k=name, K=name.upper(), v=value), value))
        out = run(self, [tool({"command": "x"}, text) for text, _ in cases])
        self.assertEqual([(text, got["tool_output"]) for (text, value), got in zip(cases, out)
                          if value in got["tool_output"] or PLACEHOLDER.sub(value, got["tool_output"]) != text], [])

    def test_located_mentions_come_out_byte_for_byte(self):
        texts = [where + form.format(k=name, K=name.upper()) for where in LOCATIONS for name in names()[:4] for form in MENTIONS]
        out = run(self, [tool({"command": "x"}, text) for text in texts])
        self.assertEqual([(text, got["tool_output"]) for text, got in zip(texts, out) if got["tool_output"] != text], [])

    def test_mentions_come_out_byte_for_byte(self):
        texts = [form.format(k=name, K=name.upper()) + tail for name in names() for form in MENTIONS for tail in ("", "\nrun()\nprint('ok')")]
        out = run(self, [tool({"command": "x"}, text) for text in texts])
        self.assertEqual([(text, got["tool_output"]) for text, got in zip(texts, out) if got["tool_output"] != text], [])


# ---------------------------------------------------------------- (c) secret-printing CLI rows


# Command positions: the shell tokenizer finds the command behind each.
WRAPS = ["{}", "sudo {}", "FOO=1 {}", "cd repo && {}", "x=$({})", "{} | cat", "timeout 5 {}", "env -u X {}", "bash -c '{}'",
         'echo "$({})"', "ls; {}", "cat <<'EOF'\nx\nEOF\n{}"]
# Not a command position: quoted text, a pattern, a quoted heredoc's body.
QUOTED = ["echo '{}'", 'git commit -m "{}"', "rg '{}' .", "cat <<'EOF'\n{}\nEOF", "grep -e '{}' x", "git log --grep='{}'"]


def cli_rows() -> list[dict]:
    return [row for item in TABLE["sensitive"]["commands"] if "rows" in item for row in item["rows"]]


def _fits(form: str, command: str) -> bool:
    """A form that quotes its command with ' or " cannot hold that quote."""
    return not any(q in command and q in form.replace("{}", "") for q in "'\"")


class CliMatrix(unittest.TestCase):
    maxDiff = None
    def test_every_row_has_examples(self):
        self.assertEqual([row["row"] for row in cli_rows() if not row.get("examples") or not row.get("not")], [])

    def test_examples_at_every_command_position(self):
        cases = [(row["row"], wrap.format(cmd)) for row in cli_rows() for cmd in row["examples"] for wrap in WRAPS if _fits(wrap, cmd)]
        out = run(self, [tool({"command": cmd}) for _, cmd in cases])
        self.assertEqual([(row, cmd, got.get("sensitive_rule")) for (row, cmd), got in zip(cases, out) if got.get("sensitive_rule") != "secret_cli"], [])

    def test_harmless_invocations_keep_the_turn(self):
        cases = [wrap.format(cmd) for row in cli_rows() for cmd in row["not"] for wrap in WRAPS if _fits(wrap, cmd)]
        out = run(self, [tool({"command": cmd}, "ordinary output") for cmd in cases])
        self.assertEqual([(cmd, got.get("sensitive_rule")) for cmd, got in zip(cases, out) if got.get("sensitive") or got["tool_output"] != "ordinary output"], [])

    def test_quoted_mentions_keep_the_turn(self):
        cases = [form.format(cmd) for row in cli_rows() for cmd in row["examples"] for form in QUOTED if _fits(form, cmd)]
        out = run(self, [tool({"command": cmd}) for cmd in cases])
        self.assertEqual([(cmd, got.get("sensitive_rule")) for cmd, got in zip(cases, out) if got.get("sensitive")], [])


# ---------------------------------------------------------------- (d) shaped keys: key and auth keep the shape test


def shaped() -> list[str]:
    """secret_keys.shaped in each case the composer spells it (lower, UPPER, Capitalized)."""
    return [case for word in TABLE["secret_keys"]["shaped"] for case in dict.fromkeys([word, word.upper(), word.capitalize()])]


# Text forms of a shaped key with a value; {k} the key, {v} the value. And the same in structured members.
SHAPED_TEXT = ['{{"{k}":"{v}"}}', '{{"{k}": "{v}"}}', '{k} = {v}', '{k}: {v}', '{k}={v}', 'cli --{k}={v} -q', 'cfg.{k} = "{v}"', "x = {{'{k}': '{v}'}}"]
# Values that are names, not credentials: they stay under a shaped key.
SHAPED_NAMES = ["name", "oauth2", "user_id_2024_v1", "oauth2_client", "primary", "Bearer", "created_at"]
# Identifiers that only end in a shaped word: never shape-tested as one (review 7, finding 5: identifier boundaries).
SHAPED_SUFFIXES = ["sort_{k}", "my{k}", "o{k}", "{k}board", "{k}_count", "x-{k}"]


class ShapedKeyMatrix(unittest.TestCase):
    maxDiff = None

    def test_shaped_words_stay_out_of_the_unconditional_lists(self):
        keys = TABLE["secret_keys"]
        self.assertEqual(set(keys["shaped"]) & set(keys["names"] + keys["whole"]), set())

    def test_credentials_lose_their_value(self):
        cases = [(form.format(k=k, v=v), v) for k in shaped() for form in SHAPED_TEXT for v in ["Ck" + _rand(14) + "7" + _rand(5)]]
        members = [({k: v}, v) for k in shaped() for v in ["Dm" + _rand(10) + "3" + _rand(9)]]
        out = run(self, [tool({"command": "x"}, text) for text, _ in cases + members])
        self.assertEqual([(text, got["tool_output"]) for (text, value), got in zip(cases + members, out) if value in json.dumps(got)], [])

    def test_names_stay(self):
        texts = [form.format(k=k, v=v) for k in shaped() for form in SHAPED_TEXT for v in SHAPED_NAMES]
        members = [{k: v} for k in shaped() for v in SHAPED_NAMES]
        out = run(self, [tool({"command": "x"}, item) for item in texts + members])
        self.assertEqual([(item, got["tool_output"]) for item, got in zip(texts + members, out) if got["tool_output"] != item], [])

    def test_suffixes_are_not_shaped_keys(self):
        value = "Ck" + _rand(14) + "7"
        keys = [suffix.format(k=k) for k in shaped() for suffix in SHAPED_SUFFIXES]
        texts = [f"{key}={value}" for key in keys] + [json.dumps({key: value}) for key in keys]
        out = run(self, [tool({"command": "x"}, text) for text in texts] + [tool({"command": "x"}, {key: value}) for key in keys])
        bad = [(item, got["tool_output"]) for item, got in zip(texts + [{key: value} for key in keys], out) if got["tool_output"] != item]
        self.assertEqual(bad, [])


def matrix_size() -> dict[str, int]:
    """How many cases the matrix generates, by family (each runs in both layers)."""
    paths = TABLE["sensitive"]["paths"]
    a = (sum(len(r["examples"]) for r in paths) * len(PATH_SOURCES) + sum(len(r["not"]) for r in paths) * len(PATH_SOURCES)
         + sum(len(r["examples"]) for r in paths) * len(PATH_MENTIONS) + len(TABLE["sensitive"]["patch"]["headers"]))
    b = (len(names()) * sum(len(forms) for forms in ASSIGNMENTS.values()) + len(names()) * len(MENTIONS) * 2
         + len(LOCATIONS) * len(names()) * len(ASSIGNMENTS["line"]) + len(LOCATIONS) * 4 * len(MENTIONS))
    rows = cli_rows()
    c = (sum(_fits(w, cmd) for r in rows for cmd in r["examples"] for w in WRAPS) + sum(_fits(w, cmd) for r in rows for cmd in r["not"] for w in WRAPS)
         + sum(_fits(f, cmd) for r in rows for cmd in r["examples"] for f in QUOTED))
    d = len(shaped()) * (2 * len(SHAPED_TEXT) + len(SHAPED_NAMES) * (len(SHAPED_TEXT) + 1) + 1) + 3 * len(shaped()) * len(SHAPED_SUFFIXES)
    return {"paths": a, "assignments": b, "cli": c, "shaped": d}


if __name__ == "__main__":
    print(json.dumps(matrix_size()))
    unittest.main()
