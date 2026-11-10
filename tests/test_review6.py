"""Regressions for the sixth review of capture: sanitized. One class per blocking finding, each in both layers (jq and its
Python mirror, which must agree) and end to end: hook, inbox, stored turn, and the HTTP body the sender posts. The rows
behind each fix are also crossed with their forms in tests/test_matrix.py.

Every fake secret is built at runtime from parts (_rand below), so no file holds one.
"""

from __future__ import annotations

import json
import random
import string
import time
import unittest
from datetime import datetime, timezone

from fake_sink import Sink, attrs, roots
from support import Sandbox
from test_matrix import codex_patch, model_call, unified
from test_review3 import everywhere, pi_run
from test_review4 import assert_parity, leaked
from test_review_fixes import tool

from judgment_core import redact

_rng = random.Random(6060)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def _iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")


def codex_transcript(path, turn: str, prompt: str, patch: str, start: int) -> None:
    """A Codex rollout of one turn: two requests, the first calling apply_patch with free-form input (custom_tool_call)."""
    usage = {"info": {"last_token_usage": {"input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 40}}}
    rows = [("session_meta", {"id": "rollout"}), ("turn_context", {"turn_id": turn, "model": "gpt-5.5-codex"}),
            ("event_msg", {"type": "task_started", "turn_id": turn}),
            ("response_item", {"type": "message", "role": "user", "content": [{"type": "input_text", "text": prompt}]}),
            ("response_item", {"type": "custom_tool_call", "name": "apply_patch", "call_id": "call-patch", "input": patch}),
            ("event_msg", {"type": "token_count", **usage}),
            ("response_item", {"type": "custom_tool_call_output", "call_id": "call-patch", "output": "Success. Updated the following files"}),
            ("response_item", {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "Patched it."}]}),
            ("event_msg", {"type": "token_count", **usage}), ("event_msg", {"type": "task_complete", "turn_id": turn})]
    path.write_text("".join(json.dumps({"timestamp": _iso(start + 100 * n), "type": kind, "payload": payload}) + "\n" for n, (kind, payload) in enumerate(rows)))


def codex_run(test: unittest.TestCase, *, prompt: str, hook_input: dict | None = None, transcript_patch: str | None = None,
              output: str = "Success. Updated the following files") -> tuple[Sandbox, Sink, str]:
    """One Codex turn through the codex hook map and the sender. ``hook_input``: the apply_patch call arrives on the hook
    route as tool_input. ``transcript_patch``: it arrives only in the rollout transcript, which enrichment reads."""
    sink = Sink().__enter__()
    test.addCleanup(sink.__exit__, None, None, None)
    sb = Sandbox(sink_url=sink.url)
    test.addCleanup(sb.cleanup)
    session = "r6-" + _rand(6)
    turn = session + "-t1"
    path = ""
    if transcript_patch is not None:
        transcript = sb.home / "rollout.jsonl"
        codex_transcript(transcript, turn, prompt, transcript_patch, int(time.time() * 1000))
        path = str(transcript)
    base = {"session_id": session, "turn_id": turn, "cwd": str(sb.project), "model": "gpt-5.5-codex", "transcript_path": path}
    done = {**base, "status": "completed", "token_usage": {"input_tokens": 900, "cached_input_tokens": 0, "output_tokens": 40}}
    payloads = [("UserPromptSubmit", {**base, "prompt": prompt}), ("PreModelRequest", base), ("PostModelRequest", done)]
    if hook_input is not None:
        call = {**base, "tool_name": "apply_patch", "tool_use_id": "call-patch", "tool_input": hook_input}
        payloads += [("PreToolUse", call), ("PostToolUse", {**call, "tool_response": output})]
    payloads += [("PreModelRequest", base), ("PostModelRequest", done), ("Stop", {**base, "last_assistant_message": "Patched it."})]
    for event, payload in payloads:
        result = sb.hook("codex", event, payload)
        test.assertEqual(result.returncode, 0, result.stderr)
    inbox = json.dumps(sb.inbox())
    test.assertEqual(sb.flush().returncode, 0)
    test.assertTrue(sink.requests)
    return sb, sink, inbox


def blanked_rule(sink: Sink) -> tuple[str, str] | None:
    root = attrs(roots(sink.spans())[0])
    reason = root.get("judgment.redacted_reason")
    return (reason, root.get("judgment.redacted_rule")) if reason else None


class PatchTargetTests(unittest.TestCase):
    """Finding 1: Codex apply_patch arrives as {"input": patch}, and input was no path source, so a patch editing .env was
    exported unchanged. A patch input is now read token by token like a command, so any path it names counts."""

    def setUp(self):
        self.cookie = "Ck" + _rand(26)  # an opaque value no rule matches: only blanking keeps it out
        self.env_patch = codex_patch(("Update File", ".env"), body=f"@@\n-SESSION_COOKIE={self.cookie}\n+SESSION_COOKIE=")

    def test_both_layers(self):
        normal = codex_patch(("Update File", "src/app.py"), body="@@\n-def f():\n+def g():")
        events = [tool({"input": self.env_patch}), model_call({"input": self.env_patch}), tool({"input": unified(".env")}),
                  tool({"input": codex_patch(("Update File", "src/app.py"), ("Update File", "deploy/id_rsa"), ("Add File", "README.md"))}),
                  tool({"input": codex_patch(("Update File", "src/app.py"), ("Move to", "config/.env.production"))}),
                  tool({"input": normal}, "Success")]
        py = assert_parity(self, events)[0]
        self.assertEqual([out.get("sensitive_rule") for out in py], ["dotenv", "dotenv", "dotenv", "ssh_private_key", "dotenv", None])
        self.assertEqual(leaked(self.cookie, json.dumps(py[:2])), [])
        self.assertEqual((py[5]["tool_input"], py[5]["tool_output"]), ({"input": normal}, "Success"))

    def test_a_normal_patch_passes_through_except_real_secrets(self):
        key = "sk-" + "ant-" + "api03-" + _rand(60)
        patch = codex_patch(("Update File", "src/app.py"), body=f"@@\n-CLIENT = make()\n+CLIENT = make(key=\"{key}\")\n+print('password: ', pw)")
        py = assert_parity(self, [tool({"input": patch}, "Success")])[0][0]
        self.assertNotIn("sensitive", py)
        self.assertEqual(leaked(key, json.dumps(py)), [])
        # key is not a secret name: the format rule takes the value.
        self.assertEqual(py["tool_input"]["input"], patch.replace(key, "[REDACTED:anthropic_key]"))

    def test_hook_route_end_to_end(self):
        marker = "please rotate " + _rand(8)
        sb, sink, inbox = codex_run(self, prompt=marker, hook_input={"input": self.env_patch})
        self.assertIn('"sensitive_rule": "dotenv"', inbox)
        for where in everywhere(sb, sink, inbox):
            self.assertEqual(leaked(self.cookie, where), [])
        # The hook blanks the call's event; the sender blanks the whole turn, prompt included.
        self.assertNotIn(marker, sink.text())
        self.assertEqual(blanked_rule(sink), ("sensitive_path", "dotenv"))

    def test_transcript_only_route_end_to_end(self):
        marker = "please rotate " + _rand(8)
        sb, sink, inbox = codex_run(self, prompt=marker, transcript_patch=self.env_patch)
        self.assertNotIn(self.cookie, inbox)  # the hook never saw the patch
        for where in everywhere(sb, sink, inbox)[1:]:
            self.assertEqual(leaked(self.cookie, where), [])
        self.assertNotIn(marker, sink.text())
        self.assertEqual(blanked_rule(sink), ("sensitive_path", "dotenv"))

    def test_normal_patch_end_to_end_keeps_the_turn(self):
        line = "+def handler_" + _rand(8) + "():"
        patch = codex_patch(("Update File", "src/app.py"), body="@@\n" + line)
        for route in ({"hook_input": {"input": patch}}, {"transcript_patch": patch}):
            sb, sink, inbox = codex_run(self, prompt="refactor the handler", **route)
            self.assertIsNone(blanked_rule(sink), route)
            self.assertIn("refactor the handler", sink.text())
            self.assertIn(json.dumps(line)[1:-1], sink.text())


# Single-line code that names a secret key inside a string, with arguments after it: byte for byte in both layers.
SINGLE_LINE = [
    'print("password:", password, "done")', 'print("password: ", password, "done")', 'print("passwd:",passwd,"ok")',
    'logging.info("secret: %s", secret, extra={"a": 1})', 'log.debug("token=%s", token, "x")', 'printf("api_key: %s\\n", api_key); exit(0);',
    'fmt.Printf("password: %s", password, "x")', 'print(f"password: {password}", "done")', "print(f'secret: {secret}', 'done')",
    'console.log(`api_key: ${apiKey}`, "done")', 'console.log("token: " + token + " ok"); next();', 'msg = "password: " + pw + ", done"',
    "msg = 'secret:' + s + 'x'", "x = 1; print('password:', pw, 'done'); y = 2", 'print("password:"); print(password); print("done")',
    'String.format("password: %s", password, "x")', '"password: {}".format(password, "x")', 'throw new Error("bad api_key: " + k + "!")',
    'echo "password:" "$PASSWORD" "done"', 'console.log("user:", u, "password:", p, "done")', 'logger.warn("Enter secret: ", s, "now")',
    'puts "token: #{token}", "done"', 'print("API_KEY:", API_KEY, "done")', 'assert resp == "password:", "done"',
]
# Known limit (code-after-name): a format placeholder or reference right after the name is replaced, and only that.
PLACEHOLDER_AFTER_NAME = ['logging.info("secret: %s", secret, extra={"a": 1})', 'log.debug("token=%s", token, "x")',
                          'printf("api_key: %s\\n", api_key); exit(0);', 'fmt.Printf("password: %s", password, "x")',
                          'console.log(`api_key: ${apiKey}`, "done")', 'String.format("password: %s", password, "x")', 'puts "token: #{token}", "done"']
# Real single-line assignments: the value goes, whatever its shape.
ASSIGNMENTS = ['password = "{v}"', '"password": "{v}"', "PASSWORD='{v}'", 'run --password="{v}"',
               'x = {{"password": "{v}", "user": "bob"}}', 'self.password = "{v}"', 'f(user, password="{v}")', 'FOO=1 PASSWORD="{v}" run',
               'secret: "{v}"', "  api_key: '{v}'", 'export API_KEY="{v}"', 'docker run -e PASSWORD="{v}" img']


class SingleLineQuoteTests(unittest.TestCase):
    """Finding 2: on one line, a string's closing quote (print("password:", ...) was still read as a value's opening quote,
    deleting the arguments between. A quoted value now cannot start with a blank, a comma, or + ) ; ] }, which follow a
    string that just closed."""

    def test_preservation_corpus_byte_for_byte(self):
        kept = [text for text in SINGLE_LINE if text not in PLACEHOLDER_AFTER_NAME]
        texts = kept + [text + "\nrun()\nprint('ok')" for text in kept]
        py = assert_parity(self, [tool({"command": "x"}, text) for text in texts])[0]
        self.assertEqual([(text, out["tool_output"]) for text, out in zip(texts, py) if out["tool_output"] != text or "redactions" in out], [])

    def test_known_limit_placeholders_after_the_name(self):
        py = assert_parity(self, [tool({"command": "x"}, text) for text in PLACEHOLDER_AFTER_NAME])[0]
        for text, out in zip(PLACEHOLDER_AFTER_NAME, py):
            self.assertEqual([item["rule"] for item in out["redactions"]], ["secret_assignment"], text)

    def test_known_limit_a_value_after_a_blank(self):
        """Known limit (space-separated): with no : or = the value is not found."""
        value = "pw" + _rand(6, string.ascii_letters)
        py = assert_parity(self, [tool({"command": "x"}, 'mysql --password "' + value + '"')])[0]
        self.assertIn(value, py[0]["tool_output"])

    def test_real_assignments_are_redacted(self):
        cases = [(form, "s3cr3t-" + _rand(10)) for form in ASSIGNMENTS] + [(form, "pw" + _rand(4, string.ascii_letters)) for form in ASSIGNMENTS]
        py = assert_parity(self, [tool({"command": "x"}, form.format(v=value)) for form, value in cases])[0]
        self.assertEqual([(form, out["tool_output"]) for (form, value), out in zip(cases, py) if value in out["tool_output"]], [])
        self.assertEqual(py[0]["tool_output"], 'password = "[REDACTED:secret_assignment]"')

    def test_end_to_end(self):
        text = "\n".join(text for text in SINGLE_LINE if text not in PLACEHOLDER_AFTER_NAME)
        sb, sink, inbox = pi_run(self, output=text)
        for where in everywhere(sb, sink, inbox):
            self.assertIn(json.dumps(text)[1:-1], where)
        value = "pw" + _rand(6, string.ascii_letters)
        sb, sink, inbox = pi_run(self, output='print("password:", password, "done")\nmysql --password="' + value + '"')
        self.assertEqual(leaked(value, *everywhere(sb, sink, inbox)), [])
        self.assertIn(json.dumps('print("password:", password, "done")')[1:-1], sink.text())


class SecretCliOperandTests(unittest.TestCase):
    """Finding 3: secret_cli blanked configuration and help requests. A help or version request still never counts.
    Known limits: a listed CLI blanks the turn whatever it was asked for (cli-any-args), and a CLI the list leaves out
    does not (unlisted-cli: gh auth status --show-token prints a token the github_token format rule redacts)."""

    HARMLESS = ["gh auth token --help", "gh auth token -h", "gh auth status", "aws sts get-caller-identity", "aws configure list", "op --version"]
    ANY_ARGS = ["aws configure get region", "aws configure get region --profile dev", "kubectl get secrets", "kubectl get secret db",
                "op item get db --fields username", "security find-generic-password -s x", "doppler secrets set X=1", "sudo kubectl get secrets"]
    PRINTS = ["aws configure get aws_secret_access_key", "aws configure get profile.dev.aws_session_token", "aws configure export-credentials",
              "gh auth token", "kubectl get secret db -o yaml", "kubectl get secret db -o jsonpath='{.data.password}'", "kubectl describe secret db",
              "gcloud auth print-access-token", "az account get-access-token", "heroku auth:token", "op read op://v/i/f", "vault kv get secret/x",
              "vault read secret/x", "doppler secrets", "doppler secrets get DB_URL", "aws sts get-session-token",
              "aws secretsmanager get-secret-value --secret-id x", "aws ssm get-parameter --name /p --with-decryption", "security find-generic-password -s x -w"]
    UNLISTED = ["gh auth status --show-token", "gh auth status -t"]

    def test_both_layers(self):
        commands = self.HARMLESS + self.ANY_ARGS + self.PRINTS + self.UNLISTED
        py = assert_parity(self, [tool({"command": c}, "out") for c in commands])[0]
        got = {c: out.get("sensitive_rule") for c, out in zip(commands, py)}
        self.assertEqual({c: got[c] for c in self.HARMLESS + self.UNLISTED if got[c]}, {})
        self.assertEqual({c: got[c] for c in self.ANY_ARGS + self.PRINTS if got[c] != "secret_cli"}, {})

    def test_end_to_end(self):
        marker = "Print the auth token " + _rand(8)
        sb, sink, inbox = pi_run(self, command="gh auth token --help", output=marker)
        for where in everywhere(sb, sink, inbox):
            self.assertIn(marker, where)
        self.assertIsNone(blanked_rule(sink))
        value = "Ak" + _rand(30)
        sb, sink, inbox = pi_run(self, command="aws configure get aws_secret_access_key", output=value)
        self.assertEqual(leaked(value, *everywhere(sb, sink, inbox)), [])
        self.assertEqual(blanked_rule(sink), ("sensitive_path", "secret_cli"))


class LinearPatchTests(unittest.TestCase):
    """Patch inputs and command rules cost linear work on adversarial input in Python (jq: tests/test_latency.py)."""

    def test_python(self):
        rs = redact.ruleset()
        for unit in ["*** Update File: a\n", "--- a/x\n+++ b/y\n", "diff --git a b ", "\n*** Move to: ", "gh auth token -t ", "aws configure get "]:
            text = (unit * (100_000 // len(unit) + 1))[:100_000]
            started = time.perf_counter()
            try:
                redact.sensitive(tool({"input": text, "command": text}), rs)
            except redact.RedactionLimit:
                pass
            self.assertLess((time.perf_counter() - started) * 1000, 2000, repr(unit))


if __name__ == "__main__":
    unittest.main()
