"""Regressions for the third review of capture: sanitized. One class per finding, each reproducing the review's case in
both layers (jq and its Python mirror, which must agree) and, where it applies, end to end: hook, inbox, stored turn, and
the HTTP body the sender posts.

Every fake secret is built at runtime from parts (support.SECRETS, or _rand below), so no file holds one.
"""

from __future__ import annotations

import json
import random
import string
import subprocess
import sys
import time
import unittest
import unittest.mock

from fake_sink import Sink, attrs, roots
from payloads import pi_turn
from support import CORE, ENV_SECRET_NAME, SECRETS, Sandbox, claude_transcript, clean_env, jq_sanitize, norm, py_sanitize
from test_review_fixes import both, tool

from judgment_core import flush, redact

_rng = random.Random(3333)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def filler(n: int) -> str:
    """Ordinary words, no rule's literal in them."""
    words = ("alpha", "beta", "gamma", "delta", "omega", "build", "step", "done")
    out, i = [], 0
    while sum(len(word) + 1 for word in out) < n:
        out.append(words[i % len(words)])
        i += 1
    return " ".join(out)[:n]


def pieces(secret: str, size: int = 8) -> list[str]:
    return [secret[i : i + size] for i in range(0, len(secret) - size + 1)]


def pi_run(test: unittest.TestCase, *, output=None, command: str = "", answer: str = "", env: dict | None = None) -> tuple[Sandbox, Sink, str]:
    """One Pi turn through the hook and the sender. Returns the sandbox, the sink, and the inbox as the hook left it.
    ``output`` may be any JSON value: the tool event's output is replaced with it."""
    sink = Sink().__enter__()
    test.addCleanup(sink.__exit__, None, None, None)
    sb = Sandbox(sink_url=sink.url)
    test.addCleanup(sb.cleanup)
    sb.env.update(env or {})
    session = "r3-" + _rand(6)
    for _, payload in pi_turn(str(sb.project), session, session + ":1", int(time.time() * 1000) - 5000, command=command, answer=answer or "Done."):
        if payload["event"] == "tool" and output is not None:
            payload = {**payload, "tool_output": output}
        test.assertEqual(sb.hook("pi", "", payload).returncode, 0)
    inbox = json.dumps(sb.inbox())
    test.assertEqual(sb.flush().returncode, 0)
    test.assertTrue(sink.requests)
    return sb, sink, inbox


def everywhere(sb: Sandbox, sink: Sink, inbox: str) -> list[str]:
    return [inbox, sb.state_text(), sink.text()]


class CompleteSpanTests(unittest.TestCase):
    """Finding 1: a shortened token was masked everywhere before the full one was found, so the full one's suffix leaked."""

    def cases(self) -> list[tuple[str, str]]:
        full = "sk-" + "proj-" + _rand(48)
        short = full[:30] + "..."
        out = []
        # Short text (one global match), one long window (the scan), and two windows.
        for before in (100, 9000, 15000):
            out.append((full, filler(before) + " " + short + " then " + full + " " + filler(9000)))
        return out

    def test_both_layers(self):
        for full, text in self.cases():
            py, js = both([tool({"command": "x"}, text)])
            self.assertEqual(norm(py), norm(js))
            for out in (py[0], js[0]):
                self.assertNotIn("redaction_error", out)
                self.assertEqual([piece for piece in pieces(full[8:]) if piece in json.dumps(out)], [])
                self.assertEqual(out["redactions"], [{"rule": "openai_key", "field": "tool_output", "count": 2}])

    def test_multibyte_text(self):
        # jq slices strings by code point, as Python does: the character after a match is the same in both layers.
        full = "sk-" + "proj-" + _rand(48)
        text = "h\u00e9llo w\u00f6rld \u2713 " * 700 + full[:30] + "\u2026 " + full + "\u00e9 " + full[:30] + "\u2713 " + "\u65e5\u672c " * 500
        py, js = both([tool({"command": "x"}, text)])
        self.assertEqual(norm(py), norm(js))
        self.assertNotIn(full[30:], json.dumps(py + js))
        self.assertEqual(py[0]["redactions"], [{"rule": "openai_key", "field": "tool_output", "count": 3}])

    def test_a_copy_that_starts_a_longer_match_is_found_whole(self):
        # jq's jp_scan blanks only copies followed by the character that ended a match; the sender's single finditer
        # finds the same distinct texts.
        rule = next(rule for rule in redact.ruleset().patterns if rule.id == "openai_key")
        full = "sk-" + "proj-" + _rand(48)
        window = filler(9000) + " " + full[:30] + "... " + full + " " + full[:30] + "."
        found = sorted({pair.found for pair in redact.found(rule, window)})
        self.assertEqual(found, sorted({full[:30], full}))

    def test_end_to_end(self):
        full, text = self.cases()[1]
        sb, sink, inbox = pi_run(self, output=text)
        for where in everywhere(sb, sink, inbox):
            self.assertEqual([piece for piece in pieces(full[8:]) if piece in where], [])


class EnrichmentProvenanceTests(unittest.TestCase):
    """Finding 2: the hook removed an env value, then a sender without that variable replaced the field with transcript text."""

    def turn(self, sb: Sandbox, value: str) -> None:
        transcript = sb.home / "t.jsonl"
        start = int(time.time() * 1000) - 60_000
        claude_transcript(transcript, "ep-1", "p1", "check the service", start, "p1-tool", "cat svc.txt", "svc " + value, "The service is up.")
        base = {"session_id": "ep-1", "cwd": str(sb.project), "transcript_path": str(transcript), "prompt_id": "p1"}
        tool_event = {**base, "tool_name": "Bash", "tool_use_id": "p1-tool", "tool_input": {"command": "cat svc.txt"}, "tool_response": {"stdout": "svc " + value}}
        for event, payload in (("UserPromptSubmit", {**base, "prompt": "check the service"}), ("PostToolUse", tool_event),
                               ("Stop", {**base, "last_assistant_message": "The service is up."})):
            self.assertEqual(sb.hook("claude", event, payload).returncode, 0)

    def other_sender(self, sb: Sandbox) -> subprocess.CompletedProcess:
        """The sender another agent started: same key, without the variable the hook redacted."""
        env = {key: value for key, value in sb.env.items() if key != ENV_SECRET_NAME}
        return subprocess.run([sys.executable, "-m", "judgment_core.flush"], capture_output=True, text=True, env={**env, "PYTHONPATH": str(CORE)}, timeout=60)

    def test_sender_without_the_variable(self):
        value = SECRETS["env_value"]["core"]
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            self.turn(sb, value)
            inbox = json.dumps(sb.inbox())
            self.assertIn(f"[REDACTED:env:{ENV_SECRET_NAME}]", inbox)
            self.assertEqual(self.other_sender(sb).returncode, 0)
            self.assertTrue(sink.requests)
            for where in (inbox, sb.state_text(), sink.text()):
                self.assertEqual([piece for piece in pieces(value) if piece in where], [])
            # The hook's sanitized output is what the tool span shows.
            self.assertIn(f"svc [REDACTED:env:{ENV_SECRET_NAME}]", sink.text())

    def test_sender_with_the_variable_still_redacts(self):
        value = SECRETS["env_value"]["core"]
        with Sink() as sink:
            sb = Sandbox(sink_url=sink.url)
            self.addCleanup(sb.cleanup)
            self.turn(sb, value)
            self.assertEqual(sb.flush().returncode, 0)
            self.assertTrue(sink.requests)
            self.assertNotIn(value, sink.text() + sb.state_text())
            self.assertIn("The service is up.", sink.text())

    def test_carry_records_rules(self):
        record = {"rule": f"env:{ENV_SECRET_NAME}", "field": "tool_output", "count": 1}
        stored = [{"_id": "a", "event": "tool", "tool_output": "svc [REDACTED]", "redactions": [record]},
                  {"_id": "b", "event": "turn_start", "prompt": "hi"}]
        prepared = [{"_id": "a", "event": "tool", "tool_output": "svc RAW"}, {"_id": "b", "event": "turn_start", "prompt": "hi there", "extra": "x"},
                    {"event": "model", "output": "RAW again", "model": "m"}]
        # The hook's field always stays. With the variable, transcript text elsewhere is the sender's to redact.
        held = flush.carry_records(stored, prepared, {ENV_SECRET_NAME})
        self.assertEqual(held[0]["tool_output"], "svc [REDACTED]")
        self.assertEqual(held[0]["redactions"], [record])
        self.assertEqual((held[1]["prompt"], held[2]["output"]), ("hi there", "RAW again"))
        # Without it, no transcript text stays: changed fields go back to the hook's, added ones and new events lose it.
        lacking = flush.carry_records(stored, prepared, set())
        self.assertEqual(lacking[0]["tool_output"], "svc [REDACTED]")
        self.assertEqual(lacking[1]["prompt"], "hi")
        self.assertNotIn("extra", lacking[1])
        self.assertNotIn("output", lacking[2])
        self.assertEqual(lacking[2]["model"], "m")


class LongQuotedValueTests(unittest.TestCase):
    """Finding 3: a quoted value over 256 characters matched nothing and passed through."""

    def test_302_character_refresh_token(self):
        token = _rand(302)
        events = [tool({"command": "x"}, '{"refresh_token": "' + token + '"}'), tool({"command": "x"}, "refresh_token = '" + token + "'"),
                  tool({"command": "x"}, json.dumps({"body": json.dumps({"refresh_token": token})}))]
        py, js = both(events)
        self.assertEqual(norm(py), norm(js))
        for out in py + js:
            self.assertNotIn("redaction_error", out)
            self.assertEqual([piece for piece in pieces(token) if piece in json.dumps(out)], [])

    def test_oversized_candidate_fails_closed(self):
        limits = redact.rules()["limits"]
        # Longer than window - step, and starting before the second window: no window holds it whole.
        token = _rand(limits["window"] - limits["step"] + 800)
        text = filler(limits["step"] - 200) + ' "refresh_token": "' + token + '" ' + filler(limits["window"])
        py, js = both([tool({"command": "x"}, text)])
        for out in (py[0], js[0]):
            self.assertTrue(out.get("redaction_error"))
            self.assertNotIn(token[100:140], json.dumps(out))

    def test_end_to_end(self):
        token = _rand(302)
        sb, sink, inbox = pi_run(self, output='{"refresh_token": "' + token + '"}')
        for where in everywhere(sb, sink, inbox):
            self.assertEqual([piece for piece in pieces(token) if piece in where], [])


class WholeReferenceTests(unittest.TestCase):
    """Finding 4: a value that started like a reference ($x, ${, process.env) was exempt whatever followed."""

    LITERALS = ["$special Password 123", "${DB_PASSWORD}tail", "process.env.X + 'abc'", "os.environ['X'] or hunter2", "$" + _rand(6) + " x"]
    REFERENCES = ["${DB_PASSWORD}", "$DB_PASSWORD", "process.env.DB_PASS", "os.environ['DB_PASS']", 'os.getenv("DB_PASS")',
                  "{{ .Values.db.password }}", "${{ secrets.DB_PASS }}", "op://vault/item/password", "[REDACTED:secret_field]"]

    def test_both_layers(self):
        events = [tool({"command": "x"}, {"password": value}) for value in self.LITERALS + self.REFERENCES]
        events += [tool({"command": "x"}, json.dumps({"password": value})) for value in self.LITERALS + self.REFERENCES]
        py, js = both(events)
        self.assertEqual(norm(py), norm(js))
        n = len(self.LITERALS) + len(self.REFERENCES)
        for structured, text in ((py[:n], False), (py[n:], True)):
            for out, value in zip(structured, self.LITERALS + self.REFERENCES):
                output = json.loads(out["tool_output"]) if text else out["tool_output"]
                if value in self.LITERALS:
                    self.assertNotIn(value, json.dumps(out), value)
                    self.assertTrue(output["password"].startswith("[REDACTED:"), value)
                else:
                    self.assertEqual(output["password"], value)

    def test_end_to_end(self):
        literal = "$special Password " + _rand(5)
        sb, sink, inbox = pi_run(self, output={"password": literal, "user": "pat"})
        for where in everywhere(sb, sink, inbox):
            self.assertNotIn(literal, where)


class UrlPasswordTests(unittest.TestCase):
    """Finding 5: a %-encoded password, or one of one or two characters, stayed in a URL."""

    def urls(self) -> tuple[dict[str, str], list[str]]:
        go = {"postgres://app:%40encodedOpaque" + _rand(10) + "@db/app": "%40encodedOpaque", "redis://u:a@h:6379": "u:a@",
              "redis://u:z9@h": "u:z9@", "redis://:pw" + _rand(4) + "@h": "://:pw"}
        stay = ["postgres://u:${DB_PASS}@h/app", "postgres://u:password@h/app", "postgres://u:****@h/app", "postgres://u:[REDACTED:url_password]@h",
                "http://localhost:8080/x@y", "https://example.com/path"]
        return go, stay

    def test_both_layers(self):
        go, stay = self.urls()
        py, js = both([tool({"command": "x"}, url) for url in list(go) + stay])
        self.assertEqual(norm(py), norm(js))
        for out, (url, piece) in zip(py, go.items()):
            self.assertNotIn(piece, out["tool_output"], url)
            self.assertIn("[REDACTED:url_password]", out["tool_output"])
        for out, url in zip(py[len(go):], stay):
            self.assertEqual(out["tool_output"], url)

    def test_end_to_end(self):
        secret = "%40encoded" + _rand(14)
        sb, sink, inbox = pi_run(self, output="dsn postgres://app:" + secret + "@db/app")
        for where in everywhere(sb, sink, inbox):
            self.assertNotIn(secret, where)


class SensitiveOperandTests(unittest.TestCase):
    """Finding 6: `git show HEAD:.env` and `rg . credentials` read secrets without blanking the turn."""

    READS = {"git show HEAD:.env": "dotenv", "git show HEAD~2:config/.env.local": "dotenv", "rg . credentials": "credentials_file",
             "git show HEAD:credentials": "credentials_file", "cat ./credentials": "credentials_file", "wc -l credentials": "credentials_file",
             "git show stash@{0}:id_rsa": "ssh_private_key"}
    MENTIONS = ["rg credentials .", "grep -rn credentials src", "git log --grep credentials", "git grep credentials", "echo credentials",
                "git show HEAD:.env.example", "git show HEAD:README.md"]

    def test_both_layers(self):
        commands = list(self.READS) + self.MENTIONS
        py, js = both([tool({"command": command}, "out") for command in commands])
        self.assertEqual(norm(py), norm(js))
        for out, command in zip(py, commands):
            self.assertEqual(out.get("sensitive_rule"), self.READS.get(command), command)

    def end_to_end(self, command: str, rule: str) -> None:
        leftover = "Lv" + _rand(18)  # matches no rule: only blanking keeps it out
        sb, sink, inbox = pi_run(self, command=command, output="VALUE=" + leftover)
        for where in everywhere(sb, sink, inbox):
            self.assertNotIn(leftover, where)
        root = attrs(roots(sink.spans())[0])
        self.assertEqual((root["judgment.redacted_reason"], root["judgment.redacted_rule"]), ("sensitive_path", rule))

    def test_git_revision_path_end_to_end(self):
        self.end_to_end("git show HEAD:.env", "dotenv")

    def test_bare_credentials_operand_end_to_end(self):
        self.end_to_end("rg . credentials", "credentials_file")


class PrintenvPositionTests(unittest.TestCase):
    """Finding 7: any mention of printenv blanked the turn."""

    RUNS = ["printenv", "printenv HOME", "sudo printenv", "ls; printenv", "env -i A=1 printenv", "sh -c 'printenv'", "x=$(printenv)",
            "/usr/bin/printenv | sort", "command printenv", "true && printenv PATH"]
    MENTIONS = ["command -v printenv", "which printenv", "grep printenv README.md", 'echo "printenv"', "man printenv", "type printenv",
                "rg -n 'printenv' src"]

    def test_both_layers(self):
        commands = self.RUNS + self.MENTIONS
        py, js = both([tool({"command": command}, "out") for command in commands])
        self.assertEqual(norm(py), norm(js))
        for out, command in zip(py, commands):
            self.assertEqual(bool(out.get("sensitive")), command in self.RUNS, command)

    def test_mention_keeps_the_turn_end_to_end(self):
        marker = "Kept" + _rand(10)
        sb, sink, inbox = pi_run(self, command="command -v printenv", output="/usr/bin/printenv " + marker)
        self.assertIn(marker, sink.text())
        self.assertNotIn("judgment.redacted_reason", attrs(roots(sink.spans())[0]))


class NumberTokenTests(unittest.TestCase):
    """Finding 8: replacing an env value in serialized JSON rewrote a number with the same digits, which broke the JSON."""

    def test_numbers_stay_and_text_goes(self):
        pin = "".join(_rng.choice("123456789") for _ in range(8))
        env = clean_env(PIN_PASSWORD=pin)
        output = {"count": int(pin), "ratio": float(pin + ".5"), "rows": [int(pin), "row " + pin], "note": "pin " + pin, "result": "unrelated text"}
        py, js = both([tool({"command": "x"}, output)], env=env)
        self.assertEqual(norm(py), norm(js))
        for out in (py[0], js[0]):
            self.assertNotIn("redaction_error", out)
            self.assertEqual(out["tool_output"], {"count": int(pin), "ratio": float(pin + ".5"), "rows": [int(pin), "row [REDACTED:env:PIN_PASSWORD]"],
                                                  "note": "pin [REDACTED:env:PIN_PASSWORD]", "result": "unrelated text"})
            self.assertEqual(out["redactions"], [{"rule": "env:PIN_PASSWORD", "field": "tool_output", "count": 2}])

    def test_numeric_secret_as_a_key_fails_closed(self):
        # Digits are replaced in string values only; a key holding them is not reached, so the event loses its text.
        pin = "".join(_rng.choice("123456789") for _ in range(8))
        py, js = both([tool({"command": "x"}, {pin: "x", "note": "pin " + pin})], env=clean_env(PIN_PASSWORD=pin))
        self.assertEqual(norm(py), norm(js))
        self.assertTrue(py[0].get("redaction_error"))
        self.assertNotIn(pin, json.dumps(py + js))

    def test_broken_json_fails_closed_in_python(self):
        # Mirror of jq's fromjson inside its try: a replacement that breaks the JSON drops the text, never raises.
        with unittest.mock.patch.object(redact, "apply_counted", side_effect=lambda text, *args, **kw: text[:-1] if text.startswith("{") else text):
            out = redact.sanitize_event(tool({"command": "x"}, {"api_key": "v" + _rand(12)}), redact.ruleset(), [])
        self.assertTrue(out.get("redaction_error"))
        self.assertNotIn("tool_output", out)

    def test_end_to_end(self):
        pin = "".join(_rng.choice("123456789") for _ in range(8))
        marker = "Unrelated" + _rand(10)
        sb, sink, inbox = pi_run(self, output={"count": int(pin), "note": "pin " + pin, "result": marker}, env={"PIN_PASSWORD": pin})
        self.assertIn(marker, sink.text())
        self.assertIn(marker, sb.state_text())
        self.assertNotIn("pin " + pin, sink.text() + sb.state_text() + inbox)


class EscapedQuoteTests(unittest.TestCase):
    """Finding 9: in doubly encoded JSON, the plain-quote alternative took the escaped quote and ate the members after it."""

    def output(self) -> tuple[str, str, str]:
        password = "pa\\ss" + _rand(10)
        marker = "fine-" + _rand(8)
        inner = json.dumps({"password": password, "rows": [1, 2, 3], "result": marker})
        return json.dumps({"body": inner}), password, marker

    def test_both_layers(self):
        text, password, marker = self.output()
        quoted = 'password = "pa\\"ss' + _rand(8) + '" next="keep-me"'
        py, js = both([tool({"command": "x"}, text), tool({"command": "x"}, quoted)])
        self.assertEqual(norm(py), norm(js))
        body = json.loads(json.loads(py[0]["tool_output"])["body"])
        self.assertEqual(body, {"password": "[REDACTED:quoted_secret]", "rows": [1, 2, 3], "result": marker})
        self.assertEqual(py[1]["tool_output"], 'password = "[REDACTED:quoted_secret]" next="keep-me"')

    def test_end_to_end(self):
        text, password, marker = self.output()
        sb, sink, inbox = pi_run(self, output=text)
        self.assertIn(marker, sink.text())
        for where in everywhere(sb, sink, inbox):
            self.assertNotIn(password[6:], where)


if __name__ == "__main__":
    unittest.main()
