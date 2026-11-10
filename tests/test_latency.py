"""Finding 8: no input makes redaction slow. Adversarial text, about 100 KB, for every rule, in both layers.

The inputs come from the rules table: each rule's needed literals, repeated with the separators and value
characters that would make an unbounded lookahead rescan the rest of the text, or a global match count
offsets over and over. The layer-1 bound is for a whole jq run, startup included.
"""

from __future__ import annotations

import json
import subprocess
import time
import unittest

from support import CORE, clean_env

from judgment_core import redact

SIZE = 100_000
JQ_MS = 100  # one jq run on one ~100 KB event, startup and compile included
PY_MS = 250  # the sender's two passes (the hook's mirror, then its own) on the same event


def fill(unit: str, size: int = SIZE) -> str:
    return (unit * (size // max(len(unit), 1) + 1))[:size]


def adversarial() -> list[tuple[str, str]]:
    table = redact.rules()
    cases = [
        ("codex: token= x16000", "token=" * 16000),
        ("token=a1", fill("token=a1")),
        ("json api_key member", fill('{"api_key":"')),
        ("json members", fill('{"api_key":"Zq81jdjskaQ8a9d2' + '",')),
        ("many distinct tokens", " ".join("gh" + "p_" + format(n, "036d") for n in range(SIZE // 41))),
        ("pem headers", fill("-----BEGIN RSA PRIV" + "ATE KEY-----\n")),
        ("pem blocks", fill("-----BEGIN RSA PRIV" + "ATE KEY-----\nMIIabc\n-----END RSA PRIV" + "ATE KEY-----\n")),
        ("words", fill("hello world ")),
        ("one long word", "a" * SIZE),
        ("@mentions", fill("@a ")),
    ]
    rules = list(table["patterns"]) + list(table["members"]) + table["sensitive"]["paths"] + table["sensitive"]["commands"]
    for rule in rules:
        needs = rule.get("needs")
        need = needs[0] if isinstance(needs, list) and needs else "key"
        for sep in ("=", ": ", "://", " ", "\n", "a1"):
            cases.append((f"{rule['id']}: {need!r}+{sep!r}", fill(need + sep)))
    return cases


def events(text: str) -> list[dict]:
    base = {"v": 1, "harness": "pi", "session_id": "s", "turn_id": "t", "event": "tool", "tool_name": "bash", "tool_call_id": "c"}
    return [{**base, "tool_input": {"command": "make"}, "tool_output": text},
            {**base, "tool_input": {"command": text}, "tool_output": "ok"},
            {**base, "event": "turn_start", "prompt": text}]


class LatencyTests(unittest.TestCase):
    def test_python_layers(self):
        rs = redact.ruleset()
        secrets = redact.env_secrets(clean_env(SERVICE_TOKEN="Sv" + "9x" * 12))
        slow = []
        for name, text in adversarial():
            for event in events(text):
                started = time.perf_counter()
                redact.sanitize_event(event, rs, secrets)
                redact.redact_event(event, rs, secrets)
                redact.first_sensitive([event], rs, secrets)
                elapsed = (time.perf_counter() - started) * 1000
                if elapsed > PY_MS:
                    slow.append((name, event["event"], round(elapsed)))
        self.assertEqual(slow, [])

    def test_jq_layer(self):
        program = 'include "prepare"; judgment_prepare({enabled: true}; "on"; "/r"; "")'
        slow = []
        for name, text in adversarial():
            for event in events(text):
                data = json.dumps(event)
                best = None
                for _ in range(2):  # the better of two runs, so a busy machine does not fail the test
                    started = time.perf_counter()
                    result = subprocess.run(["jq", "-L", str(CORE / "bin"), "-c", program], input=data, capture_output=True, text=True,
                                            env=clean_env(SERVICE_TOKEN="Sv" + "9x" * 12), timeout=30)
                    elapsed = (time.perf_counter() - started) * 1000
                    best = elapsed if best is None else min(best, elapsed)
                    if elapsed <= JQ_MS:
                        break
                self.assertEqual(result.returncode, 0, result.stderr)
                if best > JQ_MS:
                    slow.append((name, event["event"], round(best)))
        self.assertEqual(slow, [])

    def test_limits_fail_closed(self):
        """Text past the limits loses its text in both layers, and never comes out raw."""
        pem = "-----BEGIN RSA PRIV" + "ATE KEY-----\nMIIabc\n-----END RSA PRIV" + "ATE KEY-----\n"
        tokens = " ".join("gh" + "p_" + format(n, "036d") for n in range(200))
        for text in (pem * 40, tokens):
            event = events(text)[0]
            py = redact.sanitize_event(event, redact.ruleset(), [])
            out = subprocess.run(["jq", "-L", str(CORE / "bin"), "-c", 'include "prepare"; judgment_prepare({enabled: true}; "on"; "/r"; "")'],
                                 input=json.dumps(event), capture_output=True, text=True, env=clean_env(), timeout=30)
            js = json.loads(out.stdout)
            for result in (py, js):
                self.assertTrue(result.get("redaction_error"))
                self.assertNotIn("tool_output", result)


if __name__ == "__main__":
    unittest.main()
