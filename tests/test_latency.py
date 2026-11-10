"""Finding 8: no input makes redaction slow. Adversarial text, about 100 KB, for every rule, in both layers.

The inputs come from the rules table: each rule's needed literals, repeated with the separators and value
characters that would make an unbounded lookahead rescan the rest of the text, or a global match count
offsets over and over. The layer-1 bound is for a whole jq run, startup included.

The bounds catch catastrophic regressions (a rule that rescans, a quadratic match: seconds, not milliseconds) without
flaking on a loaded machine: each case takes the best of up to three runs, and its bound is the larger of a fixed one
and a multiple of what the same layer takes, at that moment, on a benign event of the same size (re-measured before a case
is called slow). On an idle machine (2026-11) the worst case is 7.5x the benign one in jq (~90 ms) and 20x in Python
(~200 ms): the bounds leave about 3x of room, and a load that slows everything scales the bound with it.
"""

from __future__ import annotations

import json
import subprocess
import time
import unittest

from support import CORE, clean_env

from judgment_core import redact

SIZE = 100_000
JQ_MS = 250  # one jq run on one ~100 KB event, startup and compile included
PY_MS = 600  # the sender's two passes (the hook's mirror, then its own) and its sensitive check on the same event
JQ_SCALE = 25  # x the benign jq run measured at the same time
PY_SCALE = 60  # x the benign Python passes measured at the same time
RUNS = 3
PROGRAM = 'include "prepare"; judgment_prepare({enabled: true}; "on"; "/r"; "")'
ENV = clean_env(SERVICE_TOKEN="Sv" + "9x" * 12)


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
        # Review 7: the code context's check after each value (bounded by assignment.after_bound), and line locations.
        ("code values on one line", fill(', password="ab"')),
        ("code values, open quote", fill(', password="ab" "')),
        ("long locations", fill("a" * 300 + ":12:password: \"x\"\n")),
        ("location digits", fill("1234567890:")),
    ]
    rules = list(table["patterns"]) + list(table["members"]) + table["sensitive"]["paths"] + table["sensitive"]["commands"]
    for rule in rules:
        needs = redact.cli_needs(rule) if "rows" in rule else rule.get("needs")
        need = needs[0] if isinstance(needs, list) and needs else "key"
        for sep in ("=", ": ", "://", " ", "\n", "a1"):
            cases.append((f"{rule['id']}: {need!r}+{sep!r}", fill(need + sep)))
    # The patch-header parser: every header row, flooded (fails closed past limits.patch_headers) and as one long line.
    for header in table["sensitive"]["patch"]["headers"]:
        cases.append((f"patch {header['prefix']!r} lines", fill(header["prefix"] + "a\n")))
        cases.append((f"patch {header['prefix']!r} one line", "\n" + header["prefix"] + fill("a/")))
    # Each CLI row's words, repeated: options and lookaheads over arguments stay bounded.
    for item in table["sensitive"]["commands"]:
        for row in item.get("rows", []):
            words = " ".join([row["cmd"].split("|")[0]] + [w.split("|")[0] for w in row["words"]])
            cases.append((f"cli {row['row']}", fill(words + " -o ")))
    return cases


def events(text: str) -> list[dict]:
    base = {"v": 1, "harness": "pi", "session_id": "s", "turn_id": "t", "event": "tool", "tool_name": "bash", "tool_call_id": "c"}
    return [{**base, "tool_input": {"command": "make"}, "tool_output": text},
            {**base, "tool_input": {"command": text}, "tool_output": "ok"},
            {**base, "event": "turn_start", "prompt": text}]


def jq_once(data: str) -> tuple[float, subprocess.CompletedProcess]:
    started = time.perf_counter()
    result = subprocess.run(["jq", "-L", str(CORE / "bin"), "-c", PROGRAM], input=data, capture_output=True, text=True, env=ENV, timeout=60)
    return (time.perf_counter() - started) * 1000, result


def median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2]


BENIGN = events(fill("hello world "))[0]  # the same size as the cases, matching no rule


def jq_bound() -> float:
    return max(JQ_MS, JQ_SCALE * median([jq_once(json.dumps(BENIGN))[0] for _ in range(5)]))


def py_bound(rs, secrets) -> float:
    def once() -> float:
        started = time.perf_counter()
        redact.sanitize_event(BENIGN, rs, secrets)
        redact.redact_event(BENIGN, rs, secrets)
        redact.first_sensitive([BENIGN], rs, secrets)
        return (time.perf_counter() - started) * 1000

    return max(PY_MS, PY_SCALE * median([once() for _ in range(5)]))


class LatencyTests(unittest.TestCase):
    def test_python_layers(self):
        rs = redact.ruleset()
        secrets = redact.env_secrets(ENV)
        bound = py_bound(rs, secrets)
        slow = []
        for name, text in adversarial():
            for event in events(text):
                best = None
                for _ in range(RUNS):
                    started = time.perf_counter()
                    redact.sanitize_event(event, rs, secrets)
                    redact.redact_event(event, rs, secrets)
                    redact.first_sensitive([event], rs, secrets)
                    elapsed = (time.perf_counter() - started) * 1000
                    best = elapsed if best is None else min(best, elapsed)
                    if best <= bound:
                        break
                if best > bound:
                    bound = py_bound(rs, secrets)  # the machine may be busier now than when the bound was taken
                if best > bound:
                    slow.append((name, event["event"], round(best), round(bound)))
        self.assertEqual(slow, [])

    def test_jq_layer(self):
        bound = jq_bound()
        slow = []
        for name, text in adversarial():
            for event in events(text):
                data = json.dumps(event)
                best = None
                for _ in range(RUNS):
                    elapsed, result = jq_once(data)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    best = elapsed if best is None else min(best, elapsed)
                    if best <= bound:
                        break
                if best > bound:
                    bound = jq_bound()
                if best > bound:
                    slow.append((name, event["event"], round(best), round(bound)))
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
