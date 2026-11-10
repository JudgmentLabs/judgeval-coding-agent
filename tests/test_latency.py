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
from typing import Callable

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
        ("code values on one line", fill(', password="ab"')),
        ("code values, open quote", fill(', password="ab" "')),
        ("long locations", fill("a" * 300 + ":12:password: \"x\"\n")),
        ("location digits", fill("1234567890:")),
    ]
    done = redact.compiled(table)
    rules = done["patterns"] + [done["names"]["assignment"], done["names"]["field"]] + done["sensitive"]["paths"] + done["sensitive"]["commands"]
    for rule in rules:
        needs = rule.get("needs")
        need = needs[0] if isinstance(needs, list) and needs else "key"
        for sep in ("=", ": ", "://", " ", "\n", "a1"):
            cases.append((f"{rule['id']}: {need!r}+{sep!r}", fill(need + sep)))
    # Each command rule's examples, repeated in one command and as one long command: the help check reads one command each.
    for item in table["sensitive"]["commands"]:
        for example in item["examples"]:
            cases.append((f"{item['id']} {example!r} repeated", fill(example + "; ")))
            cases.append((f"{item['id']} {example!r} long", example + fill(" -o")))
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


def growth_cases() -> list[tuple[str, Callable[[int], str]]]:
    """Pathological text for every built-in rule, at any size: what made a rule backtrack before (round 8 finding d), runs of
    the characters the rules stop at, and each rule's needed literal before an operator."""
    done = redact.compiled()
    rules = done["patterns"] + [done["names"]["assignment"], done["names"]["field"]] + done["sensitive"]["paths"] + done["sensitive"]["commands"]
    cases = [
        ("finding d: quote pairs after a value", lambda n: 'f(password="pw") ' + "''" * (n // 2) + ' "'),
        ("double quotes", lambda n: '"' * n),
        ("single quotes", lambda n: "'" * n),
        ("escaped quotes", lambda n: fill('\\"', n)),
        ("password=", lambda n: fill("password=", n)),
        ('"password": "', lambda n: fill('"password": "', n)),
        (":= runs", lambda n: fill(":=", n)),
        (": runs", lambda n: ":" * n),
        ("= runs", lambda n: "=" * n),
        ("token-like run", lambda n: fill("Ab3", n)),
        ("one long word after a name", lambda n: "token=" + fill("Ab3", n)),
        ("path components", lambda n: fill("a/", n) + ".env"),
        ("commands", lambda n: fill("; gh auth token -x", n)),
    ]
    for rule in rules:
        need = (rule.get("needs") or ["key"])[0]
        cases += [(f"{rule['id']}: {need!r}=", lambda n, need=need: fill(need + "=", n)), (f"{rule['id']}: {need!r}: ", lambda n, need=need: fill(need + ": ", n))]
    return cases


GROWTH_SIZE = 100_000
GROWTH_RATIO = 3  # time at twice the size, as a multiple of the time at the size: about 2 for linear work, 4 for quadratic
PY_SLACK_MS, JQ_SLACK_MS = 40, 150  # noise allowed on top of the ratio: a short run is all startup and timer jitter
PY_MAX_MS, JQ_MAX_MS = 4000, 6000  # at twice the size, a generous absolute bound


class LinearGrowthTests(unittest.TestCase):
    """Every built-in rule does linear work in both layers: on pathological text, twice the text takes about twice the time,
    and never more than a generous absolute bound. The round 8 finding d string is checked at n=5000 on its own as well.
    Project regexes (redactPatterns, redactCommands, redactPaths globs) are the user's: a slow one is a known limit."""

    def py_ms(self, text: str, rs, secrets, runs: int = 1) -> float:
        best = None
        for _ in range(runs):
            started = time.perf_counter()
            for event in events(text):
                redact.sanitize_event(event, rs, secrets)
                redact.redact_event(event, rs, secrets)
                redact.first_sensitive([event], rs, secrets)
            elapsed = (time.perf_counter() - started) * 1000
            best = elapsed if best is None else min(best, elapsed)
        return best

    def jq_ms(self, text: str, runs: int = 1) -> float:
        data = "\n".join(json.dumps(event) for event in events(text))
        best = None
        for _ in range(runs):
            elapsed, result = jq_once(data)
            self.assertEqual(result.returncode, 0, result.stderr)
            best = elapsed if best is None else min(best, elapsed)
        return best

    def check(self, measure, slack: float, most: float) -> list:
        slow = []
        for name, make in growth_cases():
            small, large = measure(make(GROWTH_SIZE)), measure(make(2 * GROWTH_SIZE))
            if large > GROWTH_RATIO * small + slack or large > most:
                # A busy machine can slow one run: measure again, best of three, before calling it slow.
                small, large = measure(make(GROWTH_SIZE), 3), measure(make(2 * GROWTH_SIZE), 3)
            if large > GROWTH_RATIO * small + slack or large > most:
                slow.append((name, round(small), round(large)))
        return slow

    def test_python(self):
        rs, secrets = redact.ruleset(), redact.env_secrets(ENV)
        self.assertEqual(self.check(lambda text, runs=1: self.py_ms(text, rs, secrets, runs), PY_SLACK_MS, PY_MAX_MS), [])

    def test_jq(self):
        self.assertEqual(self.check(self.jq_ms, JQ_SLACK_MS, JQ_MAX_MS), [])

    def test_finding_d_at_5000(self):
        text = 'f(password="pw") ' + "''" * 5000 + ' "'
        rs = redact.ruleset()
        self.assertLess(self.py_ms(text, rs, []), 1000)
        self.assertLess(self.jq_ms(text), 2000)
        out = redact.sanitize_event(events(text)[0], rs, [])
        self.assertEqual(out["tool_output"], text.replace('"pw"', '"[REDACTED:secret_assignment]"'))


if __name__ == "__main__":
    unittest.main()
