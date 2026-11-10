"""Finding 8: no input makes redaction slow. Every built-in rule does linear work in both layers.

Pathological text for each rule (its needed literal before = and before : , repeated) and the round 8 finding d string,
as tool output for a text rule and as a command for a path or command rule, at two size pairs: 2K to 8K (one window) and
64K to 256K (many). Four times the text must take at most RATIO times as long, plus a little slack for timer noise: about
4 for linear work (jq's string handling makes some cases about 6), 16 for quadratic. Each layer is timed inside one
process per size, so process start and jq's compile are not counted, and that process has a timeout, so an exponential
case fails instead of hanging. A size pair with a slow case is measured again, and each case keeps its faster time. Project regexes (redactPatterns, redactCommands, redactPaths globs) are the user's: a slow
one is a known limit.
"""

from __future__ import annotations

import json
import subprocess
import sys
import unittest

from support import CORE, clean_env

from judgment_core import redact

SIZES = ((2_000, 8_000), (64_000, 256_000))
RATIO = 8
PY_SLACK_MS, JQ_SLACK_MS = 3, 20
TIMEOUT = 60
PROGRAM = 'include "prepare"; now as $a | [judgment_prepare({enabled: true}; "on"; "/r"; "")] | (now - $a) * 1000'
BASE = {"v": 1, "harness": "pi", "session_id": "s", "turn_id": "t", "event": "tool", "tool_name": "bash", "tool_call_id": "c"}


def fill(unit: str, size: int) -> str:
    return (unit * (size // max(len(unit), 1) + 1))[:size]


def event(kind: str, text: str) -> dict:
    if kind == "output":
        return {**BASE, "tool_input": {"command": "make"}, "tool_output": text}
    return {**BASE, "tool_input": {"command": text}, "tool_output": "ok"}


def cases(size: int) -> list[tuple[str, dict]]:
    """(name, event): one per distinct text and kind."""
    done = redact.compiled()
    rules = [("output", rule) for rule in done["patterns"] + [done["names"]["assignment"], done["names"]["field"]]]
    rules += [("command", rule) for rule in done["sensitive"]["paths"] + done["sensitive"]["commands"]]
    out = {("output", 'f(password="pw") ' + "''" * (size // 2) + ' "'): "finding d: quote pairs after a value"}
    # Rounds 10 and 11: a command start that does not stop the help check, and one inside a prefix token.
    out[("command", fill("$(gh auth token ", size) + " --help")] = "command starts before --help"
    out[("command", fill("{", size) + "gh auth status")] = "braces before a command"
    for kind, rule in rules:
        need = (rule.get("needs") or ["key"])[0]
        for sep in ("=", ": "):
            out.setdefault((kind, fill(need + sep, size)), f"{rule['id']}: {need + sep!r}")
    return [(name, event(kind, text)) for (kind, text), name in out.items()]


# One Python process per size: each case through the hook's mirror and the sender's pass, best of `runs`.
PY = """
import json, sys, time
sys.path.insert(0, sys.argv[1])
from judgment_core import redact
rs, events, runs = redact.ruleset(), json.load(sys.stdin), int(sys.argv[2])
out = []
for ev in events:
    best = None
    for _ in range(runs):
        started = time.perf_counter()
        redact.sanitize_event(ev, rs, [])
        redact.redact_event(ev, rs, [])
        took = (time.perf_counter() - started) * 1000
        best = took if best is None else min(best, took)
    out.append(best)
print(json.dumps(out))
"""


def runs(size: int) -> int:
    """Best of three for a small size, where timer noise is a large part of a run; one run for a large one."""
    return 3 if size < 10_000 else 1


def py_ms(size: int) -> list[float]:
    data = json.dumps([item for _, item in cases(size)])
    result = subprocess.run([sys.executable, "-c", PY, str(CORE), str(runs(size))], input=data, capture_output=True, text=True, timeout=TIMEOUT)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


def jq_ms(size: int) -> list[float]:
    """One jq per size: each case timed inside it with now, best of `runs`."""
    events = [item for _, item in cases(size)]
    data = "\n".join(json.dumps(item) for _ in range(runs(size)) for item in events)
    result = subprocess.run(["jq", "-L", str(CORE / "bin"), "-c", PROGRAM], input=data, capture_output=True, text=True, env=clean_env(), timeout=TIMEOUT)
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    times = [float(line) for line in result.stdout.split()]
    return [min(times[i::len(events)]) for i in range(len(events))]


class LinearGrowthTests(unittest.TestCase):
    maxDiff = None

    def check(self, measure, slack: float) -> None:
        slow = []
        for small, large in SIZES:
            names = [name for name, _ in cases(small)]

            def flagged(a: list[float], b: list[float]) -> list[tuple]:
                return [(name, small, round(x, 1), round(y, 1)) for name, x, y in zip(names, a, b) if y > RATIO * x + slack]

            try:
                a, b = measure(small), measure(large)
                if flagged(a, b):
                    a, b = [list(map(min, old, new)) for old, new in ((a, measure(small)), (b, measure(large)))]
            except subprocess.TimeoutExpired:
                self.fail(f"a case at {small} or {large} characters ran over {TIMEOUT} s")
            slow += flagged(a, b)
        self.assertEqual(slow, [])

    def test_python(self):
        self.check(py_ms, PY_SLACK_MS)

    def test_jq(self):
        self.check(jq_ms, JQ_SLACK_MS)

    def test_limits_fail_closed(self):
        """Text past the limits loses its text in both layers, and never comes out raw."""
        pem = "-----BEGIN RSA PRIV" + "ATE KEY-----\nMIIabc\n-----END RSA PRIV" + "ATE KEY-----\n"
        tokens = " ".join("gh" + "p_" + format(n, "036d") for n in range(200))
        for text in (pem * 40, tokens):
            item = event("output", text)
            py = redact.sanitize_event(item, redact.ruleset(), [])
            out = subprocess.run(["jq", "-L", str(CORE / "bin"), "-c", 'include "prepare"; judgment_prepare({enabled: true}; "on"; "/r"; "")'],
                                 input=json.dumps(item), capture_output=True, text=True, env=clean_env(), timeout=30)
            js = json.loads(out.stdout)
            for result in (py, js):
                self.assertTrue(result.get("redaction_error"))
                self.assertNotIn("tool_output", result)


if __name__ == "__main__":
    unittest.main()
