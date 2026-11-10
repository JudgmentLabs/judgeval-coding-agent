"""The cross-round regression matrix, generated from tests/regressions.json: every concrete case the Codex reviews of
capture: sanitized described (rounds 1-8, and commit 7458959), one row each, run

    (a) in both layers: the hook's jq and its Python mirror, which must agree, on the event the row's route builds;
    (b) for each row of a tool route (output, structured, command, input), through the real hook of every adapter
        (claude, codex, cursor, pi) into the inbox, and then through the sender's own pass (redact_event) over what the
        hook wrote.

Each run is held to the row's expected outcome (unchanged, redacted, blanked, error, or leaks for a row a documented known
limit lets through; known_gap rows only to layer agreement). A row whose limit is set names one of TABLE["limits"].
Rule tables are data: a change to one that breaks any earlier round's case fails here, by row id.

Every fake secret is generated at run time from the row's placeholders (<<name>>); the table holds none.

    python3 tests/test_regressions.py --size     print how many rows and generated cases there are
"""

from __future__ import annotations

import json
import random
import re
import shutil
import string
import sys
import unittest
from pathlib import Path

from support import Sandbox, clean_env
from test_matrix import model_call
from test_review4 import assert_parity
from test_review_fixes import tool

from judgment_core import redact

TABLE = json.loads((Path(__file__).resolve().parent / "regressions.json").read_text(encoding="utf-8"))
ROWS = TABLE["rows"]
LIMITS = TABLE["limits"]
EXPECTS = ("unchanged", "redacted", "blanked", "error", "leaks", "known_gap")
TOOL_ROUTES = ("output", "structured", "command", "input")
ADAPTERS = ("claude", "codex", "cursor", "pi")
PLACEHOLDER = re.compile(r"<<([a-z0-9]+)(?::([0-9]+))?>>")
_rng = random.Random(7007)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


# name -> (generator, public prefix: not secret, left out of the leak check)
SECRETS = {
    "opaque": (lambda: "Ck" + _rand(26), "Ck"),
    "opaque2": (lambda: "Dm" + _rand(26), "Dm"),
    "alpha": (lambda: "".join(_rng.choice(string.ascii_lowercase) for _ in range(16)), ""),
    "phrase": (lambda: " ".join("".join(_rng.choice(string.ascii_lowercase) for _ in range(_rng.randint(5, 7))) for _ in range(4)), ""),
    "pin": (lambda: str(_rng.randint(1000, 9999)), ""),
    "digits8": (lambda: str(_rng.randint(10_000_000, 99_999_999)), ""),
    "openai": (lambda: "sk-" + "proj-" + _rand(48), "sk-proj-"),
    "anthropic": (lambda: "sk-" + "ant-" + "api03-" + _rand(60), "sk-ant-api03-"),
    "github": (lambda: "gh" + "p_" + _rand(36), "ghp_"),
    "github2": (lambda: "gh" + "p_" + _rand(36), "ghp_"),
    "long": (lambda: _rand(302), ""),
    "huge": (lambda: _rand(20000), ""),
}


def filler(n: int) -> str:
    return ("hello world " * (n // 12 + 1))[:n]


class Case:
    """One row, its placeholders filled: the event per route, and what must hold of every result."""

    def __init__(self, row: dict):
        self.row = row
        self.values: dict[str, str] = {}
        self.cores: list[str] = []
        self.input = self._fill(row["input"])
        if row.get("numbers"):
            self.input = {**self.input, **{key: int(self.input[key]) for key in row["numbers"]}}
        if row.get("layers") is not None:
            value = self.input
            for name in row["layers"]:
                value = {name: json.dumps(value)}
            self.input = json.dumps(value)
        elif row.get("encode"):
            for _ in range(row["encode"]):
                self.input = json.dumps(self.input)
        if row.get("wrap"):
            self.input = row["wrap"].replace("{}", self.input)
        self.env = {key: self._fill(value) for key, value in row.get("env", {}).items()}
        default = "<<opaque>>" if row["route"] in ("command", "input") and row["expect"] in ("blanked", "leaks") else "ok"
        self.output = self._fill(row.get("output", default))

    def _fill(self, value):
        if isinstance(value, str):
            def one(match: re.Match) -> str:
                name, size = match.group(1), match.group(2)
                if name == "filler":
                    return filler(int(size))
                if name not in self.values:
                    make, prefix = SECRETS[name]
                    self.values[name] = make()
                    self.cores.append(self.values[name][len(prefix):])
                return self.values[name][: int(size)] if size else self.values[name]
            return PLACEHOLDER.sub(one, value)
        if isinstance(value, list):
            return [self._fill(item) for item in value]
        if isinstance(value, dict):
            return {self._fill(key): self._fill(item) for key, item in value.items()}
        return value

    # ------------------------------------------------------------ the routes

    def event(self) -> dict:
        route, given = self.row["route"], self.input
        if route == "output":
            return tool({"command": "x"}, given)
        if route == "structured":
            return tool({"command": "x"}, given)
        if route == "command":
            return tool({"command": given}, self.output)
        if route == "input":
            return tool(given, self.output)
        if route == "transcript":
            return model_call(given)
        if route == "label":
            return tool({"command": "x"}, "ok", **given)
        raise ValueError(route)

    def tool_input(self):
        return {"command": "x"} if self.row["route"] in ("output", "structured") else {"command": self.input} if self.row["route"] == "command" else self.input

    def tool_output(self):
        return self.input if self.row["route"] in ("output", "structured") else self.output

    def payload(self, adapter: str, cwd: str, n: int) -> tuple[str, dict]:
        session = f"rg-{n}-{adapter}"
        given, out = self.tool_input(), self.tool_output()
        if adapter == "claude":
            return "PostToolUse", {"session_id": session, "cwd": cwd, "transcript_path": "", "tool_name": "Bash", "tool_use_id": "tu-1",
                                   "tool_input": given, "tool_response": out}
        if adapter == "codex":
            return "PostToolUse", {"session_id": session, "turn_id": session + "-t", "cwd": cwd, "model": "gpt-5.5-codex", "transcript_path": "",
                                   "tool_name": "shell", "tool_use_id": "c-1", "tool_input": given, "tool_response": out}
        if adapter == "cursor":
            return "postToolUse", {"conversation_id": session, "generation_id": "g-1", "workspace_roots": [cwd], "tool_name": "Shell",
                                   "tool_use_id": "t-1", "tool_input": given, "tool_output": out}
        return "", {**tool(given, out), "session_id": session, "turn_id": session + ":1", "cwd": cwd}

    # ------------------------------------------------------------ the expectations

    def problems(self, out: dict, where: str) -> list[str]:
        """What is wrong with one result, as readable strings (empty: it holds)."""
        expect, row = self.row["expect"], self.row
        text = json.dumps(out, ensure_ascii=False)
        found = []
        if expect in ("redacted", "blanked", "error"):
            leaked = [core[:6] + "..." for core in self.cores if _leaks(core, out, row.get("strings_only", False))]
            leaked += [item for item in (self._fill(a) for a in row.get("absent", [])) if item in text]
            if leaked:
                found.append(f"{where}: leaks {leaked}")
        if expect == "unchanged":
            if out.get("sensitive") or out.get("redaction_error") or out.get("redactions"):
                found.append(f"{where}: not unchanged: sensitive={out.get('sensitive_rule')} error={out.get('redaction_error')} records={out.get('redactions')}")
            elif row["route"] in TOOL_ROUTES and (out.get("tool_input") != self.tool_input() or out.get("tool_output") != self.tool_output()):
                found.append(f"{where}: changed: {json.dumps(out.get('tool_output'))[:160]}")
        if expect == "redacted" and (out.get("sensitive") or out.get("redaction_error")):
            found.append(f"{where}: lost its text: sensitive={out.get('sensitive_rule')} error={out.get('redaction_error')}")
        if expect == "blanked":
            if not out.get("sensitive"):
                found.append(f"{where}: not blanked")
            elif row.get("rule") and out.get("sensitive_rule") != row["rule"]:
                found.append(f"{where}: blanked by {out.get('sensitive_rule')}, not {row['rule']}")
        if expect == "leaks":
            if not any(_leaks(core, out, row.get("strings_only", False)) for core in self.cores):
                found.append(f"{where}: no longer leaks: the known limit {row.get('limit')} is fixed, so the row moves to redacted")
            if out.get("sensitive") or out.get("redaction_error"):
                found.append(f"{where}: lost its text: sensitive={out.get('sensitive_rule')} error={out.get('redaction_error')}")
        if expect == "error" and not out.get("redaction_error"):
            found.append(f"{where}: no redaction_error")
        for keep in row.get("keep", []):
            if expect != "blanked" and keep not in text:
                found.append(f"{where}: lost {keep!r}")
        return found


def _leaks(core: str, out: dict, strings_only: bool = False) -> bool:
    """Whether the secret, or (when it is long) any 8-character piece of it, is in the result (its strings, or all of it)."""
    plain = "\n".join(_strings(out))
    text = plain if strings_only else json.dumps(out, ensure_ascii=False)
    if len(core) <= 16:
        return core in text or core in plain
    pieces = {core[i : i + 8] for i in range(0, len(core) - 7, 4)}
    return any(piece in text or piece in plain for piece in pieces)


def _strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _strings(item)


def cases() -> list[Case]:
    return [Case(row) for row in ROWS]


def matrix_size() -> dict:
    tool_rows = [row for row in ROWS if row["route"] in TOOL_ROUTES and row["expect"] != "known_gap" and not row.get("layers_only")]
    rounds = sorted({row["id"].split(".")[0].split("-")[0] for row in ROWS})
    return {"rows": len(ROWS), "rounds": rounds, "known_gap": sum(row["expect"] == "known_gap" for row in ROWS),
            "layer_cases": 2 * len(ROWS), "adapter_cases": 2 * len(ADAPTERS) * len(tool_rows),
            "total": 2 * len(ROWS) + 2 * len(ADAPTERS) * len(tool_rows)}


class TableTests(unittest.TestCase):
    def test_rows_are_well_formed(self):
        bad = [row.get("id") for row in ROWS if row.get("expect") not in EXPECTS or row.get("route") not in TOOL_ROUTES + ("transcript", "label")
               or not re.fullmatch(r"(r[1-9]\.[0-9]+|c[0-9a-f]{7})-[a-z0-9-]+", row.get("id", ""))
               or (row.get("limit") is not None and row["limit"] not in LIMITS) or (row["expect"] == "leaks" and not row.get("limit"))]
        self.assertEqual(bad, [])
        self.assertEqual(len({row["id"] for row in ROWS}), len(ROWS))

    def test_every_round_has_rows(self):
        self.assertEqual(matrix_size()["rounds"], ["c7458959", "r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9"])

    def test_every_limit_is_documented(self):
        """The README's Known limits list names every limit a row may cite, so a reviewer can tell a limit from a defect."""
        readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
        known = readme[readme.index("### Known limits") :]
        self.assertEqual([name for name in LIMITS if f"(`{name}`)" not in known], [])

    def test_every_limit_is_used(self):
        self.assertEqual(sorted(set(LIMITS) - {row.get("limit") for row in ROWS}), ["space-separated"])

    def test_the_table_holds_no_secret(self):
        """Values are placeholders: a realistic token in the file would be a test bug."""
        text = json.dumps(ROWS)
        self.assertEqual([needle for needle in ("ghp_", "sk-proj-", "sk-ant-", "AKIA") if needle in text], [])


class BothLayers(unittest.TestCase):
    """(a) every row, in jq and Python: they must agree, and the result must be the row's expected outcome."""

    maxDiff = None

    def test_rows(self):
        bad = []
        groups: dict[str, list[Case]] = {}
        for case in cases():
            groups.setdefault(json.dumps([case.env, case.row.get("cfg")], sort_keys=True), []).append(case)
        for key, group in groups.items():
            env, cfg = json.loads(key)
            py, _ = assert_parity(self, [case.event() for case in group], cfg=cfg, env=clean_env(**env))
            for case, out in zip(group, py):
                if case.row["expect"] != "known_gap":
                    bad += [f"{case.row['id']}: {problem}" for problem in case.problems(out, "layers")]
        self.assertEqual(bad, [])


class Adapters(unittest.TestCase):
    """(b) every tool-route row through the real hook of each adapter into the inbox, then the sender's pass over it."""

    maxDiff = None

    def test_rows(self):
        sb = Sandbox()
        self.addCleanup(sb.cleanup)
        rs = redact.ruleset()
        bad = []
        n = 0
        for case in cases():
            if case.row["route"] not in TOOL_ROUTES or case.row["expect"] == "known_gap" or case.row.get("layers_only"):
                continue
            secrets = redact.env_secrets({**sb.env, **case.env})
            for adapter in ADAPTERS:
                n += 1
                event, payload = case.payload(adapter, str(sb.project), n)
                result = sb.hook(adapter, event, payload, env=case.env)
                self.assertEqual(result.returncode, 0, result.stderr)
                inbox = [item for item in sb.inbox() if item.get("event") == "tool"]
                shutil.rmtree(sb.state / "inbox", ignore_errors=True)
                if len(inbox) != 1:
                    bad.append(f"{case.row['id']} {adapter}: {len(inbox)} tool events in the inbox")
                    continue
                hooked = inbox[0]
                bad += [f"{case.row['id']}: {p}" for p in case.problems(hooked, f"{adapter} hook")]
                sent = redact.redact_event(hooked, rs, secrets)
                bad += [f"{case.row['id']}: {p}" for p in case.problems(sent, f"{adapter} sender")]
        self.assertEqual(bad, [])


if __name__ == "__main__":
    if "--size" in sys.argv:
        print(json.dumps(matrix_size()))
    else:
        unittest.main()
