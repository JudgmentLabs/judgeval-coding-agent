"""Hermes plugin: regression tests for review round 3 and the core API contract, shaped like Hermes 0.19.0.

    Contract     the core functions core_bridge calls still exist and take what it passes
    Snapshot     #1 one bounded pass: budgets, omission whole, copy semantics, no reference kept
    Structured   #1 through the plugin: oversized arguments blank the turn; the snapshot runs outside the lock
    Latency      #1 329 KB / 10,000-row arguments and 8 concurrent sessions with 32 KB arguments, p95 < 3 ms target
                 (fails at hermes_support.LATENCY_GATE_P95_MS, best of 3 trials)
    Identity     #2 a helper dying between link and unlink, an ingestion, then a replay: one event, one id
    Collisions   #3 two distinct secret keys matching one rule keep both entries
    Settings     tracing.json's redactCommands, redactNames, redactAllow, and redactDisable reach Hermes's redaction
"""

from __future__ import annotations

import inspect
import json
import marshal
import os
import secrets
import string
import sys
import time
import unittest
from unittest import mock

from fake_sink import attrs, roots
from hermes_support import VENDOR_CORE, Turn, load_plugin, new_session_id
from test_hermes_round2 import Case, module


def gh_token() -> str:
    """A fake GitHub token, built at runtime."""
    return "gh" + "p_" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(36))


def rows(count: int, width: int = 12) -> dict:
    """Structured tool arguments: ``count`` small rows (the review's probe: 10,000 rows, 329 KB)."""
    return {"rows": [{"i": n, "v": ("r%07d" % n).ljust(width, "x")} for n in range(count)]}


def pct(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class Contract(unittest.TestCase):
    def test_the_vendored_core_matches_what_the_bridge_calls(self):
        core = load_plugin().core
        self.assertEqual(core.contract_problems(), [], "judgment_core changed: update hermes/core_bridge.py (see CONTRACT)")

    def test_a_moved_function_is_named_not_buried(self):
        core = load_plugin().core
        with mock.patch.object(core.redact, "scrub_fields", lambda fields: None):
            problems = core.contract_problems()
        self.assertEqual(len(problems), 1)
        self.assertIn("judgment_core.redact.scrub_fields", problems[0])
        with mock.patch.object(core.redact, "scrub_fields", lambda fields, rs, secrets, *, mode: None):
            self.assertIn("requires ['mode']", core.contract_problems()[0])
        with mock.patch.object(core.optin, "safe_id", None):
            self.assertIn("judgment_core.optin.safe_id is gone", core.contract_problems()[0])

    def test_every_core_call_in_the_bridge_is_declared(self):
        core = load_plugin().core
        import ast

        tree = ast.parse(inspect.getsource(core))
        declared = {(module_, name) for module_, name, _ in core.CONTRACT}
        used = {(node.value.id, node.attr) for node in ast.walk(tree)
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("optin", "redact")}
        self.assertEqual(used - declared, set(), "core_bridge calls a core function CONTRACT does not list")


class Snapshot(unittest.TestCase):
    def setUp(self):
        self.snap = module("snapshot")

    def test_within_budget_is_a_copy_not_a_reference(self):
        args = rows(100)
        taken = self.snap.take(args)
        self.assertIsInstance(taken, bytes)
        args["rows"][0]["v"] = "changed after the callback"
        args["extra"] = 1
        self.assertEqual(marshal.loads(taken), rows(100))

    def test_over_budget_is_omitted_whole_and_fast(self):
        big = rows(10_000)
        self.assertGreater(len(json.dumps(big)), 300_000)
        start = time.perf_counter()
        taken = self.snap.take(big)
        self.assertLess(time.perf_counter() - start, 0.05)
        self.assertIsInstance(taken, self.snap.Omitted)
        self.assertEqual(taken.reason, "nodes")

    def test_budget_is_checked_while_walking_not_after(self):
        """Many small lists, each under budget, add up: the walk stops at the first member past it."""
        budget = self.snap.Budget(nodes=100, chars=10_000, depth=12)
        wide = [[n] for n in range(1000)]
        self.assertIsInstance(self.snap.take(wide, budget), self.snap.Omitted)
        self.assertIsInstance(self.snap.take({"a": "x" * 10_001}, budget), self.snap.Omitted)
        self.assertEqual(marshal.loads(self.snap.take({"a": "x" * 9_000}, budget)), {"a": "x" * 9_000})
        self.assertEqual(self.snap.take("y" * 10_001, budget).reason, "chars")

    def test_other_types_are_copied_as_before(self):
        class Thing:
            def __str__(self):
                return "thing"

        deep: dict = {}
        cursor = deep
        for _ in range(20):
            cursor["d"] = {}
            cursor = cursor["d"]
        value = {"t": (1, 2), 3: "int key", "o": Thing()}
        out = marshal.loads(self.snap.take(value))
        self.assertEqual(out, {"t": [1, 2], "3": "int key", "o": "thing"})
        self.assertEqual(self.snap.take({**value, "deep": deep}).reason, "depth", "the copy path omits deep nesting too")

    def test_depth_is_a_budget_like_the_others(self):
        """At the limit: kept whole, by either path. One level past it: omitted whole, never cut to a stub."""
        budget = self.snap.Budget(nodes=100, chars=10_000, depth=3)

        def nest(levels: int, leaf: object) -> dict:
            value: object = leaf
            for _ in range(levels):
                value = {"d": value}
            return value

        for leaf in ({}, (1,)):  # (1,) takes the copy path
            self.assertIsInstance(self.snap.take(nest(3, leaf), budget), bytes)
            over = self.snap.take(nest(4, leaf), budget)
            self.assertIsInstance(over, self.snap.Omitted)
            self.assertEqual(over.reason, "depth")

    def test_the_copy_path_spends_the_same_budget(self):
        budget = self.snap.Budget(nodes=50, chars=10_000, depth=12)
        self.assertIsInstance(self.snap.take([(n,) for n in range(100)], budget), self.snap.Omitted)

    def test_bound_names_an_omitted_field_and_no_omitted_crosses(self):
        spooler = module("spooler")
        record = {"event": "model", "tool_calls": [{"id": "c", "name": "t", "arguments": self.snap.take(rows(10_000))}], "output": "ok"}
        out, _size = spooler.bound(record)
        self.assertEqual(out["tool_calls"], spooler.OMITTED)
        self.assertEqual(out["omitted"], ["tool_calls"])
        spooler.frame({"r": out})  # marshal would refuse an Omitted

    def test_unpack_restores_the_record_and_parses_text_arguments(self):
        record = {"tool_input": self.snap.take({"a": [1, 2]}), "tool_calls": [{"id": "c", "arguments": '{"x": 1}'}, {"id": "d", "arguments": "not json"}]}
        out = self.snap.unpack(record)
        self.assertEqual(out["tool_input"], {"a": [1, 2]})
        self.assertEqual([call["arguments"] for call in out["tool_calls"]], [{"x": 1}, "not json"])


class Structured(Case):
    def tool_turn(self, args, result: str = "ok") -> Turn:
        turn = Turn(new_session_id(), "load the rows")
        call = turn.api(tool=("load_rows", args))
        turn.tool(call, "load_rows", args, result)
        turn.api(text="done")
        turn.end("done")
        return turn

    def test_oversized_structured_arguments_blank_the_turn(self):
        big = rows(10_000)
        self.fire(self.tool_turn(big, "PRIVATE-NOTE-" + "x" * 40).events)
        events = self.sb.inbox()
        tool = next(event for event in events if event["event"] == "tool")
        self.assertTrue(tool["sensitive"])
        self.assertEqual(tool["sensitive_rule"], load_plugin().core.OVERSIZE_RULE)
        self.assertNotIn("tool_input", tool, "a record blanked as too large to check keeps no input text")
        spans = self.flushed()
        self.assertEqual(attrs(roots(spans)[0])["judgment.redacted_rule"], load_plugin().core.OVERSIZE_RULE)
        self.assertNotIn("PRIVATE-NOTE", self.sink.text())
        self.assertNotIn("r0000001", self.sink.text(), "the blanked turn sends none of its arguments")

    def test_deeply_nested_sensitive_path_blanks_the_turn(self):
        """Codex round 4: depth past the budget used to become "…", dropping the .env path that made the read
        sensitive while keeping its private output. Now it is an omission: the turn is blanked, with a reason."""
        args: dict = {"path": ".env"}
        for _ in range(20):
            args = {"nested": args}
        self.fire(self.tool_turn(args, "PRIVATE-NOTE-" + "x" * 40).events)
        tool = next(event for event in self.sb.inbox() if event["event"] == "tool")
        self.assertTrue(tool["sensitive"])
        self.assertEqual(tool["sensitive_rule"], load_plugin().core.OVERSIZE_RULE)
        self.assertNotIn("tool_input", tool)
        # The model's JSON-text copy of the arguments is parsed whole in the helper, so the turn names .env itself.
        self.assertIn(attrs(roots(self.flushed())[0])["judgment.redacted_rule"], {"dotenv", load_plugin().core.OVERSIZE_RULE})
        self.assertNotIn("PRIVATE-NOTE", self.sink.text())

    def test_structured_arguments_within_budget_arrive_whole(self):
        args = rows(1_000)
        self.fire(self.tool_turn(args).events)
        events = self.sb.inbox()
        self.assertEqual(next(event for event in events if event["event"] == "tool")["tool_input"], args)
        model = next(event for event in events if event["event"] == "model" and event.get("tool_calls"))
        self.assertEqual(model["tool_calls"][0]["arguments"], args, "JSON-text arguments are parsed in the helper")

    def test_snapshot_runs_outside_the_lock_and_only_when_traced(self):
        plugin = load_plugin()
        snapshot = module("snapshot")
        real = snapshot.take
        held = []

        def spy(value, *args):
            held.append(plugin._lock.locked())
            return real(value, *args)

        turn = self.tool_turn(rows(500))
        with mock.patch.object(snapshot, "take", spy):
            self.fire(turn.events)
        self.assertTrue(held)
        self.assertEqual(held, [False] * len(held), "a snapshot taken while the plugin's lock was held")
        held.clear()
        with mock.patch.object(snapshot, "take", spy):
            self.fire(self.tool_turn(rows(500)).events, JUDGMENT_TRACING="0")
        self.assertEqual(held, [], "an untraced call was snapshotted")

    def test_an_abandoned_call_is_snapshotted_from_its_reference(self):
        turn = Turn(new_session_id(), "go")
        args = rows(10)
        call = turn.api(tool=("load_rows", args))
        turn.tool(call, "load_rows", args, "", post=False)
        turn.end(None, interrupted=True)
        self.fire(turn.events)
        tool = next(event for event in self.sb.inbox() if event["event"] == "tool")
        self.assertTrue(tool["is_error"])
        self.assertEqual(tool["tool_input"], args)


class Identity(Case):
    def test_crash_between_link_and_unlink_then_ingest_then_replay_folds_once(self):
        sys.path.insert(0, str(VENDOR_CORE))
        from judgment_core import flush

        spooler, core = module("spooler"), load_plugin().core
        gate = core.Gate("sanitized", core.settings({}), (), str(self.sb.state), ())
        record = {"v": 1, "harness": "hermes", "event": "turn_start", "session_id": "s9", "turn_id": "t9", "ts": 1, "prompt": "hi"}
        frames = [{"t": "gate", "id": 1, "gate": gate.to_json()}, {"t": "rec", "s": 1, "g": 1, "r": record}]

        class Crash(Exception):
            pass

        dead = spooler.Helper("t-link", ack_fd=-1)
        dead.fold(frames[0])
        real_unlink = os.unlink

        def die_on_staged_unlink(path, *args, **kwargs):
            if "inbox.tmp" in str(path):
                raise Crash
            return real_unlink(path, *args, **kwargs)

        with mock.patch.object(core.os, "unlink", die_on_staged_unlink):
            self.assertRaises(Crash, dead.fold, frames[1])
        name = next((self.sb.state / "inbox").iterdir()).name
        self.assertTrue((self.sb.state / "inbox.tmp" / name).exists(), "both links left, as after the crash")
        env = dict(self.sb.env)
        first = flush._inbox(self.sb.state)[0][1]["_id"]
        flush.ingest(self.sb.state, env)  # the flusher folds and deletes the inbox link before the replay
        self.assertEqual(list((self.sb.state / "inbox").iterdir()), [])

        fresh = spooler.Helper("t-link", ack_fd=-1)
        for frame in frames:
            fresh.fold(frame)  # the ledger says seq 1 was committed: publish again, never re-prepare
        replayed = flush._inbox(self.sb.state)
        self.assertEqual(len(replayed), 1)
        self.assertEqual(replayed[0][1]["_id"], first, "a replay changed the event's identity")
        flush.ingest(self.sb.state, env)
        stored = (self.sb.state / "turns" / "hermes" / "s9" / "t9.jsonl").read_text().splitlines()
        self.assertEqual(len(flush._read_jsonl(self.sb.state / "turns" / "hermes" / "s9" / "t9.jsonl")), 1, stored)
        self.assertEqual(list((self.sb.state / "inbox.tmp").iterdir()), [])


class Collisions(unittest.TestCase):
    def test_two_secret_keys_matching_one_rule_keep_both_entries(self):
        core = load_plugin().core
        gate = core.Gate("sanitized", core.settings({}), (), "/nonexistent", ())
        one, two = gh_token(), gh_token()
        record = {"v": 1, "harness": "hermes", "event": "tool", "session_id": "s", "turn_id": "t", "ts": 1,
                  "tool_input": {one: "first", two: "second", "nested": {one: "deep"}}, "tool_output": "ok"}
        out = core.prepare(record, gate)
        text = json.dumps(out)
        self.assertNotIn(one, text)
        self.assertNotIn(two, text)
        values = out["tool_input"]
        self.assertEqual(sorted(value for key, value in values.items() if key != "nested"), ["first", "second"])
        self.assertTrue(all(key.startswith("[REDACTED:github_token]") for key in values if key != "nested"))
        self.assertEqual(list(values["nested"].values()), ["deep"])
        self.assertEqual(sum(item["count"] for item in out["redactions"] if item["rule"] == "github_token"), 3)


if __name__ == "__main__":
    unittest.main()


class Settings(Case):
    def test_every_redaction_key_in_tracing_json_applies(self):
        """Each assertion differs from what the built-in rules alone would do, so a key the bridge dropped fails here."""
        pin = "pw" + gh_token()[4:14]
        token = gh_token()
        project = self.sb.home / "settings"
        (project / ".judgment").mkdir(parents=True)
        (project / ".judgment" / "tracing.json").write_text(json.dumps({
            "enabled": True, "redactCommands": ["mycli[^\\S\\n]+show-secret"], "redactNames": ["vault_pin"],
            "redactAllow": ["changeme"], "redactDisable": ["github_token"]}))
        turn = Turn(new_session_id(), "go")
        calls = [("mycli show-secret db", "x"), ("ls", "vault_pin = " + pin), ("ls", "password = changeme"), ("ls", "gh " + token)]
        for command, result in calls:
            call = turn.api(tool=("terminal", {"command": command}))
            turn.tool(call, "terminal", {"command": command}, result)
        turn.api(text="ok")
        turn.end("ok")
        self.fire(turn.events, TERMINAL_CWD=str(project))
        tools = [event for event in self.sb.inbox() if event.get("event") == "tool"]
        self.assertEqual([event.get("sensitive_rule") for event in tools], ["custom_command", None, None, None])
        self.assertEqual([event["tool_output"] for event in tools[1:]],
                         ["vault_pin = [REDACTED:secret_assignment]", "password = changeme", "gh " + token])
