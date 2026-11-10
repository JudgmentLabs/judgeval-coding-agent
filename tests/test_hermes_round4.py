"""Hermes plugin: regression tests for review round 4, one class per finding.

    Malformed    #1 a redaction setting that is not a list of strings fails closed in Hermes, as it does in the core
    Rotated      #2 /judgment on|off after a compression rotation is written under every id of the conversation
    Late         #3 an event that names a turn no longer open is dropped, never filed under the open turn
"""

from __future__ import annotations

import json
import unittest

from hermes_support import Turn, fire_all, later, new_session_id, plugin_env
from test_hermes_fixes import renamed
from test_hermes_round2 import Case
from test_hermes_round3 import gh_token


class Malformed(Case):
    BAD = ({"redactNames": "pin"}, {"redactPatterns": {"INTERNAL-[0-9]+": 1}}, {"redactCommands": "vaultctl"}, {"redactPaths": ""},
           {"redactPatterns": 5}, {"redactNames": True}, {"redactPaths": [["x"]]}, {"redactCommands": {}})

    def test_a_rule_setting_that_is_not_a_list_of_strings_drops_the_text(self):
        for number, bad in enumerate(self.BAD):
            with self.subTest(bad=bad):
                value = "pw" + gh_token()[4:16]
                project = self.sb.home / f"bad-{number}"
                (project / ".judgment").mkdir(parents=True)
                (project / ".judgment" / "tracing.json").write_text(json.dumps({"enabled": True, **bad}))
                sid = new_session_id()
                turn = Turn(sid, "pin=" + value)
                call = turn.api(tool=("terminal", {"command": "ls"}))
                turn.tool(call, "terminal", {"command": "ls"}, "pin=" + value)
                turn.api(text="ok")
                turn.end("ok")
                self.fire(turn.events, TERMINAL_CWD=str(project))
                events = [event for event in self.sb.inbox() if event.get("session_id") == sid]
                self.assertTrue(events)
                self.assertTrue(all(event.get("redaction_error") for event in events))
                self.assertNotIn(value, self.sb.state_text())

    def test_the_gate_carries_settings_as_written(self):
        from hermes_support import load_plugin

        core = load_plugin().core
        for bad in self.BAD:
            self.assertTrue(core._rules(core.settings(bad)).error, bad)
        self.assertEqual(core._rules(core.settings({})).error, "")
        gate = core.Gate("sanitized", core.settings({"redactNames": ["pin"]}), (), "/s", ())
        self.assertEqual(core.Gate.from_json(json.loads(json.dumps(gate.to_json()))), gate)


class Rotated(Case):
    def test_off_after_a_rotation_reaches_the_new_id(self):
        """Hermes rotates the id mid-turn, then the user turns tracing off: a later Hermes process resumes under the
        new id and must read off."""
        sid, rotated = new_session_id(), new_session_id()
        one = Turn(sid, "rotates")
        one.api(text="a")
        rotate_at = len(one.events)
        one.api(text="a2")
        one.end("a2")
        with plugin_env(self.sb) as (plugin, ctx):
            fire_all(ctx, renamed(one.events, sid, rotated, rotate_at), plugin)
            ctx.cli(rotated)
            self.assertIn("off for this session", ctx.commands["judgment"]["handler"]("off"))
            modes = {name: (self.sb.state / "sessions" / name).read_text().strip() for name in (sid, rotated)}
            self.assertEqual(modes, {sid: "off", rotated: "off"})
            self.assertIn("on for this session", ctx.commands["judgment"]["handler"]("on"))
        modes = {name: (self.sb.state / "sessions" / name).read_text().strip() for name in (sid, rotated)}
        self.assertEqual(modes, {sid: "on", rotated: "on"})


class Late(Case):
    def test_a_closed_turns_late_tool_result_joins_no_turn(self):
        """Turn one is off and ends with a tool still running. Its result arrives during turn two, which is traced."""
        sid = new_session_id()
        secret = "late-" + gh_token()[4:20]
        one = Turn(sid, "untraced")
        call = one.api(tool=("terminal", {"command": "sleep 60"}))
        one.tool(call, "terminal", {"command": "sleep 60"}, secret, pre=True, post=False)
        one.end("gave up")
        two = Turn(sid, "traced", first=False, start=later(one.events, 1))
        two.api(text="ok")
        late = Turn(sid, "x", first=False, start=later(two.events, 0.5))
        late.task, late.tid, late.events = one.task, one.tid, []
        late.tool(call, "terminal", {"command": "sleep 60"}, secret, pre=False, post=True)
        with plugin_env(self.sb) as (plugin, ctx):
            ctx.cli(sid)
            fire_all(ctx, one.events[:1], plugin)
            ctx.commands["judgment"]["handler"]("off")
            fire_all(ctx, one.events[1:], plugin)
            ctx.commands["judgment"]["handler"]("on")
            fire_all(ctx, two.events + late.events, plugin)
            self.assertEqual(plugin.errors["count"], 0)
            two.events = []
            two.end("ok")
            fire_all(ctx, two.events, plugin)
        inbox = self.sb.inbox()
        self.assertEqual({event["turn_id"] for event in inbox if event.get("turn_id")}, {two.tid})
        self.assertEqual([event["event"] for event in inbox if event["event"] == "tool"], [])
        self.assertNotIn(secret, self.sb.state_text())


if __name__ == "__main__":
    unittest.main()
