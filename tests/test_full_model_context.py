"""Each llm span carries exactly what the model saw: the whole history, nothing truncated."""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# First: it sets the flusher's grace period before the flusher is imported.
from test_customer_user_id import Collector, HookCase  # noqa: E402
from judgment_core import flush  # noqa: E402
from judgment_core.claude_transcript import enrich_claude_events  # noqa: E402
from judgment_core.codex_transcript import COMPACTION_PLACEHOLDER, enrich_codex_events  # noqa: E402
from judgment_core.spans import build_request  # noqa: E402
from support import SECRETS  # noqa: E402

BIG = "x" * 200_000
GH = SECRETS["github_token"]["core"]


def llm_spans(payload: dict) -> list[dict]:
    found = []
    for span in payload["resourceSpans"][0]["scopeSpans"][0]["spans"]:
        attrs = {item["key"]: next(iter(item["value"].values())) for item in span["attributes"]}
        if attrs.get("judgment.span_kind") == "llm":
            found.append(attrs)
    return found


def io(span: dict) -> tuple[list[dict], dict]:
    return json.loads(span["judgment.input"]), json.loads(span["judgment.output"])[0]


class Shared(unittest.TestCase):
    def assert_appends(self, spans: list[dict]) -> None:
        """Each request's input is the previous input, its output, and what came after."""
        for before, after in zip(spans, spans[1:]):
            inputs, output = io(before)
            output = {key: value for key, value in output.items() if key != "finish_reason"}
            self.assertEqual(json.loads(after["judgment.input"])[: len(inputs) + 1], [*inputs, output])

    def assert_no_secrets(self, payload: dict) -> None:
        text = json.dumps(payload)
        for secret in ("gAAAA", "encrypted_content", "SIGNATURE", "signature", "[truncated]", "omitted]"):
            self.assertNotIn(secret, text)


class FallbackHistoryTest(HookCase, Shared):
    """Pi has no transcript. Earlier turns' messages are kept beside the turn and replayed whole."""

    def pi(self, session: str, turn: str, start: int, prompt: str) -> None:
        base = {"v": 1, "harness": "pi", "session_id": session, "turn_id": turn, "cwd": str(self.project)}
        records = [
            {"event": "turn_start", "prompt": prompt},
            {"event": "model", "model": "claude-sonnet-4-5", "output": "reading", "tool_calls": [{"id": f"{turn}-c", "name": "read", "arguments": {"path": "a"}}]},
            {"event": "tool", "tool_name": "read", "tool_call_id": f"{turn}-c", "tool_input": {"path": "a"}, "tool_output": BIG},
            {"event": "model", "model": "claude-sonnet-4-5", "output": f"done {turn}"},
            {"event": "turn_end", "output": f"done {turn}"},
        ]
        for offset, record in enumerate(records):
            ts = start + offset * 10
            self.hook({**base, **record, "start_ts": ts, "end_ts": ts + 5, "ts": ts + 5})

    def sent_llm(self) -> list[dict]:
        found = []
        for body in Collector.bodies:
            found.extend(llm_spans(body))
        return sorted(found, key=lambda span: len(span["judgment.input"]))

    def test_earlier_turns_replay_whole(self) -> None:
        self.pi("s1", "t1", 1_000, "first")
        self.assertEqual(self.export(), 1)
        self.pi("s1", "t2", 2_000, "second")
        self.assertEqual(self.export(), 1)
        spans = self.sent_llm()
        self.assertEqual(len(spans), 4)
        self.assert_appends(spans)
        inputs, _ = io(spans[-1])
        results = [part["response"] for message in inputs for part in message["parts"] if part["type"] == "tool_call_response"]
        self.assertEqual(results, [BIG, BIG])
        self.assertEqual(inputs[0]["parts"][0]["content"], "first")
        for body in Collector.bodies:
            self.assert_no_secrets(body)

    def test_metadata_capture_keeps_no_messages(self) -> None:
        self.optin({"enabled": True, "capture": "metadata"})
        self.pi("s1", "t1", 1_000, "first")
        self.assertEqual(self.export(), 1)
        self.assertEqual(list((self.tmp / "state").rglob("*.messages.json")), [])
        self.assertNotIn("first", json.dumps(Collector.bodies))

    def test_a_request_the_server_refuses_as_too_large_does_not_hold_back_the_rest(self) -> None:
        huge = "y" * (flush.REQUEST_BYTES + 1)
        base = {"v": 1, "harness": "pi", "session_id": "s1", "turn_id": "t1", "cwd": str(self.project)}
        for offset, record in enumerate(
            [
                {"event": "turn_start", "prompt": "go"},
                {"event": "tool", "tool_name": "read", "tool_call_id": "c", "tool_input": {}, "tool_output": huge},
                {"event": "model", "model": "claude-sonnet-4-5", "output": "ok"},
                {"event": "turn_end", "output": "ok"},
            ]
        ):
            self.hook({**base, **record, "ts": 1_000 + offset})
        Collector.statuses = [413]
        self.assertEqual(self.export(), 1)
        names = [span["name"] for body in Collector.bodies for span in body["resourceSpans"][0]["scopeSpans"][0]["spans"]]
        self.assertIn("pi turn", names)


def _row(kind: str, payload: dict, second: int) -> dict:
    return {"timestamp": f"2026-01-01T00:00:{second:02d}.000Z", "type": kind, "payload": payload}


def _item(payload: dict, second: int) -> dict:
    return _row("response_item", payload, second)


def _text(role: str, text: str) -> dict:
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _usage(total: int) -> dict:
    last = {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5, "total_tokens": 15}
    return {"type": "token_count", "info": {"total_token_usage": {"total_tokens": total}, "last_token_usage": last}}


class CodexTest(Shared):
    def rollout(self) -> Path:
        tools = {
            "type": "additional_tools",
            "role": "developer",
            "tools": [
                {
                    "type": "namespace",
                    "name": "functions",
                    "tools": [
                        {"type": "function", "name": "shell", "description": "Run a command", "strict": False, "parameters": {"type": "object"}},
                        {"type": "custom", "name": "apply_patch", "description": "Patch files", "format": {"type": "grammar"}},
                    ],
                }
            ],
        }
        record = {"usage": {"input_tokens": 10, "output_tokens": 5}}
        # Codex puts the encrypted summary first and keeps the user messages after it.
        compacted = {
            "message": "",
            "replacement_history": [
                {"type": "compaction", "encrypted_content": "gAAAAsummary"},
                _text("user", "first"),
                _text("user", "second"),
            ],
        }
        rows = [
            {"timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta", "payload": {"id": "s1", "base_instructions": {"text": "You are Codex."}}},
            _row("event_msg", {"type": "task_started", "turn_id": "t1"}, 1),
            _row("turn_context", {"turn_id": "t1", "model": "gpt-5"}, 1),
            _item(tools, 1),
            _item(_text("developer", "Follow the skills."), 1),
            _item(_text("user", "<environment_context>cwd</environment_context>"), 1),
            _item(_text("user", "first"), 1),
            _item({"type": "reasoning", "summary": [{"type": "summary_text", "text": "Look at the file."}], "encrypted_content": "gAAAAsecret"}, 2),
            _item({"type": "reasoning", "summary": [], "encrypted_content": "gAAAAsecret"}, 2),
            _item({"type": "function_call", "name": "shell", "arguments": json.dumps({"cmd": "cat a"}), "call_id": "c1"}, 2),
            _row("token_usage_record", record, 3),
            _item({"type": "function_call_output", "call_id": "c1", "output": BIG}, 3),
            _row("event_msg", _usage(15), 3),
            _item(_text("assistant", "done first"), 4),
            _row("token_usage_record", record, 4),
            _row("event_msg", _usage(30), 4),
            _row("event_msg", {"type": "task_complete", "turn_id": "t1"}, 4),
            _row("event_msg", {"type": "task_started", "turn_id": "t2"}, 5),
            _row("turn_context", {"turn_id": "t2", "model": "gpt-5"}, 5),
            _item(_text("user", "second"), 5),
            _item({"type": "function_call", "name": "shell", "arguments": json.dumps({"cmd": "ls"}), "call_id": "c2"}, 6),
            _row("token_usage_record", record, 6),
            _item({"type": "function_call_output", "call_id": "c2", "output": "a"}, 6),
            _row("event_msg", _usage(45), 6),
            # The context filled up mid-turn. The model's history restarts from the compacted copy.
            _row("token_usage_record", record, 7),
            _row("compacted", compacted, 7),
            _row("event_msg", _usage(45), 7),  # repeats the total: no request behind it
            _item(_text("assistant", "still second"), 8),
            _row("token_usage_record", record, 8),
            _row("event_msg", _usage(60), 8),
            _row("token_usage_record", record, 9),
            _row("compacted", compacted, 9),
            _row("event_msg", _usage(60), 9),
            _item(_text("assistant", "done second"), 10),
            _row("token_usage_record", record, 10),
            _row("event_msg", _usage(75), 10),
            _row("event_msg", {"type": "task_complete", "turn_id": "t2"}, 10),
        ]
        path = Path(tempfile.mkdtemp()) / "rollout.jsonl"
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        return path

    def turn(self, path: Path, turn: str, prompt: str) -> dict:
        events = [
            {"v": 1, "harness": "codex", "event": "turn_start", "session_id": "s1", "turn_id": turn, "prompt": prompt, "transcript_path": str(path), "ts": 1},
            {"v": 1, "harness": "codex", "event": "turn_end", "session_id": "s1", "turn_id": turn, "ts": 2},
        ]
        result = build_request(enrich_codex_events(events, attempts=1))
        assert result is not None
        self.assertIsNone(result.messages)  # read from the rollout, not kept beside the turn
        return result.payload

    def test_inputs_are_the_rollout_history(self) -> None:
        path = self.rollout()
        first = llm_spans(self.turn(path, "t1", "first"))
        self.assertEqual(len(first), 2)
        self.assert_appends(first)
        inputs, output = io(first[0])
        self.assertEqual([message["role"] for message in inputs], ["system", "user", "user"])
        self.assertEqual(output["parts"][0], {"type": "reasoning", "content": "Look at the file."})
        self.assertEqual(len(output["parts"]), 2)  # the empty summary adds nothing
        self.assertNotIn("finish_reason", output)
        self.assertEqual(json.loads(first[0]["gen_ai.system_instructions"]), [{"type": "text", "content": "You are Codex."}])
        self.assertEqual(
            json.loads(first[0]["gen_ai.tool.definitions"]),
            [
                {"type": "function", "name": "shell", "description": "Run a command", "inputSchema": {"type": "object"}, "strict": False},
                {"type": "provider", "id": "openai.custom", "name": "apply_patch", "args": {"description": "Patch files", "format": {"type": "grammar"}}},
            ],
        )
        self.assertEqual(io(first[1])[0][-1]["parts"][0]["response"], BIG)

        second = llm_spans(self.turn(path, "t2", "second"))
        self.assertEqual(len(second), 3)  # the repeated totals after the compactions are not requests
        before, _ = io(second[0])
        self.assertEqual(before[: len(io(first[1])[0]) + 1], [*io(first[1])[0], io(first[1])[1]])
        # Each compaction restarts the history with one placeholder where the encrypted summary was.
        for span in second[1:]:
            after, _ = io(span)
            self.assertEqual(after, [{"role": "user", "parts": [{"type": "text", "content": text}]} for text in (COMPACTION_PLACEHOLDER, "first", "second")])
        self.assert_no_secrets({"spans": first + second})


def _sha(message: dict) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class ClaudeTest(Shared):
    def transcript(self, *, logged: bool) -> Path:
        prompt = {"role": "user", "content": "add a test"}
        sent_prompt = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>env</system-reminder>"}, {"type": "text", "text": "add a test"}]}
        call = {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file": "a"}}
        reply = {"role": "assistant", "content": [{"type": "thinking", "thinking": "Read it first.", "signature": "SIGNATURE"}, call]}
        result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": BIG}]}

        def assistant(uuid: str, request: str, block: dict, stop: str, second: int) -> dict:
            return {
                "type": "assistant",
                "uuid": uuid,
                "requestId": request,
                "requestRef": request,
                "timestamp": f"2026-01-01T00:00:{second:02d}.000Z",
                "message": {"id": f"msg_{request}", "role": "assistant", "model": "claude-sonnet-4-5", "content": [block], "stop_reason": stop, "usage": {"input_tokens": 10, "output_tokens": 5}},
            }

        rows = [
            {"type": "user", "uuid": "u1", "promptId": "p1", "timestamp": "2026-01-01T00:00:01.000Z", "message": prompt},
            assistant("a1", "r1", {"type": "thinking", "thinking": "Read it first.", "signature": "SIGNATURE"}, "tool_use", 2),
            # The transcript adds the call's default arguments. The request replays what the model wrote.
            assistant("a2", "r1", {**call, "input": {"file": "a", "limit": None}}, "tool_use", 2),
            {"type": "user", "uuid": "u2", "promptId": "p1", "timestamp": "2026-01-01T00:00:03.000Z", "message": result, "toolUseResult": {}},
            assistant("a3", "r2", {"type": "thinking", "thinking": "", "signature": "SIGNATURE"}, "end_turn", 4),
            assistant("a4", "r2", {"type": "text", "text": "added"}, "end_turn", 4),
        ]
        if logged:
            shape = {"system": [{"type": "text", "text": "You are Claude Code."}], "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}]}
            rows[:0] = [
                {"type": "api-request-shape", "shapeHash": "h", "shape": shape},
                {"type": "api-request-blob", "hash": _sha(sent_prompt), "message": sent_prompt},
                {"type": "api-request", "id": "r1", "shapeHash": "h", "keep": 0, "tail": [_sha(sent_prompt)]},
                {"type": "api-request-blob", "hash": _sha(reply), "message": reply},
                {"type": "api-request-blob", "hash": _sha(result), "message": result},
                {"type": "api-request", "id": "r2", "shapeHash": "h", "base": "r1", "keep": 1, "tail": [_sha(reply), _sha(result)]},
            ]
        path = Path(tempfile.mkdtemp()) / "session.jsonl"
        path.write_text("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows))
        return path

    def spans(self, path: Path) -> list[dict]:
        base = {"v": 1, "harness": "claude-code", "session_id": "s1", "turn_id": "p1", "transcript_path": str(path)}
        events = [{**base, "event": "turn_start", "prompt": "add a test", "ts": 1_767_225_601_000}, {**base, "event": "turn_end", "output": "added", "ts": 1_767_225_604_000}]
        result = build_request(enrich_claude_events(events, attempts=1))
        assert result is not None
        self.assert_no_secrets(result.payload)
        return llm_spans(result.payload)

    def test_logged_requests_are_replayed_exactly(self) -> None:
        spans = self.spans(self.transcript(logged=True))
        self.assertEqual(len(spans), 2)
        self.assert_appends(spans)
        inputs, output = io(spans[0])
        self.assertEqual(inputs[0]["parts"][0]["content"], "<system-reminder>env</system-reminder>")
        self.assertEqual(output["finish_reason"], "tool_use")
        self.assertEqual(output["parts"][0], {"type": "reasoning", "content": "Read it first."})
        self.assertEqual(output["parts"][1]["arguments"], {"file": "a"})
        self.assertEqual(json.loads(spans[0]["gen_ai.system_instructions"]), [{"type": "text", "content": "You are Claude Code."}])
        self.assertEqual(json.loads(spans[0]["gen_ai.tool.definitions"]), [{"type": "function", "name": "Read", "description": "Read a file", "inputSchema": {"type": "object"}}])
        inputs, output = io(spans[1])
        self.assertEqual(inputs[-1]["parts"][0]["response"], BIG)
        self.assertEqual(output, {"role": "assistant", "parts": [{"type": "text", "content": "added"}], "finish_reason": "end_turn"})

    def test_without_logged_requests_the_transcript_is_replayed(self) -> None:
        spans = self.spans(self.transcript(logged=False))
        self.assertEqual(len(spans), 2)
        self.assert_appends(spans)
        self.assertNotIn("gen_ai.system_instructions", spans[0])
        self.assertEqual(io(spans[1])[0][-1]["parts"][0]["response"], BIG)


if __name__ == "__main__":
    unittest.main()


class RedactionTest(HookCase):
    """The system prompt, the tool definitions, and the whole history are redacted like any other text."""

    def rollout(self) -> Path:
        tools = {"type": "additional_tools", "role": "developer",
                 "tools": [{"type": "function", "name": "shell", "description": "Run a command. Token " + GH, "parameters": {"type": "object"}}]}
        rows = [
            {"timestamp": "2026-01-01T00:00:00.000Z", "type": "session_meta", "payload": {"id": "s1", "base_instructions": {"text": "You are Codex. " + GH}}},
            _row("event_msg", {"type": "task_started", "turn_id": "t1"}, 1),
            _row("turn_context", {"turn_id": "t1", "model": "gpt-5"}, 1),
            _item(tools, 1),
            _item(_text("user", "first " + GH), 1),
            _item({"type": "function_call", "name": "shell", "arguments": json.dumps({"cmd": "cat a"}), "call_id": "c1"}, 2),
            _row("token_usage_record", {"usage": {"input_tokens": 10, "output_tokens": 5}}, 3),
            _item({"type": "function_call_output", "call_id": "c1", "output": "token " + GH}, 3),
            _row("event_msg", _usage(15), 3),
            _item(_text("assistant", "done"), 4),
            _row("token_usage_record", {"usage": {"input_tokens": 10, "output_tokens": 5}}, 4),
            _row("event_msg", _usage(30), 4),
            _row("event_msg", {"type": "task_complete", "turn_id": "t1"}, 4),
        ]
        path = self.tmp / "rollout.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return path

    def codex(self, path: Path) -> None:
        base = {"v": 1, "harness": "codex", "session_id": "s1", "turn_id": "t1", "cwd": str(self.project), "transcript_path": str(path)}
        self.hook({**base, "event": "turn_start", "prompt": "first", "ts": 1_000})
        self.hook({**base, "event": "turn_end", "ts": 5_000})

    def llm(self) -> list[dict]:
        return [span for body in Collector.bodies for span in llm_spans(body)]

    def test_sanitized_redacts_the_new_content(self) -> None:
        self.codex(self.rollout())
        self.assertEqual(self.export(), 1)
        self.assertNotIn(GH, json.dumps(Collector.bodies))
        spans = self.llm()
        self.assertTrue(spans)
        for key in ("gen_ai.system_instructions", "gen_ai.tool.definitions", "judgment.input"):
            self.assertIn("[REDACTED:github_token]", spans[-1][key], key)

    def test_metadata_sends_none_of_it(self) -> None:
        self.optin({"enabled": True, "capture": "metadata"})
        self.codex(self.rollout())
        self.assertEqual(self.export(), 1)
        sent = json.dumps(Collector.bodies)
        for text in (GH, "You are Codex", "Run a command", "first"):
            self.assertNotIn(text, sent)
        for span in self.llm():
            for key in ("gen_ai.system_instructions", "gen_ai.tool.definitions"):
                self.assertNotIn(key, span)

    def test_a_blanked_turn_sends_none_of_it(self) -> None:
        path = self.rollout()
        base = {"v": 1, "harness": "codex", "session_id": "s1", "turn_id": "t1", "cwd": str(self.project), "transcript_path": str(path)}
        self.hook({**base, "event": "turn_start", "prompt": "first", "ts": 1_000})
        self.hook({**base, "event": "tool", "tool_name": "Read", "tool_call_id": "r", "tool_input": {"file_path": ".env"}, "tool_output": "A=1", "ts": 2_000})
        self.hook({**base, "event": "turn_end", "ts": 5_000})
        self.assertEqual(self.export(), 1)
        spans = self.llm()
        self.assertIn("gen_ai.system_instructions", spans[-1])
        for span in spans:
            for key in ("gen_ai.system_instructions", "gen_ai.tool.definitions", "judgment.input", "judgment.output"):
                self.assertEqual(span.get(key, ""), "", key)

    def test_history_kept_beside_a_turn_is_redacted_before_it_is_stored_and_replayed(self) -> None:
        base = {"v": 1, "harness": "pi", "session_id": "s1", "cwd": str(self.project)}
        for turn, start, capture in (("t1", 1_000, "full"), ("t2", 2_000, "sanitized"), ("t3", 3_000, "sanitized")):
            self.optin({"enabled": True, "capture": capture})
            for offset, record in enumerate([
                {"event": "turn_start", "prompt": f"{turn} {GH}"},
                {"event": "model", "model": "claude-sonnet-4-5", "output": f"{turn} done"},
                {"event": "turn_end", "output": f"{turn} done"},
            ]):
                self.hook({**base, **record, "turn_id": turn, "ts": start + offset * 10})
            self.assertEqual(self.export(), 1)
            if turn == "t2":
                sidecars = list((self.tmp / "state").rglob("*.messages.json"))
                self.assertEqual(len(sidecars), 2)
                stored = [path.read_text() for path in sidecars]
                self.assertEqual(sum(GH in text for text in stored), 1)  # t1's, captured in full; t2's was redacted
        last = json.dumps(Collector.bodies[-1])
        self.assertNotIn(GH, last)  # t1 was captured in full; replayed in a sanitized turn, it is redacted
        self.assertIn('\\"field\\":\\"history\\"', last)
