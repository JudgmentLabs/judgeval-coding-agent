"""Each llm span carries exactly what the model saw: the whole history, nothing truncated."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "core"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

# First: it sets the flusher's grace period before the flusher is imported.
from test_customer_user_id import Collector, HookCase  # noqa: E402
from judgment_core import claude_transcript, codex_transcript, flush, redact  # noqa: E402
from judgment_core.claude_transcript import enrich_claude_events  # noqa: E402
from judgment_core.codex_transcript import COMPACTION_PLACEHOLDER, enrich_codex_events, read_thread  # noqa: E402
from judgment_core.rows import records  # noqa: E402
from judgment_core.spans import build_request  # noqa: E402
from support import SECRETS  # noqa: E402

BIG = "x" * 200_000
GH = SECRETS["github_token"]["core"]
# 2026-01-01T00:00:00Z, when the transcripts below start.
T0 = 1_767_225_600_000
ENV = "API_KEY=hunter2"


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
    def tmpdir(self) -> Path:
        path = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, path, ignore_errors=True)
        return path

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

    def names(self) -> list[list[str]]:
        """The names of the spans in each request the server took."""
        return [[span["name"] for span in body["resourceSpans"][0]["scopeSpans"][0]["spans"]] for body in Collector.bodies]

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
        self.assertIn("pi turn", [name for names in self.names() for name in names])

    def test_a_request_refused_as_too_large_is_split_and_every_span_still_sent(self) -> None:
        self.pi("s1", "t1", 1_000, "first")
        Collector.statuses = [413]  # the turn fits one request: the server refuses it once
        self.assertEqual(self.export(), 1)
        sent = self.names()
        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[-1][-1], "pi turn")  # the root still goes last
        self.assertEqual(len(llm_spans(Collector.bodies[0])) + len(llm_spans(Collector.bodies[1])), 2)

    def test_a_turn_the_server_refuses_span_by_span_is_dropped_not_retried(self) -> None:
        self.pi("s1", "t1", 1_000, "first")
        Collector.statuses = [413] * 50
        self.assertEqual(self.export(), 0)
        self.assertEqual(Collector.bodies, [])
        self.assertIn("drop a turn: post status 413 for every span", (self.tmp / "state" / "flush.log").read_text())
        Collector.statuses = []
        self.assertEqual(self.export(), 0)
        self.assertEqual(Collector.bodies, [])

    def test_a_long_history_keeps_the_newest_turns_that_fit(self) -> None:
        with mock.patch.object(flush, "HISTORY_BYTES", 3 * len(BIG) // 2):  # one earlier turn (BIG once) fits, two do not
            for number, prompt in enumerate(("first", "second", "third"), 1):
                self.pi("s1", f"t{number}", number * 1_000, prompt)
                self.assertEqual(self.export(), 1)
        inputs, _ = io(self.sent_llm()[-1])
        self.assertEqual(inputs[0], {"role": "system", "parts": [{"type": "text", "content": "[1 earlier turns omitted]"}]})
        self.assertEqual(inputs[1]["parts"][0]["content"], "second")
        self.assertNotIn("first", [part.get("content") for message in inputs for part in message["parts"]])

    def test_a_turn_without_a_model_call_keeps_its_tools_and_answer(self) -> None:
        base = {"v": 1, "harness": "pi", "session_id": "s1", "turn_id": "t1", "cwd": str(self.project)}
        for offset, record in enumerate([
            {"event": "turn_start", "prompt": "first"},
            {"event": "tool", "tool_name": "read", "tool_call_id": "c", "tool_input": {"path": "a"}, "tool_output": "contents"},
            {"event": "turn_end", "output": "the answer"},
        ]):
            self.hook({**base, **record, "ts": 1_000 + offset * 10})
        self.assertEqual(self.export(), 1)
        self.pi("s1", "t2", 2_000, "second")
        self.assertEqual(self.export(), 1)
        inputs, _ = io(self.sent_llm()[0])
        self.assertEqual([message["role"] for message in inputs], ["user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(inputs[2]["parts"][0]["response"], "contents")
        self.assertEqual(inputs[3], {"role": "assistant", "parts": [{"type": "text", "content": "the answer"}]})


def _at(second: int) -> str:
    return f"2026-01-01T00:00:{second:02d}.000Z"


def _jsonl(path: Path, rows: list[dict], *, spaced: bool = False) -> Path:
    """The harnesses write compact JSON. Spaced JSON reads the same."""
    separators = (", ", ": ") if spaced else (",", ":")
    path.write_text("".join(json.dumps(row, separators=separators) + "\n" for row in rows))
    return path


def _row(kind: str, payload: dict, second: int) -> dict:
    return {"timestamp": _at(second), "type": kind, "payload": payload}


def _item(payload: dict, second: int) -> dict:
    return _row("response_item", payload, second)


def _text(role: str, text: str) -> dict:
    kind = "output_text" if role == "assistant" else "input_text"
    return {"type": "message", "role": role, "content": [{"type": kind, "text": text}]}


def _usage(total: int) -> dict:
    last = {"input_tokens": 10, "cached_input_tokens": 0, "output_tokens": 5, "total_tokens": 15}
    return {"type": "token_count", "info": {"total_token_usage": {"total_tokens": total}, "last_token_usage": last}}


RECORD = {"usage": {"input_tokens": 10, "output_tokens": 5}}
META = {"timestamp": _at(0), "type": "session_meta", "payload": {"id": "s1", "base_instructions": {"text": "You are Codex."}}}


def _opens(turn: str, second: int) -> list[dict]:
    return [_row("event_msg", {"type": "task_started", "turn_id": turn}, second), _row("turn_context", {"turn_id": turn, "model": "gpt-5"}, second)]


def _call(call_id: str, cmd: str, second: int) -> dict:
    return _item({"type": "function_call", "name": "shell", "arguments": json.dumps({"cmd": cmd}), "call_id": call_id}, second)


def _call_output(call_id: str, output: str, second: int) -> dict:
    return _item({"type": "function_call_output", "call_id": call_id, "output": output}, second)


def _closes(turn: str, second: int) -> dict:
    return _row("event_msg", {"type": "task_complete", "turn_id": turn}, second)


def _prompt(uuid: str, prompt_id: str, text: str, second: int) -> dict:
    return {"type": "user", "uuid": uuid, "promptId": prompt_id, "timestamp": _at(second), "message": {"role": "user", "content": text}}


def _result(uuid: str, prompt_id: str, call_id: str, content: str, second: int) -> dict:
    message = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call_id, "content": content}]}
    return {"type": "user", "uuid": uuid, "promptId": prompt_id, "timestamp": _at(second), "message": message, "toolUseResult": {}}


def _assistant(uuid: str, request: str, block: dict, stop: str, second: int) -> dict:
    return {
        "type": "assistant",
        "uuid": uuid,
        "requestId": request,
        "requestRef": request,
        "timestamp": _at(second),
        "message": {"id": f"msg_{request}", "role": "assistant", "model": "claude-sonnet-4-5", "content": [block], "stop_reason": stop, "usage": {"input_tokens": 10, "output_tokens": 5}},
    }


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
        return _jsonl(self.tmpdir() / "rollout.jsonl", rows)

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

    def test_without_usage_records_a_request_keeps_its_reasoning(self) -> None:
        """Only token_count rows: the tool output ends the request's output before its usage arrives."""
        rows = [
            META,
            *_opens("t1", 1),
            _item(_text("user", "go"), 1),
            _item({"type": "reasoning", "summary": [{"type": "summary_text", "text": "Look at the file."}], "encrypted_content": "gAAAAsecret"}, 2),
            _call("c1", "cat a", 2),
            _call_output("c1", "a", 3),
            _row("event_msg", _usage(15), 3),
            _item(_text("assistant", "done"), 4),
            _row("event_msg", _usage(30), 4),
            _closes("t1", 4),
        ]
        spans = llm_spans(self.turn(_jsonl(self.tmpdir() / "rollout.jsonl", rows), "t1", "go"))
        self.assertEqual(len(spans), 2)
        self.assert_appends(spans)
        _, output = io(spans[0])
        self.assertEqual(output["parts"][0], {"type": "reasoning", "content": "Look at the file."})
        self.assertEqual(output["parts"][1]["type"], "tool_call")
        self.assert_no_secrets({"spans": spans})

    def test_a_compaction_request_is_no_span(self) -> None:
        """Only token_usage_record rows: the compaction's request has one too, and adds nothing to the history."""
        compacted = {"message": "", "replacement_history": [{"type": "compaction", "encrypted_content": "gAAAAsummary"}, _text("user", "go")]}
        rows = [
            META,
            *_opens("t1", 1),
            _item(_text("user", "go"), 1),
            _call("c1", "cat a", 2),
            _row("token_usage_record", RECORD, 2),
            _call_output("c1", "a", 3),
            _row("token_usage_record", RECORD, 4),
            _row("compacted", compacted, 4),
            _item(_text("assistant", "done"), 5),
            _row("token_usage_record", RECORD, 5),
            _closes("t1", 5),
        ]
        spans = llm_spans(self.turn(_jsonl(self.tmpdir() / "rollout.jsonl", rows), "t1", "go"))
        self.assertEqual(len(spans), 2)
        self.assertEqual(io(spans[1])[0], [{"role": "user", "parts": [{"type": "text", "content": text}]} for text in (COMPACTION_PLACEHOLDER, "go")])
        self.assertEqual(io(spans[1])[1]["parts"], [{"type": "text", "content": "done"}])


def _sha(message: dict) -> str:
    return hashlib.sha256(json.dumps(message, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class ClaudeTest(Shared):
    def transcript(self, *, logged: bool, answered: bool = True) -> Path:
        """One turn: a Read, then the answer. Without ``answered``, the answer's text row is not written yet."""
        prompt = {"role": "user", "content": "add a test"}
        sent_prompt = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>env</system-reminder>"}, {"type": "text", "text": "add a test"}]}
        call = {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file": "a"}}
        reply = {"role": "assistant", "content": [{"type": "thinking", "thinking": "Read it first.", "signature": "SIGNATURE"}, call]}
        result = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": BIG}]}
        rows = [
            {"type": "user", "uuid": "u1", "promptId": "p1", "timestamp": _at(1), "message": prompt},
            _assistant("a1", "r1", {"type": "thinking", "thinking": "Read it first.", "signature": "SIGNATURE"}, "tool_use", 2),
            # The transcript adds the call's default arguments. The request replays what the model wrote.
            _assistant("a2", "r1", {**call, "input": {"file": "a", "limit": None}}, "tool_use", 2),
            {"type": "user", "uuid": "u2", "promptId": "p1", "timestamp": _at(3), "message": result, "toolUseResult": {}},
            _assistant("a3", "r2", {"type": "thinking", "thinking": "", "signature": "SIGNATURE"}, "end_turn", 4),
        ]
        if answered:
            rows.append(_assistant("a4", "r2", {"type": "text", "text": "added"}, "end_turn", 4))
        if logged:
            # The system prompt as one string: Claude Code sends it as a string or as text blocks.
            shape = {"system": "You are Claude Code.", "tools": [{"name": "Read", "description": "Read a file", "input_schema": {"type": "object"}}]}
            rows[:0] = [
                {"type": "api-request-shape", "shapeHash": "h", "shape": shape},
                {"type": "api-request-blob", "hash": _sha(sent_prompt), "message": sent_prompt},
                {"type": "api-request", "id": "r1", "shapeHash": "h", "keep": 0, "tail": [_sha(sent_prompt)]},
                {"type": "api-request-blob", "hash": _sha(reply), "message": reply},
                {"type": "api-request-blob", "hash": _sha(result), "message": result},
                {"type": "api-request", "id": "r2", "shapeHash": "h", "base": "r1", "keep": 1, "tail": [_sha(reply), _sha(result)]},
            ]
        return _jsonl(self.tmpdir() / "session.jsonl", rows)

    def events(self, path: Path) -> list[dict]:
        base = {"v": 1, "harness": "claude-code", "session_id": "s1", "turn_id": "p1", "transcript_path": str(path)}
        return enrich_claude_events([{**base, "event": "turn_start", "prompt": "add a test", "ts": T0 + 1_000}, {**base, "event": "turn_end", "output": "added", "ts": T0 + 4_000}], attempts=1)

    def spans(self, path: Path, events: list[dict] | None = None) -> list[dict]:
        result = build_request(events or self.events(path))
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

    def test_the_answer_the_turn_reported_completes_a_request_the_transcript_cut_short(self) -> None:
        spans = self.spans(self.transcript(logged=True, answered=False))
        self.assertEqual(io(spans[1])[1], {"role": "assistant", "parts": [{"type": "text", "content": "added"}], "finish_reason": "end_turn"})

    def test_a_request_that_lost_its_input_is_rebuilt_alone(self) -> None:
        """Say the first request's input hit a redaction limit: only it is rebuilt from the hook's events."""
        path = self.transcript(logged=True)
        events = self.events(path)
        first = next(index for index, event in enumerate(events) if event.get("event") == "model")
        events[first] = {key: value for key, value in events[first].items() if key != "input_messages"}
        spans = self.spans(path, events)
        self.assertEqual(io(spans[0])[0], [{"role": "user", "parts": [{"type": "text", "content": "add a test"}]}])
        inputs, _ = io(spans[1])
        self.assertEqual(inputs[0]["parts"][0]["content"], "<system-reminder>env</system-reminder>")
        self.assertEqual(inputs[-1]["parts"][0]["response"], BIG)

    def test_a_long_chain_of_logged_requests_is_rebuilt(self) -> None:
        """Each logged request keeps messages of the one before it. A long session chains more than the recursion limit."""
        count = 1_200
        rows = [{"type": "api-request-shape", "shapeHash": "h", "shape": {"system": [{"type": "text", "text": "You are Claude Code."}]}}]
        rows += [{"type": "api-request-blob", "hash": f"b{n}", "message": {"role": "user", "content": f"m{n}"}} for n in range(count)]
        # Newest first, so the first lookup walks the whole chain.
        rows += [{"type": "api-request", "id": f"r{n}", "shapeHash": "h", "keep": n, "tail": [f"b{n}"], **({"base": f"r{n - 1}"} if n else {})} for n in reversed(range(count))]
        rows += [
            {"type": "api-request", "id": "negative", "shapeHash": "h", "base": "r1", "keep": -1, "tail": []},
            {"type": "api-request", "id": "x", "shapeHash": "h", "base": "y", "keep": 0, "tail": ["b0"]},
            {"type": "api-request", "id": "y", "shapeHash": "h", "base": "x", "keep": 0, "tail": ["b0"]},
        ]
        found, _ = claude_transcript._exact_requests(rows)
        self.assertEqual([message["content"] for message in found[f"r{count - 1}"]["messages"]], [f"m{n}" for n in range(count)])
        self.assertEqual(found["r0"]["system"], "You are Claude Code.")
        for broken in ("negative", "x", "y"):  # a negative keep, a cycle
            self.assertNotIn(broken, found)

    def test_a_subagent_replays_no_earlier_turn_of_the_main_thread(self) -> None:
        """Requests name the earlier turns they replay. A subagent's file holds only the task this turn gave it."""
        directory = self.tmpdir()
        path = _jsonl(directory / "session.jsonl", [
            _prompt("u1", "p1", "turn one plan", 1),
            _assistant("a1", "r1", {"type": "text", "text": "done first"}, "end_turn", 2),
            _prompt("u2", "p2", "turn two ask", 3),
            _assistant("a2", "r2", {"type": "tool_use", "id": "toolu_1", "name": "Agent", "input": {"prompt": "look"}}, "tool_use", 4),
            _result("u3", "p2", "toolu_1", "found", 6),
            _assistant("a3", "r3", {"type": "text", "text": "done second"}, "end_turn", 7),
        ])
        side = {"isSidechain": True, "agentId": "ag1"}
        sub = _jsonl(directory / "agent-ag1.jsonl", [
            {**_prompt("s1", "p2", "look", 4), **side},
            {**_assistant("s2", "q1", {"type": "text", "text": "found"}, "end_turn", 5), **side},
        ])
        base = {"v": 1, "harness": "claude-code", "session_id": "s1", "turn_id": "p2", "transcript_path": str(path)}
        events = enrich_claude_events([
            {**base, "event": "turn_start", "prompt": "turn two ask", "ts": T0 + 3_000},
            {**base, "event": "subagent_start", "agent_id": "ag1", "agent_transcript_path": str(sub), "ts": T0 + 4_000},
            {**base, "event": "subagent_end", "agent_id": "ag1", "agent_transcript_path": str(sub), "output": "found", "ts": T0 + 6_000},
            {**base, "event": "turn_end", "output": "done second", "ts": T0 + 7_000},
        ], attempts=1)
        models = [event for event in events if event.get("event") == "model"]
        self.assertEqual([(event.get("agent_id", ""), event["replay_turns"]) for event in models], [("", ["p1"]), ("", ["p1"]), ("ag1", [])])
        # Only a turn sent with its text to this project may be replayed. Otherwise the main thread's requests are
        # rebuilt, and the subagent's keep their transcript input.
        dest = "https://api.example.com org project"
        sent = {"replayable": True, "dest": dest}
        for earlier, replayed in (
            (None, False),  # never traced
            (sent, True),
            ({**sent, "dest": "https://api.example.com org other"}, False),
            ({**sent, "replayable": False}, False),  # not sent yet, metadata, or failed rules
        ):
            with self.subTest(earlier=earlier):
                index = {} if earlier is None else {"claude-code\ts1\tp1": earlier}
                kept = flush._replayable(index, "claude-code\ts1\tp2", dest, models)
                self.assertEqual(["input_messages" in event for event in kept], [replayed, replayed, True])
                self.assertFalse(any("replay_turns" in event for event in kept))


class ReplayTest(HookCase):
    """A transcript replays every earlier turn of the session. A request keeps that input only when each of those turns
    went to this project with its text. Otherwise it is rebuilt from the turn's own events."""

    def claude(self, *, reads_env: bool) -> Path:
        rows = [_prompt("u1", "p1", "turn one plan", 1)]
        if reads_env:
            rows += [
                _assistant("a1", "r1", {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": ".env"}}, "tool_use", 2),
                _result("u2", "p1", "toolu_1", ENV, 3),
            ]
        rows += [
            _assistant("a2", "r2", {"type": "text", "text": "done first"}, "end_turn", 4),
            _prompt("u3", "p2", "turn two ask", 5),
            _assistant("a3", "r3", {"type": "text", "text": "done second"}, "end_turn", 6),
        ]
        return _jsonl(self.tmp / "session.jsonl", rows)

    def codex(self, *, reads_env: bool) -> Path:
        rows = [META, *_opens("t1", 1), _item(_text("user", "turn one plan"), 1)]
        if reads_env:
            rows += [_call("c1", "cat .env", 2), _row("token_usage_record", RECORD, 2), _call_output("c1", ENV, 3), _row("event_msg", _usage(15), 3)]
        rows += [
            _item(_text("assistant", "done first"), 4),
            _row("token_usage_record", RECORD, 4),
            _row("event_msg", _usage(30), 4),
            _closes("t1", 4),
            *_opens("t2", 5),
            _item(_text("user", "turn two ask"), 5),
            _item(_text("assistant", "done second"), 6),
            _row("token_usage_record", RECORD, 6),
            _row("event_msg", _usage(45), 6),
            _closes("t2", 6),
        ]
        return _jsonl(self.tmp / "rollout.jsonl", rows)

    def turn(self, harness: str, path: Path, turn: str, prompt: str, output: str, start: int, *, reads_env: bool = False) -> None:
        base = {"v": 1, "harness": harness, "session_id": "s1", "turn_id": turn, "cwd": str(self.project), "transcript_path": str(path)}
        self.hook({**base, "event": "turn_start", "prompt": prompt, "ts": start})
        if reads_env:
            self.hook({**base, "event": "tool", "tool_name": "Read", "tool_call_id": "toolu_1", "tool_input": {"file_path": ".env"}, "tool_output": ENV, "ts": start + 1_000})
        self.hook({**base, "event": "turn_end", "output": output, "ts": start + 3_000})

    def second_turn(self, harness: str, before: str) -> dict:
        """Turn one, which reads .env, traced ``before`` (off, metadata, sanitized), then turn two sanitized: its llm span."""
        path = self.claude(reads_env=True) if harness == "claude-code" else self.codex(reads_env=True)
        first, second = ("p1", "p2") if harness == "claude-code" else ("t1", "t2")
        self.optin({"enabled": before != "off", "capture": "metadata" if before == "metadata" else "sanitized"})
        self.turn(harness, path, first, "turn one plan", "done first", T0 + 1_000, reads_env=True)
        self.assertEqual(self.export(), int(before != "off"))
        Collector.bodies = []
        self.optin({"enabled": True, "capture": "sanitized"})
        self.turn(harness, path, second, "turn two ask", "done second", T0 + 5_000)
        self.assertEqual(self.export(), 1)
        spans = [span for body in Collector.bodies for span in llm_spans(body)]
        self.assertEqual(len(spans), 1)
        return spans[0]

    def test_a_turn_not_sent_with_its_text_is_not_replayed(self) -> None:
        for harness in ("claude-code", "codex"):
            for before in ("off", "metadata"):
                with self.subTest(harness=harness, before=before):
                    self.setUp()  # a fresh state directory and project
                    span = self.second_turn(harness, before)
                    self.assertEqual(io(span)[0], [{"role": "user", "parts": [{"type": "text", "content": "turn two ask"}]}])
                    sent = json.dumps(Collector.bodies)
                    for text in ("hunter2", "turn one plan", "file_path"):
                        self.assertNotIn(text, sent)

    def test_a_turn_sent_with_its_text_is_replayed(self) -> None:
        for harness in ("claude-code", "codex"):
            with self.subTest(harness=harness):
                self.setUp()  # a fresh state directory and project
                inputs, _ = io(self.second_turn(harness, "sanitized"))
                texts = [part.get("content") for message in inputs for part in message["parts"]]
                self.assertEqual([texts[0], *texts[-2:]], ["turn one plan", "done first", "turn two ask"])
                # Replayed text is redacted like the turn's own.
                sent = json.dumps(Collector.bodies)
                self.assertNotIn("hunter2", sent)
                self.assertIn("[REDACTED:secret_assignment]", sent)


class TailReadTest(Shared):
    """A large transcript is read from its end: back to the last compaction before the turn, and the older rows a reader
    still needs."""

    def setUp(self) -> None:
        for name, value in (("TAIL_MIN_BYTES", 1), ("_CHUNK", 97)):
            patcher = mock.patch(f"judgment_core.rows.{name}", value)
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_claude(self) -> None:
        shape = {"system": "You are Claude Code.", "tools": []}
        old = {"role": "user", "content": "old ask"}
        new = {"role": "user", "content": "new ask"}
        for stamped in (True, False):
            compaction = {"type": "system", "subtype": "compact_boundary", **({"timestamp": _at(9)} if stamped else {})}
            rows = [
                {"type": "summary", "summary": "first line"},
                {"type": "api-request-shape", "shapeHash": "h", "shape": shape},
                {"type": "api-request-shape", "shapeHash": "unused", "shape": shape},
                {"type": "api-request-blob", "hash": "b0", "message": old},
                {"type": "api-request-blob", "hash": "unused", "message": old},
                {"type": "api-request", "id": "r0", "shapeHash": "h", "keep": 0, "tail": ["b0"]},
                {"type": "user", "uuid": "u0", "promptId": "p0", "timestamp": _at(1), "message": old},
                _result("u1", "p0", "toolu_0", "old result", 2),
                _assistant("a0", "r0", {"type": "text", "text": "old answer"}, "end_turn", 3),
                # A minute before the turn, inside the margin a turn's rows may start early by: the read stops here.
                compaction,
                {"type": "user", "uuid": "u2", "isCompactSummary": True, "timestamp": _at(9), "message": {"role": "user", "content": "summary"}},
                {"type": "user", "uuid": "u3", "promptId": "p1", "timestamp": _at(10), "message": new},
                {"type": "api-request-blob", "hash": "b1", "message": new},
                {"type": "api-request", "id": "r1", "shapeHash": "h", "base": "r0", "keep": 0, "tail": ["b1"]},
                _assistant("a1", "r1", {"type": "text", "text": "new answer"}, "end_turn", 11),
                # During the turn: its requests before this one saw what came before it.
                {"type": "system", "subtype": "compact_boundary", "timestamp": _at(12)},
                _assistant("a2", "r2", {"type": "text", "text": "after"}, "end_turn", 13),
            ]
            path = _jsonl(self.tmpdir() / "session.jsonl", rows, spaced=True)
            found = records(path, T0 + 10_000, claude_transcript._is_compaction, claude_transcript._pins)
            # Before the compaction: the shape, blob, and base request r1 needs, and the earlier prompt (in the summary).
            skipped = (2, 4, 7, 8)
            with self.subTest(stamped=stamped):
                self.assertEqual(found, [row for index, row in enumerate(rows) if index not in skipped])
                self.assertEqual(claude_transcript._exact_requests(found)[0]["r1"]["system"], "You are Claude Code.")

    def test_codex(self) -> None:
        tools = {"type": "additional_tools", "role": "developer", "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}]}
        rows = [
            {**META, "payload": {"id": "s1"}},
            *_opens("t0", 1),
            _item({**tools, "tools": []}, 1),
            _item(tools, 1),
            _item(_text("user", "old ask"), 2),
            _item(_text("assistant", "old answer"), 3),
            _row("compacted", {"message": "", "replacement_history": [_text("user", "old ask")]}, 9),
            *_opens("t1", 10),
            _item(_text("user", "new ask"), 10),
            _item(_text("assistant", "new answer"), 11),
            _row("token_usage_record", RECORD, 11),
            _row("event_msg", _usage(15), 11),
            _closes("t1", 11),
        ]
        path = _jsonl(self.tmpdir() / "rollout.jsonl", rows, spaced=True)
        found = records(path, T0 + 10_000, codex_transcript._is_compaction, codex_transcript._pins)
        # Before the compaction: the turn it carries forward, and the latest tool list.
        self.assertEqual(found, [row for index, row in enumerate(rows) if index not in (3, 5, 6)])
        thread = read_thread(path, turn_id="t1", since_ms=T0 + 10_000)
        self.assertEqual(thread["replay_turns"], ["t0"])
        self.assertEqual([tool["name"] for tool in thread["steps"][0]["tools"]], ["shell"])


class RedactionTest(HookCase):
    """The system prompt, the tool definitions, and the whole history are redacted like any other text."""

    def test_a_turn_redacted_at_once_reads_as_each_event_redacted_alone(self) -> None:
        rs = redact.ruleset()
        shared = [{"role": "user", "parts": [{"type": "text", "content": f"token {GH}"}]}]
        again = {"role": "assistant", "parts": [{"type": "text", "content": f"again {GH}"}]}
        events = [
            {"event": "model", "input_messages": shared, "output": f"saw {GH}"},
            {"event": "model", "input_messages": [*shared, again, *shared], "output": "ok"},
            {"event": "tool", "tool_name": "read", "tool_output": GH},
        ]
        pooled = redact.redact_events(events, rs, [])
        self.assertEqual(pooled, [redact.redact_event(event, rs, []) for event in events])
        self.assertNotIn(GH, json.dumps(pooled))
        self.assertEqual([item["count"] for item in pooled[1]["redactions"] if item["field"] == "input_messages"], [3])

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


if __name__ == "__main__":
    unittest.main()
