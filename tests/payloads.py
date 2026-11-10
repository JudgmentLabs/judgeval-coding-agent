"""Realistic hook payloads for each agent: a turn whose prompt, tool call, tool result, and answer hold fake secrets."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from support import REPO, SECRETS, secret_blob


def claude_turn(cwd: str, session: str, prompt_id: str, *, transcript: str = "", prompt: str = "", command: str = "", output: str = "", answer: str = "") -> list[tuple[str, dict]]:
    base = {"session_id": session, "cwd": cwd, "transcript_path": transcript, "prompt_id": prompt_id, "hook_event_name": ""}
    prompt = prompt or "deploy with key " + SECRETS["openai_key"]["core"]
    command = command or "export DEPLOY" + "_TOKEN=" + SECRETS["export_token"]["core"] + " && ./deploy.sh"
    output = output or secret_blob()
    answer = answer or "Deployed. The token " + SECRETS["github_token"]["core"] + " still works."
    return [
        ("UserPromptSubmit", {**base, "hook_event_name": "UserPromptSubmit", "prompt": prompt}),
        ("PostToolUse", {**base, "hook_event_name": "PostToolUse", "tool_name": "Bash", "tool_use_id": prompt_id + "-tool",
                         "tool_input": {"command": command, "description": "Deploy"},
                         "tool_response": {"stdout": output, "stderr": "", "interrupted": False}, "duration_ms": 420}),
        ("PostToolUseFailure", {**base, "hook_event_name": "PostToolUseFailure", "tool_name": "WebFetch", "tool_use_id": prompt_id + "-fail",
                                "tool_input": {"url": SECRETS["url_password"]["text"]}, "error": "401 for " + SECRETS["bearer_token"]["text"]}),
        ("Stop", {**base, "hook_event_name": "Stop", "last_assistant_message": answer}),
    ]


def codex_turn(cwd: str, session: str, turn: str) -> list[tuple[str, dict]]:
    base = {"session_id": session, "turn_id": turn, "cwd": cwd, "model": "gpt-5.5-codex", "transcript_path": ""}
    command = "curl -H 'Authorization: Bearer " + SECRETS["bearer_token"]["core"] + "' https://api.example.com/v1"
    return [
        ("UserPromptSubmit", {**base, "prompt": "check the api with " + SECRETS["anthropic_key"]["core"]}),
        ("PreModelRequest", {**base}),
        ("PostModelRequest", {**base, "status": "completed", "token_usage": {"input_tokens": 2000, "cached_input_tokens": 500, "output_tokens": 90}}),
        ("PreToolUse", {**base, "tool_name": "shell", "tool_use_id": turn + "-c1", "tool_input": {"command": ["bash", "-lc", command]}}),
        ("PostToolUse", {**base, "tool_name": "shell", "tool_use_id": turn + "-c1", "tool_input": {"command": ["bash", "-lc", command]},
                         "tool_response": {"output": secret_blob(), "exit_code": 0}}),
        ("Stop", {**base, "last_assistant_message": "Done, used " + SECRETS["stripe_key"]["core"]}),
    ]


def cursor_turn(cwd: str, conversation: str, generation: str) -> list[tuple[str, dict]]:
    base = {"conversation_id": conversation, "generation_id": generation, "workspace_roots": [cwd], "model": "claude-sonnet-5-5", "transcript_path": ""}
    command = "psql " + SECRETS["url_password"]["text"]
    return [
        ("beforeSubmitPrompt", {**base, "prompt": "connect using " + SECRETS["url_password"]["text"]}),
        ("preToolUse", {**base, "tool_name": "Shell", "tool_use_id": generation + "-t1", "tool_input": {"command": command}}),
        ("postToolUse", {**base, "tool_name": "Shell", "tool_use_id": generation + "-t1", "tool_input": {"command": command},
                         "tool_output": json.dumps({"output": secret_blob(), "exitCode": 0}), "duration": 1200}),
        ("afterAgentThought", {**base, "text": "The JWT " + SECRETS["jwt"]["core"] + " is expired.", "duration": 300}),
        ("afterAgentResponse", {**base, "text": "Connected with " + SECRETS["google_api_key"]["core"]}),
        ("stop", {**base, "status": "completed", "text": "Connected with " + SECRETS["google_api_key"]["core"],
                  "input_tokens": 3000, "output_tokens": 150, "cache_read_tokens": 1000}),
    ]


_PI_SCRIPT = r"""
const logic = await import(process.argv[1]);
const spec = JSON.parse(process.argv[2]);
const info = {sessionId: spec.session, cwd: spec.cwd, turnId: spec.turn, finalOutput: spec.answer, model: "claude-sonnet-5-5", provider: "anthropic"};
const t = spec.t0;
const out = [
  ...logic.recordsFor("before_agent_start", {prompt: spec.prompt, end_ts: t}, info),
  ...logic.recordsFor("message_end", {message: {role: "assistant", content: [{type: "text", text: "Running it."},
      {type: "toolCall", id: "call-1", name: "bash", arguments: {command: spec.command}}],
      usage: {input: 900, output: 40, cacheRead: 100, cost: {total: 0.0042}}}, start_ts: t + 10, end_ts: t + 500}, info),
  ...logic.recordsFor("tool_execution_end", {toolName: "bash", toolCallId: "call-1", args: {command: spec.command},
      result: {content: [{type: "text", text: spec.output}]}, isError: false, start_ts: t + 510, end_ts: t + 900}, info),
  ...logic.recordsFor("message_end", {message: {role: "assistant", content: [{type: "text", text: spec.answer}],
      usage: {input: 1200, output: 60}}, start_ts: t + 910, end_ts: t + 1500}, info),
  ...logic.recordsFor("agent_end", {end_ts: t + 1600}, info),
];
console.log(JSON.stringify(out));
"""


def pi_turn(cwd: str, session: str, turn: str, t0: int, *, prompt: str = "", command: str = "", output: str = "", answer: str = "") -> list[tuple[str, dict]]:
    """Pi records as the extension builds them, from pi/src/logic.mjs. Without node, the same shapes by hand."""
    spec = {
        "session": session, "cwd": cwd, "turn": turn, "t0": t0,
        "prompt": prompt or "rotate " + SECRETS["aws_access_key_id"]["core"],
        "command": command or "echo " + SECRETS["slack_token"]["core"],
        "output": output or secret_blob(),
        "answer": answer or "Rotated. Old key " + SECRETS["aws_access_key_id"]["core"],
    }
    if shutil.which("node"):
        result = subprocess.run(["node", "--input-type=module", "-e", _PI_SCRIPT, str(REPO / "pi" / "src" / "logic.mjs"), json.dumps(spec)],
                                capture_output=True, text=True, timeout=30, check=True)
        return [("pi", record) for record in json.loads(result.stdout)]
    base = {"v": 1, "harness": "pi", "session_id": session, "cwd": cwd, "turn_id": turn, "environment": "production"}
    return [("pi", record) for record in [
        {**base, "event": "turn_start", "prompt": spec["prompt"], "ts": t0},
        {**base, "event": "model", "model": "claude-sonnet-5-5", "provider": "anthropic", "output": "Running it.",
         "usage": {"input_tokens": 900, "output_tokens": 40, "cache_read_input_tokens": 100, "total_cost_usd": 0.0042},
         "tool_calls": [{"id": "call-1", "name": "bash", "arguments": {"command": spec["command"]}}], "start_ts": t0 + 10, "end_ts": t0 + 500, "ts": t0 + 500},
        {**base, "event": "tool", "tool_name": "bash", "tool_call_id": "call-1", "tool_input": {"command": spec["command"]}, "tool_output": spec["output"],
         "is_error": False, "duration_ms": None, "start_ts": t0 + 510, "end_ts": t0 + 900, "ts": t0 + 900},
        {**base, "event": "model", "model": "claude-sonnet-5-5", "provider": "anthropic", "output": spec["answer"],
         "usage": {"input_tokens": 1200, "output_tokens": 60}, "start_ts": t0 + 910, "end_ts": t0 + 1500, "ts": t0 + 1500},
        {**base, "event": "turn_end", "output": spec["answer"], "ts": t0 + 1600},
    ]]


def adapter_turns(cwd: str) -> dict[str, list[tuple[str, dict]]]:
    import time

    return {
        "claude": claude_turn(cwd, "cc-sess-1", "cc-p1"),
        "codex": codex_turn(cwd, "cx-sess-1", "cx-t1"),
        "cursor": cursor_turn(cwd, "cu-conv-1", "cu-gen-1"),
        "pi": pi_turn(cwd, "pi-sess-1", "pi-sess-1:1", int(time.time() * 1000) - 5000),
    }


def hook_path(repo: Path) -> Path:
    return repo / "core" / "bin" / "judgment-hook"
