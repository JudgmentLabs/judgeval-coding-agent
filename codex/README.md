# Codex

This plugin sends a Codex session to Judgment when the project opts in. Install it, then follow the project setup in the repository README.

## Install

Add this repository as a plugin marketplace, then install the plugin:

```bash
codex plugin marketplace add https://github.com/JudgmentLabs/judgeval-coding-agent.git
codex plugin add judgment-tracing@judgment-tracing
```

Codex runs a plugin's hooks only after you trust them. Start Codex and trust the Judgment hooks when it asks, or review them with `/hooks`. Codex saves that trust, so later sessions, `codex exec` included, run the hooks without asking. The plugin also adds the `$judgment` skill, so Codex suggests the session command as you type.

If you installed an earlier version by writing its hooks into `~/.codex/hooks.json`, or into a project's `.codex/hooks.json`, remove those entries and keep your other hooks. With both in place, every event is recorded twice.

Codex does not run a plugin's `PreModelRequest` and `PostModelRequest` hooks. The model spans come from the session's rollout file, so each model request still has its own span with tokens.

Hook reference: [Hooks](https://developers.openai.com/codex/hooks).

## Update

Codex checks the marketplace when a session starts and installs a new release on its own. To update right away, run:

```bash
codex plugin marketplace upgrade judgment-tracing
```

Trust carries over to the new release while its hook definitions stay the same. A release that changes a hook asks you to review it again.

## Turn a session on or off

Codex rejects `/judgment off` before the tracing hook runs. Skills are invoked with `$`. This plugin ships a `judgment` skill. Type one of these:

```text
$judgment off
$judgment on
$judgment status
```

The hook blocks that prompt, so Codex does not run the skill as a task. The reply is the tracing state for this conversation only. The skill does not run on its own. See [Build skills](https://developers.openai.com/codex/skills).

## What this plugin can see

| Field | Where it comes from |
| --- | --- |
| Prompt | `UserPromptSubmit` `prompt` |
| Final answer | `Stop` `last_assistant_message` |
| Tool arguments and result | `PostToolUse` `tool_input` and `tool_response` |
| Model requests, text, and tokens | The rollout at `transcript_path` |
| Subagent task and answer | The child rollout and `SubagentStop` `last_assistant_message` |

Each model request is one llm span. Codex writes one `token_count` event per request to the rollout, and that event is the only source of llm spans and their tokens. The span holds the assistant text, including commentary written before a tool call, and the request's tool calls. Its input is the prompt plus every earlier call and result in the turn. The rollout `input_tokens` already includes cached tokens. The exporter stores the uncached remainder in `judgment.usage.non_cached_input_tokens` and keeps the cached count in `judgment.usage.cache_read_input_tokens`. Codex 0.166 also sends `PreModelRequest` and `PostModelRequest`. Those hooks carry no text, so they are used only when no rollout can be read. Codex 0.159 does not send them.

A shell command is one tool span, built from `PreToolUse` and `PostToolUse`. In code mode the model calls `exec` with a script, and each command the script runs has its own hooks. That `exec` call stays in the llm output, and the command becomes the tool span. An `exec` call that runs no hooked command becomes its own tool span, from the rollout. Hosted tools such as web search do not run these hooks, so they do not become tool spans.

`transcript_path` is not a stable format. See [Hooks](https://developers.openai.com/codex/hooks). A resumed session reads only the current turn, matched by turn id or by the prompt.

A subagent is an agent span inside the parent turn. Its llm and tool spans sit under it, and its tokens come from its own rollout. Codex encrypts subagent tasks. With `multi_agent_v2`, the `spawn_agent` message and the child's `NEW_TASK` payload are encrypted, and only `task_name` is plain text. The agent span input is `Subagent task: <task_name>`, and the encrypted text reads `[encrypted]` in the `spawn_agent` span and in the child's first llm input. That input is the `NEW_TASK` message the child received, not the parent prompt. Without `multi_agent_v2`, the child prompt is plain text and is the agent span input.

A background child can finish after the parent turn ends. When the child is already running at `Stop`, the turn waits and is sent once, with the child, after the child stops. When the child starts after `Stop`, the turn is sent at `Stop` and sent again as the child reports, on the same trace with a higher `judgment.update_id`.

Either way the child stays in the parent trace. It is not a second `codex turn`. `spawn_agent` and `wait_agent` are tool spans on the parent.

The trace root is `codex turn`. Follow-up prompts are new traces with the same `judgment.session_id`. Anything else in the rollout is ignored. The rollout is read when the trace is sent, not on the hook path.

## Try it from the CLI

Install the plugin first, as above. Then use a temporary git repo.

```bash
mkdir /tmp/judgment-codex && cd /tmp/judgment-codex
git init
mkdir -p .judgment && printf '{"enabled": true}\n' > .judgment/tracing.json
codex exec "Read README.md if it exists, then run ls"
```

Open the project in Judgment. You should see one trace whose root is `codex turn`, with a model span for each request and a tool span for the shell call. A run in a directory with no `.judgment/tracing.json`, or with `JUDGMENT_TRACING=0`, should add no trace.
