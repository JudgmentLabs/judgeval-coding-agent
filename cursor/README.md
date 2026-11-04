# Cursor

This plugin sends a Cursor agent session to Judgment when the project opts in. Install it, then follow the project setup in the repository README.

## Install

This command copies the plugin into Cursor's local plugin folder, and replaces the old copy only after the new one is in place. Cursor does not load a symlink that points outside that folder.

```bash
d=$(mktemp -d) && git clone -q --depth 1 https://github.com/JudgmentLabs/judgeval-coding-agent.git "$d" && mkdir -p ~/.cursor/plugins/local && rm -rf ~/.cursor/plugins/.judgment-tracing.new && cp -a "$d/cursor" ~/.cursor/plugins/.judgment-tracing.new && rm -rf ~/.cursor/plugins/local/judgment-tracing && mv ~/.cursor/plugins/.judgment-tracing.new ~/.cursor/plugins/local/judgment-tracing && rm -rf "$d"
```

Reload the window. Open Customize and confirm Judgment tracing is listed. Local plugin imports can be turned off on Enterprise. An admin turns that on under **Dashboard** > **Settings** > **Security & Identity** > **Marketplace and Plugins**.

The Cursor CLI, `cursor-agent`, does not run a plugin's `beforeSubmitPrompt`, `afterAgentResponse`, and `stop` hooks. A CLI trace from the plugin has no model span or tokens. For the CLI, use project hooks instead. Copy `hooks/hooks.json` to `.cursor/hooks.json` in the repository and replace `$CURSOR_PLUGIN_ROOT` with the absolute path of this directory in a clone. `stop` and `subagentStop` run from the workspace root, so a relative script path misses them.

Hook reference: [Hooks](https://cursor.com/docs/agent/hooks). Plugin layout: [Plugins](https://cursor.com/docs/plugins).

## Update

Run the install command again, then reload the window.

## Turn a session on or off

Type one of these as the whole message:

```text
/judgment on
/judgment off
/judgment status
```

Cursor does not send that message to the model. In the IDE, the hook's `user_message` is the tracing state for this conversation. The Cursor CLI does not render that field, so `/judgment status` prints nothing there. `judgment-trace status` in the project directory prints the same state.

## What this plugin can see

| Field | Where it comes from |
| --- | --- |
| Prompt | `beforeSubmitPrompt` `prompt` |
| Assistant text | `afterAgentResponse` `text` |
| Thinking text and duration | `afterAgentThought` `text`, `duration` or `duration_ms` |
| Tool start | `preToolUse` |
| Tool arguments, result, and duration | `postToolUse` `tool_input`, `tool_output`, `duration`. For Shell, only `tool_output.output`. |
| Failed tool | `postToolUseFailure` `error_message` |
| Model name | common `model` field |
| Token counts | `stop` `input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens` |
| Subagent task and summary | `Task` `tool_input.prompt` and `tool_output.agentId`. `subagentStart` and `subagentStop` are kept for the IDE. The CLI does not fire them. |

`stop` token counts are copied onto the model span. Non-cached input is `input_tokens` minus the cache read. `preCompact` reports `context_tokens`, which is the size of the context window, not usage for one generation. This plugin does not copy that number into `judgment.usage.*`.

`afterAgentThought` runs once for each routed model. The generation id gains that model, or a suffix `-<n>-<4 chars>`. Both copies are one turn, and the repeated thought text is one span. The hook time is the end of the thought. The span starts at that time minus `duration`.

The model span uses a resolved model when the thought hook reports one. Cursor names that model `grok-4.7-high`. The suffix is reasoning effort. The span stores `grok-4.7`, which is the id on xAI's pricing page. Below 200k prompt tokens the published rates are $2 per million input tokens, $0.50 per million cached input tokens, and $6 per million output tokens. At 200k and above they are $4, $1, and $12. The span's `total_cost_usd` is those rates times the reported tokens. `auto-smart` stays unnamed until a resolved model appears, and then there is no cost. A stop with a model name and no token counts does not create a model span.

A `Task` tool is a subagent. `preToolUse` marks the task open. Child hooks then arrive with the child's `conversation_id`, before `postToolUse` reports `agentId`. Those hooks are written onto the parent turn with `judgment.agent_id` set to the child id. The agent span's input is `tool_input.prompt`. Its output is the last assistant text in `~/.cursor/projects/<slug>/agent-transcripts/<conversation>/subagents/<agentId>.jsonl` when that file is on `transcript_path`. A final `<|eos|>` line in that file is not the answer. A Task that fails reports no `agentId`, but its child already ran and wrote a transcript. That child is an agent span too, with the prompt and last answer from its own file. Child tools and thoughts nest under the agent span. There is no child model span, because the CLI does not report the child's tokens. A background Task returns quickly. The later wake-up, which has no `beforeSubmitPrompt`, continues the launching turn. `subagentStart` and `subagentStop` stay mapped for the IDE.

Headless `cursor-agent -p` fires `sessionStart`, `afterAgentThought`, `preToolUse`, `postToolUse`, and `sessionEnd`. It does not fire `beforeSubmitPrompt`, `afterAgentResponse`, or `stop`. Events between that start and end are one trace. A shell event that has no generation id stays on that trace. `cursor-agent -p --resume` fires no `sessionStart`, so an event that arrives with no open turn starts a new trace in the same session. An exported turn is not rewritten. `afterAgentResponse` can arrive a few milliseconds after `stop`. It carries that turn's generation id and stays on that turn. The prompt and the final answer come from the last user turn in `agent-transcripts/<conversation>/<conversation>.jsonl` when `transcript_path` points at it. A `<user_query>` wrapper is removed. There is no model span unless a stop reports token counts. Tab completions are not traced.

## Try it from the CLI

This runs the plugin from a clone of this repository at `~/judgment-tracing`, without installing it:

```bash
git clone https://github.com/JudgmentLabs/judgeval-coding-agent.git ~/judgment-tracing
```

Use a temporary git repo.

Each hook in `hooks/hooks.json` runs the vendored `judgment-hook` with this plugin's `hooks/map.jq`. Point the project hooks at that checkout.

```bash
mkdir /tmp/judgment-cursor && cd /tmp/judgment-cursor
git init
export JUDGMENT_API_KEY=<your-api-key>
export JUDGMENT_PROJECT_ID=<your-project-id>
mkdir -p .judgment && printf '{"enabled": true}\n' > .judgment/tracing.json
PLUGIN=$HOME/judgment-tracing/cursor
mkdir -p .cursor
sed "s|\$CURSOR_PLUGIN_ROOT|$PLUGIN|g" "$PLUGIN/hooks/hooks.json" > .cursor/hooks.json
```

Run the Cursor agent in that repo and ask it to read a file and run a shell command. Open the project in Judgment. You should see one trace whose root is `cursor turn`, with a model span and a tool span. When `stop` includes token counts, they are on the model span. A directory without `.judgment/tracing.json`, or a shell with `JUDGMENT_TRACING=0`, should add no trace.
