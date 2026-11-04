# Claude Code

This plugin sends a Claude Code session to Judgment when the project opts in. Install it, then follow the project setup in the repository README.

## Install

Add this repository as a plugin marketplace, then install the plugin:

```bash
claude plugin marketplace add JudgmentLabs/judgeval-coding-agent
claude plugin install judgment-tracing@judgment-tracing
```

The marketplace file is `.claude-plugin/marketplace.json`. You can also install from `/plugin` inside Claude Code. See [Marketplace reference](https://code.claude.com/docs/en/plugins/marketplace-reference).

Start a new Claude Code session after installing. Hooks from a plugin load with that session. To try the plugin for one session without installing it, run `claude --plugin-dir <clone>/claude-code` from a clone of this repository.

## Update

```bash
claude plugin marketplace update judgment-tracing && claude plugin update judgment-tracing@judgment-tracing
```

The first command fetches the latest release, and the second installs it. The update takes effect in the next session.

## Turn a session on or off

Claude Code treats a line that starts with `/` as a slash command. `/judgment off` is not a command, so it never reaches the tracing hook. The plugin registers three skills. Type one of these:

```text
/judgment:off
/judgment:on
/judgment:status
```

The hook blocks the skill, so Claude Code does not run it as a prompt. The reply is the tracing state for this conversation only. A leading space before `/judgment off` also reaches the hook, because Claude Code does not treat that line as a command. The skill form is the one to use.

Skill names in a plugin keep the plugin prefix. See [Slash commands](https://code.claude.com/docs/en/slash-commands).

## What this plugin can see

| Field | Where it comes from |
| --- | --- |
| Prompt | `UserPromptSubmit` `prompt` |
| Final answer | `Stop` `last_assistant_message` |
| Tool arguments and result | `PostToolUse` `tool_input` and `tool_response` |
| Failed tool | `PostToolUseFailure` `error` |
| Model name | `SessionStart` `model`, then `PostModelSwitch` `to_model` |
| Subagent text | `SubagentStop` `last_assistant_message` |
| Subagent tokens | `PostToolUse` on `Agent`, `tool_response.usage` |

`tool_response.usage` is the subagent's final request only. `totalTokens` on that result is the same request, not the whole subagent run. The main turn's hook payloads do not include token counts. After the turn is spooled, the exporter reads `transcript_path`. Each API request becomes one model span. Rows that share a message id or a request id are one request, including parallel tool calls split across rows. Model and usage come from the row that has them, and a tool call is stored once. A tool result is not a new user prompt, so later requests in the same turn are included. When a row has a timestamp, the span uses it. That read is not on the hook path. A background subagent that finishes after `Stop` is exported again with the same span ids. The root is `claude-code turn`. That resend includes the root, with `judgment.update_id`, so its start and end cover the children. The same `judgment.session_id` on every span groups the traces from one conversation. Each subagent request is read from `subagents/agent-<id>.jsonl` next to the session transcript, or from `agent_transcript_path` on `SubagentStop`. A `SubagentStop` with an empty `agent_type` and no matching Agent tool or `SubagentStart` is an internal fork, such as a prompt suggestion, and is ignored.

Tool calls inside a subagent include `agent_id` on `PostToolUse`. Those tool spans are children of the subagent span.

Assistant `tool_use` blocks and the following `tool_result` rows become the tool calls on that model span and the context of the next one. The trace root is `claude-code turn`. A later subagent flush sends that root again with `judgment.update_id` so its end time covers the new children.

Hook reference: [Claude Code hooks](https://code.claude.com/docs/en/hooks).

## Try it from the CLI

This runs the plugin from a clone of this repository at `~/judgment-tracing`, without installing it:

```bash
git clone https://github.com/JudgmentLabs/judgeval-coding-agent.git ~/judgment-tracing
```

Use a temporary git repo, not a repo you care about.

```bash
mkdir /tmp/judgment-claude && cd /tmp/judgment-claude
git init
export JUDGMENT_API_KEY=<your-api-key>
export JUDGMENT_PROJECT_ID=<your-project-id>
mkdir -p .judgment && printf '{"enabled": true}\n' > .judgment/tracing.json
claude --plugin-dir $HOME/judgment-tracing/claude-code -p "Read README.md if it exists, then run ls"
```

Open the project in Judgment. You should see one trace whose root is `claude-code turn`, with a model span and a tool span for the shell call. A second prompt in the same conversation is a new trace with the same session id. A second run in a directory without `.judgment/tracing.json` should add no trace. `JUDGMENT_TRACING=0` in an opted-in repo should add no trace either.

`/judgment off` needs an interactive session. `claude -p` submits one prompt and exits, so use a normal `claude` session to try the command.
