# Pi

This extension sends a Pi session to Judgment when the project opts in. It runs inside Pi and appends each event to the same local spool as the other plugins. Install it, then follow the project setup in the repository README.

## Install

```bash
pi install git:github.com/JudgmentLabs/judgeval-coding-agent
```

Pi clones the repository, adds it to `~/.pi/agent/settings.json`, and loads the extension in every session. Add `-l` to install it for the current project only, in `.pi/settings.json`. Project packages load after you trust the project. To try the extension for one run without installing it, run `pi -e <clone>/pi/src/index.ts` from a clone of this repository.

The extension queues each event for `judgment-hook --raw` and returns without waiting for it. No Pi event waits on the queue, `agent_settled` included. The queue runs one hook at a time so the spool keeps event order, and a hook that runs longer than 5 seconds is killed. On `session_shutdown`, Pi exits when the handlers return, so the extension waits up to 3 seconds for queued events first. `/judgment on|off` waits the same way, so events from before the switch keep their old setting. The event is already prepared, so that path does not run jq. With `JUDGMENT_TRACING=0` the extension subscribes to no Pi event. `before_provider_request` is the model span start, so the span includes time to the first token. `message_start` is the fallback when that event is absent. `tool_execution_start` is the tool span start. `durationMs` is the tool length when Pi reports it. The extension runs the hook from `pi/vendor/judgment-core`, a copy of `core/` that `scripts/sync-core.sh` refreshes.

## Update

```bash
pi update --extensions
```

Pi shows a notice at startup when an update is waiting. Plain `pi update` updates Pi itself, not the extension.

## Turn a session on or off

Pi registers a command, so the message never starts a turn:

```text
/judgment on
/judgment off
/judgment status
```

The notice is the tracing state for this conversation only.

## What this extension can see

| Field | Where it comes from |
| --- | --- |
| Prompt | `before_agent_start` `prompt` |
| Assistant text and tool calls | `message_end` on an assistant message. `content[]` entries with `type: "toolCall"` become tool calls on that model span. `toolResult` messages are the matching tool spans. |
| Model | `message.model` |
| Tokens and cost | `message.usage`, including `input`, `output`, `cacheRead`, `cacheWrite`, and `cost.total` |
| Tool arguments | `tool_execution_start` `args`, or `tool_call` `input`. `tool_execution_end` in Pi 1.5.0 does not include `args`. |
| Tool result and duration | `tool_execution_start` for the start, then `tool_execution_end` `result`, `durationMs`, `isError` |

Pi has no built-in subagent event. This extension builds the child from the `subagent` tool result. The official example puts the child conversation on `result.details.results[].messages`. `pi-subagents` puts a `sessionFile` on each result, and the task text may be `[prompt redacted]`. The extension streams that file in the background, after the handler returns. Each result becomes one agent span under the parent turn. The task is the input. The last assistant text is the output. Child model and tool spans nest under it. The session id is `ctx.sessionManager.getSessionId()`. A file named `session` is not used as an id.

A spawned child inherits `JUDGMENT_PARENT` and does not trace itself. Parallel `subagent` calls share the variable, so it stays set until the last running call ends. A background `pi-subagents` run returns before the child has done any work, so it gets no span at launch. When its `subagent-notify` completion notice arrives, the extension reads the child session file the notice names and adds the child as an agent span to the turn that launched it. That trace is sent again with a higher `judgment.update_id`. It is never its own trace. If Pi is idle, the notice starts a run that stays on the launching turn, and the root input stays the prompt that started the child. If the notice arrives inside a later turn, Pi hands it to the model between that turn's tool calls, so the model's reply stays in the later turn.

An image content part (`type: "image"`) in a prompt, tool arguments, or a tool result is stored as `[image]`, not as the base64 blob. A request that fails has no content, so its model span output is Pi's error message.

One user prompt is one trace. The root is `pi turn`. Tool rounds inside that run stay under that root. The turn ends on `agent_settled`, not `agent_end`, because Pi can retry a failed request after `agent_end`. A follow-up prompt is a new trace with the same `judgment.session_id`. That includes a follow-up or steering message typed while Pi runs. Pi delivers it as a user message inside the same run, with no `before_agent_start`, so the extension ends the current turn there and starts a new one with that message as its prompt. A prompt that is only an image is a new turn whose prompt is `[image]`. Only a custom message, such as a `pi-subagents` completion notice, stays on an earlier turn.

## Try it from the CLI

This runs the plugin from a clone of this repository at `~/judgment-tracing`, without installing it:

```bash
git clone https://github.com/JudgmentLabs/judgeval-coding-agent.git ~/judgment-tracing
```

Use a temporary git repo.

```bash
mkdir /tmp/judgment-pi && cd /tmp/judgment-pi
git init
echo 'hello' > README.md
export JUDGMENT_API_KEY=<your-api-key>
export JUDGMENT_PROJECT_ID=<your-project-id>
mkdir -p .judgment && printf '{"enabled": true}\n' > .judgment/tracing.json
pi -e $HOME/judgment-tracing/pi/src/index.ts
```

Ask Pi to read `README.md` and run `ls`.

Open the project in Judgment. You should see one trace whose root is `pi turn`, with a model span that has token counts and a tool span. A second prompt in the same Pi session is another trace with the same session id. A directory without `.judgment/tracing.json`, or a shell with `JUDGMENT_TRACING=0`, should add no trace.
