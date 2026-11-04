# Trace format

This page lists the spans and attributes the plugins send to Judgment, and how a turn is queued and sent. Use it to write a judge against these traces, or to check what leaves your machine.

## Traces and sessions

One user request and its answer are one trace. A follow-up request is a new trace.

The root span is named `<harness> turn`, for example `claude-code turn`. Its kind is `agent`. Its `judgment.input` and `judgment.output` are the prompt and the final answer as plain text. Model, tool, and subagent spans sit under the root.

Every span carries `judgment.session_id` and `gen_ai.conversation.id`. Judgment groups traces into a session from the root's `judgment.session_id`.

The root also carries `agent.workspace_name`, the name of the folder the agent runs in. It's the last part of the path only. It's left out when that folder is your home folder, and when `capture` is `metadata`.

## Spans

| Span | `judgment.span_kind` | `judgment.input` | `judgment.output` |
| --- | --- | --- | --- |
| Root | `agent` | The prompt, as plain text. | The final answer, as plain text. |
| Model call | `llm` | A messages array. See [Model input](#model-input). | The assistant message, including reasoning and tool calls. |
| Tool call | `tool` | The tool's arguments. | The tool's result. |
| Subagent | `agent` | Its task. | Its answer. |

A subagent span holds its task and answer as plain text when it has model spans, and as a one-message array each when it has none.

A tool span ends when the tool finished.

## Model input

A model span's input is the conversation the model was sent, in this order:

1. The earlier turns of the session. Each one is its prompt and final answer.
2. The user prompt.
3. Every tool call and tool result that came before this request.

A tool result is the text the model read, not the agent's raw tool response.

These limits keep a long session from growing every request:

| What | Limit | Past the limit |
| --- | --- | --- |
| Earlier turns, in all | 8 KB | The newest turns are kept. |
| Each earlier-turn message | 2 KB | The message is cut. |
| The whole input | 64 KB | The earlier turns, the prompt, and the newest messages stay whole. A system message notes how many messages were left out. |

## Agent differences

Cursor reports tokens once per stop, not per model request. Its model span covers the whole stop. The tools that ran and their results are in the input, and the final text is the output.

Each agent's setup page lists the token fields that agent exposes.

A background subagent that finishes after the turn is sent again with the same span ids, a later end time, and `judgment.update_id`. The root is included, so its start and end cover every child.

## How a turn is queued and sent

Each hook writes its event to a queue in `~/.local/state/judgment` and returns. The queue and the sender's files are readable only by your user.

Each turn records the project ID, the API URL, and a fingerprint of the API key that were set when the hook ran. It doesn't record the key itself. A turn is sent only when all of the following hold:

- Every event in it carries the same fingerprint.
- That fingerprint matches the sender's own key.
- It goes to the project it recorded.

A turn recorded under a different key stays queued until a sender with that key runs. One sender runs at a time. While it runs, one more sender per key waits for it, so a busy sender for one key never strands another key's turn.

A turn waits for its subagents to finish, for up to 10 minutes. A turn that never ends, because the agent was killed mid-turn, is sent as it stands after a day with no new events.

A turn being sent stays on disk until the send succeeds, so a crash or a failed send retries the whole turn on the next run. A large turn goes out in several compressed requests, with the root last. A turn the server rejects as invalid or too large is skipped and noted in `flush.log`, so it never holds back later turns.
