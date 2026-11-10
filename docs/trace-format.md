# Trace format

This page lists the spans and attributes the plugins send to Judgment, and how a turn is queued and sent. Use it to write a judge against these traces, or to check what leaves your machine.

## Traces and sessions

One user request and its answer are one trace. A follow-up request is a new trace.

The root span is named `<harness> turn`, for example `claude-code turn`. Its kind is `agent`. Its `judgment.input` and `judgment.output` are the prompt and the final answer as plain text. Model, tool, and subagent spans sit under the root.

Every span carries `judgment.session_id` and `gen_ai.conversation.id`. Judgment groups traces into a session from the root's `judgment.session_id`.

The root also carries `agent.workspace_name`, the name of the folder the agent runs in. It's the last part of the path only. It's left out when that folder is your home folder, and when `capture` is `metadata`.

## Redaction

With `capture` set to `sanitized`, the default, the root span records what was redacted. The attributes never hold a secret, or a hash, prefix, or length of one.

| Attribute | Type | Value |
| --- | --- | --- |
| `judgment.redacted` | bool | `true` when anything in the turn was redacted or the turn was blanked. |
| `judgment.redaction_count` | int | How many redactions the turn's text holds. |
| `judgment.redactions` | string | JSON list of `{"rule", "field", "count"}`, such as `{"rule":"github_token","field":"tool_output","count":2}`. `rule` is a pattern id, `secret_field` for a secret-named member's value in structured input or output, `env:<NAME>` for an environment variable's value, or `custom` for a `redactPatterns` rule. `field` is the event field the text was in: `prompt`, `output`, `tool_input`, `tool_output`, `tool_results`, `tool_calls`, and so on. Left out when the count is 0. |
| `judgment.redacted_reason` | string | `sensitive_path` when the turn was blanked. |
| `judgment.redacted_rule` | string | The rule that blanked it, such as `dotenv`, `credentials_file`, `env_dump`, or `custom_path`. `redaction_limit` when the turn held more to check than the limits allow. |
| `judgment.redacted_trigger` | string | The path that blanked it, such as `.env` or `~/.aws/credentials`, with the home folder as `~` and any secret in it redacted. For a command, `cmd:<rule>`, such as `cmd:env_dump`. |
| `judgment.redaction_error` | bool | A rule failed, such as an invalid `redactPatterns` regex, or an event needed more redaction than the limits allow, so the affected events were sent without text. A failed rule also drops the earlier turns' text that would ride along as context. |

A child span whose input or output holds placeholders carries its own `judgment.redaction_count`, the number of `[REDACTED:` placeholders in that span's text.

A blanked turn has the same spans, span ids, token counts, cost, and timing as it would unblanked, and every `judgment.input` and `judgment.output` is empty. A model span's `judgment.llm.tool_calls` keeps only each call's id and name. A blanked turn is never sent as context in a later turn's model input, and the next turn is not blanked.

With `capture` set to `full` the root has none of these attributes. With `metadata` it has `judgment.redacted` and `judgment.redaction_count`.

## Spans

| Span | `judgment.span_kind` | `judgment.input` | `judgment.output` |
| --- | --- | --- | --- |
| Root | `agent` | The prompt, as plain text. | The final answer, as plain text. |
| Model call | `llm` | A messages array. See [Model input](#model-input). | The assistant message, including reasoning and tool calls. |
| Tool call | `tool` | The tool's arguments. | The tool's result. |
| Subagent | `agent` | Its task. | Its answer. |

A subagent span holds its task and answer as plain text when it has model spans, and as a one-message array each when it has none.

A tool span ends when the tool finished.

A model call that failed, when the agent reports one (Hermes Agent does), is its own model span with error status. Its output is the error message. The next model call's input leaves that message out, because the model never saw it.

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

Secrets are redacted twice: by the hook, before an event reaches the queue, and by the sender, after it reads transcript text. Both use the rules in `core/bin/redact-rules.json`. The sender checks for a sensitive file before it redacts, so a placeholder in a file name cannot hide it. A turn that touched a sensitive file is saved blank before it is sent, so its text does not stay on disk.

A turn being sent stays on disk until the send succeeds, so a crash or a failed send retries the whole turn on the next run. A large turn goes out in several compressed requests, with the root last. A turn the server rejects as invalid or too large is skipped and noted in `flush.log`, so it never holds back later turns.
