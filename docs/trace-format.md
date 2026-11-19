# Trace format

This page lists the spans and attributes the plugins send to Judgment, and how a turn is queued and sent. Use it to write a judge against these traces, or to check what leaves your machine.

## Traces and sessions

One user request and its answer are one trace. A follow-up request is a new trace.

The root span is named `<harness> turn`, for example `claude-code turn`. Its kind is `agent`. Its `judgment.input` and `judgment.output` are the prompt and the final answer as plain text. Model, tool, and subagent spans sit under the root.

Every span carries `judgment.session_id` and `gen_ai.conversation.id`. Judgment groups traces into a session from the root's `judgment.session_id`.

Every span, subagent spans included, carries `judgment.customer_user_id`, the person who ran the session. It is `JUDGMENT_CUSTOMER_USER_ID` when set, from the environment or the credentials file, else `git config --global user.email`. With neither, spans have no `judgment.customer_user_id`. In Claude Code, Codex, Cursor, and Pi, the session's first traced event reads it and every turn of the session keeps it, so a change applies from the next session. Hermes reads it again every few seconds as it rereads the project settings, so a change applies to its next turn; it reads the git email once per Hermes process. It is kept under every `capture` mode. Redaction treats it as a label: no built-in rule matches an email address, so it is sent as it is unless a project's own `redactPatterns` match it. It is redacted before it is written to disk, under every `capture` mode. When redaction fails for an event, that event has no `judgment.customer_user_id`.

The root also carries `agent.workspace_name`, the name of the folder the agent runs in. It's the last part of the path only. It's left out when that folder is your home folder, and when `capture` is `metadata`.

## Redaction

With `capture` set to `sanitized`, the default, the root span records what was redacted. The attributes never hold a secret, or a hash, prefix, or length of one.

| Attribute | Type | Value |
| --- | --- | --- |
| `judgment.redacted` | bool | `true` when anything in the turn was redacted or the turn was blanked. |
| `judgment.redaction_count` | int | How many redactions the turn's text holds. |
| `judgment.redactions` | string | JSON list of `{"rule", "field", "count"}`, such as `{"rule":"github_token","field":"tool_output","count":2}`. `rule` is a format rule's id (such as `github_token` or `bearer_token`), `secret_assignment` for a value after a secret name in text, `secret_field` for a string under a secret name in structured input or output, `env:<NAME>` for an environment variable's value, or `custom` for a `redactPatterns` rule. `field` is the event field the text was in: `prompt`, `output`, `tool_input`, `tool_output`, `tool_results`, `tool_calls`, and so on, `input_messages` for the messages a request replays from a transcript, or `history` for kept earlier-turn messages redacted again while this request was built. Each layer records what it replaces as it replaces it; nothing is read back from placeholders. Left out when the count is 0. |
| `judgment.redacted_reason` | string | `sensitive_path` when the turn was blanked. |
| `judgment.redacted_rule` | string | The rule that blanked it: a path rule (`dotenv`, `ssh_private_key`, `key_file`, `credentials_file`, `secret_file`, `auth_file`, `proc_environ`, or `custom_path`) or a command rule (`env_dump`, `printenv`, `secret_cli`, or `custom_command`). Command rules are checked first. `redaction_limit` when the turn held more to check than the limits allow. |
| `judgment.redacted_trigger` | string | The path token that blanked it, such as `.env` or `~/.aws/credentials`, with the home folder as `~` and any secret in it redacted. For a command, `cmd:<rule>`, such as `cmd:env_dump`. |
| `judgment.redaction_error` | bool | A rule failed, such as an invalid `redactPatterns` regex, or an event needed more redaction than the limits allow, so the affected events were sent without text. A failed rule also drops the earlier turns' text that would ride along as context. |

A child span whose input or output holds placeholders carries its own `judgment.redaction_count`, the number of `[REDACTED:` placeholders in that span's text.

A blanked turn has the same spans, span ids, token counts, cost, and timing as it would unblanked, and every `judgment.input` and `judgment.output` is empty. A model span's `judgment.llm.tool_calls` keeps only each call's id and name. A blanked turn is never sent as context in a later turn's model input, and the next turn is not blanked.

With `capture` set to `full` the root has none of these attributes. With `metadata` it has `judgment.redacted` and `judgment.redaction_count`.

What each rule catches, the `.judgment/tracing.json` fields that add or turn off rules, and the known limits are in the README, under [Secrets are redacted by default](../README.md#secrets-are-redacted-by-default) and [Known limits](../README.md#known-limits). A rule id in these attributes is the `id` in `core/bin/redact-rules.json`.

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

A model span's input is the messages array the model was sent, whole and in order. Nothing is truncated.

- **Claude Code and Codex** read it from the agent's transcript: every earlier message since the session started or was last compacted, including earlier turns' answers, reasoning, tool calls, and results. After a compaction the input restarts from the compacted history. Codex encrypts its compaction summary, so the start of that history holds the user message `[Earlier conversation compacted by Codex; summary is encrypted by OpenAI]` in its place.
- **Claude Code** requests logged in the transcript are replayed exactly, with the system prompt in `gen_ai.system_instructions` and the tool list in `gen_ai.tool.definitions`. **Codex** sets `gen_ai.system_instructions` from the session's base instructions, and `gen_ai.tool.definitions` when the rollout lists its tools.
- **Cursor, Pi, Hermes Agent**, and any turn without a usable transcript: every message the session's earlier turns added, then the user prompt, then each tool call and result that came before the request. Those messages are kept next to the turn's file (`<turn>.messages.json`) after it is sent, already redacted under `sanitized`, and are never kept under `metadata` or for a blanked turn.

A tool result is the text the model read, not the agent's raw tool response. The output is the assistant message, including reasoning and tool calls, and, for Claude Code, the stop reason as `finish_reason`. Reasoning is the readable text only. Encrypted reasoning, redacted thinking, and signatures are never sent.

The system prompt, the tool definitions, and the earlier messages are redacted like any other text under `sanitized`, and dropped under `metadata`. A blanked turn sends none of them. After a turn of the session is blanked, later turns no longer replay the transcript, which would hold the blanked turn's text: their input is built from their own events and the messages kept for unblanked turns.

A span larger than the server accepts is dropped alone, and the rest of the turn is still sent.

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

Secrets are redacted twice: by the hook, before an event reaches the queue, and by the sender, after it reads transcript text. Both use the rules in `core/bin/redact-rules.json`, joined with the project's own from `.judgment/tracing.json` the same way. The sender checks for a sensitive file before it redacts, so a placeholder in a file name cannot hide it. A turn that touched a sensitive file is saved blank before it is sent, so its text does not stay on disk. The turn is blanked as soon as the sender stores the event that shows it, even a late subagent's event for a turn already sent: its kept messages are deleted and its saved text stripped before anything waits or is sent, and later events of that turn are stored without text. A turn with events not yet sent is not used as a later turn's history, since those events could still blank it.

The hook starts the sender in the background and waits, at most about 40 ms, until `/proc` shows it outside the hook's process group, so an agent that kills that group the moment the hook returns does not kill the sender. With job control (bash, zsh, and macOS's `/bin/sh`) that holds before `&` returns. Under dash it is `setsid(1)`, usually within a millisecond; without `setsid(1)` the sender calls `setsid()` as Python starts. When no acknowledgement comes in time, a second sender is started by bash with job control. Without bash, a sender killed with the agent leaves its turns on disk for the next one to send.

A turn being sent stays on disk until the send succeeds, so a crash or a failed send retries the whole turn on the next run. A large turn goes out in several compressed requests, with the root last. A turn the server rejects as invalid is skipped, and a request it rejects as too large is dropped alone. Both are noted in `flush.log` and never hold back later turns.
