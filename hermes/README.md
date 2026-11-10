# Hermes Agent

This plugin sends a Hermes Agent session to Judgment when the project opts in. It's a native Hermes plugin that uses only read-only observer hooks. It never blocks, changes, or delays a prompt, a tool call, or a model call. Install it, then follow the project setup in the repository README.

It's tested with Hermes Agent 0.19.0. It runs in Hermes's own Python and doesn't need `jq`.

## Install

Copy the plugin into your Hermes plugins folder, then enable it:

```bash
git clone https://github.com/JudgmentLabs/judgeval-coding-agent.git ~/judgment-tracing
mkdir -p ~/.hermes/plugins
cp -R ~/judgment-tracing/hermes ~/.hermes/plugins/judgment-tracing
hermes plugins enable judgment-tracing --no-allow-tool-override
```

The plugin registers no tools, so it never needs the tool-override grant. To follow the clone instead of a copy, use `ln -s ~/judgment-tracing/hermes ~/.hermes/plugins/judgment-tracing`. With `HERMES_HOME` set, the plugins folder is `$HERMES_HOME/plugins`.

Hermes loads user plugins in the CLI, the TUI, and the gateway. It doesn't load a project's `.hermes/plugins/` in the gateway, so install the plugin in your user folder.

## Update

Run the copy command again, or `git pull` in the clone if you linked it. Start a new Hermes session afterwards.

## Turn a session on or off

```text
/judgment on
/judgment off
/judgment status
```

The change applies from your next message, so a turn is never half recorded.

`on` and `off` work in the interactive CLI, which runs one conversation. In the gateway and the TUI, Hermes doesn't tell a plugin command which conversation sent it, so `/judgment on` and `off` change nothing there and say so. `/judgment status` shows the project setting. Use `sessionDefault` in `.judgment/tracing.json` for those sessions.

## Which sessions are traced

The plugin looks for `.judgment/tracing.json` in the folder Hermes works in for that turn, then in each parent folder. It asks Hermes the way Hermes's own tools do: a folder registered for the task (TUI, desktop, ACP), then the session's folder in a multi-session gateway, then `TERMINAL_CWD`, then the folder Hermes started in. Without one, the session isn't traced and the plugin writes no file. That includes gateway sessions, such as Telegram, Discord, or mobile, whose folder isn't inside an opted-in project. `JUDGMENT_TRACING=0` turns tracing off everywhere.

## What this plugin can see

| Span | Hermes hooks | Fields |
| --- | --- | --- |
| `hermes turn` (root) | `pre_llm_call` starts it. `post_llm_call` gives the answer. `on_session_end` ends it. | The prompt (`user_message`) and the final answer (`assistant_response`). |
| Model call | `pre_api_request`, then `post_api_request` or `api_request_error` | Model, provider, start and end, token counts (`input_tokens`, `output_tokens`, `cache_read_tokens`, `cache_write_tokens`, `reasoning_tokens`), the assistant text and tool calls. A failed attempt is its own model span with error status and the error message. Its retry is the next span. |
| Tool call | `pre_tool_call`, then `post_tool_call` | Tool name, arguments, result, and duration. A call that's `blocked`, `cancelled`, or returns an error has error status. |
| Subagent | `subagent_start`, then `subagent_stop` | An agent span under the turn that started it, with its goal as input and its summary as output. The child's model and tool calls nest under it. |

Hermes runs a delegation from the main agent in the background, so a child can finish after its turn ended. Its events still go to the turn that started it, and that trace is sent again with a higher `judgment.update_id`. If Hermes exits while a child is still running, the child is closed when the session finalizes, so the turn is still sent.

A turn interrupted before Hermes answers is sent with what it has, as an interrupted turn. A tool call or model call still running then is closed with error status.

Hermes doesn't report cost. Judgment prices the model calls from the provider, model, and token counts when it knows the model.

When Hermes compresses a long conversation mid-turn, it gives the session a new id. The plugin keeps the rest of the conversation in the same Judgment session, and `/judgment on` or `off` carries over to the new id. A manual `/compress` between turns looks the same to a plugin as `/resume` or `/branch`, so the conversation continues as a new Judgment session.

## How it records

Each callback turns Hermes's arguments into an event and writes it down a pipe to one helper process per Hermes process. The helper redacts each event under the project's `capture` setting and writes it to the same local queue as the other plugins, in `~/.local/state/judgment/inbox`. Text crosses only the pipe, which is in memory, and reaches disk only redacted. Callbacks make no network call, write no file, and never wait on redaction. Each callback took under 0.5 ms at p95 in the plugin's tests, including a 56 KB tool output holding 1,000 fake credentials and 8 threads at once. Set `JUDGMENT_HERMES_SPOOL=inline` to do the work in the callback instead.

The helper runs in its own session, so it keeps running after Hermes exits or is killed, and finishes what Hermes sent it. When a turn, a subagent, or a session ends, it starts the sender, `python3 -m judgment_core.flush`, in its own session with output to `flush.log`, and doesn't wait for it.

A text longer than 64 KB is redacted first and then cut at a word boundary, so a cut never leaves part of a secret. Dictionary keys, and names a model or provider chose (the model, provider, tool name, tool call id, and child role), are redacted like text.

The plugin uses the copy of the core in `hermes/vendor/judgment-core`, which `scripts/sync-core.sh` refreshes. `hermes/core_bridge.py` is the only file that calls into it.

To time the callbacks in a real session, set `JUDGMENT_HERMES_STATS` to a file path. The plugin appends each callback's duration, per hook, to that file when the session ends. It records no prompt or tool text.

## Try it from the CLI

Use a temporary folder:

```bash
mkdir /tmp/judgment-hermes && cd /tmp/judgment-hermes
echo 'hello' > README.md
export JUDGMENT_API_KEY=<your-api-key>
export JUDGMENT_PROJECT_ID=<your-project-id>
mkdir -p .judgment && printf '{"enabled": true}\n' > .judgment/tracing.json
hermes chat -q "Read README.md and tell me what it says"
```

Open the project in Judgment. You should see one trace whose root is `hermes turn`, with model spans that have token counts and a `read_file` tool span. A folder without `.judgment/tracing.json`, or a shell with `JUDGMENT_TRACING=0`, adds no trace.

`tests/e2e_hermes.py` runs the same flow against a local mock model and a local fake Judgment.
