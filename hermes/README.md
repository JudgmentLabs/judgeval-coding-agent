# Hermes Agent

This plugin sends a Hermes Agent session to Judgment when the project opts in. It's a native Hermes plugin that uses only read-only observer hooks. It never blocks, changes, or delays a prompt, a tool call, or a model call. Install it, then follow the project setup in the repository README.

It's tested with Hermes Agent 0.19.0 and 0.21.3. It runs in Hermes's own Python and doesn't need `jq`.

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

Pull the clone, then replace the plugin folder with a fresh copy. Copying onto the existing folder would nest the new copy inside the old one.

```bash
git -C ~/judgment-tracing pull
rm -rf ~/.hermes/.judgment-tracing.new && cp -R ~/judgment-tracing/hermes ~/.hermes/.judgment-tracing.new && rm -rf ~/.hermes/plugins/judgment-tracing && mv ~/.hermes/.judgment-tracing.new ~/.hermes/plugins/judgment-tracing
```

If you linked the clone instead, `git pull` is enough. The plugin stays enabled. Start a new Hermes session afterwards.

## Uninstall

```bash
hermes plugins disable judgment-tracing
hermes plugins remove judgment-tracing
```

`remove` deletes the plugin folder. Your `.judgment/tracing.json` files and the local queue in `~/.local/state/judgment` stay.

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

When Hermes compresses a long conversation, it gives the session a new id, either mid-turn or just before the turn's first model call. The plugin keeps the rest of the conversation in the same Judgment session, and `/judgment on` or `off` carries over to the new id. For a compression before the first model call, it reads the old id from the turn id Hermes built before compressing. That covers a conversation turned off in an earlier Hermes run, too. A manual `/compress` between turns looks the same to a plugin as `/resume` or `/branch`, so the conversation continues as a new Judgment session.

## How it records

Each callback turns Hermes's arguments into events and appends them to one in-memory queue. It doesn't touch a file or a pipe, or wait on another thread. A writer thread hands the queue to one helper process per Hermes process. The helper redacts each event under the project's `capture` setting and writes it to the same local queue as the other plugins, in `~/.local/state/judgment/inbox`. Text crosses only the pipe, which is in memory. Under `sanitized` and `metadata` capture, it reaches disk only after redaction. Under `"capture": "full"`, the helper writes text to the inbox unredacted, only cut at 64 KB. Redaction runs in the helper, not in Hermes, because a Python regex holds the interpreter lock for its whole scan. Done in Hermes, it would stall Hermes's own threads by several milliseconds per 56 KB secret-dense output.

Each callback took under 1 ms at p95 in the plugin's benchmark (`tests/bench_hermes.py`). That includes a 56 KB tool output holding 1,000 fake credentials, 8 threads at once, a stopped helper, a queue fuller than the helper can drain, and structured tool arguments up to 329 KB and 10,000 rows.

Hermes may change a tool's arguments after a callback returns, so the plugin keeps a copy, not a reference. It measures the arguments in one pass against a budget of 8,192 members and 512 KB, and stops as soon as the budget is spent. Only arguments within the budget are copied. Larger arguments are dropped whole and their turn is blanked, because their sensitivity can't be checked in time. A model's tool-call arguments arrive as JSON text, which needs no copy: the helper parses them.

The queue holds at most 64 MB. Past that, a new event is dropped whole and counted in `flush.log`, and the callback still returns at once. The helper acknowledges each event once it's written to the inbox. If the helper dies, a new one starts and gets every event that wasn't acknowledged, and each is written exactly once. When Hermes exits, it waits at most 0.5 s to hand the queue to the helper. The helper runs in its own session, so it keeps running after Hermes exits or is killed, and finishes what it was handed. When a turn, a subagent, or a session ends, it starts the sender, `python3 -m judgment_core.flush`, in its own session with output to `flush.log`, and doesn't wait for it. A forked Hermes process starts with an empty queue of its own.

The helper redacts each value whole: it checks a tool call for a sensitive file on the complete command, redacts, and only then cuts a text longer than 64 KB, dropping the last partial word. A single value larger than 512 KB, or past 1 MB in one event, is too large to redact within the helper's budget. It's dropped whole, never cut. If it's a prompt or a tool's input, the turn is blanked, as for a `.env` read, because its sensitivity can't be checked. Dictionary keys are redacted by the core. Names a model or provider chose, such as the model, provider, tool name, and child role, are redacted like text. A tool call id that redaction would change is replaced by `call_redacted_<n>`, distinct within its turn, in both the model call and the tool call. The project folder and `environment` label are redacted too. If the destination stamp would change, it's dropped and the turn isn't sent.

The plugin uses the copy of the core in `hermes/vendor/judgment-core`, which `scripts/sync-core.sh` refreshes. `hermes/core_bridge.py` is the only file that calls into it. It lists every core function it calls in `CONTRACT`, and `tests/test_hermes_round3.py` checks the vendored core against that list. When the core changes, that test names the function that moved.

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
