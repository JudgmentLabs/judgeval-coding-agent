<div align="center">

<a href="https://judgmentlabs.ai/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_darkmode.svg">
    <img src="assets/logo_lightmode.svg" alt="Judgment Logo" width="400" />
  </picture>
</a>

<br>

## Trace your coding agents in Judgment

Claude Code, Codex, Cursor and Pi sessions, one trace per turn.

[![Docs](https://img.shields.io/badge/Documentation-orange)](https://docs.judgmentlabs.ai/documentation)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE.md)

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)

</div>

## Overview

These plugins send coding-agent sessions to [Judgment](https://judgmentlabs.ai/). A session is the set of traces from one conversation. Each user turn becomes one trace, with the prompt, the model calls, and the tool calls inside it.

Tracing stays off until the project opts in. A hook writes the event to a local spool and returns. A separate process sends the spans after that. If that send fails, the agent keeps running and the spool retries later.

## Install

The plugins run on macOS and Linux. The hooks need `jq` and `python3` (3.9 or later) on `PATH`. Without either one, the hooks still return and nothing is traced.

1. Save your credentials in `~/.config/judgment/credentials`, then run `chmod 600` on that file. See [Credentials](#credentials) for every field.

   ```text
   JUDGMENT_API_KEY=<your-api-key>
   JUDGMENT_PROJECT_ID=<your-project-id>
   ```

2. Install the plugin for each agent you run. Each agent installs and updates it through its own plugin system. The setup pages have the details.

   | Agent | Install | Update | Setup page |
   | --- | --- | --- | --- |
   | Claude Code | `claude plugin marketplace add JudgmentLabs/judgeval-coding-agent`, then `claude plugin install judgment-tracing@judgment-tracing` | `claude plugin marketplace update judgment-tracing && claude plugin update judgment-tracing@judgment-tracing` | [claude-code/README.md](claude-code/README.md) |
   | Codex | `codex plugin marketplace add https://github.com/JudgmentLabs/judgeval-coding-agent.git`, then `codex plugin add judgment-tracing@judgment-tracing`, then trust the hooks when Codex starts | Automatic when a session starts | [codex/README.md](codex/README.md) |
   | Cursor | One copy command into `~/.cursor/plugins/local` | Run the copy command again | [cursor/README.md](cursor/README.md) |
   | Pi | `pi install git:github.com/JudgmentLabs/judgeval-coding-agent` | `pi update --extensions` | [pi/README.md](pi/README.md) |

3. Turn tracing on in a project, from its root:

   ```bash
   mkdir -p .judgment && printf '{"enabled": true}\n' > .judgment/tracing.json
   ```

   [Turn tracing on for a project](#turn-tracing-on-for-a-project) lists the other settings.

4. Start a new agent session in that project and send a prompt. The trace appears in the project's Traces view a few seconds after the turn ends.

## Move from an earlier install

If you installed tracing before this repository was public, from a local clone or an earlier marketplace, remove that install before you add this one. With both in place, every event is recorded twice. An earlier marketplace with the same name also blocks the new one from being added.

| Agent | Remove the earlier install |
| --- | --- |
| Claude Code | `claude plugin uninstall judgment-tracing@judgment`, then `claude plugin marketplace remove judgment`. |
| Codex | If you added an earlier marketplace, run `codex plugin remove judgment-tracing@judgment-tracing`, then `codex plugin marketplace remove judgment-tracing`. Delete any Judgment entries from `~/.codex/hooks.json`, and from `.codex/hooks.json` in projects where you added them. Keep your other hooks. |
| Cursor | Uninstall any Judgment tracing copy installed from a marketplace, under Customize. Delete Judgment entries from `.cursor/hooks.json` in projects that used project hooks. The install command replaces the local plugin folder. |
| Pi | Remove the earlier source with `pi remove`, using the path or `git:` source you installed from. Add `-l` for project installs. |

## Turn tracing on for a project

Tracing is on in a directory when `.judgment/tracing.json` is in it or in a parent of it. The file looks like this:

```json
{
  "enabled": true,
  "capture": "full",
  "sessionDefault": "on"
}
```

| Field | Values | What it does |
| --- | --- | --- |
| `enabled` | `true`, `false` | `false` turns the file off without deleting it. |
| `capture` | `full`, `metadata` | `full` stores prompt, response, and tool text. `metadata` keeps names, models, token counts, cost, timing, and ids. It drops prompts, answers, reasoning, tool arguments and results, and subagent tasks from the spool, the spans, and the saved session history, including text read back from the agent's transcript. |
| `sessionDefault` | `on`, `off` | `off` leaves each new session untraced until you turn that session on. |

`sessionDefault` set to `off` is how a long traced session and a side chat share one project. The side chat stays quiet until you turn it on.

Add `.judgment/` to `.gitignore`. The file stays on this machine. It holds no API key.

The `judgment-trace` command manages this file and checks your setup. It runs from a clone of this repository:

```bash
git clone https://github.com/JudgmentLabs/judgeval-coding-agent.git ~/judgment-tracing
export PATH="$HOME/judgment-tracing/bin:$PATH"
```

| Command | What it does |
| --- | --- |
| `judgment-trace enable` | Writes the file above, and adds `.judgment/` to `.git/info/exclude`. |
| `judgment-trace disable` | Removes the file. |
| `judgment-trace status` | Prints the mode Judgment uses from this directory. |
| `judgment-trace doctor` | Reports a missing tool, a missing credential, and whether tracing is on here. |

## Credentials

Judgment reads the API key from the environment, or from `~/.config/judgment/credentials` when a variable is unset. The environment wins. Use one `KEY=value` per line in that file, and run `chmod 600` on it.

| Variable | Required | Purpose |
| --- | --- | --- |
| `JUDGMENT_API_KEY` | Yes | Ingest API key. Create one under **Settings** > **Ingest API Keys**. |
| `JUDGMENT_PROJECT_ID` | Yes | Project ID. A project name is not accepted. |
| `JUDGMENT_ORG_ID` | Personal keys | Organization ID. An ingest key does not need it. |
| `JUDGMENT_API_URL` | No | API base URL. The default is `https://api.judgmentlabs.ai`. |
| `JUDGMENT_TRACING` | No | Set to `0` to turn tracing off in every project. |

Copy the project ID from the project URL in Judgment, or press Ctrl-K and search for the project ID. See [Send OTLP directly](https://docs.judgmentlabs.ai/documentation/connect/integrations/send-otlp-directly).

## Session command

Inside an opted-in project, turn tracing on or off for the current conversation only.

| Agent | Command |
| --- | --- |
| Claude Code | `/judgment:on`, `/judgment:off`, `/judgment:status` |
| Codex | `$judgment on`, `$judgment off`, `$judgment status` |
| Cursor | `/judgment on`, `/judgment off`, `/judgment status` |
| Pi | `/judgment on`, `/judgment off`, `/judgment status` |

Other conversations in the project keep the project default.

## What gets stored

One user request and its answer are one trace. The root span is named `<harness> turn`, for example `claude-code turn`. Its kind is `agent`, and its `judgment.input` and `judgment.output` are the prompt and the final answer as plain text. Model, tool, and subagent spans sit under that root. A follow-up request is a new trace.

Every span carries `judgment.session_id` and `gen_ai.conversation.id`. Judgment groups traces into a session from the root's `judgment.session_id`. Model spans set `judgment.span_kind` to `llm`. Their `judgment.input` is a messages array: the earlier turns of the session, then the user prompt, then any tool call and tool result that came before that request. Each earlier turn is its prompt and final answer. The newest turns are kept up to 8 KB in all, and each message is cut at 2 KB. A tool result is the text the model read, not the agent's raw tool response. An input stays under 64 KB: past that, it keeps the prompt and the newest messages whole and notes how many earlier messages it left out. `judgment.output` is the assistant message, including reasoning and tool calls. A subagent's agent span holds its task and answer as plain text when it has model spans, and as a one-message array each when it has none. Tool spans set `judgment.span_kind` to `tool`, with the arguments as input and the result as output. A tool span ends when the tool finished.

Cursor reports tokens once per stop, not per model request. Its model span covers the whole stop: the tools that ran and their results are in the input, and the final text is the output.

A background subagent that finishes after the turn is exported again with the same span ids, a later end time, and `judgment.update_id`. The root is included so its start and end cover every child. The setup page for each agent states which token fields that agent exposes.

Each turn records the project ID, API URL, and a fingerprint of the API key that were set when the hook ran. It doesn't record the key itself.

A flusher sends a turn only when every line of it carries the same fingerprint, the fingerprint matches the flusher's own key, and it sends the turn to the recorded project. A turn recorded under a different key stays queued until a flusher with that key runs. One flusher runs at a time. While it runs, one more flusher per key waits for it, so a busy flusher for one key never strands another key's turn.

A turn with a line recorded without credentials, or with lines recorded under two keys, has no single project to go to, so it is never sent. It is deleted a day after its last event, and `flush.log` notes the deletion.

The spool and the flusher's files are readable only by your user. A turn being sent stays on disk until the send succeeds, so a crash or a failed send retries the whole turn on the next run. A large turn goes out in several compressed requests, with the root last. A turn the server rejects as invalid or too large is skipped and noted in `flush.log`, so it never holds back later turns.

A turn waits until its subagents finish, for up to 10 minutes. A turn that never ends, because the agent was killed mid-turn, is sent as it stands after a day with no new events.

## If traces do not appear

Run `judgment-trace doctor` if you have it. Confirm `.judgment/tracing.json` is in the directory where the agent started, or in a parent of it. Confirm `JUDGMENT_TRACING` is unset. Confirm the session command is not off. Confirm the plugin is installed for the agent you run, and start a new session after installing it.

Then read `flush.log` in the state directory, `~/.local/state/judgment` by default (`$XDG_STATE_HOME/judgment` when that is set). It records failed sends and skipped turns, never prompt text.

Spans show up in the project's Traces view. A trace can take a short time to appear after the turn ends.

## Release a new version

Set the version in every plugin manifest, then merge to `main`:

```bash
sh scripts/version.sh 0.3.0
```

Claude Code offers an update only when the version changes. Codex and Pi follow `main`. CI fails when the manifests disagree, and warns when a Codex or Claude Code hook definition changes, because Codex users must trust a changed hook again.

## Contributing

Bug reports, feature requests, and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache 2.0. See [LICENSE.md](LICENSE.md).
