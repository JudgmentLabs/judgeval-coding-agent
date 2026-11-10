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

Tracing stays off until a project opts in. A hook writes each event to a local queue and returns, so the agent never waits on the network. A background process sends the turn after it ends. If a send fails, the agent keeps running and the send is retried later.

## Install

The plugins run on macOS and Linux. The hooks need `jq` and `python3` (3.9 or later) on `PATH`. Without either one, the hooks still return and nothing is traced. Pi without `jq` records each event's ids and timing, and none of its text.

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

4. Start a new agent session in that project and send a prompt. The trace appears in the project's **Traces** view a few seconds after the turn ends.

## Turn tracing on for a project

Tracing is on in a directory when `.judgment/tracing.json` is in it or in a parent of it. The file looks like this:

```json
{
  "enabled": true,
  "capture": "sanitized",
  "sessionDefault": "on"
}
```

| Field | Values | What it does |
| --- | --- | --- |
| `enabled` | `true`, `false` | `false` turns the file off without deleting it. |
| `capture` | `sanitized`, `full`, `metadata` | `sanitized`, the default, sends prompt, response, and tool text with secrets redacted. `full` sends the text raw. `metadata` sends only names, models, token counts, cost, timing, and ids. [What gets sent](#what-gets-sent) has the details. |
| `sessionDefault` | `on`, `off` | `off` leaves each new session untraced until you turn that session on. |
| `redactPatterns` | regexes | More text to redact, as `[REDACTED:custom]`. |
| `redactPaths` | globs | More files whose turns are blanked, such as `"secrets/*"`. `*` matches any characters, `/` included. |

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

## What gets sent

Each trace carries the following:

- The agent's name and version, the model names, token counts, cost, timing, and session and turn ids.
- The tool names, and the name of the folder the agent runs in. The full path stays on your machine, because it holds your account name. The name is left out when the agent runs in your home folder.
- With `capture` set to `sanitized` or `full`, the text: prompts, answers, reasoning, every model call's messages, tool arguments and results, and subagent tasks.

### Secrets are redacted by default

With `capture` set to `sanitized`, the default, secrets are removed from the text twice: by the hook, before an event is written to the local queue, and again by the sender, after it reads text from the agent's transcript. Everything else is sent as it is.

- **Known secrets** become `[REDACTED:<type>]`: OpenAI and Anthropic `sk-` keys, AWS access key ids, GitHub tokens, Slack tokens and webhooks, Google API keys, Stripe keys, JWTs, `Bearer` and `Authorization` values, private key blocks, passwords in URLs, and values assigned to names such as `password`, `api_key`, `secret`, or `token`. A value must look like a credential: `${TOKEN}`, `os.environ["API_KEY"]`, `config.api_key`, `<your-key>`, and `changeme` stay as they are. In a tool's structured input or output, the value of a member such as `"api_key"` or `"password"` becomes `[REDACTED:secret_field]`, and a key that is itself a secret is redacted too.
- **Your environment's secrets**: the value of every environment variable whose name holds `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `CREDENTIAL`, `PRIVATE`, or `AUTH`, and your Judgment API key, becomes `[REDACTED:env:<NAME>]` wherever it appears: the whole value, and each line of a value that spans lines. A value is replaced when it has 12 or more characters, or 8 to 11 that are not all lowercase letters, all capitals, or all digits, so a setting such as `keycloak` or `production` stays.
- **Sensitive files**: a turn whose tool call reads or names `.env` or `.env.*`, `.envrc`, `*.pem`, `*.key`, `*.p12`, `*.pfx`, an SSH private key, a `credentials` file such as `~/.aws/credentials`, `.netrc`, `.npmrc`, `.pypirc`, `.git-credentials`, `auth.json`, a file with `secret` in its name, or that dumps the environment with `env` (`env -0`, `/usr/bin/env`), `printenv`, `set`, or `export -p`, is sent blank. Agent and subagent names are redacted like text, and a blank turn drops them. Its spans, ids, token counts, cost, and timing stay. Its text does not. `.env.example`, `.env.sample`, and `.env.template` are not sensitive, and neither are source files such as `secret_manager.py`.

Each redaction is recorded on the turn's root span: which rule, in which field, how many times, and for a blank turn, the path or command that caused it. The secret itself, and any hash of it, is never recorded. [Trace format](docs/trace-format.md#redaction) lists the attributes. Add your own rules with `redactPatterns` and `redactPaths`. Rules you add cannot turn the built-in ones off. If a rule fails, or an event needs more redaction than the limits allow (more than 128 different secrets, for example), the event's text is dropped instead of sent.

> [!WARNING]
> `"capture": "full"`, or `"redact": false` without a `capture`, turns redaction off. Prompts, tool output, and file contents are then sent exactly as the agent saw them, secrets included. Use it only in a project with no secrets.

With `capture` set to `metadata`, the plugin keeps names, models, token counts, cost, timing, and ids. It drops the text and the folder name from the local queue, the spans, and the saved session history. That includes text read back from the agent's transcript.

Your API key is never sent anywhere but Judgment, and never saved with a turn. Each turn records a fingerprint of the key, so it goes only to the project it was recorded for. The local queue lives in `~/.local/state/judgment` (`$XDG_STATE_HOME/judgment` when that is set), readable only by your user. It holds the text as redacted, and a blank turn is saved blank.

[Trace format](docs/trace-format.md) lists every span and attribute, and how turns are queued and sent.

## If traces do not appear

1. Run `judgment-trace doctor` if you have it.
2. Confirm `.judgment/tracing.json` is in the directory where the agent started, or in a parent of it.
3. Confirm `JUDGMENT_TRACING` is unset, and that the session command is not off.
4. Confirm the plugin is installed for the agent you run, and start a new session after installing it.
5. Read `flush.log` in `~/.local/state/judgment`. It records failed sends and skipped turns, never prompt text.

A turn recorded with no credentials, or under two different API keys, has no single project to go to. It is never sent, and it's deleted a day after its last event. `flush.log` notes the deletion.

## Move from an earlier install

If you installed tracing before this repository was public, from a local clone or an earlier marketplace, remove that install before you add this one. With both in place, every event is recorded twice. An earlier marketplace with the same name also blocks the new one from being added.

| Agent | Remove the earlier install |
| --- | --- |
| Claude Code | `claude plugin uninstall judgment-tracing@judgment`, then `claude plugin marketplace remove judgment`. |
| Codex | If you added an earlier marketplace, run `codex plugin remove judgment-tracing@judgment-tracing`, then `codex plugin marketplace remove judgment-tracing`. Delete any Judgment entries from `~/.codex/hooks.json`, and from `.codex/hooks.json` in projects where you added them. Keep your other hooks. |
| Cursor | Uninstall any Judgment tracing copy installed from a marketplace, under Customize. Delete Judgment entries from `.cursor/hooks.json` in projects that used project hooks. The install command replaces the local plugin folder. |
| Pi | Remove the earlier source with `pi remove`, using the path or `git:` source you installed from. Add `-l` for project installs. |

## Contributing

Bug reports, feature requests, and pull requests are welcome. [CONTRIBUTING.md](CONTRIBUTING.md) covers making a change and releasing a version.

## License

Apache 2.0. See [LICENSE.md](LICENSE.md).
