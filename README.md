<div align="center">

<a href="https://judgmentlabs.ai/">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo_darkmode.svg">
    <img src="assets/logo_lightmode.svg" alt="Judgment Logo" width="400" />
  </picture>
</a>

<br>

## Trace your coding agents in Judgment

Claude Code, Codex, Cursor, Pi and Hermes Agent sessions, one trace per turn.

[![Docs](https://img.shields.io/badge/Documentation-orange)](https://docs.judgmentlabs.ai/documentation)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE.md)

[![X](https://img.shields.io/badge/-X/Twitter-000?logo=x&logoColor=white)](https://x.com/JudgmentLabs)
[![LinkedIn](https://custom-icon-badges.demolab.com/badge/LinkedIn%20-0A66C2?logo=linkedin-white&logoColor=fff)](https://www.linkedin.com/company/judgmentlabs)

</div>

## Overview

These plugins send coding-agent sessions to [Judgment](https://judgmentlabs.ai/). A session is the set of traces from one conversation. Each user turn becomes one trace, with the prompt, the model calls, and the tool calls inside it.

Tracing stays off until a project opts in. A hook writes each event to a local queue and returns, so the agent never waits on the network. A background process sends the turn after it ends. If a send fails, the agent keeps running and the send is retried later.

## Install

The plugins run on macOS and Linux. The hooks need `jq` and `python3` (3.9 or later) on `PATH`. Without either one, the hooks still return and nothing is traced. Pi without `jq` records each event's ids and timing, and none of its text. The Hermes plugin runs in Hermes's own Python and doesn't need `jq`.

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
   | Hermes Agent | Copy `hermes/` to `~/.hermes/plugins/judgment-tracing`, then `hermes plugins enable judgment-tracing --no-allow-tool-override` | `git pull` in the clone, then replace the plugin folder as [hermes/README.md](hermes/README.md#update) shows | [hermes/README.md](hermes/README.md) |

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
| `redactPatterns` | regexes | More text to redact. A match becomes `[REDACTED:custom]`. When the regex's first group is named `p`, as in `(?<p>ticket=)[0-9]+`, the text `p` matched stays before the placeholder. |
| `redactPaths` | globs | More files whose turns are blanked, such as `"secrets/*"`. `*` matches any characters, `/` included. The rule id is `custom_path`. A glob after `!`, such as `"!.env.test"`, keeps the files it matches from blanking a turn, under any path rule. |
| `redactCommands` | regexes | More commands whose turns are blanked. They match where a command starts, as the built-in ones do. The rule id is `custom_command`. |
| `redactNames` | names | More secret names, such as `"pin"`. A value after one is redacted as a value after `password` is. Write lowercase parts joined by `_`: `vault_pin` also matches `VAULT_PIN`, `vault-pin`, and `vaultPin`. A built-in name after `!`, such as `"!token"`, is no longer a secret name. |
| `redactAllow` | strings | Text that is never redacted, such as `"changeme"`. A find whose redacted part is exactly one of these stays. |
| `redactDisable` | rule ids | Built-in rules to turn off, such as `"secret_assignment"`, `"github_token"`, `"dotenv"`, or `"env_dump"`. |

Each `redact*` field is a list of strings. A setting that cannot work makes every event in the project lose its text: a `redactPatterns`, `redactPaths`, `redactCommands`, or `redactNames` value that is not a list of strings, a regex that does not compile, a name that is not letters and digits joined by `_`, a `!name` or an id that no built-in rule has. In `redactAllow` and `redactDisable`, anything that is not a string is ignored. A turn already recorded also loses its text when the sender can no longer read the file, because it was deleted or cut off: the project's rules are then unknown. `judgment-trace status` says when a setting cannot work.

A regex runs in both Oniguruma (the hook's jq) and Python's `re` (the sender), so write one both accept: a named group is `(?<p>...)`, not `(?P<p>...)`, and a POSIX class such as `[[:alpha:]]` is refused. A regex runs over all of an event's text joined into one, each string on its own line, so `^` and `$` match only at the start and end of that whole text, not of a field or a line. Leave them out. This file adds one name, keeps one placeholder, and turns off one path rule:

```json
{
  "enabled": true,
  "redactNames": ["pin"],
  "redactAllow": ["changeme"],
  "redactDisable": ["key_file"]
}
```

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
| Hermes Agent | `/judgment on`, `/judgment off`, `/judgment status` |

Other conversations in the project keep the project default.

## What gets sent

Each trace carries the following:

- The agent's name and version, the model names, token counts, cost, timing, and session and turn ids.
- The tool names, and the name of the folder the agent runs in. The full path stays on your machine, because it holds your account name. The name is left out when the agent runs in your home folder.
- With `capture` set to `sanitized` or `full`, the text: prompts, answers, reasoning, every model call's messages, tool arguments and results, and subagent tasks.

### Secrets are redacted by default

With `capture` set to `sanitized`, the default, secrets are removed from the text twice: by the hook, before an event is written to the local queue, and again by the sender, after it reads text from the agent's transcript. Everything else is sent as it is.

- **Known formats** become `[REDACTED:<rule>]`: OpenAI and Anthropic `sk-` keys, AWS access key ids, GitHub tokens, Slack tokens and webhooks, Google API keys, Stripe keys, JWTs, `Bearer` and `Authorization` header values, private key blocks, and passwords in URLs. A URL password that is a whole placeholder, such as `${DB_PASS}`, `****`, or `password`, stays.
- **A value after a secret name** becomes `[REDACTED:secret_assignment]` in text. The names are `password`, `passwd`, `pwd`, `passphrase`, `secret`, `secret_key`, `client_secret`, `token`, `api_key`, `access_key`, `private_key`, `credential`, and `credentials`, each part lowercase, UPPERCASE, or Capitalized, with `_`, `-`, `.`, or nothing between the parts (`API_KEY`, `apiKey`). A name also counts as the end of a longer one (`db_password`, `smtpPassword`). The name is followed by an optional closing quote, backtick, or Markdown `**`, then `:` or `=`, then the value, which may be quoted with `"`, `'`, or a backtick. So `**Password:** hunter2` loses its value too. Only the value goes: `password: "hunter2"` becomes `password: "[REDACTED:secret_assignment]"`. A quoted value runs to its closing quote, a bare one to the next blank. A value that is a type or a constant, such as `str`, `Optional[str`, `None`, `null`, or `true`, stays: `password: str` and `token = None` are not changed. In a tool's structured input or output, a string under a key that ends in one of these names becomes `[REDACTED:secret_field]`, as in `{"DB Password": ...}` or `{"spring:datasource:password": ...}`. `key`, `auth`, and `authorization` are not names: a value after them goes only when it has a known format.
- **Your environment's secrets**: the value of every environment variable whose name holds `KEY`, `TOKEN`, `SECRET`, `PASSWORD`, `PASSWD`, `CREDENTIAL`, `PRIVATE`, or `AUTH`, and your Judgment API key, becomes `[REDACTED:env:<NAME>]` wherever it appears: the whole value, and each line of a value that spans lines, whatever its shape. Names that end in a location or setting suffix such as `_PATH`, `_FILE`, `_DIR`, `_SOCK`, `_URL`, `_HOST`, or `_ID` are left out. Values are replaced in text and keys, never inside a JSON number. The hook redacts with the environment of the agent it runs in, and the sender may have been started by another agent. So where the hook replaced an environment value, the sender keeps the hook's text over the transcript's. When the sender does not hold a variable the hook replaced, it sends no text read from the transcript for that turn.
- **Sensitive files and commands blank the turn.** A turn is sent blank when a tool's path, command, or patch input names one of these files: a name that starts with `.env` or ends in `.env` (but not `.env.example`, `.env.sample`, `.env.template`, or `example.env`), such as `.envrc` or `prod.env`, an SSH private key (`id_rsa`, `id_ed25519`), `*.pem`, `*.key`, `*.p12`, `*.pfx`, a `credentials` or `*credentials.json` file, a `secret` or `secrets` file such as `secrets.yaml` (source files such as `secrets.py`, `secrets.sh`, or `secrets.sql` excepted), `*.tfvars`, `.netrc`, `.npmrc`, `.pypirc`, `.git-credentials`, `auth.json`, `~/.docker/config.json`, `~/.kube/config`, `kubeconfig`, or `/proc/*/environ`. A file is named by its last path component, so `git show HEAD:.env` and a Codex patch's `*** Update File: .env` count. A turn is also sent blank when a command dumps the environment (`env`, `printenv`, `set`, or `export` with no arguments, `export -p`, `declare -x`, `declare -p`) or prints a stored credential: `gh auth token`, `aws configure get` or `export-credentials`, `kubectl get secret`, `kubectl describe secret`, `gcloud auth print-access-token`, `op read`, `op item get`, `security find-generic-password`, `vault kv get`, `vault read`, `heroku config`, and the other entries of `sensitive.commands` in `core/bin/redact-rules.json`. A command counts at the start of the command text, or after `;`, `&`, `|`, `(`, `{`, a backtick, or a newline. It still counts behind `then`, `do`, `else`, `time`, `exec`, `command`, `nohup`, or `!`, behind assignments such as `FOO=1`, behind `sudo` and its short options, inside `bash -c`, `sh -lc`, or `/bin/sh -c`, and with a directory before it, as in `/usr/bin/env`. A redirect after `env`, as in `env 2>&1`, is no argument. A command with `--help`, `-h`, or `--version` before the next separator does not count. A blank turn keeps its spans, ids, token counts, cost, and timing, and drops its text and its agent and subagent names.

Each redaction is recorded on the turn's root span: which rule, in which field, how many times, and for a blank turn, the path or command that caused it. The secret itself, and any hash of it, is never recorded. [Trace format](docs/trace-format.md#redaction) lists the attributes. Add, trim, or turn off rules in `.judgment/tracing.json` ([Turn tracing on for a project](#turn-tracing-on-for-a-project)). If a rule fails, or an event needs more redaction than the limits allow, the event's text, labels, and tool names are dropped instead of sent. The limits are more than 128 different secrets in one event at the hook, more than 4,096 in the text the sender reads, a find longer than 8 KB (one can fail from 8 KB up, depending on where it starts), and JSON nested in a string more than four levels deep.

### Known limits

Redaction finds secrets by format, by a short list of names, and by a short list of files and commands. It does not read code or shell syntax, so it misses some secrets and removes some text that is not secret. A case below is a documented limit, not a defect. The id after each one tags the matching rows in `tests/regressions.json`.

Not caught:

- A quoted value is redacted up to its closing quote, a newline, a comma, or a backslash. A bare value is redacted up to a blank, a quote, a comma, `;`, `)`, `]`, `}`, or a backslash. The rest of a value that holds one of these stays: the second line of a multiline password, or the part after a backslash. A value that starts with one of these characters, or a bare value that starts with `=`, `[`, `{`, `(`, or `:`, is not redacted at all. A default after a type annotation is not read either: in `password: str = "hunter2"` the value is the type `str`, which stays, and so does the default. (`value-stops`)
- A value given without `:` or `=`, such as `--password hunter2`, is not redacted. (`space-separated`)
- A value after `key`, `auth`, or `authorization`, in text or as a structured member, goes only when it has a known format. `"Authorization": "Basic ..."` in structured input stays, as does an `Authorization: Basic` or `Bearer` value with no digit, `+`, `=`, or `/` in it, and the password in `curl -u user:pass`. (`key-auth-unformatted`)
- A sensitive command behind a wrapper the command prefix does not read is not seen: `timeout 5 env`, `env -u X`, `eval 'env'`, `FOO="a b" env`, or `sudo --user=root env`. The secret-named environment values in its output are still redacted. (`wrapped-command`)
- A command followed by `-h` is taken to ask for help, even where `-h` means something else: `gh auth token -h github.com` does not blank the turn. The `github_token` rule still redacts the token it prints. (`command-word-order`)
- A command with options between its words, such as `gh auth -t status`, is not seen as that command. (`kubectl get -o yaml secret db` still blanks the turn, because `secret` is a sensitive file name.) (`command-word-order`)
- A command not in the list does not blank the turn, even when it prints a credential, such as `gh auth status --show-token`. A format rule may still redact what it prints. (`unlisted-cli`)
- A secret shorter than 8 characters in an environment variable is not replaced, because replacing every `2024` in all text would destroy ordinary text. Keep short secrets out of the environment, or add a `redactPatterns` rule.
- A sender that holds the same environment variable name with a different value cannot tell the values apart.
- A `redactPatterns`, `redactCommands`, or `redactPaths` entry runs as written, so one with catastrophic backtracking, such as `(a+)+$`, can stall the sender on matching text and hold back the turns after it. The built-in rules do linear work (`tests/test_latency.py`).

Removed although not secret:

- A reference, placeholder, constant, or expression after a secret name is redacted like a value: `api_key = config.api_key`, `password: ${DB_PASSWORD}`, `token: ${{ secrets.GITHUB_TOKEN }}`, a type annotation other than the built-in ones (`token: TokenType`), or a call (`token = tokenizer.encode(x)`). (`reference-value`)
- A format placeholder or `$reference` right after a secret name inside a string is taken for the value: `logger.info("secret: %s", s)` becomes `logger.info("secret: [REDACTED:secret_assignment]", s)`. (`code-after-name`)

Blanked although nothing sensitive was read:

- A sensitive file name anywhere in a tool's path, command, or patch input blanks the turn, whether the tool reads it or only mentions it: `grep -rn credentials src/`, `git commit -m "rotate secrets"`, a patch whose body mentions `.env`, or `id_rsa_parser.py`. So does a sensitive command after a separator inside quoted text or on a heredoc line: `echo "foo; printenv"`. (`mention-blanks`)
- A listed command blanks the turn whatever it asks for: `aws configure get region`, or `kubectl get secret` without `-o`. (`cli-any-args`)

> [!WARNING]
> `"capture": "full"`, or `"redact": false` without a `capture`, turns redaction off. Prompts, tool output, and file contents are then sent exactly as the agent saw them, secrets included. Use it only in a project with no secrets.

> [!IMPORTANT]
> **Upgrading?** `judgment-trace enable` in earlier versions wrote `"capture": "full"` into `.judgment/tracing.json`. Such a project stays raw after you upgrade: nothing is redacted. To redact, set `"capture": "sanitized"`, delete the `capture` key, or run `judgment-trace enable` again (it rewrites the file, so add back any `redact*` fields).

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
