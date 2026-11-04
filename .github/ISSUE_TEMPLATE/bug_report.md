---
name: Bug report
about: Report a tracing problem with a coding agent
title: "[BUG]"
labels: potential bug

---

## Describe the bug
What went wrong. For example, no trace appeared, a span is missing, or a prompt or tool result looks wrong.

## To reproduce
1. Agent and how you ran it (CLI, desktop app, IDE):
2. Prompt you sent:
3. What you saw in Judgment:

## Expected behavior
What you expected to see.

## Environment
 - Agent and version: [e.g. Claude Code 2.1.311, Codex 0.159.3, Cursor 3.28.7, Pi]
 - Plugin version: [the `version` in the installed plugin's manifest]
 - OS: [e.g. macOS 15, Ubuntu 24.04]
 - `jq` and `python3` versions:

## Logs
Paste the relevant lines of `flush.log` from `~/.local/state/judgment`. It records failed sends and skipped turns, never prompt text. Don't paste API keys.

## Are you interested in contributing a fix for this bug?
The Judgment community is happy to help with guidance and review on [Discord](https://discord.com/invite/tGVFf8UBUY).

- [ ] Yes
- [ ] No
