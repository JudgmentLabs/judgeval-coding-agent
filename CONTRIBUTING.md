# Contribute to judgeval-coding-agent

You can contribute in these ways:

- File [bug reports](https://github.com/JudgmentLabs/judgeval-coding-agent/issues/new?template=bug_report.md) and [feature requests](https://github.com/JudgmentLabs/judgeval-coding-agent/issues/new?template=feature_request.md).
- Open a [pull request](https://github.com/JudgmentLabs/judgeval-coding-agent/pulls) with a fix or an improvement to the docs.
- Tell people about Judgment.

## Make a change

The shared code lives in `core/`. Each plugin carries a copy of it under `<agent>/vendor/judgment-core`, because an agent installs only that plugin's folder.

1. Edit `core/`, or the agent's own folder (`claude-code/`, `codex/`, `cursor/`, `pi/`).
2. Copy `core/` into every plugin:

   ```bash
   sh scripts/sync-core.sh
   ```

3. Try the change in the agent you touched, against a Judgment project of your own. Each agent's README has a "Try it from the CLI" section.
4. Open a pull request against `main`. CI checks that every vendored copy matches `core/` and that every manifest has the same version.

A change to `codex/hooks/hooks.json` or `claude-code/hooks/hooks.json` makes Codex users approve the hooks again. Say so in the pull request.

## Release

Maintainers release by setting one version in every manifest, then merging to `main`:

```bash
sh scripts/version.sh <version>
```

Claude Code offers an update only when the version changes. Codex and Pi follow `main`. CI fails when the manifests disagree. It warns when a Codex or Claude Code hook definition changes, because Codex users must trust a changed hook again.

[![Contributors](https://contributors-img.web.app/image?repo=JudgmentLabs/judgeval-coding-agent)](https://github.com/JudgmentLabs/judgeval-coding-agent/graphs/contributors)
