"""Redaction rules: every pattern, env values, sensitive paths, user rules, and capture modes. jq and Python must agree."""

from __future__ import annotations

import json
import unittest

from support import ENV_SECRET_NAME, SECRETS, clean_env, jq, jq_sanitize, leaks, norm, py_sanitize, secret_blob, whole_leaks

from judgment_core import optin, redact

PATTERN_CASES = {
    # rule id -> fake secret key in SECRETS
    "openai_key": "openai_key",
    "anthropic_key": "anthropic_key",
    "aws_access_key_id": "aws_access_key_id",
    "github_token": "github_token",
    "github_pat": "github_pat",
    "slack_token": "slack_token",
    "google_api_key": "google_api_key",
    "stripe_key": "stripe_key",
    "jwt": "jwt",
    "private_key": "private_key",
    "bearer_token": "bearer_token",
    "url_password": "url_password",
    "secret_assignment": "secret_assignment",
}


def tool_event(text: str, **extra) -> dict:
    return {"v": 1, "harness": "claude-code", "event": "tool", "session_id": "s", "turn_id": "t", "tool_name": "Bash",
            "tool_call_id": "c1", "tool_input": {"command": "make deploy"}, "tool_output": text, **extra}


class PatternTests(unittest.TestCase):
    def test_every_pattern_type_in_both_layers(self):
        events = [tool_event(SECRETS[key]["text"]) for key in PATTERN_CASES.values()]
        py = py_sanitize(events)
        js = jq_sanitize(events)
        self.assertEqual(norm(py), norm(js))
        for (rule, key), out in zip(PATTERN_CASES.items(), py):
            with self.subTest(rule=rule):
                self.assertNotIn(SECRETS[key]["core"], json.dumps(out))
                self.assertIn(f"[REDACTED:{rule}]" if rule != "bearer_token" else "[REDACTED:", out["tool_output"])
                self.assertTrue(out["redactions"])

    def test_assignments_in_json_shell_and_aws_forms(self):
        for key in ("json_api_key", "export_token", "aws_secret"):
            with self.subTest(form=key):
                for out in (py_sanitize([tool_event(SECRETS[key]["text"])])[0], jq_sanitize([tool_event(SECRETS[key]["text"])])[0]):
                    self.assertNotIn(SECRETS[key]["core"], out["tool_output"])
                    self.assertIn("[REDACTED:secret_assignment]", out["tool_output"])

    def test_keeps_the_text_around_a_secret(self):
        text = SECRETS["url_password"]["text"]
        out = py_sanitize([tool_event(text)])[0]["tool_output"]
        self.assertEqual(out, "postgres://admin:[REDACTED:url_password]@db.internal:5432/app")
        bearer = py_sanitize([tool_event(SECRETS["bearer_token"]["text"])])[0]["tool_output"]
        self.assertTrue(bearer.startswith("curl -H 'Authorization: Bearer [REDACTED:"))
        self.assertTrue(bearer.endswith("' https://api.example.com"))

    def test_walks_every_text_field_and_leaves_structure(self):
        secret = SECRETS["github_token"]["core"]
        event = {
            "v": 1, "harness": "pi", "event": "model", "session_id": "s", "turn_id": "t", "model": "m", "ts": 5,
            "usage": {"input_tokens": 10}, "export_fp": "abc", "tool_call_id": "c",
            "prompt": "use " + secret, "output": secret, "received": [secret], "tool_results": [{"id": "x", "response": secret}],
            "tool_calls": [{"id": "c", "name": "Bash", "arguments": {"command": "echo " + secret}}],
            "messages": [{"role": "user", "content": [{"type": "text", "text": secret}]}], "error": "bad " + secret,
        }
        for out in (py_sanitize([event])[0], jq_sanitize([event])[0]):
            self.assertEqual(whole_leaks(json.dumps(out)), [])
            for field in ("v", "harness", "event", "session_id", "turn_id", "model", "ts", "usage", "export_fp", "tool_call_id"):
                self.assertEqual(out[field], event[field])
            self.assertEqual(out["tool_calls"][0]["id"], "c")
            self.assertEqual({item["field"] for item in out["redactions"]}, {"prompt", "output", "received", "tool_results", "tool_calls", "messages", "error"})

    def test_audit_trail_has_rule_field_count_and_no_secret(self):
        events = [tool_event(secret_blob(), prompt="key " + SECRETS["openai_key"]["core"])]
        env = clean_env(**{ENV_SECRET_NAME: SECRETS["env_value"]["core"], "JUDGMENT_API_KEY": SECRETS["judgment_key"]["core"]})
        for out in (py_sanitize(events, env=env)[0], jq_sanitize(events, env=env)[0]):
            audit = out["redactions"]
            for item in audit:
                self.assertEqual(set(item), {"rule", "field", "count"})
                self.assertGreater(item["count"], 0)
            self.assertIn({"rule": "openai_key", "field": "prompt", "count": 1}, audit)
            self.assertIn({"rule": f"env:{ENV_SECRET_NAME}", "field": "tool_output", "count": 1}, audit)
            self.assertIn({"rule": "env:JUDGMENT_API_KEY", "field": "tool_output", "count": 1}, audit)
            self.assertEqual(leaks(json.dumps(audit)), [])
            self.assertEqual(leaks(json.dumps(out)), [])

    def test_redacting_twice_changes_nothing(self):
        once = py_sanitize([tool_event(secret_blob())])
        self.assertEqual(py_sanitize(once), once)
        self.assertEqual(norm(jq_sanitize(once)), norm(once))


class EnvTests(unittest.TestCase):
    def env(self, **extra):
        return clean_env(**extra)

    def test_secret_named_values_are_replaced_literally(self):
        value = "a.b*c+d(" + SECRETS["env_value"]["core"]  # regex metacharacters must not matter
        env = self.env(**{ENV_SECRET_NAME: value})
        event = tool_event("got " + value + " and again " + value)
        for out in (py_sanitize([event], env=env)[0], jq_sanitize([event], env=env)[0]):
            self.assertEqual(out["tool_output"], f"got [REDACTED:env:{ENV_SECRET_NAME}] and again [REDACTED:env:{ENV_SECRET_NAME}]")

    def test_judgment_key_always_and_names_that_are_not_secrets(self):
        keep = {
            "GIT_AUTHOR_NAME": "Pat Example Person",
            "SSH_AUTH_SOCK": "/tmp/ssh-abc/agent.1234",
            "MAX_THINKING_TOKENS": "31999999",
            "KEYBOARD_LAYOUT": "us-intl-alt",
            "AUTH_PROVIDER": "keycloak",
            "OAUTH_REDIRECT_URL": "https://example.com/callback",
            "SHORT_TOKEN": "abc",
        }
        redacted = {"JUDGMENT_API_KEY": "short123", "SERVICE_PASSWORD": "Pw-" + SECRETS["env_value"]["core"][:10], "GITHUB_TOKEN": SECRETS["github_token"]["core"]}
        env = self.env(**keep, **redacted)
        py_names = sorted(name for name, _ in redact.env_secrets(env))
        self.assertEqual(py_names, sorted(redacted))
        event = tool_event(" | ".join(list(keep.values()) + list(redacted.values())))
        for out in (py_sanitize([event], env=env)[0], jq_sanitize([event], env=env)[0]):
            for value in keep.values():
                self.assertIn(value, out["tool_output"])
            for name in redacted:
                self.assertIn(f"[REDACTED:env:{name}]", out["tool_output"])
            self.assertEqual(sorted(item["rule"] for item in out["redactions"]), sorted(f"env:{name}" for name in redacted))

    def test_names_with_odd_characters_and_casings(self):
        env = self.env(**{"npm_config_//registry.example.com/:_authToken": "npm_" + SECRETS["env_value"]["core"], "githubToken": "Gh" + SECRETS["env_value"]["core"][:12],
                          "AUTHORIZATION_HEADER": "Basic " + SECRETS["env_value"]["core"][:16], "MONKEY_ISLAND": "Guybrush123", "XAUTHORITY_X": "Cookie12345"})
        names = sorted(name for name, _ in redact.env_secrets(env))
        self.assertEqual(names, sorted(["npm_config_//registry.example.com/:_authToken", "githubToken", "AUTHORIZATION_HEADER"]))
        event = tool_event(" ".join(env[name] for name in names) + " Guybrush123 Cookie12345")
        self.assertEqual(norm(py_sanitize([event], env=env)), norm(jq_sanitize([event], env=env)))

    def test_flusher_adds_the_key_from_the_credentials_file(self):
        names = [name for name, _ in redact.env_secrets(self.env(), api_key="filekey-" + SECRETS["judgment_key"]["core"])]
        self.assertEqual(names, ["JUDGMENT_API_KEY"])


SENSITIVE = [
    ({"command": "cat .env"}, "dotenv", ".env"),
    ({"command": "source ./config/.env.local && npm start"}, "dotenv", "./config/.env.local"),
    ({"file_path": "/home/tester/proj/.env.production"}, "dotenv", "~/proj/.env.production"),
    ({"path": ".envrc"}, "dotenv", ".envrc"),
    ({"command": ["bash", "-lc", "cat ~/.aws/credentials"]}, "credentials_file", "~/.aws/credentials"),
    ({"file_path": "/home/tester/.config/judgment/credentials"}, "credentials_file", "~/.config/judgment/credentials"),
    ({"command": "cat gcp-service-account-credentials.json"}, "credentials_file", "gcp-service-account-credentials.json"),
    ({"file_path": "/home/tester/.ssh/id_ed25519"}, "ssh_private_key", "~/.ssh/id_ed25519"),
    ({"command": "openssl rsa -in server.key -text"}, "key_file", "server.key"),
    ({"target_file": "certs/client.p12"}, "key_file", "certs/client.p12"),
    ({"file_path": "deploy/secrets.yaml"}, "secret_file", "deploy/secrets.yaml"),
    ({"path": "client_secret.json"}, "secret_file", "client_secret.json"),
    ({"command": "cat ~/.netrc"}, "auth_file", "~/.netrc"),
    ({"file_path": "/home/tester/.npmrc"}, "auth_file", "~/.npmrc"),
    ({"command": "cat ~/.git-credentials"}, "auth_file", "~/.git-credentials"),
    ({"path": "/home/tester/.codex/auth.json"}, "auth_file", "~/.codex/auth.json"),
    ({"command": "cat ~/.docker/config.json"}, "auth_file", "~/.docker/config.json"),
    ({"command": "cat /proc/self/environ"}, "proc_environ", "/proc/self/environ"),
    ({"command": "printenv"}, "env_dump", "cmd:env_dump"),
    ({"command": "printenv OPENAI_API_KEY"}, "printenv", "cmd:printenv"),
    ({"command": "env | sort"}, "env_dump", "cmd:env_dump"),
    ({"command": "cd app && set"}, "env_dump", "cmd:env_dump"),
    ({"command": "export -p"}, "env_dump", "cmd:env_dump"),
    ({"command": "gh auth token"}, "secret_cli", "cmd:secret_cli"),
    ({"command": "kubectl get secret db -o yaml"}, "secret_cli", "cmd:secret_cli"),
]

NOT_SENSITIVE = [
    {"command": "cp .env.example .env.sample"},
    {"file_path": "/home/tester/proj/.env.template"},
    {"command": "python -m venv .venv && . .venv/bin/activate"},
    {"command": "node -e 'console.log(process.env.HOME)'"},
    {"command": "/usr/bin/env python3 script.py"},
    {"command": "env FOO=1 make test"},
    {"command": "set -euo pipefail; make"},
    {"command": "export PATH=$PATH:/opt/bin"},
    {"file_path": "/home/tester/.ssh/id_ed25519.pub"},
    {"command": "grep -rn credentials src/"},
    {"command": "grep -r secret_key ."},
    {"file_path": "src/secret_manager.py"},
    {"file_path": "docs/secrets.md"},
    {"command": "jq .key data.json"},
    {"file_path": "src/environment.ts"},
    {"pattern": "**/.env"},
]


class SensitiveTests(unittest.TestCase):
    def run_both(self, inputs: list[dict], cfg: dict | None = None) -> tuple[list, list]:
        events = [tool_event("", tool_input=item) for item in inputs]
        rs = redact.ruleset(cfg or {})
        py = [redact.sensitive(event, rs) for event in events]
        py = [None if hit is None else {**hit, "trigger": redact.show_trigger(hit["trigger"], rs, [], "/home/tester")} for hit in py]
        js = [out.get("sensitive_rule") and {"rule": out["sensitive_rule"], "trigger": out["sensitive_trigger"]} or None for out in jq_sanitize(events, cfg=cfg)]
        return py, js

    def test_sensitive_paths_and_env_dumps(self):
        py, js = self.run_both([item for item, _, _ in SENSITIVE])
        self.assertEqual(py, js)
        for (item, rule, trigger), hit in zip(SENSITIVE, py):
            with self.subTest(item=item):
                self.assertEqual(hit, {"rule": rule, "trigger": trigger})

    def test_lookalikes_are_not_sensitive(self):
        py, js = self.run_both(NOT_SENSITIVE)
        self.assertEqual(py, js)
        for item, hit in zip(NOT_SENSITIVE, py):
            with self.subTest(item=item):
                self.assertIsNone(hit)

    def test_sensitive_event_loses_its_text_and_keeps_its_ids(self):
        event = tool_event("DB_URL=postgres://u:Abc123xyz@h\nPLAIN=value", tool_input={"command": "cat .env"}, duration_ms=12, is_error=False)
        for out in (py_sanitize([event])[0], jq_sanitize([event])[0]):
            self.assertNotIn("tool_output", out)
            self.assertNotIn("tool_input", out)
            self.assertEqual((out["sensitive"], out["sensitive_rule"], out["sensitive_trigger"]), (True, "dotenv", ".env"))
            for field in ("tool_call_id", "tool_name", "duration_ms", "is_error", "session_id", "turn_id"):
                self.assertEqual(out[field], event[field])

    def test_tool_calls_and_prompt_mentions(self):
        model = {"v": 1, "event": "model", "session_id": "s", "turn_id": "t",
                 "tool_calls": [{"id": "c", "name": "Read", "arguments": {"file_path": "/srv/app/.env"}}]}
        prompt = {"v": 1, "event": "turn_start", "session_id": "s", "turn_id": "t", "prompt": "explain @.env.local please, mail me@example.com"}
        for out in (py_sanitize([model, prompt]), jq_sanitize([model, prompt])):
            self.assertEqual(out[0]["sensitive_trigger"], "/srv/app/.env")
            self.assertEqual(out[0]["tool_calls"], [{"id": "c", "name": "Read"}])
            self.assertEqual(out[1]["sensitive_trigger"], ".env.local")
            self.assertEqual(out[1].get("prompt", ""), "")

    def test_trigger_never_holds_a_secret(self):
        secret = SECRETS["github_token"]["core"]
        event = tool_event("", tool_input={"command": f"cat /tmp/{secret}/.env"})
        for out in (py_sanitize([event])[0], jq_sanitize([event])[0]):
            self.assertEqual(out["sensitive_trigger"], "/tmp/[REDACTED:github_token]/.env")


class UserRuleTests(unittest.TestCase):
    def test_redact_patterns_and_paths(self):
        cfg = {"redactPatterns": ["ACME-[0-9]{6}", "TCK_[A-Z]{4}", 7, ""], "redactPaths": ["vault/*.json", "prod-*.cfg", 3]}
        events = [
            tool_event("id ACME-123456 and TCK_ABCD done"),
            tool_event("", tool_input={"file_path": "/srv/app/vault/db.json"}),
            tool_event("", tool_input={"command": "cat conf/prod-eu.cfg"}),
            tool_event("", tool_input={"command": "cat conf/dev-eu.cfg"}),
        ]
        py, js = py_sanitize(events, cfg), jq_sanitize(events, cfg)
        self.assertEqual(norm(py), norm(js))
        self.assertEqual(py[0]["tool_output"], "id [REDACTED:custom] and [REDACTED:custom] done")
        self.assertEqual((py[1]["sensitive_rule"], py[1]["sensitive_trigger"]), ("custom_path", "/srv/app/vault/db.json"))
        self.assertEqual(py[2]["sensitive_trigger"], "conf/prod-eu.cfg")
        self.assertNotIn("sensitive", py[3])

    def test_invalid_user_regex_fails_closed(self):
        cfg = {"redactPatterns": ["(unclosed"]}
        event = tool_event("plain text " + SECRETS["github_token"]["core"], prompt="hello")
        for out in (py_sanitize([event], cfg)[0], jq_sanitize([event], cfg)[0]):
            self.assertTrue(out["redaction_error"])
            self.assertNotIn("tool_output", out)
            self.assertNotIn("prompt", out)
            self.assertEqual(out["tool_call_id"], "c1")

    def test_user_rules_only_add(self):
        cfg = {"redactPatterns": [], "redactPaths": [], "redact": True}
        out = py_sanitize([tool_event(SECRETS["openai_key"]["core"])], cfg)[0]
        self.assertEqual(out["tool_output"], "[REDACTED:openai_key]")


class GateTests(unittest.TestCase):
    """A rule's needs only let the layers skip its regex. They must never change a result."""

    def corpus(self) -> list[str]:
        from fp_report import CORPUS

        texts = [text for _, text in CORPUS] + [item["text"] for item in SECRETS.values()] + [secret_blob()]
        texts += [json.dumps(item) for item, _, _ in SENSITIVE] + [json.dumps(item) for item in NOT_SENSITIVE]
        texts += [text.upper() for text in texts] + [text.lower() for text in texts]
        return texts

    def test_every_match_holds_a_needed_literal(self):
        rs = redact.ruleset()
        for rule in rs.patterns + rs.paths + rs.commands:
            for text in self.corpus():
                for match in rule.pattern.finditer(text):
                    with self.subTest(rule=rule.id, match=match.group(0)[:40]):
                        self.assertTrue(rule.applies(match.group(0)))

    def test_gated_and_ungated_redaction_agree(self):
        from dataclasses import replace

        rs = redact.ruleset()
        open_rs = redact.Ruleset(tuple(replace(rule, needs=()) for rule in rs.patterns), rs.paths, rs.commands)
        for text in self.corpus():
            self.assertEqual(redact.redact_string(text, rs, []), redact.redact_string(text, open_rs, []))

    def test_env_needs_cover_the_env_rule(self):
        table = redact.rules()["env"]
        for name, _ in redact.env_secrets(clean_env(GITHUB_TOKEN="x" * 20, npm_config__authToken="Ab12345678", Some_Private_Thing="Zz98765432")):
            self.assertTrue(any(need in name for need in table["needs"]), name)


CAPTURE_MATRIX = [
    ({}, "sanitized"),
    ({"capture": "sanitized"}, "sanitized"),
    ({"capture": "full"}, "full"),
    ({"capture": "metadata"}, "metadata"),
    ({"capture": "raw"}, "sanitized"),
    ({"capture": "FULL"}, "sanitized"),
    ({"capture": None}, "sanitized"),
    ({"capture": 1}, "sanitized"),
    ({"redact": False}, "full"),
    ({"redact": "false"}, "sanitized"),
    ({"redact": True}, "sanitized"),
    ({"capture": "sanitized", "redact": False}, "sanitized"),
    ({"capture": "metadata", "redact": False}, "metadata"),
    ({"capture": "full", "redact": True}, "full"),
    ({"capture": "bogus", "redact": False}, "full"),
]


class CaptureModeTests(unittest.TestCase):
    def test_jq_and_python_resolve_the_same_mode(self):
        js = [jq('judgment_prepare({enabled: true} + $cfg; "on"; "/r"; "")', [tool_event("x")], cfg=cfg)[0]["capture"] for cfg, _ in CAPTURE_MATRIX]
        for (cfg, expected), got in zip(CAPTURE_MATRIX, js):
            with self.subTest(cfg=cfg):
                self.assertEqual(optin.resolve_capture(cfg), expected)
                self.assertEqual(got, expected)

    def test_prepare_stamps_the_mode_and_applies_it(self):
        secret = SECRETS["github_token"]["core"]
        event = tool_event("x " + secret)
        for cfg, expected in CAPTURE_MATRIX:
            with self.subTest(cfg=cfg):
                out = jq("judgment_prepare($cfg; \"on\"; \"/r\"; \"\")", [event], cfg=cfg)[0]
                self.assertEqual(out["capture"], expected)
                if expected == "full":
                    self.assertEqual(out["tool_output"], event["tool_output"])
                else:
                    self.assertNotIn(secret, json.dumps(out))

    def test_load_project_normalizes(self, ):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracing.json"
            for cfg, expected in CAPTURE_MATRIX:
                path.write_text(json.dumps({"enabled": True, **cfg}))
                self.assertEqual(optin.load_project(path)["capture"], expected)
            path.write_text("{")
            self.assertEqual(optin.load_project(path)["enabled"], False)

    def test_enable_writes_sanitized(self):
        import io
        import tempfile
        from contextlib import redirect_stdout
        from pathlib import Path

        from judgment_core import cli

        with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
            cli.main(["--cwd", folder, "enable"])
            written = json.loads((Path(folder) / ".judgment" / "tracing.json").read_text())
        self.assertEqual(written["capture"], "sanitized")


if __name__ == "__main__":
    unittest.main()
