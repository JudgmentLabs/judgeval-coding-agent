"""Redaction rules: every pattern, env values, sensitive paths, user rules, and capture modes. jq and Python must agree."""

from __future__ import annotations

import json
import shutil
import unittest

import random
import string
import tempfile
from pathlib import Path

from support import ENV_SECRET_NAME, SECRETS, assert_parity, clean_env, jq, jq_sanitize, leaks, norm, py_sanitize, secret_blob, tool, whole_leaks

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
                self.assertTrue(redact.census(out))

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
            self.assertEqual({item["field"] for item in redact.census(out)}, {"prompt", "output", "received", "tool_results", "tool_calls", "messages", "error"})

    def test_audit_trail_has_rule_field_count_and_no_secret(self):
        events = [tool_event(secret_blob(), prompt="key " + SECRETS["openai_key"]["core"])]
        env = clean_env(**{ENV_SECRET_NAME: SECRETS["env_value"]["core"], "JUDGMENT_API_KEY": SECRETS["judgment_key"]["core"]})
        for out in (py_sanitize(events, env=env)[0], jq_sanitize(events, env=env)[0]):
            audit = redact.census(out)
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
            "OAUTH_REDIRECT_URL": "https://example.com/callback",
            "SHORT_TOKEN": "abc",
        }
        # Any value of 8 or more characters under a secret name goes, whatever its shape (review 2, finding 2).
        redacted = {"JUDGMENT_API_KEY": "short123", "SERVICE_PASSWORD": "Pw-" + SECRETS["env_value"]["core"][:10], "GITHUB_TOKEN": SECRETS["github_token"]["core"],
                    "AUTH_PROVIDER": "keycloak"}
        env = self.env(**keep, **redacted)
        py_names = sorted({name for name, _ in redact.env_secrets(env)})
        self.assertEqual(py_names, sorted(redacted))
        event = tool_event(" | ".join(list(keep.values()) + list(redacted.values())))
        for out in (py_sanitize([event], env=env)[0], jq_sanitize([event], env=env)[0]):
            for value in keep.values():
                self.assertIn(value, out["tool_output"])
            for name in redacted:
                self.assertIn(f"[REDACTED:env:{name}]", out["tool_output"])
            self.assertEqual(sorted(item["rule"] for item in redact.census(out)), sorted(f"env:{name}" for name in redacted))

    def test_names_with_odd_characters_and_casings(self):
        env = self.env(**{"npm_config_//registry.example.com/:_authToken": "npm_" + SECRETS["env_value"]["core"], "githubToken": "Gh" + SECRETS["env_value"]["core"][:12],
                          "AUTHORIZATION_HEADER": "Basic " + SECRETS["env_value"]["core"][:16], "MONKEY_ISLAND": "Guybrush123", "XAUTHORITY_X": "Cookie12345"})
        names = sorted({name for name, _ in redact.env_secrets(env)})
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
    {"command": "grep -r secret_key ."},
    {"file_path": "src/secret_manager.py"},
    {"file_path": "docs/secrets.md"},
    {"command": "jq .key data.json"},
    {"file_path": "src/environment.ts"},
    {"pattern": "**/.env"},
]

OVER_BLANKED = [
    ({"command": "grep -rn credentials src/"}, "credentials_file"),
    ({"command": 'git commit -m "rotate secrets"'}, "secret_file"),
]


class SensitiveTests(unittest.TestCase):
    def run_both(self, inputs: list[dict], cfg: dict | None = None) -> tuple[list, list]:
        events = [tool_event("", tool_input=item) for item in inputs]
        rs = redact.ruleset(cfg or {})
        py = [redact.sensitive(event, rs) for event in events]
        py = [None if hit is None else {**hit, "trigger": redact.show_trigger(hit["trigger"], "/home/tester")} for hit in py]
        js = [out.get("sensitive_rule") and {"rule": out["sensitive_rule"], "trigger": redact.show_trigger(out["sensitive_trigger"], "/home/tester")} or None
              for out in jq_sanitize(events, cfg=cfg)]
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

    def test_known_limit_a_mention_blanks_the_turn(self):
        """Known limit (mention-blanks): a sensitive file name is a mention wherever it sits in a command, so a search for
        the word blanks the turn too. It loses that turn's text and never leaks."""
        py, js = self.run_both([item for item, _ in OVER_BLANKED])
        self.assertEqual(py, js)
        self.assertEqual([hit and hit["rule"] for hit in py], [rule for _, rule in OVER_BLANKED])

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
        cfg = {"redactPatterns": ["ACME-[0-9]{6}", "TCK_[A-Z]{4}", ""], "redactPaths": ["vault/*.json", "prod-*.cfg"]}
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


class ProjectRuleTests(unittest.TestCase):
    """tracing.json's redactCommands, redactNames, redactAllow, and redactDisable, in both layers. A bad entry fails closed."""

    def both(self, events: list[dict], cfg: dict) -> list[dict]:
        py, js = py_sanitize(events, cfg), jq_sanitize(events, cfg)
        self.assertEqual(norm(py), norm(js))
        return py

    def test_redact_commands(self):
        cfg = {"redactCommands": ["mycli[^\\S\\n]+show-secret"]}
        out = self.both([tool_event("x", tool_input={"command": c}) for c in ("mycli show-secret db", "cd a && mycli show-secret", "mycli show-secret --help",
                                                                             "echo 'mycli show-secret'", "mycli list")], cfg)
        self.assertEqual([item.get("sensitive_rule") for item in out], ["custom_command", "custom_command", None, None, None])

    def test_redact_names(self):
        value = "pw" + SECRETS["github_token"]["random"][:10]
        texts = [f"db_pin = {value}", f'{{"vaultPin": "{value}"}}', f"PIN: {value}"]
        events = [tool_event(text) for text in texts] + [tool_event("ok", tool_input={"pin": value})]
        out = self.both(events, {"redactNames": ["pin", "vault_pin"]})
        self.assertEqual([item["tool_output"] for item in out[:3]], ["db_pin = [REDACTED:secret_assignment]", '{"vaultPin": "[REDACTED:secret_assignment]"}',
                                                                    "PIN: [REDACTED:secret_assignment]"])
        self.assertEqual(out[3]["tool_input"], {"pin": "[REDACTED:secret_field]"})
        self.assertEqual([item["tool_output"] for item in self.both([tool_event(text) for text in texts], {})], texts)

    def test_redact_allow(self):
        token = SECRETS["github_token"]["core"]
        cfg = {"redactAllow": ["changeme", token]}
        out = self.both([tool_event("password = changeme"), tool_event("password = hunter22"), tool_event("gh " + token),
                         tool_event("ok", tool_input={"password": "changeme"})], cfg)
        self.assertEqual([item["tool_output"] for item in out[:3]], ["password = changeme", "password = [REDACTED:secret_assignment]", "gh " + token])
        self.assertEqual(out[3]["tool_input"], {"password": "changeme"})

    def test_redact_disable(self):
        token = SECRETS["github_token"]["core"]
        cfg = {"redactDisable": ["secret_assignment", "github_token", "dotenv", "env_dump"]}
        events = [tool_event("password = hunter22 " + token), tool_event("x", tool_input={"command": "cat .env"}),
                  tool_event("x", tool_input={"command": "env"}), tool_event("x", tool_input={"command": "printenv HOME"}),
                  tool_event("ok", tool_input={"password": "hunter22"})]
        out = self.both(events, cfg)
        self.assertEqual(out[0]["tool_output"], "password = hunter22 " + token)
        self.assertEqual([item.get("sensitive_rule") for item in out[1:4]], [None, None, "printenv"])
        self.assertEqual(out[4]["tool_input"], {"password": "[REDACTED:secret_field]"})

    def test_bad_entries_fail_closed(self):
        secret = SECRETS["github_token"]["core"]
        event = tool_event("plain text " + secret, prompt="hello")
        for cfg in ({"redactDisable": ["github_tokn"]}, {"redactNames": ["db pin"]}, {"redactNames": ["pin)|("]}, {"redactCommands": ["(unclosed"]}):
            with self.subTest(cfg=cfg):
                self.assertTrue(redact.ruleset(cfg).error)
                for out in (py_sanitize([event], cfg)[0], jq_sanitize([event], cfg)[0]):
                    self.assertTrue(out["redaction_error"])
                    self.assertNotIn("tool_output", out)
                    self.assertNotIn("prompt", out)

    def test_non_strings_and_non_lists_in_trims_are_skipped(self):
        """A trim that is not a list of strings trims nothing. A rule setting fails closed instead."""
        cfg = {"redactDisable": "dotenv", "redactAllow": [3, None, ""]}
        out = self.both([tool_event("pin = abc"), tool_event("x", tool_input={"command": "cat .env"})], cfg)
        self.assertEqual(out[0]["tool_output"], "pin = abc")
        self.assertEqual(out[1]["sensitive_rule"], "dotenv")

    def test_load_project_carries_every_key(self):
        import tempfile
        from pathlib import Path

        settings = {"redactPatterns": ["A"], "redactPaths": ["b"], "redactCommands": ["c"], "redactNames": ["d"], "redactAllow": ["e"],
                    "redactDisable": ["dotenv"]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tracing.json"
            path.write_text(json.dumps({**settings, "redactAllow": ["e", 4, ""]}))
            # Every setting arrives as written: redact.ruleset alone decides what a malformed one means.
            written = {**settings, "redactAllow": ["e", 4, ""], "redactNames": ["d", 4]}
            path.write_text(json.dumps(written))
            loaded = optin.load_project(path)
            self.assertEqual({key: loaded[key] for key in settings}, written)
            self.assertEqual(set(optin.REDACT_KEYS), set(settings))
            self.assertEqual(redact.ruleset(loaded).error, "redactNames")
            # Read as the hook's jq reads it: a byte order mark and a second document are fine.
            for text in ("\ufeff" + json.dumps(settings), json.dumps(settings) + "\n{}"):
                path.write_text(text)
                self.assertEqual(redact.ruleset(optin.load_project(path)).error, "", text)
            # A file that cannot be read, or that jq would refuse, has unknown rules: the rule set fails closed, so the
            # sender sends no transcript text under the built-in rules alone.
            for text in ("", "{", "[1]", json.dumps(settings) + '\n{"redactPatterns":['):
                path.write_text(text)
                rs = redact.ruleset(optin.load_project(path))
                self.assertTrue(rs.error, text)
                self.assertTrue(redact.redact_event(tool_event("ticket=12345"), rs, [])["redaction_error"])
            path.unlink()
            self.assertTrue(redact.ruleset(optin.load_project(path)).error)

    def test_kept_group_and_posix_class_agree_in_both_layers(self):
        """p is kept only as the first group, at the start of the match. A POSIX class, which the engines read differently, fails closed."""
        for regex, expected in (("(?<p>ticket=)[0-9]+", "hello ticket=[REDACTED:custom]"), ("ticket=(?<p>[0-9])[0-9]+", "hello [REDACTED:custom]"),
                                ("(ticket=)(?<p>[0-9]+)", "hello [REDACTED:custom]")):
            out = self.both([tool_event("hello ticket=12345")], {"redactPatterns": [regex]})
            self.assertEqual(out[0]["tool_output"], expected, regex)
        out = self.both([tool_event("hello ticket=12345")], {"redactPatterns": ["ticket=(?<p>[0-9])[0-9]+"], "redactAllow": ["icket=12345"]})
        self.assertEqual(out[0]["tool_output"], "hello [REDACTED:custom]")
        # jq reads the class; Python would read a set of characters, so the sender refuses the rule set instead.
        self.assertTrue(redact.ruleset({"redactPatterns": ["[[:alpha:]]+=12345"]}).error)


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
        for rule in rs.patterns + rs.paths + rs.commands + rs.members:
            for text in self.corpus():
                for match in rule.pattern.finditer(text):
                    with self.subTest(rule=rule.id, match=match.group(0)[:40]):
                        self.assertTrue(rule.applies(match.group(0)))

    def test_gated_and_ungated_redaction_agree(self):
        from dataclasses import replace

        rs = redact.ruleset()
        open_rs = redact.Ruleset(tuple(replace(rule, needs=(), pre=None) for rule in rs.patterns), rs.paths, rs.commands,
                                 tuple(replace(rule, needs=(), pre=None) for rule in rs.members), allow=rs.allow)
        for text in self.corpus():
            self.assertEqual(redact.redact_text(text, rs, []), redact.redact_text(text, open_rs, []))

    def test_member_rule_needs_cover_its_keys(self):
        """Every secret key the member rules take holds a need of the rule that took it, in the JSON the rule sees."""
        rs = redact.ruleset()
        for key in ("api_key", "x-api-key", "X-Api-Key", "DB_PASSWORD", "clientSecret", "refresh_token", "Credentials", "passphrase", "pwd",
                    "smtpPassword", "aws_secret_access_key", "token", "SESSION_TOKEN"):
            text = redact.jq_json([{key: "Zq81jdjskaQ8a9d2"}])
            hits = [rule for rule in rs.members if rule.pattern.search(text)]
            self.assertTrue(hits, key)
            for rule in hits:
                self.assertTrue(rule.applies(text), (key, rule.id))

    def test_member_and_pre_filters_cover_every_match(self):
        """needs and pre only let the layers skip a regex: each match of a rule passes both, in the corpus and the secrets."""
        rs = redact.ruleset()
        for text in self.corpus():
            json_text = redact.jq_json([text, {"k": text}])
            for rule in rs.members:
                for match in rule.pattern.finditer(json_text):
                    self.assertTrue(rule.applies(match.group(0)), (rule.id, match.group(0)[:40]))


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
        from contextlib import redirect_stdout

        from judgment_core import cli

        from support import hermetic_dir

        folder = hermetic_dir("jt-cli-")
        self.addCleanup(shutil.rmtree, folder, True)
        (folder / ".git").mkdir()  # the CLI writes at the repository root: make it this folder, whatever is above
        with redirect_stdout(io.StringIO()):
            cli.main(["--cwd", str(folder), "enable"])
        written = json.loads((folder / ".judgment" / "tracing.json").read_text())
        self.assertEqual(written["capture"], "sanitized")


_rng = random.Random(8080)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def outputs(test: unittest.TestCase, texts: list, cfg: dict | None = None, env: dict | None = None) -> list[dict]:
    return assert_parity(test, [tool({"command": "x"}, text) for text in texts], cfg, env)[0]


def rules_of(test: unittest.TestCase, commands: list[str], cfg: dict | None = None) -> dict:
    py = assert_parity(test, [tool({"command": command}, "out") for command in commands], cfg)[0]
    return {command: out.get("sensitive_rule") for command, out in zip(commands, py)}


class EscapeBoundaryTests(unittest.TestCase):
    """Finding 1: a secret right after a JSON escape (\\n, \\t, \\r) in JSON nested in a string failed every identifier
    boundary, in both layers and the sender."""

    def test_secrets_after_an_escape_go(self):
        token = "gh" + "p_" + _rand(36)
        password = "Pw" + _rand(14)
        texts = ['{"log": "started\\n' + token + '"}', '{"log": "a\\t' + token + '"}', '{"log": "a\\r' + token + '"}',
                 '{"log": "started\\npassword=' + password + '"}']
        py = outputs(self, texts)
        secrets = [token, token, token, password]
        self.assertEqual([text for text, out, s in zip(texts, py, secrets) if s in json.dumps(out)], [])
        sender = [redact.redact_event(tool({"command": "x"}, text), redact.ruleset(), []) for text in texts]
        self.assertEqual([text for text, out, s in zip(texts, sender, secrets) if s in json.dumps(out)], [])

    def test_escapes_that_are_no_boundary_stay(self):
        texts = ["C:\\new\\tokenizer.py", "a\\nmax_tokens=5"]
        py = outputs(self, texts)
        self.assertEqual([out["tool_output"] for out in py], texts)


class EnvDumpThenCommandTests(unittest.TestCase):
    """Finding 2: env_dump consumed its separator, so the help check read the next command's --version."""

    def test_dump_before_another_command(self):
        commands = ["env; node --version", "set; git --version", "env\ngit --version", "env | grep -h FOO"]
        self.assertEqual(rules_of(self, commands), {command: "env_dump" for command in commands})


class NestedParityTests(unittest.TestCase):
    """Finding 3: the sender looked for nested JSON at an escaped backslash before a quote, the hook at an escaped quote."""

    def test_env_value_in_nested_json(self):
        value = "Ab" + "c9\"d\\e/f<g"
        env = clean_env(MY_API_TOKEN=value)
        text = json.dumps({"a": "v " + value})
        py = outputs(self, [text], env=env)
        self.assertEqual(py[0]["tool_output"], '{"a": "v [REDACTED:env:MY_API_TOKEN]"}')
        sender = redact.redact_event(tool({"command": "x"}, text), redact.ruleset(), redact.env_secrets(env))
        self.assertEqual(sender["tool_output"], py[0]["tool_output"])


class AssignmentFormTests(unittest.TestCase):
    """Findings 5, 6, 7, 11: aligned config, key prefixes with spaces or colons, template literals, and type annotations."""

    def test_aligned_values_go(self):
        value = "hu" + _rand(12)
        texts = ["password" + " " * 20 + "= " + value, "api_key:" + " " * 24 + value]
        self.assertEqual([out["tool_output"] for out in outputs(self, texts) if value in out["tool_output"]], [])

    def test_keys_with_spaces_colons_and_slashes(self):
        value = "Zq" + _rand(20)
        values = [{"DB Password": value}, {"spring:datasource:password": value}, {"secrets/api_token": value}]
        py = assert_parity(self, [tool({"command": "x"}, item) for item in values])[0]
        self.assertEqual([out["tool_output"] for out in py if value in json.dumps(out)], [])

    def test_template_literal(self):
        value = "hu" + _rand(12)
        py = outputs(self, ["const password = `" + value + "`;"])
        self.assertEqual(py[0]["tool_output"], "const password = `[REDACTED:secret_assignment]`;")

    def test_types_and_constants_stay(self):
        texts = ["password: str", "def f(api_key: Optional[str] = None):", "token = None", "secret: boolean;", "let token: string = x",
                 "password = null", "PASSWORD = True", "credentials: bool"]
        py = outputs(self, texts)
        self.assertEqual([out["tool_output"] for out in py], texts)


class WrappedCommandTests(unittest.TestCase):
    """Finding 8: a command position now reads through shell keywords, assignments, sudo with options, sh -c, and a path."""

    CAUGHT = ['bash -c "env"', "bash -lc 'printenv'", "FOO=1 printenv", "sudo -u root env", "sudo -E env", "/usr/bin/env", "if x; then env; fi",
              "for a in b; do printenv; done", "{ env; }", "nohup env", "command env"]
    KEPT = ["which env", "x=$(which env)", "command -v env", "env --help", "bash -c 'make env'", "conda env list", "env FOO=1 make"]

    def test_caught(self):
        got = rules_of(self, self.CAUGHT)
        self.assertEqual([command for command, rule in got.items() if rule not in ("env_dump", "printenv")], [])

    def test_kept(self):
        self.assertEqual({command: rule for command, rule in rules_of(self, self.KEPT).items() if rule}, {})


class PathGapTests(unittest.TestCase):
    """Finding 9: dotenv files named another way, and secret-named source files."""

    def test_paths(self):
        blanked = ["cat prod.env", "cat .env-prod", "cat config/.envrc.local"]
        kept = ["cat example.env", "cat src/secrets.sh", "vim secret.c", "cat templates/secret.html", "cat db/secrets.sql"]
        got = rules_of(self, blanked + kept)
        self.assertEqual({c: got[c] for c in blanked}, {c: "dotenv" for c in blanked})
        self.assertEqual({c: got[c] for c in kept if got[c]}, {})


class ConfigTypeTests(unittest.TestCase):
    """Finding 10: a setting that is not a list of strings was skipped, so a rule the project asked for silently did nothing."""

    BAD = [{"redactPatterns": "hunter[0-9]+"}, {"redactPaths": {"a": 1}}, {"redactCommands": ["ok", 5]}, {"redactNames": "pin"}]

    def test_both_layers_fail_closed(self):
        for cfg in self.BAD:
            py = assert_parity(self, [tool({"command": "x"}, "hello")], cfg)[0]
            self.assertTrue(py[0].get("redaction_error"), cfg)

    def test_load_project_keeps_the_bad_setting(self):
        for cfg in self.BAD:
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "tracing.json"
                path.write_text(json.dumps({"enabled": True, **cfg}))
                self.assertTrue(redact.ruleset(optin.load_project(path)).error, cfg)

    def test_trims_stay_lenient(self):
        py = assert_parity(self, [tool({"command": "x"}, "hello")], {"redactAllow": "x", "redactDisable": 3})[0]
        self.assertEqual(py[0]["tool_output"], "hello")


class TrimBuiltinsTests(unittest.TestCase):
    """Finding 12: !name drops a built-in name, !glob excepts a path from blanking."""

    def test_drop_a_name(self):
        value = "hu" + _rand(12)
        py = outputs(self, ["token=" + value, "password=" + value], {"redactNames": ["!token", "pin"]})
        self.assertEqual([out["tool_output"] for out in py], ["token=" + value, "password=[REDACTED:secret_assignment]"])

    def test_drop_an_unknown_name_fails_closed(self):
        py = outputs(self, ["hello"], {"redactNames": ["!pin"]})
        self.assertTrue(py[0].get("redaction_error"))

    def test_except_a_path(self):
        got = rules_of(self, ["cat .env.test", "cat app/.env.test", "cat .env"], {"redactPaths": ["!.env.test"]})
        self.assertEqual(got, {"cat .env.test": None, "cat app/.env.test": None, "cat .env": "dotenv"})


if __name__ == "__main__":
    unittest.main()
