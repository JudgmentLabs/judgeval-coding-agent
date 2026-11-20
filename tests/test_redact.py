"""Redaction rules: every pattern, env values, user rules, and capture modes. jq and Python must agree."""

from __future__ import annotations

import json
import io
import shutil
import unittest
from contextlib import redirect_stderr, redirect_stdout

import random
import string
import tempfile
from pathlib import Path

from support import ENV_SECRET_NAME, SECRETS, Sandbox, assert_parity, clean_env, jq, jq_sanitize, leaks, norm, py_sanitize, secret_blob, tool, whole_leaks

from judgment_core import cli, optin, redact

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


class StructuredSecretTests(unittest.TestCase):
    def test_non_string_members_are_masked_without_exposing_their_contents(self):
        token = SECRETS["github_token"]["core"]
        value = {"password": 12345678, "token": [token], "secret": {"value": "private"},
                 "nested": [{"dbPassword": -1.25}], "ordinary": [0], "other": [1]}
        py, _ = assert_parity(self, [tool_event(value, prompt=token)])
        out = py[0]
        marker = "[REDACTED:secret_field]"
        self.assertEqual(out["tool_output"], {"password": marker, "token": marker, "secret": marker,
                                               "nested": [{"dbPassword": marker}], "ordinary": [0], "other": [1]})
        self.assertEqual(out["redactions"], [{"rule": "github_token", "field": "prompt", "count": 1},
                                            {"rule": "secret_field", "field": "tool_output", "count": 4}])
        self.assertEqual(value["password"], 12345678)

    def test_empty_containers_and_non_secret_primitives(self):
        py, _ = assert_parity(self, [tool_event({"password": {}, "token": [], "pwd": None, "secret": False},
                                              usage={"password": 123})])
        self.assertEqual(py[0]["tool_output"], {"password": "[REDACTED:secret_field]", "token": "[REDACTED:secret_field]",
                                                "pwd": None, "secret": False})
        self.assertEqual(py[0]["usage"], {"password": 123})

    def test_marker_cannot_collide_with_negative_zero(self):
        py, _ = assert_parity(self, [tool_event({"password": 7, "ordinary": [-0.0]})])
        self.assertEqual(py[0]["tool_output"], {"password": "[REDACTED:secret_field]", "ordinary": [-0.0]})

    def test_name_policy_and_disabled_member_rule_apply_to_typed_values(self):
        for cfg, expected in (({"redactNames": ["pin", "!password"]}, {"pin": "[REDACTED:secret_field]", "password": 2}),
                              ({"redactDisable": ["secret_field"]}, {"pin": [1], "password": 2})):
            with self.subTest(cfg=cfg):
                py, _ = assert_parity(self, [tool_event({"pin": [1], "password": 2})], cfg=cfg)
                self.assertEqual(py[0]["tool_output"], expected)

    def test_structural_and_text_replacements_share_the_hook_budget(self):
        for value, extra in (([{"password": n} for n in range(129)], ""),
                             ([{"password": n} for n in range(128)], SECRETS["github_token"]["core"])):
            with self.subTest(extra=bool(extra)):
                py, _ = assert_parity(self, [tool_event(value, prompt=extra)])
                self.assertTrue(py[0].get("redaction_error"))
                self.assertNotIn("tool_output", py[0])

    def test_placeholder_is_restored_after_custom_matching(self):
        py, _ = assert_parity(self, [tool_event({"password": 123, "note": "secret"})], cfg={"redactPatterns": ["secret"]})
        self.assertEqual(py[0]["tool_output"], {"password": "[REDACTED:secret_field]", "note": "[REDACTED:custom]"})

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
        # Any value of 8 or more characters under a secret name goes, whatever its shape.
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


class UserRuleTests(unittest.TestCase):
    def test_redact_patterns(self):
        cfg = {"redactPatterns": ["ACME-[0-9]{6}", "TCK_[A-Z]{4}", ""]}
        events = [tool_event("id ACME-123456 and TCK_ABCD done")]
        py, js = py_sanitize(events, cfg), jq_sanitize(events, cfg)
        self.assertEqual(norm(py), norm(js))
        self.assertEqual(py[0]["tool_output"], "id [REDACTED:custom] and [REDACTED:custom] done")

    def test_invalid_user_regex_fails_closed(self):
        cfg = {"redactPatterns": ["(unclosed"]}
        event = tool_event("plain text " + SECRETS["github_token"]["core"], prompt="hello")
        for out in (py_sanitize([event], cfg)[0], jq_sanitize([event], cfg)[0]):
            self.assertTrue(out["redaction_error"])
            self.assertNotIn("tool_output", out)
            self.assertNotIn("prompt", out)
            self.assertEqual(out["tool_call_id"], "c1")

    def test_user_rules_only_add(self):
        cfg = {"redactPatterns": [], "redact": True}
        out = py_sanitize([tool_event(SECRETS["openai_key"]["core"])], cfg)[0]
        self.assertEqual(out["tool_output"], "[REDACTED:openai_key]")


class OverlappingFindTests(unittest.TestCase):
    def both(self, text, cfg=None):
        py, js = assert_parity(self, [tool_event(text)], cfg)
        return py[0]

    def test_private_key_head_overlaps_formatted_tokens(self):
        head = "-----BEGIN OPENSSH PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC7\n"
        for token in ("ghp_" + "a1B2c3D4e5" * 4, SECRETS["jwt"]["core"]):
            with self.subTest(token=token[:4]):
                out = self.both(head + token + "\nmore")
                self.assertEqual(out["tool_output"], "[REDACTED:private_key]\nmore")
                self.assertEqual(out["redactions"], [{"rule": "private_key", "field": "tool_output", "count": 1}])

    def test_contained_assignments_charge_only_merged_pairs(self):
        text = "\n".join(f"SERVICE_{i}_API_KEY={SECRETS['stripe_key']['core']}{i}" for i in range(100))
        out = self.both(text)
        self.assertNotIn("redaction_error", out)
        self.assertEqual(out["tool_output"], "\n".join(f"SERVICE_{i}_API_KEY=[REDACTED:secret_assignment]" for i in range(100)))
        self.assertEqual(out["redactions"], [{"rule": "secret_assignment", "field": "tool_output", "count": 100}])

    def test_member_and_format_finds_share_the_pair_budget(self):
        out = self.both([{f"SERVICE_{i}_API_KEY": SECRETS["stripe_key"]["core"] + str(i)} for i in range(100)])
        self.assertNotIn("redaction_error", out)
        self.assertEqual(out["redactions"], [{"rule": "secret_field", "field": "tool_output", "count": 100}])

    def test_raw_find_count_does_not_exhaust_merged_budget(self):
        text = "outer=" + " ".join(SECRETS["stripe_key"]["core"] + str(i) for i in range(129))
        out = self.both(text, {"redactPatterns": ["(?<p>outer=)[^\\n]+"]})
        self.assertEqual(out["tool_output"], "outer=[REDACTED:custom]")
        self.assertEqual(out["redactions"], [{"rule": "custom", "field": "tool_output", "count": 1}])

    def test_overlap_chains_keep_earliest_label_and_preserved_prefix(self):
        cfg = {"redactPatterns": ["(?<p>ticket=)abc", "bcde", "defg"]}
        out = self.both("é ticket=abcdefg then ticket=abcdefg!", cfg)
        self.assertEqual(out["tool_output"], "é ticket=[REDACTED:custom] then ticket=[REDACTED:custom]!")
        self.assertEqual(out["redactions"], [{"rule": "custom", "field": "tool_output", "count": 2}])

    def test_a_find_in_a_kept_prefix_is_not_exposed(self):
        out = self.both("ticket=12345", {"redactPatterns": ["(?<p>ticket=)[0-9]+", "ticket"]})
        self.assertEqual(out["tool_output"], "[REDACTED:custom]")

    def test_empty_and_adjacent_finds(self):
        self.assertEqual(self.both("")["tool_output"], "")
        out = self.both("abcxyz", {"redactPatterns": ["abc", "xyz"]})
        self.assertEqual(out["tool_output"], "[REDACTED:custom][REDACTED:custom]")
        self.assertEqual(out["redactions"], [{"rule": "custom", "field": "tool_output", "count": 2}])

    def test_ambiguous_literal_position_fails_closed(self):
        out = self.both("ababa", {"redactPatterns": ["(?<=ab)aba"]})
        self.assertTrue(out["redaction_error"])
        self.assertNotIn("tool_output", out)

    def test_a_find_spanning_two_fields_fails_closed(self):
        py, js = assert_parity(self, [tool_event("suffix", prompt="prefix")], {"redactPatterns": ["suffix\\nprefix"]})
        self.assertTrue(py[0]["redaction_error"])
        self.assertNotIn("tool_output", py[0])


class ProjectRuleTests(unittest.TestCase):
    """tracing.json's redactNames, redactAllow, and redactDisable, in both layers. A bad entry fails closed."""

    def both(self, events: list[dict], cfg: dict) -> list[dict]:
        py, js = py_sanitize(events, cfg), jq_sanitize(events, cfg)
        self.assertEqual(norm(py), norm(js))
        return py

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
        cfg = {"redactDisable": ["secret_assignment", "github_token"]}
        out = self.both([tool_event("password = hunter22 " + token), tool_event("ok", tool_input={"password": "hunter22"})], cfg)
        self.assertEqual(out[0]["tool_output"], "password = hunter22 " + token)
        self.assertEqual(out[1]["tool_input"], {"password": "[REDACTED:secret_field]"})

    def test_redact_paths_and_commands_hide_the_call(self):
        """Only what the project names is hidden: the call's arguments and result. A !glob excepts a file; nothing is
        built in, and an id older versions had for a built-in file or command rule may still be in redactDisable."""
        cfg = {"redactPaths": ["data/*", ".env*", "!.env.test"], "redactCommands": ["vaultctl[^\\S\\n]+dump"], "redactDisable": ["dotenv", "env_dump"]}
        events = [tool_event("rows", tool_input={"file_path": "data/eu/customers.csv"}), tool_event("dump", tool_input={"command": "sudo vaultctl dump"}),
                  tool_event("A=1", tool_input={"file_path": ".env.test"}), tool_event("B=2", tool_input={"command": "cat app/.env.local"}),
                  tool_event("usage", tool_input={"command": "vaultctl dump --help"})]
        blank = lambda rule: (f"[REDACTED:{rule}]", f"[REDACTED:{rule}]")  # noqa: E731
        out = self.both(events, cfg)
        self.assertEqual([(event["tool_input"], event["tool_output"]) for event in out],
                         [blank("custom_path"), blank("custom_command"), ({"file_path": ".env.test"}, "A=1"), blank("custom_path"), ({"command": "vaultctl dump --help"}, "usage")])
        self.assertEqual(out[0]["redactions"], [{"rule": "custom_path", "field": "tool_input", "count": 1}, {"rule": "custom_path", "field": "tool_output", "count": 1}])
        # Without the settings every one of these calls is sent, redacted by format and name like any other text.
        self.assertEqual([(event["tool_input"], event["tool_output"]) for event in self.both(events, {})], [(event["tool_input"], event["tool_output"]) for event in events])
        # A setting that cannot work fails closed, as every other rule setting does.
        for bad in ({"redactPaths": {"a": 1}}, {"redactCommands": ["(unclosed"]}):
            self.assertTrue(redact.ruleset(bad).error, bad)
            self.assertTrue(self.both([tool_event("x")], bad)[0]["redaction_error"])

    def test_bad_entries_fail_closed(self):
        secret = SECRETS["github_token"]["core"]
        event = tool_event("plain text " + secret, prompt="hello")
        for cfg in ({"redactDisable": ["github_tokn"]}, {"redactNames": ["db pin"]}, {"redactNames": ["pin)|("]}):
            with self.subTest(cfg=cfg):
                self.assertTrue(redact.ruleset(cfg).error)
                for out in (py_sanitize([event], cfg)[0], jq_sanitize([event], cfg)[0]):
                    self.assertTrue(out["redaction_error"])
                    self.assertNotIn("tool_output", out)
                    self.assertNotIn("prompt", out)

    def test_non_strings_and_non_lists_in_trims_are_skipped(self):
        """A trim that is not a list of strings trims nothing. A rule setting fails closed instead."""
        cfg = {"redactDisable": "secret_assignment", "redactAllow": [3, None, ""]}
        out = self.both([tool_event("pin = abc"), tool_event("password = hunter22")], cfg)
        self.assertEqual(out[0]["tool_output"], "pin = abc")
        self.assertEqual(out[1]["tool_output"], "password = [REDACTED:secret_assignment]")

    def test_load_project_carries_every_key(self):
        import tempfile
        from pathlib import Path

        settings = {"redactPatterns": ["A"], "redactPaths": ["b/*"], "redactCommands": ["c"], "redactNames": ["d"], "redactAllow": ["e"], "redactDisable": ["github_token"]}
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
        texts += [text.upper() for text in texts] + [text.lower() for text in texts]
        return texts

    def test_every_match_holds_a_needed_literal(self):
        rs = redact.ruleset()
        for rule in rs.patterns + rs.members:
            for text in self.corpus():
                for match in rule.pattern.finditer(text):
                    with self.subTest(rule=rule.id, match=match.group(0)[:40]):
                        self.assertTrue(rule.applies(match.group(0)))

    def test_gated_and_ungated_redaction_agree(self):
        from dataclasses import replace

        rs = redact.ruleset()
        open_rs = redact.Ruleset(tuple(replace(rule, needs=(), pre=None) for rule in rs.patterns),
                                 tuple(replace(rule, needs=(), pre=None) for rule in rs.members), allow=rs.allow)
        for text in self.corpus():
            self.assertEqual(redact.scrub_fields({"text": text}, rs, []), redact.scrub_fields({"text": text}, open_rs, []))

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

    def test_enable_preserves_existing_policy(self):
        config = {"enabled": False, "capture": "full", "sessionDefault": "off", "redactPaths": ["*.vault"],
                  "redactPatterns": ["private-marker"], "environment": "staging", "custom": {"keep": True}}
        sb = Sandbox(config)
        self.addCleanup(sb.cleanup)
        with redirect_stdout(io.StringIO()):
            result = cli.main(["--cwd", str(sb.project), "enable", "--session-default", "on"])
        self.assertEqual(result, 0)
        written = json.loads((sb.project / ".judgment" / "tracing.json").read_text())
        self.assertEqual(written, {**config, "enabled": True, "capture": "sanitized", "sessionDefault": "on"})

    def test_enable_refuses_malformed_existing_policy(self):
        sb = Sandbox()
        self.addCleanup(sb.cleanup)
        path = sb.project / ".judgment" / "tracing.json"
        for content in ("", "{", "[]", "null", "42"):
            with self.subTest(content=content):
                path.write_text(content)
                stderr = io.StringIO()
                with redirect_stdout(io.StringIO()), redirect_stderr(stderr):
                    result = cli.main(["--cwd", str(sb.project), "enable"])
                self.assertEqual(result, 1)
                self.assertIn("Cannot enable tracing", stderr.getvalue())
                self.assertEqual(path.read_text(), content)


_rng = random.Random(8080)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    return "".join(_rng.choice(alphabet) for _ in range(n - 1)) + _rng.choice(string.digits)


def outputs(test: unittest.TestCase, texts: list, cfg: dict | None = None, env: dict | None = None) -> list[dict]:
    return assert_parity(test, [tool({"command": "x"}, text) for text in texts], cfg, env)[0]


class EscapeBoundaryTests(unittest.TestCase):
    """A secret right after a JSON escape (\\n, \\t, \\r) in JSON nested in a string failed every identifier
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


class HiddenCallTests(unittest.TestCase):
    """A tool call that names a file in the project's redactPaths is blanked wherever the turn holds it, and only that call."""

    CFG = {"redactPaths": [".env*", "secrets.yaml"]}
    BLANK = "[REDACTED:custom_path]"

    def test_a_patch_counts_by_the_files_it_names(self):
        head = "*** Begin Patch\n*** Update File: "
        events = [tool_event("ok", tool_name="apply_patch", tool_input={"input": head + "config/.env.local\n+A=1\n"}),
                  tool_event("ok", tool_name="apply_patch", tool_input={"input": head + "src/a.ts\n+// values come from .env and secrets.yaml\n"}),
                  tool_event("ok", tool_input={"patch": "--- a/x.yaml\n+++ b/secrets.yaml\n@@\n+a: 1\n"}),
                  tool_event("ok", tool_name="Grep", tool_input={"pattern": "TODO", "glob": ".env*"}),
                  tool_event("ok", tool_name="Grep", tool_input={"pattern": "secrets.yaml", "path": "src"})]
        py, _ = assert_parity(self, events, self.CFG)
        self.assertEqual([event["tool_input"] == self.BLANK for event in py], [True, False, True, True, False])

    def test_only_the_call_is_blanked_and_it_is_recorded(self):
        secret = "Qx7Lm2Tz9Rk4" + "Wp8Hn3Vb6Yc1"
        py, _ = assert_parity(self, [
            {"v": 1, "harness": "pi", "event": "model", "session_id": "s", "turn_id": "t", "output": "reading", "tool_calls": [
                {"id": "a", "name": "read", "arguments": {"path": ".env"}}, {"id": "b", "name": "bash", "arguments": {"command": "ls"}}],
             "tool_results": [{"id": "a", "response": "COOKIE=" + secret}, {"id": "b", "response": "README.md"}]},
            tool_event("COOKIE=" + secret, tool_input={"command": "cat .env"}, prompt="why does the cookie fail"),
            {"v": 1, "harness": "pi", "event": "tool", "session_id": "s", "turn_id": "t", "tool_name": "read", "tool_call_id": "c", "tool_input": {"path": "secrets.yaml"}},
        ], self.CFG)
        self.assertEqual(py[0]["tool_calls"], [{"id": "a", "name": "read", "arguments": self.BLANK}, {"id": "b", "name": "bash", "arguments": {"command": "ls"}}])
        self.assertEqual(py[0]["tool_results"], [{"id": "a", "response": self.BLANK}, {"id": "b", "response": "README.md"}])
        self.assertEqual(py[0]["redactions"], [{"rule": "custom_path", "field": "tool_calls", "count": 1}, {"rule": "custom_path", "field": "tool_results", "count": 1}])
        self.assertEqual((py[0]["output"], py[1]["prompt"], py[1]["tool_name"]), ("reading", "why does the cookie fail", "Bash"))
        self.assertEqual(py[2]["tool_input"], self.BLANK)
        self.assertNotIn("tool_output", py[2])
        self.assertNotIn(secret, json.dumps(py))

    def test_the_sender_blanks_a_call_wherever_the_turn_holds_it(self):
        """What the hook never saw: a transcript's output over the hook's blanked event, and the messages that replay the call."""
        secret = "Qx7Lm2Tz9Rk4" + "Wp8Hn3Vb6Yc1"
        call = {"type": "tool_call", "id": "t1", "name": "Read", "arguments": {"file_path": "/app/.env"}}
        other = {"type": "tool_call", "id": "t2", "name": "Bash", "arguments": {"command": "ls"}}
        messages = [{"role": "user", "parts": [{"type": "text", "content": "fix the cookie"}]}, {"role": "assistant", "parts": [call, other]},
                    {"role": "tool", "parts": [{"type": "tool_call_response", "id": "t1", "response": "COOKIE=" + secret}]},
                    {"role": "tool", "parts": [{"type": "tool_call_response", "id": "t2", "response": "README.md"}]}]
        events = [
            # The hook blanked this event; enrichment then put the transcript's output back.
            tool_event("COOKIE=" + secret, tool_call_id="t1", tool_name="Read", tool_input=self.BLANK),
            # The two requests share the assistant message, as a transcript reader builds them.
            {"event": "model", "input_messages": messages[:1], "output_message": messages[1]},
            {"event": "model", "input_messages": messages, "output_message": {"role": "assistant", "parts": [{"type": "text", "content": "done"}]}},
        ]
        rs = redact.ruleset(self.CFG)
        out = redact.redact_events(events, rs, [])
        self.assertNotIn(secret, json.dumps(out))
        self.assertEqual(out[0]["tool_output"], self.BLANK)
        self.assertEqual([part.get("arguments") for part in out[1]["output_message"]["parts"]], [self.BLANK, {"command": "ls"}])
        self.assertEqual([message["parts"][0].get("response") for message in out[2]["input_messages"][2:]], [self.BLANK, "README.md"])
        self.assertEqual(out[2]["redactions"], [{"rule": "custom_path", "field": "input_messages", "count": 2}])
        self.assertEqual(events[2]["input_messages"][2]["parts"][0]["response"], "COOKIE=" + secret)  # the input is not changed in place
        self.assertEqual(redact.redact_events(out, rs, []), out)  # a second pass changes and counts nothing
        # A project that names no file hides no call.
        self.assertIn(secret, json.dumps(redact.redact_events(events[1:], redact.ruleset(), [])))


class NestedParityTests(unittest.TestCase):
    """The sender looked for nested JSON at an escaped backslash before a quote, the hook at an escaped quote."""

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


class ConfigTypeTests(unittest.TestCase):
    """A setting that is not a list of strings was skipped, so a rule the project asked for silently did nothing."""

    BAD = [{"redactPatterns": "hunter[0-9]+"}, {"redactPatterns": ["ok", 5]}, {"redactNames": "pin"}]

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
    """Finding 12: !name drops a built-in name."""

    def test_drop_a_name(self):
        value = "hu" + _rand(12)
        py = outputs(self, ["token=" + value, "password=" + value], {"redactNames": ["!token", "pin"]})
        self.assertEqual([out["tool_output"] for out in py], ["token=" + value, "password=[REDACTED:secret_assignment]"])

    def test_drop_an_unknown_name_fails_closed(self):
        py = outputs(self, ["hello"], {"redactNames": ["!pin"]})
        self.assertTrue(py[0].get("redaction_error"))


if __name__ == "__main__":
    unittest.main()
