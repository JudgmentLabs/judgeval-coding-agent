"""Shared test helpers. Every fake secret is built at runtime from parts, so no file holds a realistic token."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import string
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CORE = REPO / "core"
HOOK = CORE / "bin" / "judgment-hook"
MAPS = {name: REPO / folder / "hooks" / "map.jq" for name, folder in (("claude", "claude-code"), ("codex", "codex"), ("cursor", "cursor"))}

sys.path.insert(0, str(CORE))

_rng = random.Random(1729)


def _rand(n: int, alphabet: str = string.ascii_letters + string.digits) -> str:
    # Mixed case with at least one digit, like a real token.
    body = "".join(_rng.choice(alphabet) for _ in range(n - 1))
    return body + _rng.choice(string.digits)


def build_secrets() -> dict[str, dict[str, str]]:
    """Fake secrets by rule: ``text`` is how it appears in agent text, ``core`` is the secret part itself."""
    found: dict[str, dict[str, str]] = {}

    def add(rule: str, core: str, text: str | None = None) -> None:
        # ``random`` drops the token's public prefix, such as "github_pat_", which names the rule and is no secret.
        prefix = next((p for p in ("sk-proj-", "sk-ant-api03-", "github_pat_", "xoxb-", "sk_live_", "AKIA", "AIza", "ghp_", "jud") if core.startswith(p)), "")
        found[rule] = {"core": core, "text": text if text is not None else core, "random": core[len(prefix):]}

    add("openai_key", "sk-" + "proj-" + _rand(48))
    add("anthropic_key", "sk-" + "ant-" + "api03-" + _rand(60))
    add("aws_access_key_id", "AK" + "IA" + "".join(_rng.choice(string.ascii_uppercase + string.digits) for _ in range(15)) + "7")
    add("github_token", "gh" + "p_" + _rand(36))
    add("github_pat", "github" + "_pat_" + _rand(22) + "_" + _rand(40))
    add("slack_token", "xo" + "xb-" + "".join(_rng.choice(string.digits) for _ in range(11)) + "-" + _rand(24))
    add("google_api_key", "AI" + "za" + _rand(35, string.ascii_letters + string.digits + "_"))
    add("stripe_key", "sk" + "_live_" + _rand(24))
    jwt = "ey" + "J" + _rand(20) + "." + "ey" + "J" + _rand(30) + "." + _rand(43, string.ascii_letters + string.digits + "_-")
    add("jwt", jwt)
    pem_body = "MII" + _rand(60, string.ascii_letters + string.digits + "+/") + "\n" + _rand(64, string.ascii_letters + string.digits + "+/")
    begin, end = "-----BEGIN " + "RSA PRIV" + "ATE KEY-----", "-----END " + "RSA PRIV" + "ATE KEY-----"
    add("private_key", pem_body, begin + "\n" + pem_body + "\n" + end)
    bearer = "tok" + _rand(29)
    add("bearer_token", bearer, "curl -H 'Authorization: Bearer " + bearer + "' https://api.example.com")
    url_pass = "Pw" + _rand(14)
    add("url_password", url_pass, "postgres://admin:" + url_pass + "@db.internal:5432/app")
    password = "hu" + "nter" + _rand(10)
    add("secret_assignment", password, "DB_PASS" + "WORD=" + password)
    api_value = "Zq" + _rand(30)
    add("json_api_key", api_value, '{"api' + '_key": "' + api_value + '"}')
    export_value = _rand(28)
    add("export_token", export_value, "export DEPLOY" + "_TOKEN=" + export_value)
    aws_secret = _rand(40, string.ascii_letters + string.digits + "/+")
    add("aws_secret", aws_secret, "aws_secret" + "_access_key = " + aws_secret)
    add("env_value", _rand(32))
    add("judgment_key", "jud" + _rand(37))
    return found


SECRETS = build_secrets()
ENV_SECRET_NAME = "MY_CUSTOM_SERVICE" + "_TOKEN"


def secret_cores() -> list[str]:
    return [item["core"] for item in SECRETS.values()]


def secret_blob() -> str:
    """Every fake secret, as an agent would see it in text, one per line."""
    lines = [item["text"] for rule, item in SECRETS.items() if rule not in ("env_value", "judgment_key")]
    lines.append("my service token is " + SECRETS["env_value"]["core"])
    lines.append("the judgment key " + SECRETS["judgment_key"]["core"])
    return "\n".join(lines)


def leaks(text: str, window: int = 6, hashes: bool = True) -> list[str]:
    """Which fake secrets appear in ``text``, whole or as any ``window``-long piece, or as a common hash.

    The state and the requests hold the API key's fingerprint on purpose, so they check with ``hashes=False``.
    """
    found = []
    for rule, item in SECRETS.items():
        core = item["random"].replace("\n", "")
        pieces = {core[i : i + window] for i in range(0, max(len(core) - window + 1, 1))}
        digests = set()
        for algo in ("sha256", "sha1", "md5") if hashes else ():
            digest = hashlib.new(algo, item["core"].encode()).hexdigest()
            digests |= {digest, digest[:8], digest[:12], digest[:16], digest[:32]}
        if item["core"] in text or any(piece in text for piece in pieces) or any(d in text for d in digests):
            found.append(rule)
    return found


def whole_leaks(text: str) -> list[str]:
    return [rule for rule, item in SECRETS.items() if item["core"] in text]


class Sandbox:
    """A temp HOME, state dir, and opted-in project. The env holds no real variable but PATH."""

    def __init__(self, tracing: dict | None = {"enabled": True}, sink_url: str = "http://127.0.0.1:9"):
        self.dir = Path(tempfile.mkdtemp(prefix="jt-test-"))
        self.home = self.dir / "home"
        self.state = self.dir / "state"
        self.project = self.home / "proj"
        self.project.mkdir(parents=True)
        (self.home / ".config" / "judgment").mkdir(parents=True)
        if tracing is not None:
            (self.project / ".judgment").mkdir()
            (self.project / ".judgment" / "tracing.json").write_text(json.dumps(tracing) + "\n")
        self.env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.home),
            "LANG": "C.UTF-8",
            "JUDGMENT_STATE_DIR": str(self.state),
            "JUDGMENT_API_URL": sink_url,
            "JUDGMENT_API_KEY": SECRETS["judgment_key"]["core"],
            "JUDGMENT_PROJECT_ID": "proj-test-1",
            "JUDGMENT_ORG_ID": "org-test-1",
            "JUDGMENT_FLUSH": "0",
            "JUDGMENT_FLUSH_GRACE_MS": "0",
            ENV_SECRET_NAME: SECRETS["env_value"]["core"],
            "PYTHONDONTWRITEBYTECODE": "1",
        }

    def set_tracing(self, tracing: dict) -> None:
        (self.project / ".judgment" / "tracing.json").write_text(json.dumps(tracing) + "\n")

    def hook(self, adapter: str, event: str, payload: dict, *, env: dict | None = None, hook: Path = HOOK) -> subprocess.CompletedProcess:
        if adapter == "pi":
            args = [str(hook), "--raw"]
        else:
            block = {"claude": "claude", "codex": "codex", "cursor": "cursor"}[adapter]
            args = [str(hook), "--map", str(hook.parent.parent.parent / MAPS[adapter].relative_to(REPO)), "--event", event, "--block", block]
        return subprocess.run(args, input=json.dumps(payload), capture_output=True, text=True, env={**self.env, **(env or {})}, timeout=30)

    def flush(self, env: dict | None = None) -> subprocess.CompletedProcess:
        full = {**self.env, "PYTHONPATH": str(CORE), **(env or {})}
        return subprocess.run([sys.executable, "-m", "judgment_core.flush"], capture_output=True, text=True, env=full, timeout=60)

    def state_text(self) -> str:
        """Everything under the state dir: inbox, turns, index, fold, flush.log."""
        parts = []
        for path in sorted(self.state.rglob("*")):
            if path.is_file():
                parts.append(f"== {path.relative_to(self.state)}\n" + path.read_text(encoding="utf-8", errors="replace"))
        return "\n".join(parts)

    def inbox(self) -> list[dict]:
        folder = self.state / "inbox"
        return [json.loads(path.read_text()) for path in sorted(folder.iterdir())] if folder.is_dir() else []

    def cleanup(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


def iso(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def claude_transcript(path: Path, session: str, prompt_id: str, prompt: str, start_ms: int, tool_id: str, command: str, tool_text: str, answer: str) -> None:
    rows = [
        {"type": "user", "sessionId": session, "promptId": prompt_id, "uuid": prompt_id + "-u", "timestamp": iso(start_ms), "message": {"role": "user", "content": prompt}},
        {
            "type": "assistant", "sessionId": session, "promptId": prompt_id, "requestId": prompt_id + "-r1", "timestamp": iso(start_ms + 1000),
            "message": {"id": prompt_id + "-m1", "role": "assistant", "model": "claude-sonnet-5-5", "content": [
                {"type": "text", "text": "Let me look."}, {"type": "tool_use", "id": tool_id, "name": "Bash", "input": {"command": command}}],
                "usage": {"input_tokens": 1200, "output_tokens": 80, "cache_read_input_tokens": 300}},
        },
        {"type": "user", "sessionId": session, "promptId": prompt_id, "timestamp": iso(start_ms + 2000), "toolUseResult": {"stdout": tool_text},
         "message": {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_id, "content": tool_text}]}},
        {
            "type": "assistant", "sessionId": session, "promptId": prompt_id, "requestId": prompt_id + "-r2", "timestamp": iso(start_ms + 3000),
            "message": {"id": prompt_id + "-m2", "role": "assistant", "model": "claude-sonnet-5-5", "content": [{"type": "text", "text": answer}],
                        "usage": {"input_tokens": 1500, "output_tokens": 120}},
        },
    ]
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def wait_for(predicate, timeout: float = 20.0, step: float = 0.1) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(step)
    return predicate()


def clean_env(**extra: str) -> dict:
    return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": "/home/tester", "LANG": "C.UTF-8", **extra}


def jq(filter_text: str, inputs: list, *, cfg: dict | None = None, env: dict | None = None, bin_dir: Path = CORE / "bin") -> list:
    """Run ``filter_text`` over each input in one jq, with prepare.jq loaded. An error becomes {"__error": msg}."""
    program = 'include "prepare"; . as $in | try (' + filter_text + ') catch {"__error": .}'
    result = subprocess.run(
        ["jq", "-L", str(bin_dir), "-c", "--argjson", "cfg", json.dumps(cfg or {}), program],
        input="\n".join(json.dumps(item) for item in inputs),
        capture_output=True,
        text=True,
        env=env or clean_env(),
        timeout=60,
    )
    if result.returncode != 0:
        raise AssertionError(result.stderr)
    return [json.loads(line) for line in result.stdout.splitlines() if line.strip()]


def py_sanitize(events: list[dict], cfg: dict | None = None, env: dict | None = None) -> list[dict]:
    from judgment_core import redact

    env = env or clean_env()
    rs = redact.ruleset(cfg or {})
    secrets = redact.env_secrets(env)
    return [redact.sanitize_event(event, rs, secrets) for event in events]


_STAMPS = ("capture", "environment", "project_root")


def jq_sanitize(events: list[dict], cfg: dict | None = None, env: dict | None = None) -> list[dict]:
    """Layer 1 as the hook runs it, without the stamps prepare adds after redacting."""
    out = jq('judgment_prepare({enabled: true} + $cfg; "on"; "/r"; "")', events, cfg=cfg, env=env)
    cleaned = []
    for event, result in zip(events, out):
        result = {key: value for key, value in result.items() if key not in _STAMPS and not (key == "ts" and "ts" not in event)}
        cleaned.append(result)
    return cleaned


def norm(events: list[dict]) -> list[dict]:
    """Events with ``redactions`` in a stable order, for comparing jq with Python."""
    out = []
    for event in events:
        event = dict(event)
        if "redactions" in event:
            event["redactions"] = sorted(event["redactions"], key=lambda item: (item["field"], item["rule"]))
        out.append(event)
    return out
