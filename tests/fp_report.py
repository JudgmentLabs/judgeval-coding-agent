"""Normal content that must pass both redaction layers byte for byte, except the cases in OVER_REDACTED.

    python3 tests/fp_report.py     print case -> redacted? for the hook (jq) and the sender (Python)
"""

from __future__ import annotations

import base64
import hashlib
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from support import jq_sanitize, py_sanitize  # noqa: E402

_sha1 = hashlib.sha1(b"commit").hexdigest()
_sha256 = hashlib.sha256(b"artifact").hexdigest()
_sha512 = hashlib.sha512(b"artifact").digest()
_uuid = str(uuid.UUID(int=0x123E4567E89B12D3A456426614174000))

CORPUS: list[tuple[str, str]] = [
    ("git sha", f"commit {_sha1}\nAuthor: Pat <pat@example.com>\n\n    Fix the build"),
    ("short sha", f"HEAD is now at {_sha1[:7]} Fix the build"),
    ("sha256 digest", f"{_sha256}  dist/app.tar.gz"),
    ("sha512 hex", hashlib.sha512(b"x").hexdigest()),
    ("uuid", f"request_id={_uuid} session {_uuid.upper()}"),
    ("base64 id", "id: " + base64.b64encode(b"user:1234:profile:settings").decode()),
    ("lockfile integrity", '"integrity": "sha512-' + base64.b64encode(_sha512).decode() + '"'),
    ("yarn lock", "  integrity sha512-" + base64.b64encode(_sha512).decode()),
    ("docker digest", f"FROM python:3.13-slim@sha256:{_sha256}"),
    ("max_tokens", "response = client.messages.create(model=MODEL, max_tokens=4096, messages=messages)"),
    ("tokenizer", "tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')\ntoken_count = len(tokenizer.encode(text))"),
    ("get_token", "def get_token():\n    return session.token\n\ntoken = get_token()"),
    ("password_field", "password_field = forms.CharField(widget=forms.PasswordInput)"),
    ("api_key env ref", 'api_key = os.environ["API_KEY"]\napi_key = os.getenv("API_KEY")'),
    ("process.env ref", "const apiKey = process.env.OPENAI_API_KEY;\nconst token = process.env.GITHUB_TOKEN"),
    ("k8s secret name", "secret_name: my-k8s-secret\nsecretName: db-credentials-v2"),
    ("bearer placeholder", 'curl -H "Authorization: Bearer ${TOKEN}" https://api.example.com'),
    ("auth header ref", "headers = {'Authorization': f'Bearer {token}'}"),
    ("placeholders", "API_KEY=<your-key>\nTOKEN=xxx\nPASSWORD=changeme\nSECRET=your_secret_here\npassword: ********"),
    ("github actions", "token: ${{ secrets.GITHUB_TOKEN }}\npassword: ${{ secrets.DB_PASSWORD }}"),
    ("prose", "The access token expires after an hour. Keep the secret safe, and rotate the password."),
    ("markdown", "## Tokens\n\n- `max_tokens`: the most tokens to sample.\n- Secrets go in the vault.\n\n| Key | Value |\n| --- | --- |\n| a | 1 |"),
    ("json numbers", '{"input_tokens": 1200, "output_tokens": 80, "cost": 0.0123, "token_count": 42, "secret": false, "password": null}'),
    ("url without password", "git clone https://github.com/JudgmentLabs/judgeval-coding-agent.git\nssh://git@github.com/org/repo.git"),
    ("url env password", "postgres://app:${DB_PASSWORD}@localhost:5432/app"),
    ("task words", "ask-the-team-about-the-release-plan task-runner-settings-file"),
    ("aws doc words", "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in your shell."),
    ("pem public cert", "-----BEGIN CERTIFICATE-----\nMIIB" + "A" * 60 + "\n-----END CERTIFICATE-----"),
    ("ssh public key", "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI" + "B" * 40 + " pat@laptop"),
    ("python dict", "config = {'token_limit': 1000, 'secret_rotation_days': 30}"),
    ("env var names", "export PATH=$PATH:/usr/local/bin\nexport NODE_ENV=production"),
    ("unicode", "Résumé — naïve café: token budget ✓"),
    ("member expressions", "api_key = config2.api_key\ntoken = response_v2.token\nsecret = settings.SECRET_KEY_V2\npassword = form2.cleaned_data"),
    ("versioned identifiers", "api_key = api_key_v2\ntoken: tokenV2\naccess_token = self.client_v3.auth.token"),
    ("structured members", '{"token": "the", "max_tokens": 4096, "key": "user_id_2024_v1", "password_hint": "ask Pat", "api_key": "${API_KEY}"}'),
    ("env commands", "env FOO=1 make test && /usr/bin/env python3 run.py\ngrep -rn credentials src/"),
]

# Known limit (reference-value): a reference, placeholder, constant, or expression after a secret name is redacted like a
# value. These cases change, and only by secret_assignment.
OVER_REDACTED = {"get_token", "api_key env ref", "process.env ref", "placeholders", "github actions", "member expressions",
                 "versioned identifiers", "structured members"}


def check() -> list[dict]:
    """For each case: did jq or Python change it, and how many redactions each recorded."""
    events = [{"v": 1, "event": "tool", "session_id": "s", "turn_id": "t", "tool_output": text, "prompt": text} for _, text in CORPUS]
    js = jq_sanitize(events)
    py = py_sanitize(events)
    rows = []
    for (name, text), j, p in zip(CORPUS, js, py):
        rows.append(
            {
                "case": name,
                "jq": j.get("tool_output") != text or j.get("prompt") != text,
                "py": p.get("tool_output") != text or p.get("prompt") != text,
                "count": sum(item["count"] for item in j.get("redactions", []) + p.get("redactions", [])),
                "rules": sorted({item["rule"] for item in j.get("redactions", []) + p.get("redactions", [])}),
                "sensitive": bool(j.get("sensitive") or p.get("sensitive")),
            }
        )
    return rows


def table(rows: list[dict]) -> str:
    width = max(len(row["case"]) for row in rows)
    lines = [f"{'case'.ljust(width)}  redacted? jq  redacted? python", f"{'-' * width}  ------------  ----------------"]
    for row in rows:
        lines.append(f"{row['case'].ljust(width)}  {'yes' if row['jq'] else 'no':<12}  {'yes' if row['py'] else 'no'}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(table(check()))
