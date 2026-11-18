from __future__ import annotations

import hashlib

from judgment_core.redact import jq_json


def text_id(value: str) -> str:
    return hashlib.sha256(jq_json(value).encode("utf-8")).hexdigest()[:32]
