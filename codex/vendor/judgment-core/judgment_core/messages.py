"""OpenTelemetry GenAI messages (``{role, parts}``), as AI SDK v7 writes them on a model call span.

The span builder and the transcript readers both build messages here, so an
llm span's output and the copy of it that the next request's input replays
are the same message.
"""

from __future__ import annotations

from typing import Any


def text_message(role: str, text: str) -> dict:
    return {"role": role, "parts": [{"type": "text", "content": text}]}


def tool_call(call_id: str, name: str, arguments: Any) -> dict:
    return {"type": "tool_call", "id": call_id, "name": name, "arguments": arguments}


def assistant_message(text: str, calls: list[dict], reasoning: list[str] | None = None) -> dict:
    """Reasoning, then text, then tool calls. An empty reply keeps no parts, so the view draws no blank bubble."""
    parts: list[dict] = [{"type": "reasoning", "content": thought} for thought in reasoning or [] if thought]
    if text:
        parts.append({"type": "text", "content": text})
    parts.extend(tool_call(call["id"], call["name"], call["arguments"]) for call in calls)
    return {"role": "assistant", "parts": parts}


def tool_message(call_id: str, response: Any) -> dict:
    return {"role": "tool", "parts": [{"type": "tool_call_response", "id": call_id, "response": response}]}


def media_part(modality: str, source: str, mime_type: str | None = None) -> dict:
    """An image or file the model read: a base64 data URL becomes a blob, anything else a uri."""
    if source.startswith("data:") and ";base64," in source:
        header, data = source[5:].split(";base64,", 1)
        return {"type": "blob", "modality": modality, "mime_type": header or mime_type, "content": data}
    return {"type": "uri", "modality": modality, "mime_type": mime_type, "uri": source}
