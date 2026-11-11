"""A scripted OpenAI-compatible chat server for the real-Hermes test. Stdlib only.

Each request gets a reply from ``script``: the next item of a list, or ``script(body, n)`` when it
is a function. A reply is ``{"tool": name, "args": {...}}`` for one tool call, ``{"text": "..."}``
for a final answer, or ``{"status": 500}`` for an HTTP error. With no script left it answers
"done". Streaming (SSE) and plain JSON are both served, with usage on the last chunk.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "mock-model-1"


class MockLLM:
    def __init__(self, script=None):
        self.script = script if callable(script) else list(script or [])
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        llm = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, status: int, body: bytes, kind: str = "application/json") -> None:
                self.send_response(status)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):  # noqa: N802
                if self.path.rstrip("/").endswith("/models"):
                    self._send(200, json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model", "owned_by": "mock", "context_length": 128000}]}).encode())
                else:
                    self._send(404, b"{}")

            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    body = json.loads(raw or b"{}")
                except ValueError:
                    body = {}
                if not self.path.rstrip("/").endswith("/chat/completions"):
                    self._send(404, b"{}")
                    return
                with llm._lock:
                    llm.requests.append(body)
                    n = len(llm.requests)
                    if callable(llm.script):
                        reply = llm.script(body, n) or {"text": "done"}
                    else:
                        reply = llm.script.pop(0) if llm.script else {"text": "done"}
                if "status" in reply:
                    error = {"error": {"message": reply.get("message", "mock upstream error"), "type": "server_error", "code": reply["status"]}}
                    self._send(int(reply["status"]), json.dumps(error).encode())
                    return
                message: dict = {"role": "assistant", "content": None}
                finish = "stop"
                if "tool" in reply:
                    call = {"id": f"call_{n}", "type": "function", "function": {"name": reply["tool"], "arguments": json.dumps(reply.get("args") or {})}}
                    message["tool_calls"] = [call]
                    finish = "tool_calls"
                else:
                    message["content"] = reply.get("text", "done")
                usage = {"prompt_tokens": 1000 + n, "completion_tokens": 20 + n, "total_tokens": 1020 + 2 * n}
                created = int(time.time())
                if body.get("stream"):
                    chunks = []
                    base = {"id": f"chatcmpl-{n}", "object": "chat.completion.chunk", "created": created, "model": MODEL}
                    if message.get("tool_calls"):
                        call = message["tool_calls"][0]
                        delta = {"role": "assistant", "tool_calls": [{"index": 0, "id": call["id"], "type": "function", "function": {"name": call["function"]["name"], "arguments": call["function"]["arguments"]}}]}
                    else:
                        delta = {"role": "assistant", "content": message["content"]}
                    chunks.append({**base, "choices": [{"index": 0, "delta": delta, "finish_reason": None}]})
                    chunks.append({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
                    chunks.append({**base, "choices": [], "usage": usage})
                    data = "".join(f"data: {json.dumps(chunk)}\n\n" for chunk in chunks) + "data: [DONE]\n\n"
                    self._send(200, data.encode(), "text/event-stream")
                else:
                    out = {"id": f"chatcmpl-{n}", "object": "chat.completion", "created": created, "model": MODEL,
                           "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
                    self._send(200, json.dumps(out).encode())

            def log_message(self, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "MockLLM":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()
