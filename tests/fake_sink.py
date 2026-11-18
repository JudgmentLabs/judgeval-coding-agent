"""A local OTLP/HTTP sink that records each request. Stdlib only.

    python3 tests/fake_sink.py [port]    serve until killed, printing one line per request
"""

from __future__ import annotations

import gzip
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Sink:
    def __init__(self, port: int = 0, status: int = 200):
        self.requests: list[dict] = []
        self.status = status
        self._lock = threading.Lock()
        sink = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):  # noqa: N802
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if self.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                with sink._lock:
                    sink.requests.append({"path": self.path, "headers": dict(self.headers), "body": raw.decode("utf-8", "replace")})
                self.send_response(sink.status)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"{}")

            def log_message(self, *args):
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "Sink":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()

    def bodies(self) -> list[str]:
        with self._lock:
            return [item["body"] for item in self.requests]

    def text(self) -> str:
        """Every request body and header except the API key header."""
        with self._lock:
            return "\n".join(item["body"] + json.dumps({k: v for k, v in item["headers"].items() if k.lower() != "authorization"}) for item in self.requests)

    def spans(self) -> list[dict]:
        found = []
        for body in self.bodies():
            for resource in json.loads(body).get("resourceSpans", []):
                for scope in resource.get("scopeSpans", []):
                    found.extend(scope.get("spans", []))
        return found


def attrs(span: dict) -> dict:
    out = {}
    for item in span.get("attributes", []):
        value = item["value"]
        out[item["key"]] = next(iter(value.values())) if value else None
    return out


def roots(spans: list[dict]) -> list[dict]:
    return [span for span in spans if not span.get("parentSpanId")]


if __name__ == "__main__":
    with Sink(int(sys.argv[1]) if len(sys.argv) > 1 else 0) as sink:
        print(sink.url, flush=True)
        seen = 0
        try:
            while True:
                threading.Event().wait(0.5)
                while seen < len(sink.requests):
                    print(f"POST {sink.requests[seen]['path']} {len(sink.requests[seen]['body'])} bytes", flush=True)
                    seen += 1
        except KeyboardInterrupt:
            pass
