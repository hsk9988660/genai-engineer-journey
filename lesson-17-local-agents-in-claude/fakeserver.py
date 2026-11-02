"""A tiny OpenAI-compatible server on 127.0.0.1, for testing the HTTP client and the evaluation script without any real model.

`responder(request_json, call_index)` returns `(status, body, delay_s)`; `body` is a dict (sent as JSON) or bytes (sent as is, for malformed-response tests).
It records every request it receives.
"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeLocalServer:
    def __init__(self, responder):
        self.responder, self.requests, self._lock = responder, [], threading.Lock()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                try:
                    req = json.loads(raw)
                except ValueError:
                    req = None
                with outer._lock:
                    idx = len(outer.requests)
                    outer.requests.append({"path": self.path, "headers": dict(self.headers), "json": req})
                if self.path != "/v1/chat/completions":
                    status, body, delay = 404, {"error": "not found"}, 0
                else:
                    status, body, delay = outer.responder(req, idx)
                if delay:
                    time.sleep(delay)
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except OSError:  # the client hung up (a timeout test does this on purpose)
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
