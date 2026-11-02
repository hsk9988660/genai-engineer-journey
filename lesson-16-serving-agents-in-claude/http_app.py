"""A thin HTTP front end on the standard library: POST /v1/chat, GET /healthz, /readyz, /metrics.

It exists to show the parts of serving that are about HTTP, not about the model:
  * WHO the caller is comes from the Authorization header (a token mapped to a user id). A `user_id` in the body is ignored
  * an overloaded server answers 429 with Retry-After, a draining one 503, and a body that is too large is answered 413 after a bounded drain (answering before reading can reset the connection)
  * error bodies carry a status and a short code, never a traceback or user text
  * /healthz says the process is alive. /readyz says it should receive traffic (not draining, provider circuit not open, store answering)
  * /metrics needs an admin token
This is the standard library's threaded server. It is fine for the lesson and not what you would put in front of the internet: use a real server, TLS, and a gateway.
"""

import hashlib
import hmac
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from service import AgentService, Request

MAX_BODY = 16_384
DRAIN_LIMIT = 1_048_576


class Authenticator:
    """Maps bearer tokens to user ids. Only hashes of tokens are held, and comparison is constant-time."""

    def __init__(self, tokens: dict[str, str]):
        self._by_hash = {hashlib.sha256(t.encode()).hexdigest(): u for t, u in tokens.items()}

    def user_for(self, header: str | None) -> str | None:
        if not header or not header.startswith("Bearer "):
            return None
        h = hashlib.sha256(header[7:].strip().encode()).hexdigest()
        for known, user in self._by_hash.items():
            if hmac.compare_digest(known, h):
                return user
        return None


def make_server(service: AgentService, auth: Authenticator, admin_token: str, host: str = "127.0.0.1", port: int = 0) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # the default logs the request line to stderr, which can contain data
            pass

        def _send(self, code: int, body: dict | str, headers: dict | None = None):
            raw = (body if isinstance(body, str) else json.dumps(body)).encode()
            self.send_response(code)
            self.send_header("Content-Type", "text/plain" if isinstance(body, str) else "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in (headers or {}).items():
                self.send_header(k, str(v))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path == "/healthz":
                return self._send(200, "ok")
            if self.path == "/readyz":
                r = service.ready()
                return self._send(200 if r["ready"] else 503, r)
            if self.path == "/metrics":
                if not hmac.compare_digest(self.headers.get("X-Admin-Token", ""), admin_token):
                    return self._send(403, {"error": "forbidden"})
                return self._send(200, service.metrics.lines())
            self._send(404, {"error": "not_found"})

        def do_POST(self):
            if self.path != "/v1/chat":
                return self._send(404, {"error": "not_found"})
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                length = -1
            if length < 0:
                return self._send(411, {"error": "length_required"}, {"Connection": "close"})
            if length > MAX_BODY:
                self.rfile.read(min(length, DRAIN_LIMIT))
                return self._send(413, {"error": "too_large"}, {"Connection": "close"})
            raw = self.rfile.read(length)
            user = auth.user_for(self.headers.get("Authorization"))
            if user is None:
                return self._send(401, {"error": "unauthorized"}, {"WWW-Authenticate": "Bearer"})
            try:
                body = json.loads(raw)
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                return self._send(400, {"error": "invalid_json"})
            try:
                resp = service.handle(Request(user_id=user, text=body.get("text"), thread_id=body.get("thread_id"), idempotency_key=body.get("idempotency_key"),
                                              deadline_s=min(float(body.get("deadline_s", 30)), 120) if isinstance(body.get("deadline_s", 30), (int, float)) else 30))
            except Exception:  # belt and braces: the service already catches everything, but a front end must not drop the connection either
                return self._send(500, {"error": "internal"})
            headers = {"X-Trace-Id": resp.trace_id}
            if resp.retry_after_s is not None:
                headers["Retry-After"] = str(max(1, int(resp.retry_after_s)))
            payload = {"status": resp.status, "text": resp.text, "thread_id": resp.thread_id, "version": resp.version, "tier": resp.tier, "cached": resp.cached, "trace_id": resp.trace_id}
            if resp.status != "ok":
                payload["detail"] = resp.detail
            self._send(resp.http, payload, headers)

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    return server
