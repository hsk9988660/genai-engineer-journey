"""An A2A-INSPIRED task protocol over HTTP. It is NOT wire-compatible with the real A2A specification.

I am not certain of the real spec's exact method names, field names and well-known paths (they have changed between
versions), so this deliberately uses its own names (task.create, task.get, ...) instead of claiming conformance. It exists to
teach the ideas that make agent-to-agent delegation different from calling a tool, and to test the security properties.
For real interoperability use the official A2A SDK.

The ideas:
  Agent card   a public JSON description of the agent (name, skills, endpoint, how to authenticate). Discovery is cheap
               and unauthenticated, so the client must not blindly trust it: here the card's endpoint must be on the same origin.
  Task         a unit of delegated work with a LIFECYCLE: submitted -> working -> (input_required <-> working) -> completed |
               failed | canceled. Terminal states are final. This is the difference from a tool call: the work can take a
               long time, ask a question mid-way, and be cancelled.
  Artifact     what a task produces.
  Auth         the RPC endpoint needs a bearer token. A client can only see its own tasks. The card is public.
  Limits       request body cap, per-client task cap, text length cap.
Remote agent output is UNTRUSTED text: render_for_model() wraps it as data, the same rule as a web page or an MCP result.
"""

import hmac
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import jsonrpc as jr

CARD_PATH = "/.well-known/agent-card.json"  # A2A-inspired: the real spec's path has varied between versions
UNAUTHORIZED = -32001
DRAIN_LIMIT = 1_048_576  # at most this much of a rejected request body is read and discarded
TERMINAL = {"completed", "failed", "canceled"}
TRANSITIONS = {"submitted": {"working", "canceled", "failed"},
               "working": {"input_required", "completed", "failed", "canceled"},
               "input_required": {"working", "canceled", "failed"},
               "completed": set(), "failed": set(), "canceled": set()}


@dataclass
class Task:
    id: str
    client: str
    messages: list
    state: str = "submitted"
    artifacts: list = field(default_factory=list)
    question: str = ""
    error: str = ""

    def view(self) -> dict:
        return {"id": self.id, "state": self.state, "question": self.question, "artifacts": list(self.artifacts), "error": self.error}


class TaskStore:
    def __init__(self, max_active_per_client: int):
        self.max_active, self._tasks, self._lock = max_active_per_client, {}, threading.RLock()

    def create(self, client: str, text: str) -> Task:
        with self._lock:
            active = sum(1 for t in self._tasks.values() if t.client == client and t.state not in TERMINAL)
            if active >= self.max_active:
                raise jr.JsonRpcError(jr.INVALID_REQUEST, f"too many active tasks (limit {self.max_active})")
            task = Task(uuid.uuid4().hex, client, [{"role": "user", "text": text}])
            self._tasks[task.id] = task
            return task

    def get(self, client: str, task_id) -> Task | None:
        with self._lock:
            task = self._tasks.get(task_id) if isinstance(task_id, str) else None
            return task if task and task.client == client else None  # another client's task looks exactly like a missing one

    def transition(self, task: Task, new_state: str) -> bool:
        with self._lock:
            if new_state not in TRANSITIONS[task.state]:
                return False
            task.state = new_state
            return True


class A2AServer:
    def __init__(self, card: dict, handler, tokens: dict, max_body: int = 65_536, max_active_per_client: int = 5, max_text: int = 4000):
        """handler(messages) -> ("completed", text) | ("input_required", question) | ("failed", reason)"""
        self.card, self.handler, self.tokens = dict(card), handler, tokens  # tokens: client_id -> bearer token
        self.max_body, self.max_text = max_body, max_text
        self.store = TaskStore(max_active_per_client)
        self.httpd = None

    def start(self) -> str:
        a2a = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, code: int, obj: dict) -> None:
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path == CARD_PATH:
                    self._send(200, a2a.card)
                else:
                    self._send(404, {"error": "not found"})

            def _drain(self, length: int) -> None:
                """Read (and discard) up to DRAIN_LIMIT bytes of the request body before answering a rejection.

                If a server replies and closes while the client is still sending, the OS can reset the connection and
                destroy the reply, so the client sees a network error instead of the 401 or 413. The drain is bounded so
                a hostile client cannot make the server read forever.
                """
                remaining = min(max(length, 0), DRAIN_LIMIT)
                while remaining > 0:
                    chunk = self.rfile.read(min(remaining, 65_536))
                    if not chunk:
                        break
                    remaining -= len(chunk)

            def do_POST(self):
                if self.path != "/rpc":
                    return self._send(404, {"error": "not found"})
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    length = -1
                if length < 0:
                    return self._send(411, jr.error(None, jr.INVALID_REQUEST, "Content-Length required"))
                client = a2a._authenticate(self.headers.get("Authorization", ""))
                if client is None:
                    self._drain(length)
                    return self._send(401, jr.error(None, UNAUTHORIZED, "missing or invalid bearer token"))
                if length > a2a.max_body:
                    self._drain(length)
                    self.close_connection = True
                    return self._send(413, jr.error(None, jr.INVALID_REQUEST, f"request body over {a2a.max_body} bytes"))
                try:
                    msg = jr.decode(self.rfile.read(length).decode("utf-8", "replace"))
                except jr.JsonRpcError as exc:
                    return self._send(200, jr.error(None, exc.code, exc.message))
                self._send(200, a2a._handle(client, msg))

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.card.setdefault("url", base + "/rpc")
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        return base

    def stop(self) -> None:
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()

    def _authenticate(self, header: str) -> str | None:
        scheme, _, token = header.partition(" ")
        if scheme.lower() != "bearer" or not token:
            return None
        found = None
        for client, expected in self.tokens.items():  # compare against every token: no early exit that leaks which one matched
            if hmac.compare_digest(token.encode(), expected.encode()):
                found = client
        return found

    def _handle(self, client: str, msg: dict) -> dict:
        if jr.kind(msg) != "request":
            return jr.error(None, jr.INVALID_REQUEST, "expected a request")
        try:
            return jr.result(msg["id"], self._dispatch(client, msg["method"], msg.get("params") or {}))
        except jr.JsonRpcError as exc:
            return jr.error(msg["id"], exc.code, exc.message)
        except Exception:
            return jr.error(msg["id"], jr.INTERNAL_ERROR, "internal error")

    def _text(self, params: dict) -> str:
        text = params.get("text")
        if not isinstance(text, str) or not text.strip() or len(text) > self.max_text:
            raise jr.JsonRpcError(jr.INVALID_PARAMS, f"text must be a non-empty string up to {self.max_text} characters")
        return text

    def _task(self, client: str, params: dict) -> Task:
        task = self.store.get(client, params.get("id"))
        if task is None:
            raise jr.JsonRpcError(jr.INVALID_PARAMS, "no such task")
        return task

    def _dispatch(self, client: str, method: str, params: dict):
        if method == "task.create":
            task = self.store.create(client, self._text(params))
            threading.Thread(target=self._run, args=(task,), daemon=True).start()
            return task.view()
        if method == "task.get":
            return self._task(client, params).view()
        if method == "task.reply":
            task, text = self._task(client, params), self._text(params)
            if task.state != "input_required":
                raise jr.JsonRpcError(jr.INVALID_PARAMS, "the task is not waiting for input")
            task.messages.append({"role": "user", "text": text})
            self.store.transition(task, "working")
            threading.Thread(target=self._run, args=(task, False), daemon=True).start()
            return task.view()
        if method == "task.cancel":
            task = self._task(client, params)
            if not self.store.transition(task, "canceled"):
                raise jr.JsonRpcError(jr.INVALID_PARAMS, f"the task already finished ({task.state})")
            return task.view()
        raise jr.JsonRpcError(jr.METHOD_NOT_FOUND, f"Method not found: {method}")

    def _run(self, task: Task, first: bool = True) -> None:
        if first:
            self.store.transition(task, "working")
        try:
            kind, payload = self.handler(list(task.messages))
        except Exception:
            kind, payload = "failed", "The agent failed."  # never leak internals to the caller
        with self.store._lock:
            if task.state in TERMINAL:  # cancelled while the handler was running: the late result is discarded
                return
            if kind == "completed":
                task.artifacts.append({"name": "result", "text": str(payload)})
            elif kind == "input_required":
                task.question = str(payload)
            else:
                task.error = str(payload)
            self.store.transition(task, kind if kind in ("completed", "input_required", "failed") else "failed")


class UntrustedCard(Exception):
    pass


class A2AError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None  # never follow a redirect: it could lead a credentialed request somewhere else


class A2AClient:
    def __init__(self, base_url: str, token: str, timeout_s: float = 5.0):
        self.base, self.token, self.timeout_s = base_url.rstrip("/"), token, timeout_s
        self._open = urllib.request.build_opener(_NoRedirect).open

    def discover(self) -> dict:
        """Fetch the public card. Refuse a card whose endpoint is on a different origin than the one we asked."""
        try:
            with self._open(self.base + CARD_PATH, timeout=self.timeout_s) as resp:
                card = json.loads(resp.read())
        except (urllib.error.URLError, ValueError, OSError) as exc:
            raise A2AError(jr.INTERNAL_ERROR, f"could not fetch the agent card: {exc}") from exc
        if not isinstance(card, dict) or not all(k in card for k in ("name", "description", "url", "skills")):
            raise UntrustedCard("the agent card is missing required fields")
        a, b = urllib.parse.urlparse(card["url"]), urllib.parse.urlparse(self.base)
        if (a.scheme, a.netloc) != (b.scheme, b.netloc):
            raise UntrustedCard(f"the card points to {a.netloc!r}, not {b.netloc!r}: refusing to send credentials there")
        return card

    def _rpc(self, method: str, params: dict) -> dict:
        body = json.dumps(jr.request(1, method, params)).encode()
        req = urllib.request.Request(self.base + "/rpc", data=body, method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.token}"})
        try:
            with self._open(req, timeout=self.timeout_s) as resp:
                msg = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            try:
                msg = json.loads(exc.read())
            except ValueError:
                raise A2AError(exc.code, "http error") from exc
        except (urllib.error.URLError, ValueError, OSError) as exc:
            raise A2AError(jr.INTERNAL_ERROR, f"request failed: {exc}") from exc
        if "error" in msg:
            raise A2AError(msg["error"].get("code", jr.INTERNAL_ERROR), msg["error"].get("message", "error"))
        return msg["result"]

    def create(self, text: str) -> dict:
        return self._rpc("task.create", {"text": text})

    def get(self, task_id: str) -> dict:
        return self._rpc("task.get", {"id": task_id})

    def reply(self, task_id: str, text: str) -> dict:
        return self._rpc("task.reply", {"id": task_id, "text": text})

    def cancel(self, task_id: str) -> dict:
        return self._rpc("task.cancel", {"id": task_id})

    def wait(self, task_id: str, timeout_s: float = 5.0, poll_s: float = 0.02) -> dict:
        """Poll until the task finishes or needs input."""
        import time
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            task = self.get(task_id)
            if task["state"] in TERMINAL or task["state"] == "input_required":
                return task
            time.sleep(poll_s)
        raise TimeoutError(f"task {task_id} did not finish within {timeout_s}s")


def render_for_model(task: dict) -> str:
    """Remote agent output is untrusted text. Wrap it as data before it reaches your model."""
    texts = "\n".join(a.get("text", "") for a in task.get("artifacts", []))
    return f'<untrusted_remote_agent_output state="{task["state"]}">\n{texts or task.get("question") or task.get("error", "")}\n</untrusted_remote_agent_output>'
