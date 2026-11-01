"""An MCP client written from scratch. It spawns a server as a child process and speaks JSON-RPC 2.0 over its stdio.

What a client has to get right (and what this one does):
  * the handshake: initialize, check the negotiated version, then send notifications/initialized
  * matching responses to requests by id, while notifications can arrive in between
  * timeouts. A hung server must not hang your host. On timeout it sends notifications/cancelled, and the late reply
    (if it ever comes) is ignored, because ids only increase and a stale id never matches a newer request.
  * a server that dies mid-call must raise, not block
  * a server that writes garbage (or a log line) to stdout is a PROTOCOL error, reported with the offending text
  * a message-size cap, so a hostile server cannot flood your memory

Two transports: StdioTransport (a real subprocess) and LoopbackTransport (calls a server object directly, for fast tests).
Not implemented: the HTTP transports, authorization, server-initiated requests (sampling, elicitation), progress.
"""

import queue
import subprocess
import threading
import time
from dataclasses import dataclass

import jsonrpc as jr
from mcp_server import LATEST, SUPPORTED_VERSIONS, MCPServer


class ProtocolError(Exception):
    pass


class MessageTooLarge(ProtocolError):
    pass


class ServerClosed(Exception):
    pass


class ToolTimeout(TimeoutError):
    pass


class RpcError(Exception):
    def __init__(self, code: int, message: str, data=None):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.data = code, message, data


@dataclass
class ToolResult:
    text: str
    is_error: bool
    raw: dict


_EOF = object()


class StdioTransport:
    def __init__(self, command: list[str], max_message_chars: int = 1_000_000, cwd=None, env=None):
        self.max_message_chars = max_message_chars
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, encoding="utf-8", bufsize=1, cwd=cwd, env=env)
        self._q: queue.Queue = queue.Queue()
        self.stderr_tail = ""
        self._threads = [threading.Thread(target=self._read_stdout, daemon=True),
                         threading.Thread(target=self._drain_stderr, daemon=True)]
        for t in self._threads:
            t.start()

    def _read_stdout(self) -> None:
        limit = self.max_message_chars
        while True:
            try:
                line = self.proc.stdout.readline(limit + 1)
            except (ValueError, OSError):
                self._q.put(_EOF)
                return
            if not line:
                self._q.put(_EOF)
                return
            if len(line) > limit and not line.endswith("\n"):
                self._q.put(MessageTooLarge(f"server sent a message larger than {limit} characters"))
                self.proc.kill()
                return
            line = line.strip()
            if not line:
                continue
            try:
                self._q.put(jr.decode(line))
            except jr.JsonRpcError as exc:
                self._q.put(ProtocolError(f"server wrote something that is not JSON-RPC to stdout ({exc.message}): {line[:80]!r}"))

    def _drain_stderr(self) -> None:
        try:
            for line in self.proc.stderr:  # always drain stderr, or a chatty server fills the pipe and blocks
                self.stderr_tail = (self.stderr_tail + line)[-2000:]
        except (ValueError, OSError):
            pass

    def send(self, msg: dict) -> None:
        try:
            self.proc.stdin.write(jr.encode(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError, ValueError) as exc:
            raise ServerClosed(f"server is not accepting input: {self.stderr_tail.strip()[-200:]}") from exc

    def recv(self, timeout_s: float) -> dict:
        try:
            item = self._q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError from None
        if item is _EOF:
            self._q.put(_EOF)  # stay closed for later callers
            raise ServerClosed(f"server exited: {self.stderr_tail.strip()[-200:]}")
        if isinstance(item, Exception):
            raise item
        return item

    def close(self) -> None:
        try:
            self.proc.stdin.close()
        except (OSError, ValueError):
            pass
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        for pipe in (self.proc.stdout, self.proc.stderr):
            try:
                pipe.close()
            except (OSError, ValueError):
                pass


class LoopbackTransport:
    """Talks to an in-process MCPServer. Messages still round-trip through the real encoder and decoder."""

    def __init__(self, server: MCPServer):
        self.server, self._q, self.sent = server, queue.Queue(), []

    def send(self, msg: dict) -> None:
        msg = jr.decode(jr.encode(msg))
        self.sent.append(msg)
        response = self.server.handle(msg)
        for out in ([response] if response is not None else []) + self.server.drain_notifications():
            self._q.put(jr.decode(jr.encode(out)))

    def recv(self, timeout_s: float) -> dict:
        try:
            return self._q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError from None

    def close(self) -> None:
        pass


class MCPClient:
    def __init__(self, transport, name: str = "lesson-client", version: str = "0.1.0", timeout_s: float = 10.0):
        self.transport, self.name, self.version, self.timeout_s = transport, name, version, timeout_s
        self.server_info: dict = {}
        self.capabilities: dict = {}
        self.protocol_version: str | None = None
        self.tools_changed = False  # set when the server sends notifications/tools/list_changed
        self._next_id, self._lock = 1, threading.RLock()

    # --- lifecycle ------------------------------------------------------------------------------------
    def initialize(self, timeout_s: float | None = None) -> dict:
        """The handshake can take much longer than a normal call (a child process has to start), so it may have its own timeout."""
        result = self._request("initialize", {"protocolVersion": LATEST, "capabilities": {},
                                              "clientInfo": {"name": self.name, "version": self.version}}, timeout_s)
        version = result.get("protocolVersion")
        if version not in SUPPORTED_VERSIONS:
            raise ProtocolError(f"server speaks an unsupported protocol version: {version!r}")
        self.protocol_version, self.capabilities = version, result.get("capabilities") or {}
        self.server_info = result.get("serverInfo") or {}
        self._notify("notifications/initialized")
        return result

    def close(self) -> None:
        self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # --- API --------------------------------------------------------------------------------------------
    def ping(self) -> None:
        self._request("ping")

    def list_tools(self, timeout_s: float | None = None) -> list[dict]:
        tools, cursor = [], None
        for _ in range(100):  # a server that never stops paginating must not trap us
            result = self._request("tools/list", {"cursor": cursor} if cursor else {}, timeout_s)
            page = result.get("tools")
            if not isinstance(page, list):
                raise ProtocolError("tools/list result has no tools list")
            tools += page
            cursor = result.get("nextCursor")
            if not cursor:
                return tools
        raise ProtocolError("tools/list pagination did not end")

    def call_tool(self, name: str, arguments: dict | None = None, timeout_s: float | None = None) -> ToolResult:
        result = self._request("tools/call", {"name": name, "arguments": arguments or {}}, timeout_s)
        content = result.get("content")
        if not isinstance(content, list):
            raise ProtocolError("tools/call result has no content list")
        parts = [c.get("text", "") if c.get("type") == "text" else f"[{c.get('type', 'unknown')} content omitted]" for c in content]
        return ToolResult("\n".join(parts), bool(result.get("isError", False)), result)

    def list_resources(self) -> list[dict]:
        return self._request("resources/list").get("resources", [])

    def read_resource(self, uri: str) -> str:
        contents = self._request("resources/read", {"uri": uri}).get("contents", [])
        return "\n".join(c.get("text", "") for c in contents)

    # --- plumbing -----------------------------------------------------------------------------------------
    def _notify(self, method: str, params=None) -> None:
        self.transport.send(jr.notification(method, params))

    def _on_notification(self, msg: dict) -> None:
        if msg.get("method") == "notifications/tools/list_changed":
            self.tools_changed = True

    def _request(self, method: str, params=None, timeout_s: float | None = None) -> dict:
        timeout_s = self.timeout_s if timeout_s is None else timeout_s
        with self._lock:
            request_id, self._next_id = self._next_id, self._next_id + 1
            self.transport.send(jr.request(request_id, method, params))
            deadline = time.monotonic() + timeout_s
            while True:
                remaining = deadline - time.monotonic()
                try:
                    if remaining <= 0:
                        raise TimeoutError
                    msg = self.transport.recv(remaining)
                except TimeoutError:
                    try:
                        self._notify("notifications/cancelled", {"requestId": request_id, "reason": "client timeout"})
                    except ServerClosed:
                        pass
                    raise ToolTimeout(f"{method} timed out after {timeout_s}s") from None
                k = jr.kind(msg)
                if k == "notification":
                    self._on_notification(msg)
                elif k == "request":  # server-to-client requests (sampling, elicitation) are not supported here
                    self.transport.send(jr.error(msg["id"], jr.METHOD_NOT_FOUND, "client does not support this request"))
                elif msg["id"] == request_id:  # a LATE reply to a request we gave up on has an older id, which can never
                    #                           match a newer request (ids only increase), so it is ignored by this check
                    if "error" in msg:
                        err = msg["error"] if isinstance(msg["error"], dict) else {}
                        raise RpcError(err.get("code", jr.INTERNAL_ERROR), err.get("message", "unknown error"), err.get("data"))
                    result = msg.get("result")
                    if not isinstance(result, dict):
                        raise ProtocolError(f"{method} result is not an object")
                    return result
