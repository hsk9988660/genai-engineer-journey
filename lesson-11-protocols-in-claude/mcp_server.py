"""A minimal MCP server, written from scratch so you can see the protocol. Speaks JSON-RPC 2.0 over stdio.

Implemented: initialize (version negotiation), notifications/initialized, ping, tools/list (with cursor pagination),
tools/call, resources/list, resources/read, and the notifications/tools/list_changed notification.
NOT implemented: prompts, sampling, roots, elicitation, subscriptions, the HTTP transports, authorization, progress,
cancellation handling. It is single-threaded: a tool that blocks, blocks the whole server. Use the official SDK for real
servers. This one exists to teach and to test clients and gateways against.

Two kinds of failure, which people mix up:
  * PROTOCOL error  -> a JSON-RPC `error` object (unknown method, unknown tool, malformed params)
  * TOOL error      -> a normal result with "isError": true, so the MODEL can read it and recover
Here, invalid tool arguments are reported as a TOOL error so the model can fix them. Check how your spec version treats this.

stdout is the protocol channel: ONLY JSON-RPC lines may be written to it. Logging goes to stderr. A stray print() to stdout
corrupts the stream, and is the most common first bug in stdio servers.
"""

import json
import logging
import sys
from dataclasses import dataclass, field
from typing import Callable

import jsonrpc as jr
from schema import validate

SUPPORTED_VERSIONS = ["2025-06-18", "2025-03-26", "2024-11-05"]
LATEST = SUPPORTED_VERSIONS[0]
log = logging.getLogger("mcp_server")


class ToolError(Exception):
    """Raise from a handler to report a tool-level failure with a message the model may see."""


@dataclass
class ToolDef:
    name: str
    description: str
    input_schema: dict
    handler: Callable[[dict], object]
    title: str | None = None
    annotations: dict | None = None  # HINTS only (readOnlyHint, destructiveHint, ...). A client must not trust them from an untrusted server.


@dataclass
class ResourceDef:
    uri: str
    name: str
    reader: Callable[[], str]
    description: str = ""
    mime_type: str = "text/plain"


@dataclass
class MCPServer:
    name: str
    version: str = "0.1.0"
    instructions: str | None = None
    page_size: int = 50
    tools: dict = field(default_factory=dict)
    resources: dict = field(default_factory=dict)
    initialized: bool = False
    client_ready: bool = False
    before_list: Callable[["MCPServer"], None] | None = None  # test hook: lets a hostile server change itself between listings
    _outbox: list = field(default_factory=list)

    # --- registration -------------------------------------------------------------------------------
    def add_tool(self, name: str, description: str, input_schema: dict, handler, title=None, annotations=None) -> None:
        self.tools[name] = ToolDef(name, description, input_schema, handler, title, annotations)

    def add_resource(self, uri: str, name: str, reader, description: str = "", mime_type: str = "text/plain") -> None:
        self.resources[uri] = ResourceDef(uri, name, reader, description, mime_type)

    def notify_tools_changed(self) -> None:
        self._outbox.append(jr.notification("notifications/tools/list_changed"))

    def drain_notifications(self) -> list:
        out, self._outbox = self._outbox, []
        return out

    # --- message handling ---------------------------------------------------------------------------
    def handle(self, msg: dict) -> dict | None:
        """Take one decoded message; return the response to send, or None (notifications, stray responses)."""
        k = jr.kind(msg)
        if k == "response":
            return None  # this server never sends requests, so there is nothing to match
        try:
            if k == "notification":
                if msg["method"] == "notifications/initialized":
                    self.client_ready = True
                return None
            return jr.result(msg["id"], self._dispatch(msg["method"], msg.get("params") or {}))
        except jr.JsonRpcError as exc:
            return None if k == "notification" else jr.error(msg.get("id"), exc.code, exc.message, exc.data)
        except Exception:  # a bug in a handler must not take the server down, and must not leak a traceback to the client
            log.exception("unexpected error handling %s", msg.get("method"))
            return None if k == "notification" else jr.error(msg.get("id"), jr.INTERNAL_ERROR, "internal error")

    def _dispatch(self, method: str, params: dict):
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if not self.initialized:
            raise jr.JsonRpcError(jr.INVALID_REQUEST, "server not initialized: send initialize first")
        handlers = {"tools/list": self._tools_list, "tools/call": self._tools_call,
                    "resources/list": self._resources_list, "resources/read": self._resources_read}
        if method not in handlers:
            raise jr.JsonRpcError(jr.METHOD_NOT_FOUND, f"Method not found: {method}")
        return handlers[method](params)

    def _initialize(self, params: dict) -> dict:
        if self.initialized:
            raise jr.JsonRpcError(jr.INVALID_REQUEST, "already initialized")
        client_version = params.get("protocolVersion")
        client_info = params.get("clientInfo")
        if not isinstance(client_version, str) or not isinstance(client_info, dict) or "name" not in client_info:
            raise jr.JsonRpcError(jr.INVALID_PARAMS, "initialize needs protocolVersion and clientInfo.name")
        self.initialized = True
        capabilities: dict = {}
        if self.tools:
            capabilities["tools"] = {"listChanged": True}
        if self.resources:
            capabilities["resources"] = {"subscribe": False, "listChanged": False}
        out = {"protocolVersion": client_version if client_version in SUPPORTED_VERSIONS else LATEST,  # negotiation
               "capabilities": capabilities, "serverInfo": {"name": self.name, "version": self.version}}
        if self.instructions:
            out["instructions"] = self.instructions
        return out

    def _tools_list(self, params: dict) -> dict:
        if self.before_list:
            self.before_list(self)
        cursor = params.get("cursor", "0")
        if not isinstance(cursor, str) or not cursor.isdigit():
            raise jr.JsonRpcError(jr.INVALID_PARAMS, "invalid cursor")
        start = int(cursor)
        tools = list(self.tools.values())
        page = tools[start:start + self.page_size]
        items = []
        for t in page:
            item = {"name": t.name, "description": t.description, "inputSchema": t.input_schema}
            if t.title:
                item["title"] = t.title
            if t.annotations:
                item["annotations"] = t.annotations
            items.append(item)
        out: dict = {"tools": items}
        if start + self.page_size < len(tools):
            out["nextCursor"] = str(start + self.page_size)
        return out

    def _tools_call(self, params: dict) -> dict:
        name, arguments = params.get("name"), params.get("arguments", {})
        if not isinstance(name, str) or not isinstance(arguments, dict):
            raise jr.JsonRpcError(jr.INVALID_PARAMS, "tools/call needs a string name and an object arguments")
        tool = self.tools.get(name)
        if tool is None:
            raise jr.JsonRpcError(jr.INVALID_PARAMS, f"Unknown tool: {name}")
        problems = validate(arguments, tool.input_schema)
        if problems:
            return self._tool_result(f"Invalid arguments: {'; '.join(problems)}", is_error=True)
        try:
            value = tool.handler(arguments)
        except ToolError as exc:
            return self._tool_result(str(exc), is_error=True)
        except Exception:
            log.exception("tool %s failed", name)
            return self._tool_result("The tool failed.", is_error=True)  # no traceback, no internals
        return self._tool_result(value if isinstance(value, str) else json.dumps(value, default=str), is_error=False)

    @staticmethod
    def _tool_result(text: str, is_error: bool) -> dict:
        return {"content": [{"type": "text", "text": text}], "isError": is_error}

    def _resources_list(self, params: dict) -> dict:
        return {"resources": [{"uri": r.uri, "name": r.name, "description": r.description, "mimeType": r.mime_type}
                              for r in self.resources.values()]}

    def _resources_read(self, params: dict) -> dict:
        uri = params.get("uri")
        resource = self.resources.get(uri) if isinstance(uri, str) else None
        if resource is None:
            raise jr.JsonRpcError(jr.RESOURCE_NOT_FOUND, "Resource not found", {"uri": uri})
        return {"contents": [{"uri": resource.uri, "mimeType": resource.mime_type, "text": resource.reader()}]}


def serve_stdio(server: MCPServer, stdin=None, stdout=None) -> None:
    """Read one JSON-RPC message per line from stdin and write responses, one per line, to stdout. Ends at EOF."""
    stdin, stdout = stdin or sys.stdin, stdout or sys.stdout

    def write(msg: dict) -> None:
        stdout.write(jr.encode(msg) + "\n")
        stdout.flush()

    while True:
        line = stdin.readline()
        if not line:
            return
        line = line.strip()
        if not line:
            continue
        try:
            msg = jr.decode(line)
        except jr.JsonRpcError as exc:
            write(jr.error(None, exc.code, exc.message))  # per the spec, an unparseable message gets an error with id null
            continue
        response = server.handle(msg)
        if response is not None:
            write(response)
        for note in server.drain_notifications():
            write(note)
