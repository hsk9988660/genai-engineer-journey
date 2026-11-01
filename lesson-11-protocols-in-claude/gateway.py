"""An MCP gateway: the trust layer between your agent and third-party MCP servers.

A protocol makes a server CONNECTABLE. It does not make it TRUSTWORTHY. Connecting a server means (1) calling someone
else's code and (2) putting their text (tool names, descriptions, results) into your model's prompt. This gateway enforces:

  allowlist        only tools you name are ever considered. The default is NO tools.
  pinning          a tool is exposed only if the hash of its (name, description, input schema) matches the hash a HUMAN
                   approved. A new tool is quarantined as "pending". A changed definition (a RUG PULL) is blocked.
  re-verification  definitions are re-listed before each call, because a server can change them at any time, silently.
  namespacing      the model sees `server__tool`, so two servers can't collide and one can't impersonate another.
  your metadata    which tools have side effects or return attacker-controlled text is declared in YOUR config. A server's own
                   annotations (readOnlyHint, ...) are hints and are ignored for policy.
  approval + taint side-effect tools need approval, and are refused after untrusted content was read in the run.
  limits           result size caps, a tool-count cap, call timeouts. A hung or dead server is dropped, not waited on.

scan_description() is a HEURISTIC to help a human reviewer. It is easy to evade and is NOT a defense: pinning is the gate.
"""

import hashlib
import json
import re
from dataclasses import dataclass, field

from mcp_client import MCPClient, ProtocolError, RpcError, ServerClosed, StdioTransport, ToolTimeout
from schema import validate

SEP = "__"
TOOL_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
SERVER_RE = re.compile(r"^[A-Za-z0-9-]{1,20}$")  # no underscore, so the first "__" always separates server from tool

SUSPICIOUS = [
    (re.compile(r"ignore (all |any |the )?(previous|prior|above) (instructions|rules)", re.I), "tells the model to ignore its instructions"),
    (re.compile(r"<\s*important\s*>", re.I), "uses an <IMPORTANT> block to address the model"),
    (re.compile(r"do not (tell|mention|reveal|inform)", re.I), "asks the model to hide something from the user"),
    (re.compile(r"\.ssh|id_rsa|/etc/passwd|\.env\b|api[_ -]?key|password", re.I), "mentions sensitive files or credentials"),
    (re.compile(r"before (using|calling) this tool", re.I), "sets preconditions that change how the model behaves"),
    (re.compile(r"\b(whenever|always)\b.{0,80}\b(any|other|every) (tool|call)", re.I | re.S), "tries to set rules for other tools"),
    (re.compile(r"\bbcc\b|attacker|exfiltrat", re.I), "mentions covert recipients"),
    (re.compile(r"(send|post|upload|forward)\b.{0,60}https?://", re.I | re.S), "tells the model to send data to a URL"),
    (re.compile(r"system prompt", re.I), "refers to the system prompt"),
]


def _schema_texts(node) -> list[str]:
    """Every description string inside a schema: poison can hide in parameter descriptions too."""
    out = []
    if isinstance(node, dict):
        for k, v in node.items():
            if k in ("description", "title") and isinstance(v, str):
                out.append(v)
            else:
                out += _schema_texts(v)
    elif isinstance(node, list):
        for v in node:
            out += _schema_texts(v)
    return out


def scan_description(description: str, schema: dict | None = None) -> list[str]:
    text = " ".join([description, *_schema_texts(schema or {})])
    return [label for pattern, label in SUSPICIOUS if pattern.search(text)]


def definition_hash(name: str, description: str, input_schema: dict) -> str:
    canonical = json.dumps({"name": name, "description": description, "inputSchema": input_schema}, sort_keys=True,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def deny_all(summary: dict) -> str:
    return "deny"


@dataclass
class ServerConfig:
    name: str
    command: list[str] | None = None
    allow_tools: frozenset | None = None  # None means NO tools: deny by default
    pins: dict = field(default_factory=dict)  # tool name -> approved definition hash
    side_effect_tools: frozenset = frozenset()  # declared by YOU
    untrusted_output_tools: frozenset = frozenset()  # tools whose results may carry attacker text, declared by YOU
    max_result_chars: int = 4000
    timeout_s: float = 5.0  # for each call and each re-listing once connected
    connect_timeout_s: float = 15.0  # for process start-up and the handshake, which can be far slower than a normal call
    max_message_chars: int = 1_000_000


@dataclass
class RunState:
    tainted: bool = False  # True once a tool declared untrusted_output has returned in this run


@dataclass
class GatewayResult:
    status: str  # ok | tool_error | unknown_tool | unavailable | invalid_arguments | denied | blocked_taint | timeout | server_closed | protocol_error
    text: str = ""
    truncated: bool = False
    tool: str = ""
    untrusted: bool = True  # everything a third-party server returns is untrusted text


@dataclass
class _Server:
    config: ServerConfig
    client: MCPClient
    exposed: dict = field(default_factory=dict)
    pending: dict = field(default_factory=dict)
    blocked: dict = field(default_factory=dict)
    healthy: bool = True


class PinMismatch(Exception):
    pass


class MCPGateway:
    def __init__(self, transport_factory=None, approver=deny_all, recheck: str = "always", max_tools_per_server: int = 50):
        """recheck: "always" re-lists tools before every call (safest, one extra round trip per call),
        "on_notification" only after the server announces a change (a hostile server simply won't), "never" trusts the first listing."""
        self._factory = transport_factory or (lambda cfg: StdioTransport(cfg.command, cfg.max_message_chars))
        self.approver, self.recheck, self.max_tools = approver, recheck, max_tools_per_server
        self.servers: dict[str, _Server] = {}
        self.events: list[dict] = []

    def _event(self, kind: str, server: str, tool: str = "", detail: str = "") -> None:
        self.events.append({"event": kind, "server": server, "tool": tool, "detail": detail})

    # --- connecting and verifying -------------------------------------------------------------------------
    def add_server(self, config: ServerConfig) -> None:
        if not SERVER_RE.match(config.name) or config.name in self.servers:
            raise ValueError(f"invalid or duplicate server name: {config.name!r}")
        client = MCPClient(self._factory(config), timeout_s=config.timeout_s)
        try:
            client.initialize(timeout_s=config.connect_timeout_s)
        except Exception:
            client.close()
            raise
        self.servers[config.name] = _Server(config, client)
        self._event("connected", config.name, detail=str(client.server_info))
        self.refresh(config.name)

    def refresh(self, name: str | None = None) -> None:
        for srv in ([self.servers[name]] if name else list(self.servers.values())):
            self._refresh_one(srv)

    def _refresh_one(self, srv: _Server) -> None:
        cfg = srv.config
        try:
            listed = srv.client.list_tools(timeout_s=cfg.timeout_s)
        except (ServerClosed, ToolTimeout, ProtocolError, RpcError) as exc:
            self._drop(srv, "server_unavailable", detail=str(exc))
            return
        srv.client.tools_changed = False
        exposed, pending, blocked, seen = {}, {}, {}, set()
        for i, d in enumerate(listed):
            name = d.get("name") if isinstance(d, dict) else None
            if not isinstance(name, str) or not TOOL_RE.match(name):
                blocked[str(name)[:40]] = "invalid tool name"
                continue
            if name in seen:
                blocked[name] = "duplicate tool name in one listing"
                continue
            seen.add(name)
            if i >= self.max_tools:
                blocked[name] = f"more than {self.max_tools} tools listed: extra tools ignored"
                continue
            if cfg.allow_tools is None or name not in cfg.allow_tools:
                blocked[name] = "not on the allowlist"
                continue
            description = d.get("description") if isinstance(d.get("description"), str) else ""
            schema = d.get("inputSchema") if isinstance(d.get("inputSchema"), dict) else {"type": "object", "properties": {}}
            digest = definition_hash(name, description, schema)
            entry = {"name": name, "description": description, "inputSchema": schema, "hash": digest,
                     "annotations": d.get("annotations")}
            pin = cfg.pins.get(name)
            if pin == digest:
                exposed[name] = entry
            elif pin is not None:
                blocked[name] = "definition changed since it was approved (possible rug pull)"
                self._event("rug_pull", cfg.name, name, f"approved {pin[:12]}, now {digest[:12]}")
            else:
                pending[name] = {**entry, "findings": scan_description(description, schema)}
                self._event("pending_approval", cfg.name, name, digest[:12])
        srv.exposed, srv.pending, srv.blocked, srv.healthy = exposed, pending, blocked, True

    def _drop(self, srv: _Server, kind: str, detail: str = "") -> None:
        srv.healthy, srv.exposed = False, {}
        self._event(kind, srv.config.name, detail=detail)
        srv.client.close()

    # --- human actions and views ----------------------------------------------------------------------------
    def pending_approval(self) -> list[dict]:
        """What a human must review before anything is exposed: the full text, its hash, and any suspicious findings."""
        return [{"server": s.config.name, "tool": t["name"], "hash": t["hash"], "description": t["description"],
                 "schema": t["inputSchema"], "findings": t["findings"], "annotations_claimed": t["annotations"]}
                for s in self.servers.values() for t in s.pending.values()]

    def pin(self, server: str, tool: str, digest: str) -> None:
        """A human approves one exact definition. If the server has changed since they looked, this fails."""
        srv = self.servers[server]
        entry = srv.pending.get(tool)
        if entry is None or entry["hash"] != digest:
            raise PinMismatch("the definition is not the one you reviewed (changed, or not pending)")
        srv.config.pins[tool] = digest
        srv.exposed[tool] = srv.pending.pop(tool)
        self._event("pinned", server, tool, digest[:12])

    def exposed_tools(self) -> list[dict]:
        """The tools the MODEL may see, in Anthropic tool format, namespaced as server__tool."""
        return [{"name": f"{srv.config.name}{SEP}{t['name']}",
                 "description": f"[MCP server '{srv.config.name}'] {t['description']}", "input_schema": t["inputSchema"]}
                for srv in self.servers.values() if srv.healthy for t in srv.exposed.values()]

    # --- calling --------------------------------------------------------------------------------------------
    def call(self, namespaced: str, arguments: dict, state: RunState | None = None) -> GatewayResult:
        state = state if state is not None else RunState()
        server, _, tool = namespaced.partition(SEP)
        srv = self.servers.get(server)
        if srv is None or not tool:
            return GatewayResult("unknown_tool", f"No such tool: {namespaced}", tool=namespaced)
        if not srv.healthy:
            return GatewayResult("unavailable", "The server for this tool is not available.", tool=namespaced)
        if self.recheck == "always" or (self.recheck == "on_notification" and srv.client.tools_changed):
            self._refresh_one(srv)  # a server can change its tools at any time, so look again before trusting them
            if not srv.healthy:
                return GatewayResult("unavailable", "The server for this tool is not available.", tool=namespaced)
        entry = srv.exposed.get(tool)
        if entry is None:
            reason = srv.blocked.get(tool) or ("awaiting human approval" if tool in srv.pending else "not available")
            return GatewayResult("unavailable", f"This tool is not available ({reason}).", tool=namespaced)

        problems = validate(arguments, entry["inputSchema"])  # check locally: do not send obviously bad calls
        if problems:
            return GatewayResult("invalid_arguments", "; ".join(problems), tool=namespaced)

        cfg = srv.config
        if tool in cfg.side_effect_tools:
            if state.tainted:
                self._event("blocked_taint", cfg.name, tool)
                return GatewayResult("blocked_taint", "Not allowed: untrusted content was read earlier in this run.", tool=namespaced)
            try:
                approved = self.approver({"server": cfg.name, "tool": tool, "arguments": arguments,
                                          "reason": "declared as having side effects in this gateway's configuration"}) == "approve"
            except Exception:
                approved = False  # fail closed
            if not approved:
                self._event("denied", cfg.name, tool)
                return GatewayResult("denied", "The user did not approve this action.", tool=namespaced)

        try:
            result = srv.client.call_tool(tool, arguments, timeout_s=cfg.timeout_s)
        except ToolTimeout:
            self._drop(srv, "timeout", detail=tool)
            return GatewayResult("timeout", "The tool did not respond in time and its server was disconnected.", tool=namespaced)
        except ServerClosed as exc:
            self._drop(srv, "server_closed", detail=str(exc))
            return GatewayResult("server_closed", "The server exited unexpectedly.", tool=namespaced)
        except (ProtocolError, RpcError) as exc:
            self._drop(srv, "protocol_error", detail=str(exc))
            return GatewayResult("protocol_error", "The server broke the protocol and was disconnected.", tool=namespaced)

        text, truncated = result.text, False
        if len(text) > cfg.max_result_chars:
            text, truncated = text[:cfg.max_result_chars] + f"\n[truncated: {len(result.text) - cfg.max_result_chars} more characters omitted]", True
            self._event("truncated", cfg.name, tool, f"{len(result.text)} chars")
        if tool in cfg.untrusted_output_tools:
            state.tainted = True
        return GatewayResult("tool_error" if result.is_error else "ok", text, truncated, namespaced)

    def close(self) -> None:
        for srv in self.servers.values():
            srv.client.close()
