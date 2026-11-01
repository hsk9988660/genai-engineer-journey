"""Test support: a scripted model client, a scriptable transport, and helpers to build gateways."""

import queue
import sys
from pathlib import Path
from types import SimpleNamespace as NS

from demo_servers import BUILDERS
from gateway import MCPGateway, ServerConfig, definition_hash
from mcp_client import LoopbackTransport

HERE = Path(__file__).resolve().parent


def use(id_, tool_name, **inp):
    return NS(type="tool_use", id=id_, name=tool_name, input=inp)


def turn(*blocks):
    return NS(content=list(blocks), stop_reason="tool_use", usage=NS(input_tokens=10, output_tokens=5))


def answer(text="done"):
    return turn(use("fin", "submit_answer", answer=text))


class FakeModel:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})
        return self.replies.pop(0)


class FakeTransport:
    """A transport whose replies are scripted: on_send(message, transport) may call transport.push(...)."""

    def __init__(self, on_send=None):
        self.on_send, self.sent, self._q = on_send, [], queue.Queue()

    def push(self, *msgs):
        for m in msgs:
            self._q.put(m)

    def send(self, msg):
        self.sent.append(msg)
        if self.on_send:
            self.on_send(msg, self)

    def recv(self, timeout_s):
        try:
            return self._q.get(timeout=timeout_s)
        except queue.Empty:
            raise TimeoutError from None

    def close(self):
        pass


def tool_hash(server, tool_name):
    t = server.tools[tool_name]
    return definition_hash(t.name, t.description, t.input_schema)


def pins_for(server, *names):
    return {n: tool_hash(server, n) for n in (names or server.tools)}


def loopback_gateway(servers: dict, approver=None, **kw):
    """A gateway whose servers are in-process objects (fast, deterministic). Messages still use the real encoder/decoder."""
    gw = MCPGateway(transport_factory=lambda cfg: LoopbackTransport(servers[cfg.name]),
                    **({"approver": approver} if approver else {}), **kw)
    return gw


def stdio_config(name, **kw):
    """A config that launches the real demo server as a subprocess over stdio."""
    return ServerConfig(name, [sys.executable, str(HERE / "server_main.py"), name], **kw)


def fresh(name):
    return BUILDERS[name]()
