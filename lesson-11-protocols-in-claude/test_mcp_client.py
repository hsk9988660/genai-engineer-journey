import sys
import time

import pytest

import jsonrpc as jr
from demo_servers import flights_server, many_tools_server
from fakes import HERE, FakeTransport
from mcp_client import (LoopbackTransport, MCPClient, MessageTooLarge, ProtocolError, RpcError, ServerClosed, StdioTransport,
                        ToolTimeout)
from mcp_server import LATEST


def reply_to(msg, result=None, error=None):
    return jr.error(msg["id"], *error) if error else jr.result(msg["id"], result if result is not None else {})


def init_result(version=LATEST):
    return {"protocolVersion": version, "capabilities": {"tools": {}}, "serverInfo": {"name": "fake", "version": "1"}}


def scripted(handlers: dict):
    """A transport that answers each request method with handlers[method](msg) -> list of messages to push."""
    def on_send(msg, t):
        if jr.kind(msg) == "request" and msg["method"] in handlers:
            t.push(*handlers[msg["method"]](msg))
    return FakeTransport(on_send)


# --- the handshake -----------------------------------------------------------------------------------------------
def test_the_handshake_sends_initialize_then_initialized_and_records_what_the_server_said():
    t = LoopbackTransport(flights_server())
    c = MCPClient(t)
    c.initialize()
    assert [m["method"] for m in t.sent] == ["initialize", "notifications/initialized"]
    assert "id" not in t.sent[1]  # the second one is a notification
    assert c.protocol_version == LATEST and c.server_info["name"] == "flights" and "tools" in c.capabilities


def test_a_server_speaking_an_unknown_protocol_version_is_refused():
    c = MCPClient(scripted({"initialize": lambda m: [reply_to(m, init_result("1999-01-01"))]}))
    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        c.initialize()


def test_the_client_works_against_the_server_over_the_loopback():
    with MCPClient(LoopbackTransport(flights_server())) as c:
        c.initialize()
        assert [t["name"] for t in c.list_tools()] == ["search_flights", "book_flight"]
        r = c.call_tool("search_flights", {"origin": "SEA", "destination": "HNL"})
        assert r.text == "HA 20, 09:30-12:10, $389" and not r.is_error
        assert c.call_tool("search_flights", {"origin": "SEA"}).is_error
        assert "$40" in c.read_resource("policy://baggage") and c.list_resources()[0]["uri"] == "policy://baggage"
        c.ping()


def test_a_protocol_error_from_the_server_raises_rpcerror_with_the_code():
    with MCPClient(LoopbackTransport(flights_server())) as c:
        c.initialize()
        with pytest.raises(RpcError) as e:
            c.call_tool("no_such_tool")
        assert e.value.code == jr.INVALID_PARAMS
        with pytest.raises(RpcError) as e2:
            c.read_resource("policy://missing")
        assert e2.value.code == jr.RESOURCE_NOT_FOUND


# --- pagination, results and notifications -----------------------------------------------------------------------------
def test_list_tools_follows_every_cursor():
    pages = {None: ([{"name": "a"}, {"name": "b"}], "1"), "1": ([{"name": "c"}], None)}

    def handler(msg):
        tools, nxt = pages[msg.get("params", {}).get("cursor")]
        return [reply_to(msg, {"tools": tools, **({"nextCursor": nxt} if nxt else {})})]

    assert [t["name"] for t in MCPClient(scripted({"tools/list": handler})).list_tools()] == ["a", "b", "c"]


def test_a_server_that_never_stops_paginating_cannot_trap_the_client():
    c = MCPClient(scripted({"tools/list": lambda m: [reply_to(m, {"tools": [], "nextCursor": "again"})]}))
    with pytest.raises(ProtocolError, match="pagination did not end"):
        c.list_tools()


def test_call_results_join_text_parts_and_summarize_non_text_parts():
    content = [{"type": "text", "text": "one"}, {"type": "image", "data": "AAAA"}, {"type": "text", "text": "two"}]
    c = MCPClient(scripted({"tools/call": lambda m: [reply_to(m, {"content": content, "isError": True})]}))
    r = c.call_tool("x")
    assert r.text == "one\n[image content omitted]\ntwo" and r.is_error


def test_malformed_results_are_protocol_errors_not_crashes():
    with pytest.raises(ProtocolError):
        MCPClient(scripted({"tools/call": lambda m: [reply_to(m, {"content": "not a list"})]})).call_tool("x")
    with pytest.raises(ProtocolError):
        MCPClient(scripted({"tools/list": lambda m: [reply_to(m, {"tools": "nope"})]})).list_tools()


def test_a_notification_that_arrives_before_the_response_is_processed_and_does_not_confuse_matching():
    note = jr.notification("notifications/tools/list_changed")
    c = MCPClient(scripted({"ping": lambda m: [note, reply_to(m, {})]}))
    assert c.tools_changed is False
    c.ping()
    assert c.tools_changed is True


def test_a_server_to_client_request_gets_a_method_not_found_reply():
    server_request = jr.request("srv-1", "sampling/createMessage", {})
    t = scripted({"ping": lambda m: [server_request, reply_to(m, {})]})
    MCPClient(t).ping()
    reply = next(m for m in t.sent if m.get("id") == "srv-1")
    assert reply["error"]["code"] == jr.METHOD_NOT_FOUND


# --- timeouts and late replies -------------------------------------------------------------------------------------------
def test_a_timeout_sends_a_cancel_notification_and_a_late_reply_is_ignored():
    t = FakeTransport()
    c = MCPClient(t, timeout_s=0.05)
    with pytest.raises(ToolTimeout):
        c.call_tool("slow")
    cancel = [m for m in t.sent if m.get("method") == "notifications/cancelled"]
    assert len(cancel) == 1 and cancel[0]["params"]["requestId"] == 1
    # the slow reply to request 1 finally arrives, followed by the real reply to the NEXT request (id 2)
    t.push(jr.result(1, {"content": [{"type": "text", "text": "LATE"}], "isError": False}),
           jr.result(2, {"content": [{"type": "text", "text": "fresh"}], "isError": False}))
    assert c.call_tool("fast").text == "fresh"  # the stale reply was discarded instead of being returned


def test_request_ids_increase_and_replies_are_matched_by_id():
    t = scripted({"ping": lambda m: [jr.result(999, {}), reply_to(m, {})]})  # a stray reply for an unknown id comes first
    c = MCPClient(t)
    c.ping()
    c.ping()
    assert [m["id"] for m in t.sent if m.get("method") == "ping"] == [1, 2]


# --- real subprocesses over real stdio -----------------------------------------------------------------------------------
def stdio(name, **kw):
    return StdioTransport([sys.executable, str(HERE / "server_main.py"), name], **kw)


def test_a_full_session_with_a_real_server_process():
    with MCPClient(stdio("flights")) as c:
        c.initialize()
        assert c.server_info == {"name": "flights", "version": "1.0.0"}
        assert c.call_tool("search_flights", {"origin": "PDX", "destination": "HNL"}).text == "AS 810, 08:00-10:45, $412"
        assert "$40" in c.read_resource("policy://baggage")
        c.ping()


def test_pagination_works_over_a_real_process():
    with MCPClient(stdio("many")) as c:
        c.initialize()
        assert len(c.list_tools()) == 120  # three pages of 40


def test_a_log_line_on_stdout_is_a_clear_protocol_error_naming_the_offending_text():
    with MCPClient(stdio("noisy")) as c:
        with pytest.raises(ProtocolError, match="not JSON-RPC") as e:
            c.initialize()
        assert "debug: starting up" in str(e.value)


def test_a_server_that_dies_mid_call_raises_instead_of_blocking():
    with MCPClient(stdio("crash"), timeout_s=5) as c:
        c.initialize()
        start = time.monotonic()
        with pytest.raises(ServerClosed):
            c.call_tool("die")
        assert time.monotonic() - start < 4


def test_a_hung_tool_times_out_quickly_and_closing_kills_the_process():
    transport = stdio("hang")
    c = MCPClient(transport, timeout_s=0.5)
    c.initialize()
    start = time.monotonic()
    with pytest.raises(ToolTimeout):
        c.call_tool("slow", timeout_s=0.5)
    assert time.monotonic() - start < 3
    c.close()
    assert time.monotonic() - start < 8 and transport.proc.poll() is not None  # the sleeping child is really gone


def test_an_oversized_message_is_refused_and_the_server_is_killed():
    transport = stdio("flood", max_message_chars=100_000)
    c = MCPClient(transport, timeout_s=5)
    c.initialize()
    with pytest.raises(MessageTooLarge):
        c.call_tool("dump")  # 300,000 characters
    transport.proc.wait(timeout=5)
    assert transport.proc.poll() is not None
    c.close()


def test_loopback_and_stdio_give_identical_results_for_the_same_call():
    args = {"origin": "PDX", "destination": "HNL"}
    with MCPClient(LoopbackTransport(flights_server())) as a, MCPClient(stdio("flights")) as b:
        a.initialize(), b.initialize()
        assert a.call_tool("search_flights", args).raw == b.call_tool("search_flights", args).raw
        assert a.list_tools() == b.list_tools()


def test_the_handshake_can_have_a_longer_timeout_than_ordinary_calls():
    """Regression: process start-up counted against the SHORT tool timeout, so a slow interpreter start killed the connection."""
    import threading

    def slow_init(msg, t):
        if msg.get("method") == "initialize":
            threading.Timer(0.3, lambda: t.push(reply_to(msg, init_result()))).start()

    c = MCPClient(FakeTransport(slow_init), timeout_s=0.1)  # tool calls would time out after 0.1s...
    c.initialize(timeout_s=2.0)                             # ...but the handshake is allowed 2s
    assert c.protocol_version == LATEST
    with pytest.raises(ToolTimeout):
        MCPClient(FakeTransport(slow_init), timeout_s=0.1).initialize()  # and without the longer allowance it does time out
