import io
import json

import jsonrpc as jr
from demo_servers import flights_server
from mcp_server import LATEST, SUPPORTED_VERSIONS, MCPServer, ToolError, serve_stdio

INIT = {"protocolVersion": LATEST, "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}


def call(server, method, params=None, id_=1):
    return server.handle(jr.request(id_, method, params))


def ready(server=None):
    server = server or flights_server()
    call(server, "initialize", INIT)
    return server


def err(response):
    return response["error"]


# --- initialize and version negotiation -------------------------------------------------------------------------
def test_initialize_returns_the_negotiated_version_capabilities_and_server_info():
    s = flights_server()
    r = call(s, "initialize", INIT)["result"]
    assert r["protocolVersion"] == LATEST and r["serverInfo"] == {"name": "flights", "version": "1.0.0"}
    assert r["capabilities"]["tools"] == {"listChanged": True} and "resources" in r["capabilities"]
    assert r["instructions"] == "Search and book flights."


def test_a_supported_older_version_is_echoed_and_an_unknown_one_gets_the_latest():
    for v in SUPPORTED_VERSIONS:
        assert call(flights_server(), "initialize", {**INIT, "protocolVersion": v})["result"]["protocolVersion"] == v
    assert call(flights_server(), "initialize", {**INIT, "protocolVersion": "1999-01-01"})["result"]["protocolVersion"] == LATEST


def test_capabilities_only_advertise_what_the_server_actually_has():
    empty = MCPServer("bare")
    assert call(empty, "initialize", INIT)["result"]["capabilities"] == {}


def test_initialize_needs_a_version_and_client_info_and_cannot_be_repeated():
    assert err(call(flights_server(), "initialize", {}))["code"] == jr.INVALID_PARAMS
    assert err(call(flights_server(), "initialize", {"protocolVersion": LATEST, "clientInfo": {}}))["code"] == jr.INVALID_PARAMS
    s = ready()
    assert err(call(s, "initialize", INIT))["code"] == jr.INVALID_REQUEST


def test_nothing_but_ping_works_before_initialize():
    s = flights_server()
    assert err(call(s, "tools/list"))["code"] == jr.INVALID_REQUEST and "initialize" in err(call(s, "tools/list"))["message"]
    assert call(s, "ping")["result"] == {}


# --- tools/list ----------------------------------------------------------------------------------------------------
def test_tools_list_has_name_description_schema_and_optional_annotations_and_title():
    s = ready()
    s.add_tool("titled", "d", {"type": "object"}, lambda a: "x", title="Nice Name", annotations={"readOnlyHint": True})
    tools = {t["name"]: t for t in call(s, "tools/list")["result"]["tools"]}
    assert set(tools["search_flights"]) == {"name", "description", "inputSchema", "annotations"}
    assert tools["titled"]["title"] == "Nice Name" and tools["titled"]["annotations"] == {"readOnlyHint": True}


def test_pagination_walks_every_tool_exactly_once_via_cursors():
    s = MCPServer("p", page_size=2)
    for i in range(5):
        s.add_tool(f"t{i}", "d", {"type": "object"}, lambda a: "x")
    call(s, "initialize", INIT)
    seen, cursor = [], None
    for _ in range(10):
        r = call(s, "tools/list", {"cursor": cursor} if cursor else {})["result"]
        seen += [t["name"] for t in r["tools"]]
        cursor = r.get("nextCursor")
        if not cursor:
            break
    assert seen == ["t0", "t1", "t2", "t3", "t4"]


def test_a_malformed_cursor_is_an_invalid_params_error():
    assert err(call(ready(), "tools/list", {"cursor": "banana"}))["code"] == jr.INVALID_PARAMS
    assert err(call(ready(), "tools/list", {"cursor": 5}))["code"] == jr.INVALID_PARAMS


# --- tools/call: PROTOCOL errors vs TOOL errors -------------------------------------------------------------------------
def test_a_successful_call_returns_text_content_and_is_error_false():
    r = call(ready(), "tools/call", {"name": "search_flights", "arguments": {"origin": "PDX", "destination": "HNL"}})["result"]
    assert r == {"content": [{"type": "text", "text": "AS 810, 08:00-10:45, $412"}], "isError": False}


def test_an_unknown_tool_is_a_PROTOCOL_error_not_a_tool_result():
    r = call(ready(), "tools/call", {"name": "nope", "arguments": {}})
    assert "result" not in r and err(r)["code"] == jr.INVALID_PARAMS and "Unknown tool: nope" in err(r)["message"]


def test_invalid_arguments_are_a_TOOL_error_so_the_model_can_fix_them():
    r = call(ready(), "tools/call", {"name": "search_flights", "arguments": {"origin": "PDX"}})
    assert "error" not in r and r["result"]["isError"] is True and "destination" in r["result"]["content"][0]["text"]


def test_a_tool_that_raises_toolerror_shows_its_message_with_is_error_true():
    r = call(ready(), "tools/call", {"name": "search_flights", "arguments": {"origin": "AAA", "destination": "BBB"}})["result"]
    assert r["isError"] is True and "No flights found for AAA-BBB" in r["content"][0]["text"]


def test_an_unexpected_exception_never_leaks_internals_to_the_client():
    s = ready()

    def boom(args):
        raise RuntimeError("secret database password is hunter2 at /srv/db.py line 12")

    s.add_tool("boom", "d", {"type": "object"}, boom)
    r = call(s, "tools/call", {"name": "boom", "arguments": {}})["result"]
    assert r["isError"] is True and r["content"][0]["text"] == "The tool failed."
    assert "hunter2" not in json.dumps(r) and "Traceback" not in json.dumps(r)
    assert call(s, "ping")["result"] == {}  # and the server is still alive


def test_non_string_results_are_json_encoded_and_bad_params_are_rejected():
    s = ready()
    s.add_tool("data", "d", {"type": "object"}, lambda a: {"n": 3})
    assert json.loads(call(s, "tools/call", {"name": "data", "arguments": {}})["result"]["content"][0]["text"]) == {"n": 3}
    assert err(call(s, "tools/call", {"name": 5}))["code"] == jr.INVALID_PARAMS
    assert err(call(s, "tools/call", {"name": "data", "arguments": [1]}))["code"] == jr.INVALID_PARAMS


# --- resources ---------------------------------------------------------------------------------------------------------
def test_resources_list_and_read_and_an_unknown_uri_has_its_own_error_code():
    s = ready()
    listed = call(s, "resources/list")["result"]["resources"]
    assert listed == [{"uri": "policy://baggage", "name": "Baggage policy", "description": "", "mimeType": "text/plain"}]
    read = call(s, "resources/read", {"uri": "policy://baggage"})["result"]["contents"][0]
    assert read["text"].startswith("One carry-on") and read["uri"] == "policy://baggage"
    e = err(call(s, "resources/read", {"uri": "policy://nope"}))
    assert e["code"] == jr.RESOURCE_NOT_FOUND and e["data"] == {"uri": "policy://nope"}


# --- everything else -----------------------------------------------------------------------------------------------------
def test_unknown_methods_get_method_not_found():
    assert err(call(ready(), "sampling/createMessage"))["code"] == jr.METHOD_NOT_FOUND


def test_notifications_and_stray_responses_never_get_a_reply():
    s = flights_server()
    assert s.handle(jr.notification("notifications/initialized")) is None and s.client_ready
    assert s.handle(jr.notification("something/unknown")) is None
    assert s.handle(jr.result(99, {})) is None


def test_a_bug_in_a_resource_reader_is_an_internal_error_and_the_server_survives():
    s = ready()
    s.add_resource("x://bad", "bad", lambda: 1 / 0)
    e = err(call(s, "resources/read", {"uri": "x://bad"}))
    assert e["code"] == jr.INTERNAL_ERROR and e["message"] == "internal error"
    assert call(s, "ping")["result"] == {}


# --- the stdio loop ------------------------------------------------------------------------------------------------------
def run_stdio(server, *lines):
    out = io.StringIO()
    serve_stdio(server, io.StringIO("\n".join(lines) + "\n"), out)
    return [json.loads(l) for l in out.getvalue().splitlines()]


def test_stdout_carries_only_json_rpc_lines_and_garbage_gets_an_id_null_error():
    s = flights_server()
    out = run_stdio(s, "total garbage", "", json.dumps(jr.request(1, "initialize", INIT)),
                    json.dumps(jr.notification("notifications/initialized")), json.dumps(jr.request(2, "tools/list")))
    assert out[0]["id"] is None and out[0]["error"]["code"] == jr.PARSE_ERROR
    assert [o["id"] for o in out[1:]] == [1, 2] and len(out) == 3  # the notification and the blank line produced nothing
    assert all(o["jsonrpc"] == "2.0" for o in out)


def test_list_changed_notifications_are_flushed_after_the_triggering_response():
    s = ready()
    s.add_tool("trigger", "d", {"type": "object"}, lambda a: s.notify_tools_changed() or "ok")
    out = run_stdio(s, json.dumps(jr.request(5, "tools/call", {"name": "trigger", "arguments": {}})))
    assert out[0]["id"] == 5 and out[1] == {"jsonrpc": "2.0", "method": "notifications/tools/list_changed"}


def test_the_loop_ends_cleanly_at_end_of_input():
    assert run_stdio(flights_server()) == []


def test_toolerror_is_the_documented_way_to_report_a_tool_level_failure():
    assert issubclass(ToolError, Exception)
