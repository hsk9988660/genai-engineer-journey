import json

import pytest

import jsonrpc as jr
from schema import validate


# --- the wire format ----------------------------------------------------------------------------------
def test_an_encoded_message_is_always_exactly_one_ascii_line():
    msg = jr.request(1, "tools/call", {"text": "line one\nline two\r\nünïcode ✓ 日本"})
    line = jr.encode(msg)
    assert "\n" not in line and "\r" not in line and line.isascii()
    assert jr.decode(line)["params"]["text"] == "line one\nline two\r\nünïcode ✓ 日本"  # and it round-trips exactly


def test_the_three_message_kinds_have_the_right_shape():
    assert jr.request(7, "ping") == {"jsonrpc": "2.0", "id": 7, "method": "ping"}
    assert "id" not in jr.notification("notifications/initialized")  # a notification has no id, so it can never be answered
    assert jr.result(7, {})["result"] == {} and jr.error(7, -32601, "nope", {"x": 1})["error"] == {"code": -32601, "message": "nope", "data": {"x": 1}}
    assert [jr.kind(m) for m in (jr.request(1, "a"), jr.notification("a"), jr.result(1, {}), jr.error(1, 1, "x"))] == \
        ["request", "notification", "response", "response"]


@pytest.mark.parametrize("line,code", [
    ("{not json", jr.PARSE_ERROR),
    ("debug: starting up", jr.PARSE_ERROR),  # a log line on the protocol channel
    ("[]", jr.INVALID_REQUEST),  # batches are not supported
    ('"just a string"', jr.INVALID_REQUEST),
    ('{"id":1,"method":"x"}', jr.INVALID_REQUEST),  # no "jsonrpc": "2.0"
    ('{"jsonrpc":"2.0","id":1,"method":5}', jr.INVALID_REQUEST),
    ('{"jsonrpc":"2.0","id":true,"method":"x"}', jr.INVALID_REQUEST),  # a bool is not a valid id
    ('{"jsonrpc":"2.0","id":1,"method":"x","params":5}', jr.INVALID_REQUEST),
    ('{"jsonrpc":"2.0","result":{}}', jr.INVALID_REQUEST),  # a response needs an id
    ('{"jsonrpc":"2.0","id":1}', jr.INVALID_REQUEST),  # neither method, result nor error
])
def test_malformed_messages_are_rejected_with_the_right_error_code(line, code):
    with pytest.raises(jr.JsonRpcError) as e:
        jr.decode(line)
    assert e.value.code == code


def test_string_and_integer_ids_are_both_valid():
    assert jr.decode('{"jsonrpc":"2.0","id":"abc","method":"ping"}')["id"] == "abc"
    assert jr.decode('{"jsonrpc":"2.0","id":0,"method":"ping"}')["id"] == 0


# --- the argument validator ---------------------------------------------------------------------------------
SCHEMA = {"type": "object", "properties": {"city": {"type": "string", "minLength": 2, "maxLength": 5},
                                           "n": {"type": "integer", "minimum": 1, "maximum": 3},
                                           "mode": {"enum": ["a", "b"]}, "tags": {"type": "array", "items": {"type": "string"}}},
          "required": ["city"], "additionalProperties": False}


def test_valid_arguments_produce_no_problems():
    assert validate({"city": "Rome", "n": 2, "mode": "a", "tags": ["x"]}, SCHEMA) == []


@pytest.mark.parametrize("args,fragment", [
    ({}, "missing required property 'city'"),
    ({"city": 5}, "expected string"),
    ({"city": "R"}, "shorter than 2"),
    ({"city": "Rome-City"}, "longer than 5"),
    ({"city": "Rome", "n": 0}, "below the minimum"),
    ({"city": "Rome", "n": 9}, "above the maximum"),
    ({"city": "Rome", "n": 1.5}, "expected integer"),
    ({"city": "Rome", "n": True}, "expected integer"),  # a bool is not an integer
    ({"city": "Rome", "mode": "z"}, "must be one of"),
    ({"city": "Rome", "tags": ["x", 3]}, "arguments.tags[1]"),
    ({"city": "Rome", "extra": 1}, "unexpected property 'extra'"),
])
def test_each_kind_of_bad_argument_is_reported_with_its_location(args, fragment):
    assert any(fragment in p for p in validate(args, SCHEMA)), validate(args, SCHEMA)


def test_the_validator_only_enforces_a_subset_and_ignores_the_rest():
    """Documented limit: unsupported keywords like `pattern` are NOT enforced, so a server's own validation still matters."""
    assert validate({"s": "anything"}, {"type": "object", "properties": {"s": {"type": "string", "pattern": "^x$"}}}) == []
    assert validate(5, {"type": ["integer", "null"]}) == [] and validate(None, {"type": ["integer", "null"]}) == []
    assert validate("x", {"type": ["integer", "null"]})
