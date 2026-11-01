"""JSON-RPC 2.0: the message format underneath MCP (and, in this lesson, the A2A-inspired protocol).

Three kinds of message:
  request       has "id" and "method". Expects exactly one response with the same id.
  notification  has "method" but NO "id". Never gets a response.
  response      has "id" and either "result" or "error".

Over MCP's stdio transport, each message is ONE line of JSON. The encoder therefore guarantees no raw newline can appear
inside a message: it escapes them, and escapes non-ASCII too, so a line can never be split or corrupted.
Batches (JSON arrays) are not supported here.
"""

import json

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002  # the code MCP uses for an unknown resource (check your spec version)


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, data=None):
        super().__init__(message)
        self.code, self.message, self.data = code, message, data


def request(id_, method: str, params=None) -> dict:
    msg = {"jsonrpc": "2.0", "id": id_, "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def notification(method: str, params=None) -> dict:
    msg = {"jsonrpc": "2.0", "method": method}
    if params is not None:
        msg["params"] = params
    return msg


def result(id_, value) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": value}


def error(id_, code: int, message: str, data=None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": err}


def encode(msg: dict) -> str:
    """One line, no raw newlines, ASCII only."""
    return json.dumps(msg, separators=(",", ":"), ensure_ascii=True)


def kind(msg: dict) -> str:
    if "method" in msg:
        return "request" if "id" in msg else "notification"
    return "response"


def decode(line: str) -> dict:
    """Parse and structurally validate one message. Raises JsonRpcError(PARSE_ERROR or INVALID_REQUEST)."""
    try:
        msg = json.loads(line)
    except (json.JSONDecodeError, ValueError) as exc:
        raise JsonRpcError(PARSE_ERROR, f"parse error: {exc.msg if hasattr(exc, 'msg') else exc}") from exc
    if isinstance(msg, list):
        raise JsonRpcError(INVALID_REQUEST, "batches are not supported")
    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
        raise JsonRpcError(INVALID_REQUEST, 'not a JSON-RPC 2.0 message (missing "jsonrpc": "2.0")')
    if "method" in msg:
        if not isinstance(msg["method"], str):
            raise JsonRpcError(INVALID_REQUEST, "method must be a string")
        if "params" in msg and not isinstance(msg["params"], (dict, list)):
            raise JsonRpcError(INVALID_REQUEST, "params must be an object or array")
        if "id" in msg and (isinstance(msg["id"], bool) or not isinstance(msg["id"], (str, int))):
            raise JsonRpcError(INVALID_REQUEST, "id must be a string or integer")
    elif "result" in msg or "error" in msg:
        if "id" not in msg:
            raise JsonRpcError(INVALID_REQUEST, "a response needs an id")
    else:
        raise JsonRpcError(INVALID_REQUEST, "message has no method, result or error")
    return msg
