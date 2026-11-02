"""Reading a small model's tool calls without trusting them.

The lesson's loop does `json.loads(tc.function.arguments)` and then `TOOL_IMPL[name](**args)` with no `try`. A 7B model run at 4-bit will sometimes emit broken JSON, an argument that
does not exist, or a tool name it was never given. In that loop the first such call raises and the whole run is lost. Here every call is parsed and checked, and each failure becomes
a short message the MODEL can read and correct.

Some local models (Qwen's raw chat template among them) put the call in the message TEXT, as <tool_call>{...}</tool_call>, instead of the structured `tool_calls` field. That is
accepted when asked for, and marked `from_text` so the loop knows it has no call id to answer.
"""

import json
import re
from dataclasses import dataclass, field

MAX_ARGS_CHARS = 10_000
MAX_CALLS_PER_MESSAGE = 4
_TEXT_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class ToolCallError(Exception):
    pass


@dataclass(frozen=True)
class ParsedCall:
    id: str
    name: str
    args: dict
    from_text: bool = False


@dataclass(frozen=True)
class BadCall:
    id: str
    name: str
    reason: str


def validate_args(schema: dict, args) -> list[str]:
    """A small JSON-schema subset: object properties with type, enum, maxLength, minimum/exclusiveMinimum, and `required`. Extra properties are ALWAYS a problem."""
    if not isinstance(args, dict):
        return ["arguments must be a JSON object"]
    props, problems = schema.get("properties", {}), []
    for name in schema.get("required", []):
        if name not in args:
            problems.append(f"missing required argument {name!r}")
    for name, value in args.items():
        spec = props.get(name)
        if spec is None:
            problems.append(f"unexpected argument {name!r}; allowed: {sorted(props)}")
            continue
        kind = spec.get("type")
        if kind == "string":
            if not isinstance(value, str):
                problems.append(f"{name!r} must be a string")
            elif len(value) > spec.get("maxLength", 1000):
                problems.append(f"{name!r} is longer than {spec.get('maxLength', 1000)} characters")
        elif kind == "integer":
            if not isinstance(value, int) or isinstance(value, bool):
                problems.append(f"{name!r} must be an integer")
        elif kind == "number":
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                problems.append(f"{name!r} must be a number")
        elif kind == "boolean":
            if not isinstance(value, bool):
                problems.append(f"{name!r} must be true or false")
        if "enum" in spec and value not in spec["enum"]:
            problems.append(f"{name!r} must be one of {spec['enum']}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            if "minimum" in spec and value < spec["minimum"]:
                problems.append(f"{name!r} must be at least {spec['minimum']}")
            if "exclusiveMinimum" in spec and value <= spec["exclusiveMinimum"]:
                problems.append(f"{name!r} must be more than {spec['exclusiveMinimum']}")
    return problems


def parse_arguments(raw) -> dict:
    """Arguments arrive as a JSON string (OpenAI style) or already as an object (some servers). Anything else is an error, not an exception for the caller to meet."""
    if raw is None or raw == "":
        return {}
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        raise ToolCallError(f"arguments must be a JSON string or object, not {type(raw).__name__}")
    if len(raw) > MAX_ARGS_CHARS:
        raise ToolCallError(f"arguments are longer than {MAX_ARGS_CHARS} characters")
    try:
        value = json.loads(raw)
    except ValueError as exc:
        raise ToolCallError(f"arguments are not valid JSON ({exc.__class__.__name__})") from exc
    if not isinstance(value, dict):
        raise ToolCallError("arguments must be a JSON object")
    return value


def extract_calls(message: dict, accept_text: bool = True) -> list:
    """Returns a list of ParsedCall / BadCall from one assistant message. Never raises."""
    out = []
    structured = message.get("tool_calls") if isinstance(message, dict) else None
    if isinstance(structured, list) and structured:
        for i, tc in enumerate(structured[:MAX_CALLS_PER_MESSAGE]):
            cid = str(tc.get("id") or f"call_{i}") if isinstance(tc, dict) else f"call_{i}"
            fn = tc.get("function") if isinstance(tc, dict) else None
            if not isinstance(fn, dict) or not isinstance(fn.get("name"), str) or not fn["name"]:
                out.append(BadCall(cid, "", "the tool call has no function name"))
                continue
            try:
                out.append(ParsedCall(cid, fn["name"], parse_arguments(fn.get("arguments"))))
            except ToolCallError as exc:
                out.append(BadCall(cid, fn["name"], str(exc)))
        return out
    content = message.get("content") if isinstance(message, dict) else None
    if accept_text and isinstance(content, str) and "<tool_call>" in content:
        for i, m in enumerate(_TEXT_CALL.finditer(content)):
            if i >= MAX_CALLS_PER_MESSAGE:
                break
            try:
                obj = json.loads(m.group(1))
                name, args = obj.get("name"), parse_arguments(obj.get("arguments", {}))
                if not isinstance(name, str) or not name:
                    raise ToolCallError("the tool call has no name")
                out.append(ParsedCall(f"text_{i}", name, args, from_text=True))
            except ToolCallError as exc:
                out.append(BadCall(f"text_{i}", "", str(exc)))
            except (ValueError, AttributeError) as exc:
                out.append(BadCall(f"text_{i}", "", f"could not read the tool call ({exc.__class__.__name__})"))
    return out
