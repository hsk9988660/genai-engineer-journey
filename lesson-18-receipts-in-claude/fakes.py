"""Scripted stand-in for the Anthropic client."""

from types import SimpleNamespace as NS


def submit(payload: dict):
    block = NS(type="tool_use", id="sub", name="submit_result", input=payload)
    return NS(content=[block], stop_reason="tool_use", usage=NS(input_tokens=10, output_tokens=5))


def text_reply(text: str):
    return NS(content=[NS(type="text", text=text)], stop_reason="end_turn", usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})
        return self.replies.pop(0)


def tool_act(**inp):
    """A scripted model reply that calls the `act` tool once (lesson 15)."""
    block = NS(type="tool_use", id="act1", name="act", input=inp)
    return NS(content=[block], stop_reason="tool_use", usage=NS(input_tokens=100, output_tokens=20))


class RuleClient:
    """A scripted 'model' whose reply is computed from the request by `fn(kwargs) -> reply`. A TEST DOUBLE: it has the behavior its function gives it, nothing more."""

    def __init__(self, fn):
        self.fn, self.calls = fn, []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})
        return self.fn(kw)


# --- lesson 17: OpenAI-compatible chat doubles ---------------------------------------------------------------------------------------
def chat_reply(content=None, calls=(), prompt_tokens=20, completion_tokens=10):
    """An OpenAI chat-completions response dict. `calls` is a list of (name, arguments) where arguments is a JSON string, a dict, or anything (to build bad calls)."""
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [{"id": f"call_{i}", "type": "function", "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(calls)]
    return {"model": "scripted", "choices": [{"index": 0, "message": msg}], "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens}}


class ScriptedChat:
    """TEST DOUBLE for a local model: `.chat(model, messages, tools)` replays a list of responses, or computes one with `fn(messages, tools, call_index)`. Records every request."""

    def __init__(self, responses=None, fn=None):
        self.responses, self.fn, self.calls = list(responses or []), fn, []

    def chat(self, model, messages, tools=None, temperature=0.0, max_tokens=512):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages], "tools": tools})
        if self.fn is not None:
            out = self.fn(messages, tools, len(self.calls) - 1)
        else:
            out = self.responses.pop(0)
        if isinstance(out, Exception):
            raise out
        return out
