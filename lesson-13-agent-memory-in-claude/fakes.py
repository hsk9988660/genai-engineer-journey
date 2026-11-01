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
