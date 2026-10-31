"""Scripted stand-in for the Anthropic client, so tests run offline."""

from types import SimpleNamespace as NS


def text(t):
    return NS(type="text", text=t)


def use(id_, tool_name, **inp):
    return NS(type="tool_use", id=id_, name=tool_name, input=inp)


def turn(*blocks, stop="tool_use"):
    return NS(content=list(blocks), stop_reason=stop, usage=NS(input_tokens=1, output_tokens=1))


def final(t="done"):
    return turn(text(t), stop="end_turn")


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})
        return self.replies.pop(0)

    def last_results(self):
        """The tool_result blocks the model saw on its most recent call."""
        return self.calls[-1]["messages"][-1]["content"]
