"""Scripted stand-in for the Anthropic client, so tests run offline."""

from types import SimpleNamespace as NS


def use(id_, tool_name, **inp):
    return NS(type="tool_use", id=id_, name=tool_name, input=inp)


def turn(*blocks):
    return NS(content=list(blocks), stop_reason="tool_use", usage=NS(input_tokens=10, output_tokens=5))


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
