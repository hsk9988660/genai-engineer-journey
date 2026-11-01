"""Scripted stand-in for the Anthropic client, so tests run offline."""

from types import SimpleNamespace as NS


def submit(payload: dict, id_="s"):
    """A model reply that calls the forced `submit_result` tool with `payload`."""
    block = NS(type="tool_use", id=id_, name="submit_result", input=payload)
    return NS(content=[block], stop_reason="tool_use", usage=NS(input_tokens=100, output_tokens=50))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})
        return self.replies.pop(0)

    def last_feedback(self):
        """The problems the model was told about on its most recent call (empty if none)."""
        last = self.calls[-1]["messages"][-1]
        return last["content"][0]["content"] if isinstance(last["content"], list) else ""
