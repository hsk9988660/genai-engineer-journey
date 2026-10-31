"""Tests for the agent loop. They use a fake client, so no API key or network is needed.

Run:  python -m pytest -q
"""

from types import SimpleNamespace as NS

import agent


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def reply(content, stop_reason):
    return NS(content=content, stop_reason=stop_reason, usage=NS(input_tokens=10, output_tokens=5))


class FakeClient:
    """Returns scripted responses in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs["messages"][:])  # snapshot of history at call time
        return self.replies.pop(0)


def test_answers_directly_without_tools():
    client = FakeClient([reply([text("Hi!")], "end_turn")])
    res = agent.run_agent("hello", client=client, verbose=False)
    assert res.answer == "Hi!" and res.steps == 1 and not res.stopped_early


def test_tool_result_is_fed_back_to_the_model():
    client = FakeClient(
        [
            reply([tool_use("t1", "get_destinations")], "tool_use"),
            reply([text("Try Bali")], "end_turn"),
        ]
    )
    res = agent.run_agent("beach?", client=client, verbose=False)
    assert res.answer == "Try Bali" and res.steps == 2
    last_msg = client.calls[1][-1]  # what the model saw on step 2
    assert last_msg["role"] == "user"
    assert last_msg["content"][0]["tool_use_id"] == "t1"
    assert "Bali" in last_msg["content"][0]["content"]
    assert res.input_tokens == 20 and res.output_tokens == 10


def test_unknown_tool_is_reported_as_error_not_crash():
    client = FakeClient(
        [
            reply([tool_use("t1", "launch_rocket")], "tool_use"),
            reply([text("Sorry")], "end_turn"),
        ]
    )
    res = agent.run_agent("go", client=client, verbose=False)
    assert res.answer == "Sorry"
    assert client.calls[1][-1]["content"][0]["is_error"] is True


def test_tool_exception_is_caught():
    client = FakeClient(
        [
            reply([tool_use("t1", "get_destination_details", wrong_arg="x")], "tool_use"),
            reply([text("ok")], "end_turn"),
        ]
    )
    res = agent.run_agent("go", client=client, verbose=False)
    assert res.answer == "ok"
    assert client.calls[1][-1]["content"][0]["is_error"] is True


def test_step_limit_stops_runaway_loop():
    loop = [reply([tool_use(f"t{i}", "get_destinations")], "tool_use") for i in range(10)]
    client = FakeClient(loop)
    res = agent.run_agent("go", client=client, max_steps=3, verbose=False)
    assert res.stopped_early and res.steps == 3
