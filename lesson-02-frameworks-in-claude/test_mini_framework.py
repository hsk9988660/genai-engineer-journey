"""Offline tests: a fake client replaces the API, so no key or network is needed."""

from types import SimpleNamespace as NS

import pytest

import examples
from mini_framework import Agent, tool


def text(t):
    return NS(type="text", text=t)


def tool_use(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def reply(content, stop_reason="end_turn"):
    return NS(content=content, stop_reason=stop_reason, usage=NS(input_tokens=10, output_tokens=5))


class FakeStream:
    def __init__(self, resp, chunks):
        self.resp, self.text_stream = resp, iter(chunks)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.resp


class FakeClient:
    def __init__(self, replies, chunks=()):
        self.replies, self.chunks, self.calls = list(replies), chunks, []
        self.messages = NS(create=self._create, stream=self._stream)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})  # snapshot of what the model saw
        return self.replies.pop(0)

    def _stream(self, **kw):
        return FakeStream(self._create(**kw), self.chunks)


# --- @tool ---------------------------------------------------------------------------------
def test_tool_builds_schema_from_signature_and_docstring():
    @tool
    def f(city: str, days: int, tags: list[str], rush: bool = False) -> str:
        """Plan a stay."""

    s = f.schema
    assert s["name"] == "f" and s["description"] == "Plan a stay."
    props = s["input_schema"]["properties"]
    assert props["city"]["type"] == "string" and props["days"]["type"] == "integer"
    assert props["tags"]["type"] == "array" and props["rush"]["type"] == "boolean"
    assert s["input_schema"]["required"] == ["city", "days", "tags"]  # rush has a default


def test_tool_without_docstring_is_rejected():
    with pytest.raises(ValueError):

        @tool
        def f(x: str):
            return x


# --- Agent loop ----------------------------------------------------------------------------
def test_plain_answer_and_no_tools_param_when_agent_has_none():
    client = FakeClient([reply([text("Hi!")])])
    assert Agent("a", "sys", client=client).run("hello") == "Hi!"
    assert "tools" not in client.calls[0]


def test_tool_roundtrip():
    client = FakeClient([reply([tool_use("t1", "get_weather", location="Lisbon")], "tool_use"),
                         reply([text("Sunny, no jacket")])])
    agent = Agent("w", "sys", tools=[examples.get_weather], client=client)
    assert agent.run("jacket?") == "Sunny, no jacket"
    sent = client.calls[1]["messages"][-1]["content"][0]
    assert sent["tool_use_id"] == "t1" and "sunny" in sent["content"]
    assert agent.input_tokens == 20 and agent.output_tokens == 10


def test_history_persists_across_runs_like_a_thread():
    client = FakeClient([reply([text("one")]), reply([text("two")])])
    agent = Agent("a", "sys", client=client)
    agent.run("first")
    agent.run("second")
    seen = client.calls[1]["messages"]  # second call must include the first exchange
    assert [m["role"] for m in seen] == ["user", "assistant", "user"]
    assert seen[0]["content"] == "first"


def test_approval_denied_blocks_the_tool():
    ran = []

    @tool(requires_approval=True)
    def book(x: str) -> str:
        """Book something."""
        ran.append(x)
        return "booked"

    client = FakeClient([reply([tool_use("t1", "book", x="Paris")], "tool_use"), reply([text("ok, not booked")])])
    agent = Agent("a", "sys", tools=[book], client=client, approve=lambda n, a: False)
    agent.run("book it")
    assert ran == []  # the function never executed
    assert client.calls[1]["messages"][-1]["content"][0]["is_error"] is True


def test_approval_granted_runs_the_tool():
    ran = []

    @tool(requires_approval=True)
    def book(x: str) -> str:
        """Book something."""
        ran.append(x)
        return "booked"

    client = FakeClient([reply([tool_use("t1", "book", x="Paris")], "tool_use"), reply([text("done")])])
    Agent("a", "sys", tools=[book], client=client, approve=lambda n, a: True).run("book it")
    assert ran == ["Paris"]


def test_unknown_tool_and_tool_exception_do_not_crash():
    client = FakeClient([reply([tool_use("t1", "nope"), tool_use("t2", "get_weather", wrong=1)], "tool_use"),
                         reply([text("recovered")])])
    agent = Agent("a", "sys", tools=[examples.get_weather], client=client)
    assert agent.run("go") == "recovered"
    results = client.calls[1]["messages"][-1]["content"]
    assert all(r["is_error"] for r in results)


def test_step_limit():
    client = FakeClient([reply([tool_use(f"t{i}", "get_weather", location="x")], "tool_use") for i in range(9)])
    agent = Agent("a", "sys", tools=[examples.get_weather], client=client, max_steps=3)
    assert agent.run("go").startswith("Stopped")
    assert len(client.calls) == 3


def test_streaming_passes_chunks_to_callback():
    client = FakeClient([reply([text("Hello world")])], chunks=["Hello ", "world"])
    got = []
    out = Agent("a", "sys", client=client).run("hi", on_text=got.append)
    assert got == ["Hello ", "world"] and out == "Hello world"


# --- The lesson examples use real tools ----------------------------------------------------
def test_example_tools_work_and_book_flight_requires_approval():
    assert examples.calculate_growth(120_000, 150_000) == 25.0
    assert examples.retrieve_sales_data("Q4") == {"sales": 150_000}
    assert examples.book_flight.requires_approval is True
    assert examples.get_weather.requires_approval is False
    assert set(examples.EXAMPLES) == {"hello", "travel", "weather", "chain", "plan_and_execute", "thread", "streaming"}
