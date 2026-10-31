"""Offline tests with a scripted fake client (no API key, no network)."""

from types import SimpleNamespace as NS

import pytest
from pydantic import BaseModel

import patterns
from patterns import (Destination, Itinerary, NoAvailableDestination, Shortlist,
                      StructuredOutputError, plan_trip, run_structured)


def tool_use(id_, tool_name, **inp):
    return NS(type="tool_use", id=id_, name=tool_name, input=inp)


def reply(*blocks):
    return NS(content=list(blocks), stop_reason="tool_use", usage=NS(input_tokens=1, output_tokens=1))


class FakeClient:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []
        self.messages = NS(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw | {"messages": kw["messages"][:]})
        return self.replies.pop(0)


class Person(BaseModel):
    name: str
    age: int


def dest(name, available=True, budget=2000):
    return Destination(destination=name, available=available, best_season="x",
                       highlights=["a"], estimated_budget_usd=budget).model_dump()


ITIN = dict(destination="Barcelona", days=["Day 1: tapas"], estimated_total_usd=1900, visa_note="None for EU")


# --- Pattern 2 -----------------------------------------------------------------------------
def test_valid_output_is_parsed_and_tool_use_is_forced():
    client = FakeClient([reply(tool_use("t1", "submit_result", name="Ana", age=30))])
    out = run_structured(client, "sys", "who?", Person)
    assert out == Person(name="Ana", age=30)
    assert client.calls[0]["tool_choice"] == {"type": "any"}
    assert client.calls[0]["tools"][-1]["name"] == "submit_result"


def test_invalid_output_is_sent_back_and_retried():
    client = FakeClient([reply(tool_use("t1", "submit_result", name="Ana", age="old")),
                         reply(tool_use("t2", "submit_result", name="Ana", age=30))])
    assert run_structured(client, "sys", "who?", Person).age == 30
    feedback = client.calls[1]["messages"][-1]["content"][0]
    assert feedback["is_error"] is True and "validation" in feedback["content"].lower()


def test_gives_up_after_max_retries():
    client = FakeClient([reply(tool_use(f"t{i}", "submit_result", name="Ana")) for i in range(5)])
    with pytest.raises(StructuredOutputError):
        run_structured(client, "sys", "who?", Person, max_retries=2)
    assert len(client.calls) == 3  # first try + 2 retries


def test_tool_runs_before_submit_and_result_is_fed_back():
    client = FakeClient([reply(tool_use("t1", "get_destination_details", destination="Lisbon")),
                         reply(tool_use("t2", "submit_result", picks=[dest("Lisbon")], reasoning="ok"))])
    out = run_structured(client, "sys", "go", Shortlist, tools=[patterns.get_destination_details])
    assert out.picks[0].destination == "Lisbon"
    assert "Lisbon" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_tool_error_and_unknown_tool_are_reported_not_raised():
    client = FakeClient([reply(tool_use("t1", "get_destination_details", destination="Atlantis"),
                               tool_use("t2", "nope")),
                         reply(tool_use("t3", "submit_result", name="Ana", age=1))])
    run_structured(client, "sys", "go", Person, tools=[patterns.get_destination_details])
    results = client.calls[1]["messages"][-1]["content"]
    assert [r["is_error"] for r in results] == [True, True]


def test_free_text_reply_is_an_error():
    client = FakeClient([NS(content=[NS(type="text", text="hi")], stop_reason="end_turn",
                            usage=NS(input_tokens=1, output_tokens=1))])
    with pytest.raises(StructuredOutputError):
        run_structured(client, "sys", "go", Person)


def test_tool_schema_generated_from_signature():
    s = patterns.get_destination_details.schema
    assert s["name"] == "get_destination_details"
    assert s["input_schema"]["properties"]["destination"]["type"] == "string"


# --- Pattern 3: the pipeline ---------------------------------------------------------------
def test_pipeline_filters_with_ground_truth_not_model_claims():
    # The model claims Cape Town is available and cheap. The database says it is unavailable.
    # Tokyo is real but over budget. Only Barcelona survives, and the planner sees only it.
    picks = [dest("Cape Town", available=True, budget=100), dest("Tokyo"), dest("Barcelona")]
    client = FakeClient([reply(tool_use("t1", "submit_result", picks=picks, reasoning="r")),
                         reply(tool_use("t2", "submit_result", **ITIN))])
    out = plan_trip(client, "culture and food", budget_usd=2200)
    assert isinstance(out, Itinerary) and out.destination == "Barcelona"
    brief = client.calls[1]["messages"][0]["content"]
    assert "Barcelona" in brief and "Tokyo" not in brief and "Cape Town" not in brief


def test_pipeline_stops_before_planner_if_nothing_qualifies():
    client = FakeClient([reply(tool_use("t1", "submit_result",
                                        picks=[dest("Cape Town"), dest("Atlantis")], reasoning="r"))])
    with pytest.raises(NoAvailableDestination):
        plan_trip(client, "anything", budget_usd=5000)
    assert len(client.calls) == 1  # the planner agent was never called, saving a model call


def test_planner_prompt_has_the_do_not_constraint():
    assert "Do NOT choose" in patterns.PLANNER_INSTRUCTIONS
    assert "Do NOT plan" in patterns.EXPERT_INSTRUCTIONS
