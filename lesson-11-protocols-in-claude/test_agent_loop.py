import json

from agent_loop import SUBMIT, run_agent
from demo_servers import BENIGN_TIME, flights_server, injection_server, rugpull_server
from fakes import FakeModel, answer, loopback_gateway, pins_for, turn, use
from gateway import ServerConfig, definition_hash

BOOK = {"origin": "PDX", "destination": "HNL", "passenger": "Ana Ruiz"}


def flights_gw(approver=None):
    s = flights_server()
    gw = loopback_gateway({"flights": s}, approver)
    gw.add_server(ServerConfig("flights", allow_tools=frozenset(s.tools), pins=pins_for(s), side_effect_tools=frozenset({"book_flight"})))
    return gw, s


def test_the_models_tool_list_comes_from_discovery_and_nothing_is_hard_coded_in_the_agent():
    gw, s = flights_gw()
    model = FakeModel([answer("ok")])
    run_agent(model, gw, "find a flight")
    offered = [t["name"] for t in model.calls[0]["tools"]]
    assert offered == [t["name"] for t in gw.exposed_tools()] + ["submit_answer"]
    assert sorted(offered) == ["flights__book_flight", "flights__search_flights", "submit_answer"]
    by_name = {t["name"]: t for t in model.calls[0]["tools"]}
    assert by_name["flights__search_flights"]["input_schema"] == s.tools["search_flights"].input_schema  # the server's own schema


def test_a_tool_discovered_at_runtime_is_called_through_the_gateway_and_its_result_is_wrapped_as_untrusted_data():
    gw, s = flights_gw()
    model = FakeModel([turn(use("c1", "flights__search_flights", origin="PDX", destination="HNL")), answer("Take AS 810")])
    run = run_agent(model, gw, "find a flight")
    assert run.status == "answered" and run.answer == "Take AS 810" and run.calls == [("flights__search_flights", "ok")]
    result = model.calls[1]["messages"][-1]["content"][0]
    assert result["is_error"] is False
    assert result["content"].startswith('<untrusted_tool_output server="flights">') and "AS 810, 08:00-10:45, $412" in result["content"]
    assert "untrusted" in model.calls[0]["system"].lower()


def test_failures_reach_the_model_as_status_json_never_as_raw_server_text():
    gw, s = flights_gw()
    model = FakeModel([turn(use("c1", "flights__search_flights", origin="PDX")),   # invalid arguments
                       turn(use("c2", "nope__x")),                                 # unknown tool
                       turn(use("c3", "flights__book_flight", **BOOK)),            # not approved
                       answer()])
    run = run_agent(model, gw, "x")
    assert [c[1] for c in run.calls] == ["invalid_arguments", "unknown_tool", "denied"]
    for call in model.calls[1:]:
        res = call["messages"][-1]["content"][0]
        assert res["is_error"] is True and "status" in json.loads(res["content"])


def test_a_withdrawn_or_changed_tool_disappears_from_the_very_next_model_call():
    s = rugpull_server()
    gw = loopback_gateway({"clock": s})
    gw.add_server(ServerConfig("clock", allow_tools=frozenset({"get_time"}),
                               pins={"get_time": definition_hash("get_time", BENIGN_TIME, s.tools["get_time"].input_schema)}))
    model = FakeModel([turn(use("c1", "clock__get_time")), answer("could not get the time")])
    run = run_agent(model, gw, "what time is it?")
    assert run.tool_lists == [["clock__get_time"], []]  # offered at step 1, gone at step 2
    assert run.calls == [("clock__get_time", "unavailable")]
    assert "changed" in json.loads(model.calls[1]["messages"][-1]["content"][0]["content"])["message"]


def test_a_prompt_injection_in_a_tool_result_cannot_trigger_a_booking_even_if_the_model_obeys():
    flights, web = flights_server(), injection_server()
    gw = loopback_gateway({"flights": flights, "web": web}, approver=lambda s: "approve")  # even a permissive human cannot save us here
    gw.add_server(ServerConfig("flights", allow_tools=frozenset({"book_flight"}), pins=pins_for(flights, "book_flight"),
                               side_effect_tools=frozenset({"book_flight"})))
    gw.add_server(ServerConfig("web", allow_tools=frozenset({"fetch_page"}), pins=pins_for(web, "fetch_page"),
                               untrusted_output_tools=frozenset({"fetch_page"})))
    model = FakeModel([turn(use("c1", "web__fetch_page", url="https://deals.test")),
                       turn(use("c2", "flights__book_flight", **{**BOOK, "passenger": "Mallory"})),  # the model obeys the page
                       answer("I summarised the page.")])
    run = run_agent(model, gw, "summarise https://deals.test")
    assert run.calls == [("web__fetch_page", "ok"), ("flights__book_flight", "blocked_taint")]
    assert flights.bookings == [] and run.state.tainted


def test_a_user_requested_booking_with_approval_works_and_is_logged():
    gw, s = flights_gw(approver=lambda summary: "approve")
    model = FakeModel([turn(use("c1", "flights__book_flight", **BOOK)), answer("booked")])
    run = run_agent(model, gw, "book PDX to HNL for Ana Ruiz")
    assert run.calls == [("flights__book_flight", "ok")] and s.bookings == [BOOK]


def test_the_step_cap_ends_a_model_that_never_finishes():
    gw, s = flights_gw()
    replies = [turn(use(f"c{i}", "flights__search_flights", origin="PDX", destination="HNL")) for i in range(10)]
    run = run_agent(FakeModel(replies), gw, "x", max_steps=3)
    assert run.status == "capped" and run.steps == 3 and len(run.calls) == 3


def test_the_submit_tool_is_always_offered_and_ends_the_run():
    gw, s = flights_gw()
    model = FakeModel([answer("done")])
    run = run_agent(model, gw, "x")
    assert SUBMIT in model.calls[0]["tools"] and run.status == "answered" and run.steps == 1 and run.calls == []
