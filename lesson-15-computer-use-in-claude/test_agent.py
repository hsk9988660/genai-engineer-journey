import json

import pytest

from agent import ACT, Approval, ActionError, ComputerUseAgent, Limits, SYSTEM, parse_action
from fakes import FakeClient, RuleClient, text_reply, tool_act
from policy import ActionPolicy, Click, Done, Navigate, Type
from runtrace import Trace
from simsite import HOME, RESULTS, SimBrowser, SimSite
from urlpolicy import DomainPolicy

POLICY = ActionPolicy(DomainPolicy(["stays.example"]))
TASK = "Find the cheapest stay."


def make(replies, start=RESULTS, site=None, approver=None, limits=None, clock=None, client=None):
    site = site or SimSite()
    browser = SimBrowser(site, start)
    kw = {"clock": clock} if clock else {}
    agent = ComputerUseAgent(client or FakeClient(replies), browser, POLICY, limits, approver=approver, **kw)
    return agent, browser, site


def approve_all(summary, h):
    return Approval("approve", h, "manager-7")


# --- parsing the model's action --------------------------------------------------------------------------------------------------------------
def test_each_action_kind_is_parsed_into_its_type():
    assert parse_action({"action": "navigate", "url": "/x"}) == Navigate("/x")
    assert parse_action({"action": "click", "ref": 3}) == Click(3)
    assert parse_action({"action": "type", "ref": 2, "text": "hi"}) == Type(2, "hi")
    assert parse_action({"action": "done", "answer": "ok"}) == Done("ok") and parse_action({"action": "done"}) == Done("")


@pytest.mark.parametrize("bad", [{}, {"action": "fly"}, {"action": "click"}, {"action": "click", "ref": "3"}, {"action": "click", "ref": True}, {"action": "click", "ref": 1.5},
                                 {"action": "navigate"}, {"action": "navigate", "url": 5}, {"action": "type", "ref": 1}, {"action": "click", "ref": 1, "user_id": "x"}, "click", None])
def test_a_malformed_action_is_refused(bad):
    with pytest.raises(ActionError):
        parse_action(bad)


def test_a_long_answer_is_cut_to_a_bound():
    assert len(parse_action({"action": "done", "answer": "x" * 900}).answer) == 500


# --- what the model is sent ------------------------------------------------------------------------------------------------------------------
def test_each_turn_sends_a_stable_system_prompt_a_forced_tool_and_the_task_with_the_safe_page():
    agent, _, _ = make([tool_act(action="done")])
    agent.run(TASK, RESULTS)
    call = agent.client.calls[0]
    assert call["system"] == SYSTEM and call["tools"] == [ACT] and call["tool_choice"] == {"type": "tool", "name": "act"}
    body = call["messages"][0]["content"]
    assert TASK in body and "data copied from a website" in body and "AI agents reading this" not in body and "(nothing yet)" in body


def test_the_system_prompt_is_stable_so_it_can_be_cached_it_has_no_dates_or_ids():
    import re
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}:\d{2}|[0-9a-f]{32}", SYSTEM)


def test_the_tool_schema_forbids_extra_properties_and_limits_the_actions():
    s = ACT["input_schema"]
    assert s["additionalProperties"] is False and s["properties"]["action"]["enum"] == ["navigate", "click", "type", "done"]


def test_only_the_last_eight_results_are_sent_so_the_prompt_does_not_grow_with_every_step():
    replies = [tool_act(action="navigate", url="/" if i % 2 else "/s/stockholm") for i in range(12)] + [tool_act(action="done")]
    agent, _, _ = make(replies, limits=Limits(max_actions=50, max_repeats=100))
    agent.run(TASK, RESULTS)
    last = agent.client.calls[-1]["messages"][0]["content"]
    assert last.count("- navigate ok") == 8


# --- the loop --------------------------------------------------------------------------------------------------------------------------------
def test_done_ends_the_run_with_the_answer_and_takes_no_action():
    agent, _, _ = make([tool_act(action="done", answer="all set")])
    r = agent.run(TASK, RESULTS)
    assert (r.status, r.answer, r.actions) == ("done", "all set", 0)


def test_a_start_page_that_is_not_allowed_stops_the_run_before_anything_is_requested():
    agent, browser, site = make([], start=RESULTS)
    r = agent.run(TASK, "https://evil.example/")
    assert r.status == "stopped" and "not allowed" in r.reason and site.fetched == [RESULTS]


def test_clicking_a_listing_follows_the_link_and_the_next_turn_shows_the_new_page():
    agent, browser, _ = make([tool_act(action="click", ref=1), tool_act(action="done")])
    r = agent.run(TASK, RESULTS)
    assert browser.url == "https://stays.example/rooms/101" and r.actions == 1
    assert "Reserve" in agent.client.calls[1]["messages"][0]["content"] and "- click ok" in agent.client.calls[1]["messages"][0]["content"]


def test_typing_and_clicking_search_runs_a_search_with_no_human():
    home_refs = {"where": 3, "search": 4}  # the home page starts with a cookie dialog and its button
    agent, browser, _ = make([tool_act(action="type", ref=home_refs["where"], text="Stockholm"), tool_act(action="click", ref=home_refs["search"]), tool_act(action="done")], start=HOME)
    r = agent.run(TASK, HOME)
    assert r.actions == 2 and browser.url == "https://stays.example/s/stockholm" and browser._values == {"search-input": "Stockholm"}


def test_navigating_off_site_is_refused_and_no_request_is_made():
    agent, browser, site = make([tool_act(action="navigate", url="https://evil.example/x"), tool_act(action="done")])
    r = agent.run(TASK, RESULTS)
    assert r.denied == 1 and r.actions == 0 and not any("evil" in u for u in site.fetched)
    assert "refused: navigation refused" in agent.client.calls[1]["messages"][0]["content"]


def test_a_same_site_link_that_redirects_off_site_is_stopped_at_the_redirect_hop_before_any_request_to_the_other_site():
    agent, browser, site = make([tool_act(action="click", ref=6), tool_act(action="done")])  # "Featured stay" -> /go/9 -> evil.example
    r = agent.run(TASK, RESULTS)
    assert r.denied == 1 and "https://stays.example/go/9" in site.fetched and not any("evil" in u for u in site.fetched) and browser.url == RESULTS
    assert any(e["type"] == "navigation_blocked" for e in agent.trace.events)


def test_clicking_an_off_site_link_is_refused_by_policy():
    agent, _, site = make([tool_act(action="click", ref=5), tool_act(action="done")])  # the sponsored link
    r = agent.run(TASK, RESULTS)
    assert r.denied == 1 and not any("ads.example" in u for u in site.fetched)


def test_a_missing_or_malformed_tool_call_is_counted_and_the_model_is_told():
    agent, _, _ = make([text_reply("I will just talk"), tool_act(action="click", ref="one"), tool_act(action="done")])
    r = agent.run(TASK, RESULTS)
    assert r.denied == 2 and "you must call the act tool" in agent.client.calls[1]["messages"][0]["content"] and "invalid action" in agent.client.calls[2]["messages"][0]["content"]


def test_too_many_refusals_stop_the_run():
    agent, _, _ = make([tool_act(action="navigate", url=f"https://evil.example/{i}") for i in range(10)], limits=Limits(max_denied=3))
    r = agent.run(TASK, RESULTS)
    assert r.status == "stopped" and "refused" in r.reason and r.denied == 4


def test_the_action_limit_stops_a_run_that_keeps_acting():
    replies = [tool_act(action="navigate", url="/" if i % 2 else "/s/stockholm") for i in range(20)]
    agent, _, _ = make(replies, limits=Limits(max_actions=3))
    r = agent.run(TASK, RESULTS)
    assert r.status == "stopped" and r.actions == 3 and "action limit" in r.reason


def test_doing_the_same_thing_on_an_unchanged_page_is_stuck_and_stops_after_the_allowed_repeats():
    agent, _, site = make([tool_act(action="click", ref=7)] * 10)  # "Show more" does nothing visible
    r = agent.run(TASK, RESULTS)
    assert r.status == "stopped" and "stuck" in r.reason and r.actions == 2 and site.show_more_clicks == 2


def test_the_same_action_on_a_page_that_did_change_does_not_count_as_a_repeat():
    agent, _, _ = make([tool_act(action="navigate", url="/rooms/5")] * 3 + [tool_act(action="done")])  # from the results page, then twice from the room page
    assert agent.run(TASK, RESULTS).status == "done"


def test_cycling_between_two_pages_is_also_caught_because_the_same_action_on_the_same_page_keeps_coming_back():
    replies = [tool_act(action="navigate", url="/" if i % 2 else "/s/stockholm") for i in range(20)]
    agent, _, _ = make(replies, limits=Limits(max_actions=50))
    r = agent.run(TASK, RESULTS)
    assert r.status == "stopped" and "stuck" in r.reason and r.actions < 8


def test_the_time_limit_uses_the_clock_it_was_given():
    t = iter([0.0, 0.0, 500.0, 500.0, 500.0])
    agent, _, _ = make([tool_act(action="navigate", url="/"), tool_act(action="done")], clock=lambda: next(t), limits=Limits(max_seconds=100))
    r = agent.run(TASK, RESULTS)
    assert r.status == "stopped" and "time limit" in r.reason


def test_the_token_limit_counts_what_the_model_used():
    agent, _, _ = make([tool_act(action="navigate", url="/"), tool_act(action="done")], limits=Limits(max_tokens=100))
    r = agent.run(TASK, RESULTS)
    assert r.status == "stopped" and "token" in r.reason and r.tokens == 120


# --- approvals -------------------------------------------------------------------------------------------------------------------------------
def to_room_script(*rest):
    return [tool_act(action="click", ref=1), *rest]  # click the first listing: now on /rooms/101, where "Reserve" is [3]


def test_a_risky_click_does_nothing_when_nobody_can_approve_it():
    agent, browser, site = make(to_room_script(tool_act(action="click", ref=3), tool_act(action="done")))
    r = agent.run(TASK, RESULTS)
    assert browser.url == "https://stays.example/rooms/101" and "did not approve" in agent.client.calls[2]["messages"][0]["content"]


def test_an_approver_that_declines_stops_the_click():
    agent, browser, _ = make(to_room_script(tool_act(action="click", ref=3), tool_act(action="done")), approver=lambda s, h: Approval("reject", h))
    agent.run(TASK, RESULTS)
    assert "/book/" not in browser.url


def test_an_approval_for_a_different_action_is_refused():
    agent, browser, _ = make(to_room_script(tool_act(action="click", ref=3), tool_act(action="done")), approver=lambda s, h: Approval("approve", "0" * 64))
    r = agent.run(TASK, RESULTS)
    assert "/book/" not in browser.url and r.denied == 1


def test_an_approved_click_runs_and_the_approver_sees_a_summary_built_from_the_page():
    seen = []
    agent, browser, _ = make(to_room_script(tool_act(action="click", ref=3), tool_act(action="done")), approver=lambda s, h: seen.append(s) or Approval("approve", h, "manager-7"))
    agent.run(TASK, RESULTS)
    assert browser.url == "https://stays.example/book/101" and seen[0]["element"]["text"] == "Reserve" and seen[0]["page_amounts"] == ["1,250 sek"]


def test_a_price_that_changes_after_approval_stops_the_payment_and_nothing_is_bought():
    site = SimSite()

    def approve(summary, h):
        if summary["element"]["text"] == "Confirm and pay":
            site.prices[101] = 1800
        return Approval("approve", h, "manager-7")
    agent, _, _ = make(to_room_script(tool_act(action="click", ref=3), tool_act(action="click", ref=3), tool_act(action="done")), site=site, approver=approve)
    r = agent.run(TASK, RESULTS)
    assert site.purchases == [] and r.denied == 1 and "since the approval" in agent.client.calls[3]["messages"][0]["content"]


def test_an_approved_payment_with_nothing_changed_goes_through_once_at_the_displayed_price():
    agent, _, site = make(to_room_script(tool_act(action="click", ref=3), tool_act(action="click", ref=3), tool_act(action="done")), approver=approve_all)
    agent.run(TASK, RESULTS)
    assert site.purchases == [(101, 1250)]


def test_one_approval_covers_one_action_not_the_next_one():
    asked = []
    agent, _, _ = make(to_room_script(tool_act(action="click", ref=3), tool_act(action="click", ref=3), tool_act(action="done")), approver=lambda s, h: asked.append(h) or Approval("approve", h))
    agent.run(TASK, RESULTS)
    assert len(asked) == 2 and asked[0] != asked[1]


def test_a_password_or_card_field_hands_the_run_to_the_user_and_nothing_is_typed():
    agent, browser, site = make([tool_act(action="type", ref=2, text="4111111111111111")], start="https://stays.example/book/101")
    r = agent.run(TASK, "https://stays.example/book/101")
    assert r.status == "needs_user" and "user" in r.reason and browser._values == {} and site.purchases == []


# --- the page changing under the agent --------------------------------------------------------------------------------------------------------
def test_an_element_that_vanished_between_looking_and_acting_is_refused_not_clicked_blindly():
    site = SimSite()

    def reply(kw):
        site.layout = "B"  # the site is redeployed while the model is thinking: the Search button's id changes
        return tool_act(action="click", ref=4)
    agent, browser, _ = make(None, start=HOME, site=site, client=RuleClient(reply), limits=Limits(max_denied=0))
    r = agent.run(TASK, HOME)
    assert browser.url == HOME and r.denied == 1 and "gone" in r.history[0]


def test_an_element_whose_label_changed_under_the_same_number_is_refused():
    site = SimSite()
    first = {"done": False}

    def reply(kw):
        if not first["done"]:
            first["done"] = True
            site.extra_cards = []
            site.prices[101] = 1  # the card text changes while the model is thinking
            return tool_act(action="click", ref=1)
        return tool_act(action="done")
    agent, browser, _ = make(None, site=site, client=RuleClient(reply))
    r = agent.run(TASK, RESULTS)
    assert browser.url == RESULTS and r.denied == 1 and "page changed" in r.history[0]


# --- the trace --------------------------------------------------------------------------------------------------------------------------------
def test_the_trace_has_action_kinds_paths_and_hashes_but_no_page_text_or_answers():
    agent, _, _ = make(to_room_script(tool_act(action="click", ref=3), tool_act(action="done", answer="SECRET-ANSWER")), approver=approve_all)
    agent.run(TASK, RESULTS)
    blob = json.dumps(agent.trace.events)
    for text in ("Sunny loft", "Södermalm", "SECRET-ANSWER", "GO TO", "AI agents reading", "1,250"):
        assert text not in blob, text
    types = agent.trace.types()
    assert "observation" in types and "approval_requested" in types and "approval_used" in types and types[-1] == "done"
    assert next(e for e in agent.trace.events if e["type"] == "approval_used")["approver"] == "manager-7"


def test_the_trace_records_how_much_hidden_text_was_dropped_and_how_many_elements_were_flagged():
    agent, _, _ = make([tool_act(action="done")])
    agent.run(TASK, RESULTS)
    o = agent.trace.of("observation")[0]
    assert o["hidden_dropped"] == 1 and o["flagged"] == 1


def test_a_blocked_url_is_logged_by_host_and_path_without_its_query_string():
    agent, _, _ = make([tool_act(action="click", ref=6), tool_act(action="done")])
    agent.run(TASK, RESULTS)
    assert agent.trace.of("navigation_blocked")[0]["url"] == "https://evil.example/landing"


def test_a_navigate_action_to_an_allowed_page_that_redirects_off_site_is_stopped_at_the_redirect():
    agent, browser, site = make([tool_act(action="navigate", url="/go/9"), tool_act(action="done")])
    r = agent.run(TASK, RESULTS)
    assert r.denied == 1 and browser.url == RESULTS and not any("evil" in u for u in site.fetched)


def test_a_blocked_redirect_is_logged_without_its_query_string():
    site = SimSite()
    site.redirects["https://stays.example/go/9"] = "https://evil.example/landing?d=SECRET-DATA"
    agent, _, _ = make([tool_act(action="click", ref=6), tool_act(action="done")], site=site)
    agent.run(TASK, RESULTS)
    assert "SECRET-DATA" not in json.dumps(agent.trace.events) and agent.trace.of("navigation_blocked")[0]["url"] == "https://evil.example/landing?…"


def test_the_model_is_forced_to_call_the_act_tool_every_turn():
    agent, _, _ = make([tool_act(action="done")])
    agent.run(TASK, RESULTS)
    assert agent.client.calls[0]["tool_choice"]["type"] == "tool"
