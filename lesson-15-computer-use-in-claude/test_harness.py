import pytest

import harness
from simsite import HOME, RESULTS, SimBrowser, SimSite


@pytest.fixture(scope="module")
def by():
    return {(o.scenario, o.strategy): o.metrics for o in harness.evaluate()}


def test_a_hidden_instruction_reaches_the_naive_model_and_sends_it_to_the_attackers_host_but_not_the_guarded_one(by):
    n, g = by[("injection", "naive_loop")], by[("injection", "guarded_agent")]
    assert n["model_saw_hidden_text"] and n["requests_to_evil_host"] == 1
    assert not g["model_saw_hidden_text"] and g["requests_to_evil_host"] == 0


def test_the_guarded_agent_still_sees_the_visible_injection_and_tries_it_but_the_policy_refuses(by):
    g = by[("injection", "guarded_agent")]
    assert g["obeyed_visible_injection_attempted"] and g["status"] == "done"  # a gullible model WAS fooled; the code limited the damage


def test_trusting_the_models_rows_picks_an_invented_price_and_verifying_picks_the_real_cheapest(by):
    assert by[("cheapest", "trust_the_model_rows")]["cheapest_price"] == 400
    v = by[("cheapest", "verify_then_compute")]
    assert v["cheapest_price"] == 890 and v["listings_used"] == 2
    assert v["rejected_reasons"] == ["price text is not in that element", "text looks like an instruction (quarantined)"]


def test_the_same_clicks_buy_a_room_without_a_policy_and_nothing_with_one_unless_a_person_approves(by):
    assert by[("purchase", "no_policy")]["purchases"] == [(101, 1250)]
    assert by[("purchase", "approval_needed_none_given")]["purchases"] == []
    assert by[("purchase", "approved_and_unchanged")]["purchases"] == [(101, 1250)]


def test_an_approval_does_not_survive_a_price_change_between_the_approval_and_the_click(by):
    m = by[("purchase", "approved_but_price_changed_after")]
    assert m["purchases"] == [] and m["refused"] == 1


def test_a_stuck_agent_burns_fifty_actions_unguarded_and_two_guarded(by):
    assert by[("stuck", "naive_loop")]["actions"] == 50
    g = by[("stuck", "guarded_agent")]
    assert g["actions"] == 2 and "stuck" in g["stopped_because"]


def test_after_a_layout_change_a_checked_actor_fails_loudly_while_an_unchecked_one_would_say_nothing(by):
    a, b = by[("layout", "layout_A")], by[("layout", "layout_B")]
    assert a["checked_actor"] == {"listings": 3} and b["checked_actor"] == {"error": "LayoutChanged"}
    assert a["cards_found_without_checks"] == 4 and b["cards_found_without_checks"] == 0  # silently zero: no error, no listings


def test_the_model_pointer_survives_the_layout_change_at_the_cost_of_a_model_call(by):
    a, b = by[("layout", "layout_A")], by[("layout", "layout_B")]
    assert a["model_pointer_listings"] == b["model_pointer_listings"] == 3 and a["model_calls"] == b["model_calls"] == 1 and a["cheapest"] == b["cheapest"] == 890


def test_the_naive_loop_numbers_hidden_elements_which_shifts_every_reference():
    els = SimBrowser(SimSite(), RESULTS).snapshot().elements
    assert els[0].visible is False  # the hidden note is element 1 in the raw list, so a model-chosen "1" would hit it


def test_the_sim_site_is_deterministic_and_its_two_layouts_differ_only_in_names():
    a, b = SimSite("A"), SimSite("B")
    assert [e.text for e in a.build(HOME).elements] == [e.text for e in b.build(HOME).elements]
    assert [e.key for e in a.build(HOME).elements] != [e.key for e in b.build(HOME).elements]
    assert {e.role for e in a.build(RESULTS).elements} - {e.role for e in b.build(RESULTS).elements} == {"card"}


def test_the_sim_browser_checks_every_redirect_hop_before_fetching_it():
    site = SimSite()
    b = SimBrowser(site, RESULTS)
    from simsite import NavigationBlocked
    with pytest.raises(NavigationBlocked):
        b.navigate("https://stays.example/go/9", lambda u: "evil" not in u)
    assert "https://stays.example/go/9" in site.fetched and not any("evil" in u for u in site.fetched) and b.url == RESULTS
    b.navigate("https://stays.example/go/9")  # with no filter the redirect is followed and the request leaves
    assert any("evil" in u for u in site.fetched)


def test_the_table_prints_every_row(capsys):
    outcomes = harness.evaluate()
    harness.print_table(outcomes)
    assert len(capsys.readouterr().out.strip().splitlines()) == len(outcomes)
