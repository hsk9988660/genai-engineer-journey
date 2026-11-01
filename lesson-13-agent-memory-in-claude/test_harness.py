import pytest

import harness


@pytest.fixture(scope="module")
def by():
    out = {}
    for o in harness.evaluate():
        out.setdefault(o.scenario, {})[(o.strategy, o.metrics.get("sessions"))] = o.metrics
    return out


def test_the_notebook_approach_keeps_the_stale_budget_and_diet_next_to_the_new_ones(by):
    m = by["drift"][("naive_list", None)]
    assert m["current_budget_present"] and m["stale_budget_present"] and m["stale_diet_present"]  # a clash the model has to untangle


def test_the_managed_store_carries_only_the_current_values_and_keeps_the_allergy(by):
    m = by["drift"][("managed_store", None)]
    assert m["current_budget_present"] and m["allergy_kept"] and not m["stale_budget_present"] and not m["stale_diet_present"]


def test_a_generic_query_still_surfaces_the_pinned_allergy_but_misses_the_unpinned_budget(by):
    m = by["drift"][("managed_generic_query", None)]
    assert m["allergy_kept"] and not m["current_budget_present"]  # the honest limit of relevance retrieval


def test_the_naive_context_grows_with_sessions_while_the_managed_block_does_not(by):
    naive = [by["growth"][("naive_list", n)]["tokens"] for n in (10, 50, 200)]
    managed = [by["growth"][("managed_store", n)]["tokens"] for n in (10, 50, 200)]
    assert naive[0] < naive[1] < naive[2] and naive[2] > 20 * managed[2]
    assert max(managed) - min(managed) == 0


def test_the_safety_fact_survives_200_sessions_under_both_strategies(by):
    assert by["growth"][("managed_store", 200)]["allergy_kept"] and by["growth"][("naive_list", 200)]["allergy_kept"]


def test_a_model_supplied_user_id_leaks_another_users_data_in_the_notebook_design_but_not_here(by):
    assert by["cross_user"][("naive_list", None)]["leaked"] and not by["cross_user"][("managed_store", None)]["leaked"]
    assert by["cross_user"][("managed_store", None)]["victim_data_intact"]


def test_a_poisoned_memory_reaches_the_context_in_the_notebook_design_and_is_quarantined_here(by):
    assert by["poisoning"][("naive_list", None)]["in_context"]
    m = by["poisoning"][("managed_store", None)]
    assert not m["in_context"] and m["quarantined"] == 1


def test_the_table_prints_every_row(capsys):
    outcomes = harness.evaluate()
    harness.print_table(outcomes)
    out = capsys.readouterr().out
    assert len(out.strip().splitlines()) == len(outcomes) and "naive_list" in out and "managed_store" in out
