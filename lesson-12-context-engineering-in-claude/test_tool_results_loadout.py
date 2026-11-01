import pytest

from tool_loadout import Loadout, ToolIndex, ToolSpec, recall_at_k
from tool_results import ToolItem, ToolResultStore, stub_text


# --- offloading large tool results ---------------------------------------------------------------------------------
def test_a_small_result_goes_into_the_context_unchanged():
    item = ToolResultStore(offload_over_tokens=100).add("search", "three flights found")
    assert item.handle is None and item.text == "three flights found" and item.full_tokens == 5


def test_a_large_result_is_stored_outside_and_the_model_gets_a_preview_and_a_handle():
    store = ToolResultStore(offload_over_tokens=100, preview_chars=50)
    big = "ROW-DATA " * 500
    item = store.add("sql_query", big)
    assert item.handle.startswith("res_") and item.handle in item.text and "Preview: ROW-DATA" in item.text
    assert len(item.text) < len(big) / 10 and item.full_tokens == (len(big) + 3) // 4


def test_fetching_returns_bounded_pages_and_a_caller_cannot_ask_for_everything_at_once():
    store = ToolResultStore(offload_over_tokens=10, page_chars=100)
    text = "".join(str(i % 10) for i in range(1000))
    handle = store.add("dump", text).handle
    assert store.fetch(handle) == text[:100] and store.fetch(handle, 100) == text[100:200]
    assert len(store.fetch(handle, 0, length=999_999)) == 100  # the page cap wins over a huge request
    assert store.fetch(handle, 990) == text[990:]  # the last, short page


def test_unknown_handles_and_negative_offsets_are_rejected():
    store = ToolResultStore(offload_over_tokens=10)
    handle = store.add("dump", "x" * 500).handle
    with pytest.raises(KeyError):
        store.fetch("res_0000000000000000")
    with pytest.raises(ValueError):
        store.fetch(handle, -1)


def test_handles_are_unguessable_and_unique():
    store = ToolResultStore(offload_over_tokens=1)
    handles = {store.add("t", "x" * 100).handle for _ in range(200)}
    assert len(handles) == 200 and all(len(h) == len("res_") + 16 for h in handles)


def test_handles_are_not_predictable_two_independent_sessions_never_produce_the_same_first_handle():
    """Unique is not the same as unguessable: a simple counter is unique but trivially guessable by another session."""
    first = {ToolResultStore(offload_over_tokens=1).add("t", "x" * 100).handle for _ in range(50)}
    assert len(first) == 50  # 50 brand-new stores each issued their FIRST handle, and no two matched


def test_a_stub_says_what_the_result_was_how_big_and_where_the_full_text_is():
    assert stub_text(ToolItem("tr1", "search", "x", None, 412)) == "[tool result cleared: search, about 412 tokens]"
    assert "res_abc" in stub_text(ToolItem("tr2", "sql", "x", "res_abc", 9000))


# --- tool loadout selection --------------------------------------------------------------------------------------------
def make_tools(with_tags: bool) -> list[ToolSpec]:
    tag = lambda *t: t if with_tags else ()
    return [ToolSpec("search_flights", "Search flights between two airports.", tag("airline", "fly", "plane")),
            ToolSpec("book_flight", "Book a flight for a passenger.", tag("airline", "fly", "plane")),
            ToolSpec("search_hotels", "Search hotels in a city."), ToolSpec("book_hotel", "Book a hotel room."),
            ToolSpec("rent_car", "Rent a car at a destination.", tag("drive", "car rental")),
            ToolSpec("public_transport_info", "Look up metro, bus and tram routes in a city.", tag("transit", "getting around", "get around", "commute")),
            ToolSpec("currency_converter", "Convert an amount between currencies."), ToolSpec("weather_forecast", "Get the weather forecast for a city."),
            ToolSpec("restaurant_reservations", "Reserve a table at a restaurant."), ToolSpec("find_tours", "Find guided tours and activities."),
            ToolSpec("translate_text", "Translate text between languages."), ToolSpec("get_visa_rules", "Look up visa requirements for a country.")]


GAP = "how do I get around Paris without a car"
CASES = [("book me a flight to Rome", {"book_flight"}), ("find a hotel in Rome", {"search_hotels"}), (GAP, {"public_transport_info"}),
         ("what is the weather in Rome", {"weather_forecast"}), ("what do I need to enter Japan", {"get_visa_rules"})]


def test_the_loadout_offers_the_relevant_tools_and_records_the_rest_as_dropped():
    lo = ToolIndex(make_tools(False)).select("what is the weather in Rome", k=3)
    assert lo.names == ["weather_forecast"] and len(lo.dropped) == 11 and "weather_forecast" not in lo.dropped


def test_k_limits_the_loadout_and_always_tools_are_included_regardless_of_the_query():
    lo = ToolIndex(make_tools(False)).select("book me a flight to Rome", k=1, always=("currency_converter",))
    assert lo.names[0] == "currency_converter" and len(lo.names) == 2 and len(lo.selected) == 1


def test_plurals_are_stemmed_so_a_singular_query_finds_a_plural_description():
    assert "search_hotels" in ToolIndex(make_tools(False)).select("find a hotel in Rome", k=3).names


def test_generic_verbs_in_tool_names_do_not_pull_in_unrelated_tools():
    assert "get_visa_rules" not in ToolIndex(make_tools(False)).select("what do I get when I fly", k=5).names


def test_the_lessons_own_example_get_around_paris_picks_the_wrong_tool_without_tags():
    names = ToolIndex(make_tools(False)).select(GAP, k=3).names
    assert "rent_car" in names and "public_transport_info" not in names  # 'car' matches, 'get around' matches nothing


def test_tags_close_the_vocabulary_gap():
    assert "public_transport_info" in ToolIndex(make_tools(True)).select(GAP, k=3).names


def test_recall_is_measured_per_query_and_tags_improve_it():
    assert recall_at_k(ToolIndex(make_tools(False)), CASES, k=3) == 0.6
    assert recall_at_k(ToolIndex(make_tools(True)), CASES, k=3) == 0.8


def test_a_query_with_no_word_overlap_gives_an_empty_loadout_so_a_fallback_is_essential():
    """'what do I need to enter Japan' shares no words with 'visa requirements', so retrieval returns nothing at all."""
    idx = ToolIndex(make_tools(True))
    lo = idx.select("what do I need to enter Japan", k=5)
    assert lo.selected == [] and lo.names == []  # the agent would have NO tools: `always` must include a search_tools fallback
    assert idx.select("what do I need to enter Japan", k=5, always=("get_visa_rules",)).names == ["get_visa_rules"]


def test_the_search_tools_fallback_lets_the_model_find_a_tool_that_was_not_offered():
    idx = ToolIndex(make_tools(False))
    assert "public_transport_info" in idx.search_tools("metro bus") and idx.search_tools("zzzz") == "No matching tools."
    assert idx.search_tools("flight", k=1).count("\n") == 0  # k is respected


def test_loadout_names_list_always_tools_first():
    a, b = ToolSpec("a", "x"), ToolSpec("b", "y")
    assert Loadout(selected=[b], always=[a]).names == ["a", "b"]
