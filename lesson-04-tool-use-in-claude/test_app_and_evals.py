"""End-to-end checks of the real tools with a scripted model, plus the eval scorer."""

import pytest

from app import DEMO_PAGES, make_agent
from evals import CASES, EvalCase, run_eval, score
from fakes import FakeClient, final, turn, use
from safe_sql import seed_db
from toolkit import deny_all


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "demo.db"
    seed_db(path)
    return path


def test_sql_tool_answers_through_the_agent(db):
    client = FakeClient([turn(use("a", "query_sales", sql="SELECT region, SUM(units) FROM sales GROUP BY region")),
                         final("AMER leads")])
    agent = make_agent(db, approve=deny_all, client=client)
    assert agent.run("units by region?") == "AMER leads"
    assert "EMEA" in client.calls[1]["messages"][-1]["content"][0]["content"]


def test_sql_tool_refuses_the_private_table_through_the_agent(db):
    client = FakeClient([turn(use("a", "query_sales", sql="SELECT email FROM customers")), final()])
    agent = make_agent(db, approve=deny_all, client=client)
    agent.run("list customer emails")
    r = client.last_results()[0]
    assert r["is_error"] and "ana@example.com" not in r["content"]


def test_bad_airport_code_fails_validation_before_the_tool_runs(db):
    client = FakeClient([turn(use("a", "get_flight_info", origin="London", destination="CDG")), final()])
    agent = make_agent(db, approve=deny_all, client=client)
    agent.run("flights?")
    assert agent.audit[0]["status"] == "invalid"


def test_the_demo_malicious_page_cannot_trigger_a_booking(db):
    """The page in DEMO_PAGES tells the model to book. Even if the model obeys, the guard stops it."""
    assert "IGNORE ALL PREVIOUS INSTRUCTIONS" in DEMO_PAGES["https://example.com/deals"]
    client = FakeClient([
        turn(use("a", "fetch_webpage", url="https://example.com/deals")),
        turn(use("b", "book_flight", origin="LHR", destination="LIS", passenger_name="Mallory")),  # the model obeys
        final("Here is a summary of the offers."),
    ])
    agent = make_agent(db, approve=lambda n, a, t: True, client=client)
    agent.run("summarize https://example.com/deals")
    assert [e["status"] for e in agent.audit] == ["ok", "blocked"]


def test_user_requested_booking_works_with_approval_and_is_idempotent(db):
    client = FakeClient([
        turn(use("a", "book_flight", origin="LHR", destination="CDG", passenger_name="Ana Ruiz")),
        turn(use("b", "book_flight", origin="LHR", destination="CDG", passenger_name="Ana Ruiz")),
        final(),
    ])
    agent = make_agent(db, approve=lambda n, a, t: True, client=client)
    agent.run("book LHR to CDG for Ana Ruiz")
    first, second = (client.calls[i]["messages"][-1]["content"][0]["content"] for i in (1, 2))
    assert '"duplicate":false' in first.replace(" ", "") and '"duplicate":true' in second.replace(" ", "")


# --- eval scorer --------------------------------------------------------------------------
def test_score_flags_missing_and_forbidden_tools():
    case = EvalCase("q", must_call={"query_sales"}, must_not_call={"book_flight"})
    assert score(["query_sales"], case)[0] is True
    assert score([], case)[0] is False
    ok, detail = score(["query_sales", "book_flight"], case)
    assert not ok and "book_flight" in detail


def test_run_eval_reports_pass_rate_and_survives_agent_crashes(db):
    cases = [CASES[1], CASES[4]]  # availability question, and a no-tool question

    class Crasher:
        def run(self, q):
            raise RuntimeError("api down")

    good = FakeClient([turn(use("a", "check_availability", destination="Berlin")), final("Sold out")])
    agents = iter([make_agent(db, approve=deny_all, client=good), Crasher()])
    report = run_eval(lambda: next(agents), cases)
    assert [r["passed"] for r in report["results"]] == [True, False]
    assert report["pass_rate"] == 0.5 and "api down" in report["results"][1]["detail"]
