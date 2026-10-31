"""The SQL tool is tested against a real SQLite file: these are security properties, so no mocks."""

import sqlite3
import time

import pytest

from bookings import BookingService
from safe_sql import UnsafeQuery, run_readonly_query, seed_db


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "demo.db"
    seed_db(path)
    return path


def test_select_works(db):
    out = run_readonly_query(db, "SELECT region, SUM(revenue_usd) AS total FROM sales GROUP BY region ORDER BY region")
    assert out["columns"] == ["region", "total"]
    assert out["rows"][0] == ["AMER", 14800.0] and out["truncated"] is False


def test_count_star_and_cte_work(db):
    assert run_readonly_query(db, "SELECT count(*) FROM sales")["rows"] == [[8]]
    assert run_readonly_query(db, "WITH t AS (SELECT units FROM sales) SELECT max(units) FROM t")["rows"] == [[310]]


@pytest.mark.parametrize("sql", [
    "INSERT INTO sales (region) VALUES ('X')",
    "UPDATE sales SET revenue_usd = 0",
    "DELETE FROM sales",
    "DROP TABLE sales",
    "SELECT 1; DROP TABLE sales",
    "PRAGMA table_info(sales)",
    "ATTACH DATABASE ':memory:' AS x",
])
def test_non_select_statements_are_rejected(db, sql):
    with pytest.raises(UnsafeQuery):
        run_readonly_query(db, sql)
    assert run_readonly_query(db, "SELECT count(*) FROM sales")["rows"] == [[8]]  # data untouched


@pytest.mark.parametrize("sql", [
    "SELECT * FROM customers",  # private table: least privilege
    "SELECT email FROM customers",
    "SELECT name FROM sqlite_master",  # schema introspection
    "SELECT s.region FROM sales s JOIN customers c ON c.id = s.id",
])
def test_private_tables_and_schema_are_blocked(db, sql):
    with pytest.raises(UnsafeQuery):
        run_readonly_query(db, sql)


def test_a_write_hidden_inside_a_cte_is_blocked(db):
    with pytest.raises(UnsafeQuery):
        run_readonly_query(db, "WITH x AS (SELECT 1) DELETE FROM sales")


def test_file_is_opened_read_only_at_os_level(db):
    """Even with no authorizer at all, the connection itself cannot write."""
    from pathlib import Path

    conn = sqlite3.connect(Path(db).resolve().as_uri() + "?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM sales")
    conn.close()


def test_row_limit_truncates(db):
    out = run_readonly_query(db, "SELECT * FROM sales", max_rows=3)
    assert len(out["rows"]) == 3 and out["truncated"] is True


def test_runaway_query_is_aborted(db):
    start = time.monotonic()
    with pytest.raises(UnsafeQuery, match="time limit"):
        run_readonly_query(db, "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) SELECT count(*) FROM c",
                           timeout_s=0.3)
    assert time.monotonic() - start < 3


def test_syntax_error_message_is_returned_not_raised_as_sqlite_error(db):
    with pytest.raises(UnsafeQuery, match="rejected or failed"):
        run_readonly_query(db, "SELECT FROM WHERE")


# --- idempotent bookings -----------------------------------------------------------------
def test_booking_is_idempotent_and_confirmation_is_stable():
    svc = BookingService()
    first = svc.book_flight("LHR", "CDG", "Ana Ruiz")
    again = svc.book_flight("lhr", "cdg", "  ana ruiz ")  # same request, different formatting
    assert first["duplicate"] is False and again["duplicate"] is True
    assert first["confirmation"] == again["confirmation"] and svc.count() == 1
    assert BookingService().book_flight("LHR", "CDG", "Ana Ruiz")["confirmation"] == first["confirmation"]  # stable across instances
    assert svc.book_flight("LHR", "CDG", "Bo Li")["confirmation"] != first["confirmation"]


@pytest.mark.parametrize("sql", [
    "WITH t AS (SELECT * FROM customers) SELECT * FROM t",  # private table hidden behind a CTE alias
    "WITH customers AS (SELECT 1 AS email) SELECT * FROM customers",  # alias shadowing a private table name
    "SELECT * FROM (SELECT email FROM customers)",  # subquery
])
def test_cte_and_subquery_aliases_cannot_be_used_to_reach_private_tables(db, sql):
    with pytest.raises(UnsafeQuery):
        run_readonly_query(db, sql)


def test_legitimate_cte_and_subquery_aliases_still_work(db):
    assert run_readonly_query(db, "WITH t AS (SELECT region, units FROM sales) SELECT sum(units) FROM t")["rows"] == [[1000]]
    assert run_readonly_query(db, "SELECT max(u) FROM (SELECT units AS u FROM sales)")["rows"] == [[310]]
