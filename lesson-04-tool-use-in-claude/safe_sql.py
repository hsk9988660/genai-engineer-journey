"""A read-only SQL tool an LLM can use without being able to hurt you.

A "read-only role" alone is not enough, so there are several independent layers (defense in depth):
  1. Statement must start with SELECT/WITH (cheap, clear error messages for the model).
  2. The file is opened read-only at the OS level (mode=ro): writes are impossible.
  3. A SQLite authorizer allows only SELECT-type actions and only ALLOWED tables, which also blocks
     PRAGMA, ATTACH, sqlite_master and private tables such as `customers` (least privilege).
  4. A progress handler aborts queries that run longer than `timeout_s`.
  5. At most `max_rows` rows come back, so one query can't flood the model's context.
"""

import sqlite3
import time
from pathlib import Path

ALLOWED_TABLES = frozenset({"sales"})


class UnsafeQuery(ValueError):
    """The query was rejected or failed. The message is safe to show to the model."""


def seed_db(path) -> None:
    """Create a small demo database: a public `sales` table and a private `customers` table."""
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE sales (id INTEGER PRIMARY KEY, region TEXT, product TEXT, units INTEGER,
                            revenue_usd REAL, sold_on TEXT);
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT);
        INSERT INTO sales (region, product, units, revenue_usd, sold_on) VALUES
          ('EMEA', 'Widget', 120, 2400.0, '2025-01-15'), ('EMEA', 'Gadget', 80, 3200.0, '2025-02-10'),
          ('APAC', 'Widget', 200, 4000.0, '2025-01-20'), ('APAC', 'Gadget', 50, 2000.0, '2025-03-05'),
          ('AMER', 'Widget', 310, 6200.0, '2025-02-25'), ('AMER', 'Gadget', 140, 5600.0, '2025-03-18'),
          ('AMER', 'Gizmo', 60, 3000.0, '2025-03-22'), ('EMEA', 'Gizmo', 40, 2000.0, '2025-03-30');
        INSERT INTO customers (name, email) VALUES ('Ana Ruiz', 'ana@example.com'), ('Bo Li', 'bo@example.com');
        """
    )
    conn.commit()
    conn.close()


def _authorizer(allowed_tables, real_tables):
    def check(action, arg1, arg2, dbname, source):
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:  # arg1 = table, arg2 = column
            name = (arg1 or "").lower()
            if name in allowed_tables:
                return sqlite3.SQLITE_OK
            # A name that is not a real table is a CTE or subquery alias. Reading it exposes nothing by
            # itself: the base tables behind it are authorized separately. Real tables that are not
            # allowed, and sqlite_* internals (schema introspection), stay denied.
            if name not in real_tables and not name.startswith("sqlite_"):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        return sqlite3.SQLITE_DENY  # INSERT, UPDATE, DELETE, DROP, ATTACH, PRAGMA, ...

    return check


def run_readonly_query(db_path, sql: str, *, max_rows: int = 50, timeout_s: float = 2.0,
                       allowed_tables=ALLOWED_TABLES) -> dict:
    sql = sql.strip().rstrip(";").strip()
    if not sql.lower().startswith(("select", "with")):
        raise UnsafeQuery("Only a single SELECT query is allowed.")
    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    try:
        deadline = time.monotonic() + timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        real_tables = {r[0].lower() for r in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
        conn.set_authorizer(_authorizer(allowed_tables, real_tables))  # set AFTER reading the schema
        try:
            cursor = conn.execute(sql)
            rows = cursor.fetchmany(max_rows + 1)
            columns = [d[0] for d in cursor.description or []]
        except sqlite3.DatabaseError as exc:  # syntax errors, denied actions, multi-statement, timeouts
            msg = f"Query exceeded the {timeout_s}s time limit." if "interrupted" in str(exc) else f"Query rejected or failed: {exc}"
            raise UnsafeQuery(msg) from exc
        return {"columns": columns, "rows": [list(r) for r in rows[:max_rows]], "truncated": len(rows) > max_rows}
    finally:
        conn.close()
