"""Conversation threads kept OUTSIDE the process, so any instance can serve any request.

The lesson says to externalise state and then keeps `ORDERS`, `response_cache`, `CUSTOMER_MEMORY` and the rest in module-level dicts. This is the persistent
version: SQLite on a file (or `:memory:` for tests). Every query is scoped to the user who owns the thread, so another user's thread id looks exactly like a
missing one, and thread ids are random rather than counters.
"""

import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass


class ThreadError(Exception):
    pass


class NotFound(ThreadError):
    pass


class ThreadFull(ThreadError):
    pass


class InvalidMessage(ThreadError):
    pass


@dataclass(frozen=True)
class Scope:
    user_id: str

    def __post_init__(self):
        if not isinstance(self.user_id, str) or not self.user_id or len(self.user_id) > 200:
            raise ValueError("user_id must be a non-empty string")


class ThreadStore:
    def __init__(self, path=":memory:", max_messages: int = 200, max_chars: int = 8000, clock=time.time):
        self.max_messages, self.max_chars, self.clock = max_messages, max_chars, clock
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("PRAGMA secure_delete=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, user_id TEXT NOT NULL, created REAL NOT NULL);
            CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, thread_id TEXT NOT NULL, role TEXT NOT NULL, content TEXT NOT NULL, ts REAL NOT NULL);
            CREATE INDEX IF NOT EXISTS idx_threads_user ON threads(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_thread ON messages(thread_id);""")
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    def ping(self) -> bool:
        with self._lock:
            return self._db.execute("SELECT 1").fetchone() == (1,)

    def create(self, scope: Scope) -> str:
        tid = secrets.token_urlsafe(12)
        with self._lock, self._db:
            self._db.execute("INSERT INTO threads(id, user_id, created) VALUES (?,?,?)", (tid, scope.user_id, self.clock()))
        return tid

    def _owns(self, scope: Scope, thread_id: str) -> bool:
        return self._db.execute("SELECT 1 FROM threads WHERE id = ? AND user_id = ?", (thread_id, scope.user_id)).fetchone() is not None

    def append(self, scope: Scope, thread_id: str, role: str, content: str) -> None:
        if role not in ("user", "assistant"):
            raise InvalidMessage("role must be 'user' or 'assistant'")
        if not isinstance(content, str) or not content or len(content) > self.max_chars:
            raise InvalidMessage(f"content must be 1 to {self.max_chars} characters")
        with self._lock, self._db:
            if not self._owns(scope, thread_id):
                raise NotFound("no such thread")
            if self._db.execute("SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)).fetchone()[0] >= self.max_messages:
                raise ThreadFull(f"a thread holds at most {self.max_messages} messages")
            self._db.execute("INSERT INTO messages(thread_id, role, content, ts) VALUES (?,?,?,?)", (thread_id, role, content, self.clock()))

    def messages(self, scope: Scope, thread_id: str, last: int = 20) -> list[dict]:
        with self._lock:
            if not self._owns(scope, thread_id):
                raise NotFound("no such thread")
            rows = self._db.execute("SELECT role, content FROM (SELECT id, role, content FROM messages WHERE thread_id = ? ORDER BY id DESC LIMIT ?) ORDER BY id",
                                    (thread_id, last)).fetchall()
        return [{"role": r, "content": c} for r, c in rows]

    def count(self, scope: Scope, thread_id: str) -> int:
        with self._lock:
            if not self._owns(scope, thread_id):
                raise NotFound("no such thread")
            return self._db.execute("SELECT COUNT(*) FROM messages WHERE thread_id = ?", (thread_id,)).fetchone()[0]

    def delete(self, scope: Scope, thread_id: str) -> bool:
        with self._lock, self._db:
            if not self._owns(scope, thread_id):
                return False
            self._db.execute("DELETE FROM messages WHERE thread_id = ?", (thread_id,))
            self._db.execute("DELETE FROM threads WHERE id = ?", (thread_id,))
            return True

    def delete_user(self, scope: Scope) -> int:
        with self._lock, self._db:
            ids = [r[0] for r in self._db.execute("SELECT id FROM threads WHERE user_id = ?", (scope.user_id,))]
            for tid in ids:
                self._db.execute("DELETE FROM messages WHERE thread_id = ?", (tid,))
            self._db.execute("DELETE FROM threads WHERE user_id = ?", (scope.user_id,))
            return len(ids)
