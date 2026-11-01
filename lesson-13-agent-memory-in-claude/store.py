"""A persistent agent memory store on SQLite (standard library only).

What the lesson's notebook did, and what this does instead:
  a Python dict in RAM                     a real database file: memories survive closing and reopening the process
  the MODEL passes a user_id to the tools  identity is a `MemoryScope` made by YOUR code from your auth system. No method takes a
                                           user id as a string, and the model-facing tools have no such parameter.
  append-only free text                    typed keys with a cardinality: single-valued keys are SUPERSEDED (budget), set-valued keys
                                           accumulate and can be retracted (allergies, diet)
  anything the model says is saved         writes need a trusted source, a verbatim quote of what the user said, a key from a registry,
                                           and (for health data) the user's consent; everything else is QUARANTINED, never active
  no way to forget                         soft forget, hard delete, per-user delete and export, with SECURE deletion

Deletion is easy to get wrong. With SQLite's default settings the text of a deleted row STAYS IN THE FILE (I checked: the bytes were
still there after DELETE). This store turns on `secure_delete` and runs VACUUM after hard deletes, and a test scans the raw file.

Every query is parameterized. The audit log records actions, ids and key names but NEVER memory content, and refers to users by a
salted hash, so deleting a user's memories does not leave their text behind in the audit trail.
"""

import hashlib
import hmac
import json
import re
import sqlite3
import threading
import time
from dataclasses import dataclass, field

TRUSTED_SOURCES = frozenset({"user", "system"})
MAX_VALUE_CHARS = 300
MAX_PER_USER = 500


class MemoryStoreError(Exception):
    pass


class NotFoundError(MemoryStoreError):
    pass


class UnknownKey(MemoryStoreError):
    pass


class InvalidMemory(MemoryStoreError):
    pass


class MemoryLimit(MemoryStoreError):
    pass


@dataclass(frozen=True)
class KeySpec:
    cardinality: str  # "single": a new value supersedes the old one. "set": values accumulate.
    sensitive: bool = False  # health, finances and similar: needs the user's consent before it becomes active


KEYS = {
    "budget": KeySpec("single"), "dates": KeySpec("single"), "destination": KeySpec("single"), "class": KeySpec("single"),
    "diet": KeySpec("set"), "interest": KeySpec("set"), "dislike": KeySpec("set"),
    "allergy": KeySpec("set", sensitive=True), "mobility": KeySpec("single", sensitive=True),
}

_INSTRUCTION_PATTERNS = [re.compile(p, re.I) for p in (
    r"ignore (all |any |the )?(previous|prior|above)", r"\b(always|never)\b.{0,40}\b(send|email|reveal|share|forward|delete)\b",
    r"system prompt", r"\bdo not (tell|mention)\b", r"https?://", r"you (must|should) now")]


def looks_like_instruction(text: str) -> bool:
    """A HEURISTIC, easy to evade. The real defenses are source trust, grounding, the key registry, and rendering memory as data."""
    return any(p.search(text) for p in _INSTRUCTION_PATTERNS)


def norm(value: str) -> str:
    return " ".join(value.lower().split())


@dataclass(frozen=True)
class MemoryScope:
    """WHO the memory belongs to. Build it from your auth system. The model must never choose or change it."""
    user_id: str
    session: str = ""

    def __post_init__(self):
        if not isinstance(self.user_id, str) or not self.user_id or len(self.user_id) > 200:
            raise ValueError("user_id must be a non-empty string")


@dataclass
class Memory:
    id: int
    user_id: str
    kind: str
    key: str
    value: str
    cardinality: str
    status: str  # active | superseded | quarantined | retracted
    source: str
    evidence: str | None
    turn: int | None
    session: str
    sensitive: bool
    importance: float
    created_at: float
    updated_at: float
    expires_at: float | None
    superseded_by: int | None
    reason: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT NOT NULL, kind TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
  cardinality TEXT NOT NULL, status TEXT NOT NULL, source TEXT NOT NULL, evidence TEXT, turn INTEGER, session TEXT NOT NULL DEFAULT '',
  sensitive INTEGER NOT NULL DEFAULT 0, importance REAL NOT NULL DEFAULT 0.5, created_at REAL NOT NULL, updated_at REAL NOT NULL,
  expires_at REAL, superseded_by INTEGER, reason TEXT NOT NULL DEFAULT '');
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, status);
CREATE TABLE IF NOT EXISTS audit (
  seq INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL, user_hash TEXT NOT NULL, action TEXT NOT NULL, memory_id INTEGER,
  detail TEXT NOT NULL, prev TEXT NOT NULL, hash TEXT NOT NULL);
"""
_COLUMNS = ("id, user_id, kind, key, value, cardinality, status, source, evidence, turn, session, sensitive, importance, created_at, "
            "updated_at, expires_at, superseded_by, reason")


def _memory(row) -> Memory:
    r = list(row)
    r[11] = bool(r[11])
    return Memory(*r)


class MemoryStore:
    def __init__(self, path, keys: dict | None = None, clock=time.time, salt: bytes = b"change-me"):
        self.path, self.keys, self.clock, self.salt = str(path), keys or KEYS, clock, salt
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.execute("PRAGMA secure_delete=ON")  # overwrite deleted content with zeros instead of leaving it in the file
        self._db.executescript(_SCHEMA)
        self._db.commit()

    def close(self) -> None:
        self._db.close()

    # --- audit (content-free, hash-chained) ---------------------------------------------------------------------------------
    def _user_hash(self, user_id: str) -> str:
        return hmac.new(self.salt, user_id.encode(), hashlib.sha256).hexdigest()[:16]

    def _audit(self, user_id: str, action: str, memory_id: int | None, detail: str) -> None:
        last = self._db.execute("SELECT hash FROM audit ORDER BY seq DESC LIMIT 1").fetchone()
        prev = last[0] if last else "0" * 64
        ts, uh = round(self.clock(), 3), self._user_hash(user_id)
        digest = hashlib.sha256(json.dumps([prev, ts, uh, action, memory_id, detail]).encode()).hexdigest()
        self._db.execute("INSERT INTO audit(ts, user_hash, action, memory_id, detail, prev, hash) VALUES (?,?,?,?,?,?,?)",
                         (ts, uh, action, memory_id, detail, prev, digest))

    def verify_audit(self) -> bool:
        prev = "0" * 64
        for ts, uh, action, mid, detail, p, h in self._db.execute("SELECT ts,user_hash,action,memory_id,detail,prev,hash FROM audit ORDER BY seq"):
            if p != prev or h != hashlib.sha256(json.dumps([p, ts, uh, action, mid, detail]).encode()).hexdigest():
                return False
            prev = h
        return True

    def audit_rows(self) -> list[tuple]:
        return self._db.execute("SELECT action, memory_id, detail FROM audit ORDER BY seq").fetchall()

    # --- reading --------------------------------------------------------------------------------------------------------------
    def _select(self, where: str, params: tuple) -> list[Memory]:
        return [_memory(r) for r in self._db.execute(f"SELECT {_COLUMNS} FROM memories WHERE {where} ORDER BY id", params)]

    def get(self, scope: MemoryScope, memory_id: int) -> Memory | None:
        with self._lock:
            rows = self._select("user_id = ? AND id = ?", (scope.user_id, memory_id))
            return rows[0] if rows else None  # another user's id looks exactly like a missing one

    def active(self, scope: MemoryScope) -> list[Memory]:
        """Active, unexpired memories for THIS user. Quarantined, superseded and retracted ones are never returned."""
        with self._lock:
            return self._select("user_id = ? AND status = 'active' AND (expires_at IS NULL OR expires_at > ?)", (scope.user_id, self.clock()))

    def quarantined(self, scope: MemoryScope) -> list[Memory]:
        with self._lock:
            return self._select("user_id = ? AND status = 'quarantined'", (scope.user_id,))

    def history(self, scope: MemoryScope, key: str) -> list[Memory]:
        with self._lock:
            return self._select("user_id = ? AND key = ?", (scope.user_id, key))

    # --- writing --------------------------------------------------------------------------------------------------------------
    def remember(self, scope: MemoryScope, key: str, value: str, source: str, evidence: str | None = None, kind: str = "semantic",
                 consent: bool = False, importance: float = 0.5, ttl_s: float | None = None, turn: int | None = None) -> Memory:
        spec = self.keys.get(key)
        if spec is None:
            raise UnknownKey(f"{key!r} is not a registered memory key")
        value = " ".join(str(value).split())
        if not value or len(value) > MAX_VALUE_CHARS:
            raise InvalidMemory(f"value must be 1 to {MAX_VALUE_CHARS} characters")
        if source == "user" and (not evidence or len(evidence.strip()) < 3):
            raise InvalidMemory("a fact attributed to the user needs evidence: a quote of what they said")
        status, reason = "active", ""
        if source not in TRUSTED_SOURCES:
            status, reason = "quarantined", "untrusted source: needs confirmation"
        elif looks_like_instruction(value):
            status, reason = "quarantined", "looks like an instruction, not a fact"
        elif spec.sensitive and not consent:
            status, reason = "quarantined", "sensitive: needs the user's consent"
        with self._lock, self._db:
            now = self.clock()
            existing = self._select("user_id = ? AND key = ? AND status = 'active'", (scope.user_id, key))
            if status == "active":
                same = [m for m in existing if norm(m.value) == norm(value)]
                if same:  # nothing new: just refresh it
                    self._db.execute("UPDATE memories SET updated_at = ?, expires_at = COALESCE(?, expires_at) WHERE id = ?",
                                     (now, now + ttl_s if ttl_s else None, same[0].id))
                    return self._select("id = ?", (same[0].id,))[0]
            total = self._db.execute("SELECT COUNT(*) FROM memories WHERE user_id = ? AND status IN ('active','quarantined')", (scope.user_id,)).fetchone()[0]
            if total >= MAX_PER_USER:  # checked only when a row would be ADDED: refreshing a known fact never fails
                raise MemoryLimit(f"at most {MAX_PER_USER} memories per user")
            cur = self._db.execute(
                "INSERT INTO memories(user_id, kind, key, value, cardinality, status, source, evidence, turn, session, sensitive, importance,"
                " created_at, updated_at, expires_at, reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (scope.user_id, kind, key, value, spec.cardinality, status, source, evidence, turn, scope.session, int(spec.sensitive),
                 importance, now, now, now + ttl_s if ttl_s else None, reason))
            new_id = cur.lastrowid
            if status == "active" and spec.cardinality == "single":
                for old in existing:
                    self._db.execute("UPDATE memories SET status = 'superseded', superseded_by = ?, updated_at = ? WHERE id = ?", (new_id, now, old.id))
            self._audit(scope.user_id, f"remember:{status}", new_id, key)
            return self._select("id = ?", (new_id,))[0]

    def retract(self, scope: MemoryScope, key: str, value: str | None = None, source: str = "user") -> int:
        """The user says a fact is no longer true ('I am not vegetarian anymore'). Only a trusted source may do this."""
        if source not in TRUSTED_SOURCES:
            raise InvalidMemory("only a trusted source can retract a fact")
        with self._lock, self._db:
            n = 0
            for m in self._select("user_id = ? AND key = ? AND status = 'active'", (scope.user_id, key)):
                if value is None or norm(m.value) == norm(value):
                    self._db.execute("UPDATE memories SET status = 'superseded', reason = 'retracted by the user', updated_at = ? WHERE id = ?",
                                     (self.clock(), m.id))
                    self._audit(scope.user_id, "retract", m.id, key)
                    n += 1
            return n

    def confirm(self, scope: MemoryScope, memory_id: int, consent: bool = False) -> Memory:
        """A person approves a quarantined memory. Sensitive ones need consent. Single-valued keys supersede the current value."""
        with self._lock, self._db:
            m = self.get(scope, memory_id)
            if m is None or m.status != "quarantined":
                raise NotFoundError("no quarantined memory with that id")
            if m.sensitive and not consent:
                raise InvalidMemory("this memory is sensitive and needs the user's consent")
            if looks_like_instruction(m.value):
                raise InvalidMemory("this memory looks like an instruction and cannot be confirmed")
            now = self.clock()
            if m.cardinality == "single":
                for old in self._select("user_id = ? AND key = ? AND status = 'active'", (scope.user_id, m.key)):
                    self._db.execute("UPDATE memories SET status = 'superseded', superseded_by = ?, updated_at = ? WHERE id = ?", (m.id, now, old.id))
            self._db.execute("UPDATE memories SET status = 'active', reason = '', updated_at = ? WHERE id = ?", (now, m.id))
            self._audit(scope.user_id, "confirm", m.id, m.key)
            return self.get(scope, memory_id)

    # --- forgetting ------------------------------------------------------------------------------------------------------------
    def forget(self, scope: MemoryScope, memory_id: int, hard: bool = False) -> bool:
        """Soft forget hides a memory but can be undone. A hard forget deletes the row AND overwrites its bytes in the file."""
        with self._lock:
            m = self.get(scope, memory_id)
            if m is None:
                return False
            with self._db:
                if hard:
                    self._db.execute("UPDATE memories SET superseded_by = NULL WHERE superseded_by = ?", (memory_id,))
                    self._db.execute("DELETE FROM memories WHERE id = ? AND user_id = ?", (memory_id, scope.user_id))
                else:
                    self._db.execute("UPDATE memories SET status = 'retracted', updated_at = ? WHERE id = ? AND user_id = ?",
                                     (self.clock(), memory_id, scope.user_id))
                self._audit(scope.user_id, "forget:hard" if hard else "forget:soft", memory_id, m.key)
            if hard:
                self._db.execute("VACUUM")
            return True

    def restore(self, scope: MemoryScope, memory_id: int) -> bool:
        with self._lock, self._db:
            m = self.get(scope, memory_id)
            if m is None or m.status != "retracted":
                return False
            self._db.execute("UPDATE memories SET status = 'active', updated_at = ? WHERE id = ?", (self.clock(), memory_id))
            self._audit(scope.user_id, "restore", memory_id, m.key)
            return True

    def delete_user(self, scope: MemoryScope) -> int:
        """Erase EVERYTHING stored for this user, including superseded and quarantined rows, and scrub the file."""
        with self._lock:
            with self._db:
                n = self._db.execute("DELETE FROM memories WHERE user_id = ?", (scope.user_id,)).rowcount
                self._audit(scope.user_id, "delete_user", None, f"{n} rows")
            self._db.execute("VACUUM")
            return n

    def export_user(self, scope: MemoryScope) -> list[dict]:
        """Everything held about this user, all statuses: data portability and a way to SEE what the agent remembers."""
        with self._lock:
            rows = self._select("user_id = ?", (scope.user_id,))
        return [m.__dict__.copy() for m in rows]

    def purge_expired(self) -> int:
        with self._lock:
            with self._db:
                n = self._db.execute("DELETE FROM memories WHERE expires_at IS NOT NULL AND expires_at <= ?", (self.clock(),)).rowcount
                if n:
                    self._audit("system", "purge_expired", None, f"{n} rows")
            if n:
                self._db.execute("VACUUM")
            return n

