"""Durable run state and a ledger that makes side effects happen once.

Two separate things, because they fail differently:
  CheckpointStore  one JSON file per run, replaced atomically after every completed step. A crash leaves the previous complete file, never a half-written one.
  EffectLedger     a SQLite table of effect results keyed by `run:node:visit`. A step that books a flight does it through `ctx.once`, so when a crash happens AFTER
                   the booking but BEFORE the checkpoint, the resumed step gets the recorded result instead of booking again.

The remaining window is a crash INSIDE the effect, after the outside world acted but before the ledger row is written. No local file can close that: pass the
key to the remote system (`fn(key)`) so the remote side can de-duplicate. This lesson does not model remote systems.
"""

import json
import os
import re
import sqlite3
import tempfile
import threading
from pathlib import Path

RUN_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def check_run_id(run_id: str) -> str:
    if not isinstance(run_id, str) or not RUN_ID.match(run_id):
        raise ValueError("run_id must be 1 to 64 letters, digits, '-' or '_'")
    return run_id


class CheckpointStore:
    def __init__(self, directory):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.dir / f"{check_run_id(run_id)}.json"  # the id is validated, so it cannot contain '/' or '..'

    def load(self, run_id: str) -> dict | None:
        p = self._path(run_id)
        return json.loads(p.read_text(encoding="utf8")) if p.exists() else None

    def save(self, state: dict) -> None:
        p = self._path(state["run_id"])
        fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf8") as f:
                json.dump(state, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)  # atomic on the same filesystem
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise


class EffectLedger:
    def __init__(self, path):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS effects (key TEXT PRIMARY KEY, result TEXT NOT NULL)")
        self._db.commit()
        self._lock = threading.Lock()

    def once(self, key: str, fn):
        """Run `fn(key)` the first time this key is seen and record its JSON result. Later calls return the record without calling `fn`."""
        with self._lock:
            row = self._db.execute("SELECT result FROM effects WHERE key = ?", (key,)).fetchone()
            if row is not None:
                return json.loads(row[0])
            result = fn(key)
            self._db.execute("INSERT INTO effects(key, result) VALUES (?, ?)", (key, json.dumps(result)))
            self._db.commit()
            return result

    def keys(self) -> list[str]:
        return [r[0] for r in self._db.execute("SELECT key FROM effects ORDER BY rowid")]

    def close(self) -> None:
        self._db.close()
