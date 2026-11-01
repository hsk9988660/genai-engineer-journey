"""A tamper-evident audit log: each entry includes the hash of the previous entry.

Editing, deleting or reordering any entry breaks every hash after it, and `verify` finds the first bad one.
Limits you should know about:
  * Truncating the END of the log is not detectable from the log alone. Anchor `head()` somewhere else
    (a separate system, a ticket, a daily email) to catch that.
  * Whoever can rewrite the whole file AND recompute all hashes can forge it. This detects casual tampering and
    accidents. Against a determined insider, write to append-only storage you do not control.
  * Entries are redacted for PII before hashing, because logs are sensitive too.
"""

import hashlib
import json
import time

from redact import redact_obj

GENESIS = "0" * 64


def _digest(entry: dict) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def verify_entries(entries: list[dict]) -> tuple[bool, int | None]:
    """Return (ok, index_of_first_bad_entry)."""
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i or e.get("prev") != prev or e.get("hash") != _digest(e):
            return False, i
        prev = e["hash"]
    return True, None


class AuditLog:
    def __init__(self, path=None, clock=time.time):
        self.path, self.clock = path, clock
        self.entries: list[dict] = []

    def append(self, event: str, data: dict) -> dict:
        entry = {"seq": len(self.entries), "ts": round(self.clock(), 3), "event": event,
                 "data": redact_obj(data), "prev": self.head()}
        entry["hash"] = _digest(entry)
        self.entries.append(entry)
        if self.path:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else GENESIS

    def verify(self) -> tuple[bool, int | None]:
        return verify_entries(self.entries)

    @staticmethod
    def verify_file(path) -> tuple[bool, int | None]:
        with open(path, encoding="utf-8") as f:
            return verify_entries([json.loads(line) for line in f if line.strip()])
