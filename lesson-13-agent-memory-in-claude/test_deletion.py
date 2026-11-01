"""Deletion is verified against the RAW FILE, not just the API. With SQLite's defaults, deleted text stays in the file (I measured it)."""

import glob
import os
import sqlite3

from store import MemoryScope, MemoryStore

BOB, ALICE = MemoryScope("bob"), MemoryScope("alice")
SECRET = "ZEBRA-MARKER-9f3a1c"


def raw_bytes(path) -> bytes:
    """The main file plus any journal/WAL siblings."""
    return b"".join(open(p, "rb").read() for p in [str(path), *glob.glob(str(path) + "-*")] if os.path.exists(p))


def fill(path, secret=SECRET):
    s = MemoryStore(path)
    m = s.remember(BOB, "interest", f"likes {secret}", "user", evidence="I like it")
    s.remember(ALICE, "interest", "likes harbours", "user", evidence="I like it")
    return s, m


def test_the_marker_really_is_in_the_file_before_deletion_so_the_checks_below_are_meaningful(tmp_path):
    s, _ = fill(tmp_path / "m.db")
    s.close()
    assert SECRET.encode() in raw_bytes(tmp_path / "m.db")


def test_hard_forget_removes_the_text_from_the_database_file(tmp_path):
    s, m = fill(tmp_path / "m.db")
    assert s.forget(BOB, m.id, hard=True) is True
    assert SECRET.encode() not in raw_bytes(tmp_path / "m.db")
    s.close()
    assert SECRET.encode() not in raw_bytes(tmp_path / "m.db")


def test_soft_forget_keeps_the_text_in_the_file_on_purpose_because_it_can_be_restored(tmp_path):
    s, m = fill(tmp_path / "m.db")
    s.forget(BOB, m.id)
    s.close()
    assert SECRET.encode() in raw_bytes(tmp_path / "m.db")


def test_delete_user_erases_every_row_including_superseded_and_quarantined_ones_from_the_file(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    s.remember(BOB, "budget", "OLDVALUE-77", "user", evidence="my budget")
    s.remember(BOB, "budget", "NEWVALUE-88", "user", evidence="my budget")  # supersedes: the old row is kept as history
    s.remember(BOB, "interest", "QUARANTINED-99", "tool")
    s.remember(ALICE, "interest", "ALICE-KEEPS-THIS", "user", evidence="I like it")
    assert s.delete_user(BOB) == 3
    s.close()
    data = raw_bytes(tmp_path / "m.db")
    for gone in (b"OLDVALUE-77", b"NEWVALUE-88", b"QUARANTINED-99"):
        assert gone not in data
    assert b"ALICE-KEEPS-THIS" in data  # and another user's data is untouched
    s2 = MemoryStore(tmp_path / "m.db")
    assert s2.export_user(BOB) == [] and len(s2.active(ALICE)) == 1
    s2.close()


def test_deleting_a_user_leaves_no_trace_of_their_content_or_id_in_the_audit_log(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    s.remember(MemoryScope("user-4242"), "interest", SECRET, "user", evidence="I like it")
    s.delete_user(MemoryScope("user-4242"))
    s.close()
    data = raw_bytes(tmp_path / "m.db")
    assert SECRET.encode() not in data and b"user-4242" not in data


def test_hard_forgetting_a_superseding_memory_does_not_leave_a_dangling_pointer(tmp_path):
    s = MemoryStore(tmp_path / "m.db")
    s.remember(BOB, "budget", "3000", "user", evidence="my budget")
    new = s.remember(BOB, "budget", "4000", "user", evidence="my budget")
    s.forget(BOB, new.id, hard=True)
    assert [m.superseded_by for m in s.history(BOB, "budget")] == [None]
    s.close()


def test_purging_expired_memories_also_scrubs_the_file(tmp_path):
    t = [100.0]
    s = MemoryStore(tmp_path / "m.db", clock=lambda: t[0])
    s.remember(BOB, "destination", SECRET, "user", evidence="going there", ttl_s=10)
    t[0] += 20
    assert s.purge_expired() == 1
    s.close()
    assert SECRET.encode() not in raw_bytes(tmp_path / "m.db")


def test_sqlite_defaults_really_do_leave_deleted_text_in_the_file_which_is_why_this_store_changes_them(tmp_path):
    """Control experiment on a plain connection with default settings. If SQLite ever changes this default, the premise of the lesson changes."""
    path = tmp_path / "plain.db"
    db = sqlite3.connect(path)
    db.execute("CREATE TABLE t(v TEXT)")
    db.execute("INSERT INTO t VALUES (?)", (SECRET,))
    db.commit()
    db.execute("DELETE FROM t")
    db.commit()
    db.close()
    assert SECRET.encode() in raw_bytes(path)
