"""Binding a human's approval to the EXACT action, and spending it once.

An action receipt says "this agent key signed this". It never says "a person authorised this". For risky actions (refunds, deletions, payments) a separate APPROVER key signs the full
canonical action and its digest. The agent's later intent receipt carries the same digest and a reference to the approval. The approver registry is separate from the agent registry,
so a key that can sign tool calls can never sign an approval.

`verify_approval` refuses, each with its own reason:
  malformed, bad_signature, unknown_key / wrong_role / key_revoked / key_expired / key_not_yet_valid (stale authority: the key was rotated out),
  digest_mismatch (the action text was altered), action_mismatch (a real approval for a DIFFERENT action: digest substitution),
  expired, stale_policy (the policy moved since it was approved), issued_in_future
`ApprovalLedger` and `GuardedExecutor` add what verification alone cannot: the approval is RESERVED atomically before the tool runs, so the same approval cannot be spent twice, even by two
threads. A crash leaves it reserved, and `mark_unfinished_indeterminate` turns that into "outcome unknown", which is never retried automatically.
Verification is read-only and repeatable. It does not spend anything.
"""

import os
import sqlite3
import threading
import time

from canon import CanonError, digest
from keys import KeyProblem, KeyRegistry
from receipt import APPROVAL, ALG, receipt_hash, shape_problems, sign, verify_signature
from recorder import Indeterminate, ToolRecorder

CLOCK_SKEW = 300


class ApprovalRefused(Exception):
    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason


def action_digest(tool: str, args: dict) -> str:
    return digest({"tool": tool, "args": args})


def issue_approval(signer, *, approver_id: str, tool: str, args: dict, policy_version: int, expires_at: int, now: int, nonce: str | None = None) -> dict:
    if signer.role != "approver":
        raise ValueError("an approval is signed with an 'approver' key")
    return sign({"type": APPROVAL, "alg": ALG, "kid": signer.kid, "iat": int(now), "approver_id": approver_id, "tool": tool, "args": args, "action_digest": action_digest(tool, args),
                 "policy_version": policy_version, "expires_at": int(expires_at), "nonce": nonce or os.urandom(8).hex()}, signer)


def verify_approval(approval, registry: KeyRegistry, *, now: int, current_policy_version: int, expected: tuple | None = None) -> dict:
    """Returns the approval if it is acceptable now, else raises ApprovalRefused with a specific reason. `expected` is (tool, args): what is about to run."""
    problems = shape_problems(approval)
    if problems:
        raise ApprovalRefused("malformed", "; ".join(problems))
    if approval["type"] != APPROVAL:
        raise ApprovalRefused("malformed", "not an approval receipt")
    try:
        verify_signature(approval, registry, at=now)  # key status is judged at VERIFICATION time: a key rotated out since is stale authority
    except KeyProblem as exc:
        raise ApprovalRefused(exc.reason, str(exc)) from None
    try:
        recomputed = action_digest(approval["tool"], approval["args"])
    except CanonError as exc:
        raise ApprovalRefused("malformed", str(exc)) from None
    if recomputed != approval["action_digest"]:
        raise ApprovalRefused("digest_mismatch", "the action in the approval does not match its digest")
    if approval["iat"] > now + CLOCK_SKEW:
        raise ApprovalRefused("issued_in_future")
    if now >= approval["expires_at"]:
        raise ApprovalRefused("expired")
    if approval["policy_version"] != current_policy_version:
        raise ApprovalRefused("stale_policy", f"approved under policy {approval['policy_version']}, current is {current_policy_version}")
    if expected is not None:
        try:
            wanted = action_digest(*expected)
        except CanonError as exc:
            raise ApprovalRefused("action_mismatch", str(exc)) from None
        if wanted != approval["action_digest"]:
            raise ApprovalRefused("action_mismatch", "this approval is for a different action")
    return approval


class ApprovalLedger:
    """Which approvals have been spent. SQLite, so a reservation is atomic across threads and survives a restart."""

    def __init__(self, path=":memory:"):
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS approvals (hash TEXT PRIMARY KEY, status TEXT NOT NULL, request_id TEXT NOT NULL, ts REAL NOT NULL)")
        self._db.commit()
        self._lock = threading.Lock()

    def reserve(self, approval_hash: str, request_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("INSERT OR IGNORE INTO approvals(hash, status, request_id, ts) VALUES (?, 'reserved', ?, ?)", (approval_hash, request_id, time.time()))
            self._db.commit()
            return cur.rowcount == 1

    def complete(self, approval_hash: str, status: str) -> None:
        if status not in ("completed", "failed", "indeterminate"):
            raise ValueError("status must be completed, failed or indeterminate")
        with self._lock:
            self._db.execute("UPDATE approvals SET status = ? WHERE hash = ? AND status = 'reserved'", (status, approval_hash))
            self._db.commit()

    def status(self, approval_hash: str) -> str | None:
        with self._lock:
            row = self._db.execute("SELECT status FROM approvals WHERE hash = ?", (approval_hash,)).fetchone()
            return row[0] if row else None

    def mark_unfinished_indeterminate(self) -> int:
        """Run at startup: anything still 'reserved' belongs to a run that died. Its outcome is unknown, so it becomes 'indeterminate', never 'retry'."""
        with self._lock:
            cur = self._db.execute("UPDATE approvals SET status = 'indeterminate' WHERE status = 'reserved'")
            self._db.commit()
            return cur.rowcount


class GuardedExecutor:
    def __init__(self, recorder: ToolRecorder, registry: KeyRegistry, ledger: ApprovalLedger, effect_tools: set, policy_version: int, clock=lambda: int(time.time())):
        self.recorder, self.registry, self.ledger = recorder, registry, ledger
        self.effect_tools, self.policy_version, self.clock = frozenset(effect_tools), policy_version, clock

    def execute(self, tool: str, fn, args: dict, approval: dict | None = None):
        if tool not in self.effect_tools:
            return self.recorder.call(tool, fn, args)
        if approval is None:
            raise ApprovalRefused("approval_required", f"{tool!r} changes the world and needs a human approval")
        verify_approval(approval, self.registry, now=self.clock(), current_policy_version=self.policy_version, expected=(tool, args))
        h = receipt_hash(approval)
        if not self.ledger.reserve(h, os.urandom(6).hex()):
            status = self.ledger.status(h)
            raise ApprovalRefused({"indeterminate": "indeterminate_outcome", "reserved": "in_progress"}.get(status, "already_used"), f"this approval is {status}")
        try:
            result = self.recorder.call(tool, fn, args, approval=h, action_digest=approval["action_digest"])
        except Indeterminate:
            self.ledger.complete(h, "indeterminate")
            raise
        except Exception:
            self.ledger.complete(h, "failed")
            raise
        self.ledger.complete(h, "completed")
        return result


def check_intents_against_approvals(receipts: list, approvals: dict, registry: KeyRegistry, *, now: int, current_policy_version: int) -> list[tuple]:
    """For each intent that cites an approval: the approval must be supplied, valid, for the SAME action digest and tool, and used only once. Returns (index, code, detail) findings."""
    findings, used = [], {}
    for i, r in enumerate(receipts):
        if not isinstance(r, dict) or r.get("type") != "agent.tool_intent.v2" or r.get("approval_ref") is None:
            continue
        ref = r["approval_ref"]
        if ref in used:
            findings.append((i, "approval_reused", f"the approval was already cited by receipt {used[ref]}"))
        used.setdefault(ref, i)
        approval = approvals.get(ref)
        if approval is None:
            findings.append((i, "approval_missing", "the cited approval was not supplied"))
            continue
        try:
            verify_approval(approval, registry, now=now, current_policy_version=current_policy_version)
        except ApprovalRefused as exc:
            findings.append((i, f"approval_{exc.reason}", str(exc)))
            continue
        if approval["action_digest"] != r.get("action_digest"):
            findings.append((i, "approval_digest_substitution", "the intent's action digest differs from the approved action"))
        if approval["tool"] != r.get("tool"):
            findings.append((i, "approval_tool_mismatch", "the intent is for a different tool than the approval"))
    return findings
