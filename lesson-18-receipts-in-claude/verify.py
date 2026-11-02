"""Verifying a whole chain against pinned keys, with a specific reason for every failure.

`ChainVerifier.verify` returns a `VerifyReport`. Two different questions are kept apart:
  valid     nothing is WRONG: every signature checks out under a pinned key, the links and sequence are right, results match their intents, checkpoints and approvals agree
  complete  valid AND every intent has a result. An intent with no result is not an error, it is "outcome unknown": the process died, or is still running
Per-receipt checks (error codes):
  malformed, algorithm, bad_signature, unknown_key, wrong_role, key_expired, key_revoked, key_not_yet_valid,
  unexpected_type, sequence_gap, duplicate_sequence, broken_link, mixed_identity,
  orphan_result, tool_mismatch, duplicate_result
Chain-level (error codes): truncated, head_mismatch, truncated_vs_checkpoint, forked_vs_checkpoint, merkle_mismatch, checkpoint_*, approval_*
Warnings (never make a chain invalid): time_goes_backwards, future_time, signed_by_key_revoked_later.
A chain with no checkpoint and no pinned head CAN be silently cut short at the end. The report says nothing about that, because it cannot know.
"""

from dataclasses import dataclass, field

from canon import CanonError
from checkpoint import merkle_root_of, verify_checkpoint_chain
from approval import check_intents_against_approvals
from keys import KeyProblem, KeyRegistry
from receipt import INTENT, RESULT, receipt_hash, shape_problems, verify_signature

CLOCK_SKEW = 300


@dataclass(frozen=True)
class Finding:
    index: int | None
    code: str
    detail: str = ""


@dataclass
class VerifyReport:
    valid: bool
    complete: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    unresolved: list = field(default_factory=list)
    statuses: list = field(default_factory=list)
    length: int = 0
    head: str | None = None

    def codes(self) -> list[str]:
        return [f.code for f in self.errors]


class ChainVerifier:
    def __init__(self, registry: KeyRegistry, checkpoint_registry: KeyRegistry | None = None, approver_registry: KeyRegistry | None = None):
        """Three registries, three authorities: agent keys, checkpoint keys, approver keys. They may be the same object only if you want one authority to cover all three, which you should not."""
        self.registry = registry
        self.checkpoint_registry = checkpoint_registry or registry
        self.approver_registry = approver_registry or registry

    def verify(self, receipts: list, *, expected_head: str | None = None, min_length: int | None = None, checkpoints: list | tuple = (), pinned_latest_checkpoint: str | None = None,
               approvals: dict | None = None, now: int | None = None, current_policy_version: int | None = None) -> VerifyReport:
        errors, warnings, bad = [], [], set()
        intents: dict[str, tuple] = {}
        resulted: set[str] = set()
        prev_hash, identity, last_iat, hashes = None, None, None, []

        def err(i, code, detail=""):
            errors.append(Finding(i, code, detail))
            if i is not None:
                bad.add(i)

        for i, r in enumerate(receipts):
            problems = shape_problems(r)
            if problems:
                err(i, "malformed", "; ".join(problems))
                hashes.append(None)
                prev_hash = None
                continue
            if r["type"] not in (INTENT, RESULT):
                err(i, "unexpected_type", f"{r['type']!r} does not belong in a tool-call chain")
                hashes.append(None)
                prev_hash = None
                continue
            try:
                entry = verify_signature(r, self.registry)
                if entry.revoked_at is not None:
                    warnings.append(Finding(i, "signed_by_key_revoked_later", "the signing time is only the signer's claim"))
            except KeyProblem as exc:
                err(i, exc.reason, str(exc))
            if r["seq"] != i:
                err(i, "sequence_gap" if r["seq"] > i else "duplicate_sequence", f"position {i} carries sequence {r['seq']}")
            if r["prev"] != prev_hash:
                err(i, "broken_link", "the previous-receipt hash does not match the receipt before it")
            ident = (r["agent_id"], r["run_id"])
            if identity is None:
                identity = ident
            elif ident != identity:
                err(i, "mixed_identity", "agent or run id differs from the start of the chain")
            if last_iat is not None and r["iat"] < last_iat:
                warnings.append(Finding(i, "time_goes_backwards", "claimed time is earlier than the receipt before it"))
            last_iat = r["iat"] if last_iat is None else max(last_iat, r["iat"])
            if now is not None and r["iat"] > now + CLOCK_SKEW:
                warnings.append(Finding(i, "future_time", "claimed time is in the future"))
            try:
                h = receipt_hash(r)
            except CanonError as exc:
                err(i, "malformed", str(exc))
                h = None
            if r["type"] == INTENT and h is not None:
                intents[h] = (i, r)
            elif r["type"] == RESULT:
                ref = r["intent_ref"]
                if ref not in intents:
                    err(i, "orphan_result", "no earlier intent in this chain has that hash")
                else:
                    if intents[ref][1]["tool"] != r["tool"]:
                        err(i, "tool_mismatch", "the result names a different tool than its intent")
                    if ref in resulted:
                        err(i, "duplicate_result", "that intent already has a result")
                    resulted.add(ref)
            hashes.append(h)
            prev_hash = h

        length, head = len(receipts), (hashes[-1] if hashes else None)
        unresolved = sorted(idx for h, (idx, _) in intents.items() if h not in resulted)
        if min_length is not None and length < min_length:
            err(None, "truncated", f"{length} receipts presented, at least {min_length} expected")
        if expected_head is not None and head != expected_head:
            err(None, "head_mismatch", "the last receipt is not the pinned head (the chain was cut short, extended or changed)")
        for k, c in enumerate(checkpoints):
            for idx, code, detail in verify_checkpoint_chain([c], self.checkpoint_registry):
                if code not in ("checkpoint_sequence", "checkpoint_link"):
                    err(None, code, f"checkpoint {k}: {detail}")
            if not isinstance(c, dict) or shape_problems(c):
                continue
            if identity is not None and (c["agent_id"], c["run_id"]) != identity:
                continue
            if c["length"] > length:
                err(None, "truncated_vs_checkpoint", f"checkpoint {k} commits to {c['length']} receipts, only {length} presented")
                continue
            covered = hashes[:c["length"]]
            if None in covered:
                continue
            if c["length"] and covered[-1] != c["head"]:
                err(None, "forked_vs_checkpoint", f"the receipt at position {c['length'] - 1} is not the one checkpoint {k} committed to")
            elif merkle_root_of(covered) != c["merkle_root"]:
                err(None, "merkle_mismatch", f"receipts before checkpoint {k} do not match its Merkle root")
        if len(checkpoints) > 1 or pinned_latest_checkpoint is not None:
            for idx, code, detail in verify_checkpoint_chain(list(checkpoints), self.checkpoint_registry, pinned_latest=pinned_latest_checkpoint):
                if code in ("checkpoint_sequence", "checkpoint_link", "checkpoint_rollback_or_fork"):
                    err(None, code, detail)
        if approvals is not None:
            for idx, code, detail in check_intents_against_approvals(receipts, approvals, self.approver_registry, now=now if now is not None else 0,
                                                                      current_policy_version=current_policy_version if current_policy_version is not None else -1):
                err(idx, code, detail)
        statuses = ["invalid" if i in bad else "ok" for i in range(length)]
        return VerifyReport(not errors, not errors and not unresolved, errors, warnings, unresolved, statuses, length, head)
