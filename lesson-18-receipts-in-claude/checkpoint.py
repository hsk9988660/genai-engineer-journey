"""Checkpoints: a signed statement "this chain had N receipts and this head and this Merkle root", made by a DIFFERENT key.

A hash chain on its own cannot show that the END is missing: cut off the last receipts and what remains still links up. A checkpoint, signed by a separate `checkpoint` key and stored
somewhere the agent's host cannot rewrite (a transparency log, a second system, a timestamp service), pins what existed at that moment. A verifier that holds a checkpoint can then say:
  truncated_vs_checkpoint   the chain presented is shorter than the checkpoint
  forked_vs_checkpoint      the receipt at that position is not the one the checkpoint committed to
  merkle_mismatch           the receipts before the checkpoint were rewritten
Checkpoints are linked to each other (`prev_checkpoint`) with their own sequence, so an older checkpoint cannot be passed off as the latest: give the verifier the hash of the latest one you
know and it will refuse a history that does not contain it.
Nothing here stores or anchors a checkpoint anywhere. That is a deployment decision, and it is where the protection actually comes from.
"""

import time

from canon import CanonError
from keys import KeyProblem, KeyRegistry
from merkle import root
from receipt import ALG, CHECKPOINT, receipt_hash, shape_problems, sign, verify_signature


def merkle_root_of(receipt_hashes: list[str]) -> str:
    return root([h.encode("ascii") for h in receipt_hashes]).hex()


def make_checkpoint(signer, *, agent_id: str, run_id: str, receipts: list[dict], ckpt_seq: int = 0, prev_checkpoint: str | None = None, clock=lambda: int(time.time())) -> dict:
    if signer.role != "checkpoint":
        raise ValueError("a checkpoint is signed with a 'checkpoint' key")
    hashes = [receipt_hash(r) for r in receipts]
    return sign({"type": CHECKPOINT, "alg": ALG, "kid": signer.kid, "iat": int(clock()), "agent_id": agent_id, "run_id": run_id, "length": len(hashes),
                 "head": hashes[-1] if hashes else None, "merkle_root": merkle_root_of(hashes), "ckpt_seq": ckpt_seq, "prev_checkpoint": prev_checkpoint}, signer)


def verify_checkpoint_chain(checkpoints: list[dict], registry: KeyRegistry, *, pinned_latest: str | None = None) -> list[tuple]:
    """Returns (index, code, detail) findings. `pinned_latest` is the hash of the newest checkpoint you hold independently: a history that does not include it is a rollback or a fork."""
    findings, prev, seen = [], None, []
    for i, c in enumerate(checkpoints):
        problems = shape_problems(c)
        if problems or c.get("type") != CHECKPOINT:
            findings.append((i, "malformed", "; ".join(problems) or "not a checkpoint"))
            continue
        try:
            verify_signature(c, registry)
        except KeyProblem as exc:
            findings.append((i, exc.reason, str(exc)))
        if c["ckpt_seq"] != i:
            findings.append((i, "checkpoint_sequence", f"expected {i}, found {c['ckpt_seq']}"))
        if c["prev_checkpoint"] != prev:
            findings.append((i, "checkpoint_link", "the link to the previous checkpoint is wrong"))
        try:
            prev = receipt_hash(c)
        except CanonError:
            prev = None
        seen.append(prev)
    if pinned_latest is not None and pinned_latest not in seen:
        findings.append((None, "checkpoint_rollback_or_fork", "the newest known checkpoint is not part of this history"))
    return findings
