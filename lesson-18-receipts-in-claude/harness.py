"""The lesson's receipt patterns versus the hardened ones, on seven scenarios. Offline and deterministic.

The "lesson's pattern" is my re-statement of the notebook's behaviour, built on the same primitives (Ed25519 over canonical JSON), not its exact code: it cannot be imported, and I did not run it.
Nothing here uses a model. These scenarios test the cryptographic and bookkeeping mechanisms.
"""

import hashlib
import hmac
import json
import os
import tempfile
import threading
from dataclasses import dataclass, field

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from approval import ApprovalLedger, ApprovalRefused, GuardedExecutor, issue_approval
from canon import canonicalize, digest
from checkpoint import make_checkpoint
from keys import KeyRegistry, Signer
from receipt import b64u, b64u_decode, commit, open_commit, new_salt, receipt_hash
from recorder import ChainWriter, ToolRecorder, read_chain
from verify import ChainVerifier

NOW = 1_800_000_000


@dataclass
class Outcome:
    scenario: str
    strategy: str
    metrics: dict = field(default_factory=dict)


class SimulatedCrash(BaseException):
    """The process dying. A BaseException so nothing in the recorder's `except Exception` can swallow it."""


# --- what the lesson's code does, restated ---------------------------------------------------------------------------------------------
def lesson_sign(payload: dict, signer: Signer) -> dict:
    return {**payload, "signature": {"alg": "EdDSA", "sig": b64u(signer.sign(canonicalize(payload))), "public_key": b64u(signer.public_bytes)}}


def lesson_verify(receipt: dict, public_key_b64: str) -> bool:
    try:
        payload = {k: v for k, v in receipt.items() if k != "signature"}
        Ed25519PublicKey.from_public_bytes(b64u_decode(public_key_b64)).verify(b64u_decode(receipt["signature"]["sig"]), canonicalize(payload))
        return True
    except Exception:
        return False


def lesson_verify_trusting_embedded_key(receipt: dict) -> bool:
    """The trap: 'verify' against the key the receipt itself carries."""
    return lesson_verify(receipt, receipt["signature"]["public_key"])


def lesson_hash(obj) -> str:
    return "sha256:" + hashlib.sha256(canonicalize(obj)).hexdigest()


class LessonReceiptedTool:
    """One tool, its own chain, receipt written AFTER the call, plain hash of the arguments, sequence = len(receipts). As in the notebook's `ReceiptedTool`."""

    def __init__(self, name, fn, signer):
        self.name, self.fn, self.signer, self.receipts = name, fn, signer, []

    def __call__(self, **kwargs):
        result = self.fn(**kwargs)
        prev = lesson_hash(self.receipts[-1]) if self.receipts else None
        self.receipts.append(lesson_sign({"type": "agent.tool_call.v1", "tool_name": self.name, "tool_args_hash": lesson_hash(kwargs), "result_hash": lesson_hash(result),
                                          "sequence": len(self.receipts), "previous_receipt_hash": prev}, self.signer))
        return result


def lesson_verify_chain(chain: list, public_key_b64: str) -> bool:
    return all(lesson_verify(r, public_key_b64) and r["previous_receipt_hash"] == (lesson_hash(chain[i - 1]) if i else None) and r["sequence"] == i for i, r in enumerate(chain))


# --- helpers -------------------------------------------------------------------------------------------------------------------------
def setup(tmp: str, **kw):
    agent, ckpt = Signer.generate("agent"), Signer.generate("checkpoint")
    reg, creg = KeyRegistry(), KeyRegistry()
    reg.add_signer(agent)
    creg.add_signer(ckpt)
    writer = ChainWriter(os.path.join(tmp, "chain.jsonl"), agent, agent_id="support-bot", run_id="run-1", agent_version="v-3f9a2c", model_id="claude-sonnet-5-5", clock=lambda: NOW, fsync=False, **kw)
    return agent, ckpt, reg, creg, writer, ToolRecorder(writer, "refund-policy", "sha256:policyhash")


# --- 1. a forged receipt ---------------------------------------------------------------------------------------------------------------
def scenario_forgery() -> list[Outcome]:
    issuer, attacker = Signer.generate("agent"), Signer.generate("agent")
    genuine = lesson_sign({"type": "agent.tool_call.v1", "tool_name": "issue_refund", "policy_id": "strict-v3"}, issuer)
    forged = lesson_sign({"type": "agent.tool_call.v1", "tool_name": "issue_refund", "policy_id": "permissive"}, attacker)
    out = [Outcome("forgery", "lesson_verify_with_the_embedded_key", {"forgery_accepted": lesson_verify_trusting_embedded_key(forged)}),
           Outcome("forgery", "lesson_verify_with_a_pinned_key", {"forgery_accepted": lesson_verify(forged, b64u(issuer.public_bytes)), "genuine_accepted": lesson_verify(genuine, b64u(issuer.public_bytes))})]
    with tempfile.TemporaryDirectory() as tmp:
        agent, _, reg, _, writer, rec = setup(tmp)
        rec.call("issue_refund", lambda **a: "ok", {"order": "A1"})
        # the attacker writes a whole chain of their own, claiming the same agent and run, signed with their own key
        evil = ChainWriter(os.path.join(tmp, "evil.jsonl"), attacker, agent_id="support-bot", run_id="run-1", agent_version="v-3f9a2c", model_id="claude-sonnet-5-5", clock=lambda: NOW, fsync=False)
        ToolRecorder(evil, "permissive", "x").call("issue_refund", lambda **a: "ok", {"order": "A1"})
        v = ChainVerifier(reg)
        r1 = v.verify(read_chain(evil.path))
        # an attacker who copies the issuer's kid but cannot sign
        copied = read_chain(evil.path)
        for rc in copied:
            rc["kid"] = agent.kid
        r2 = v.verify(copied)
        out.append(Outcome("forgery", "pinned_registry_chain_verifier", {"own_key_chain_accepted": r1.valid, "own_key_reason": sorted(set(r1.codes())), "copied_kid_accepted": r2.valid, "copied_kid_reason": sorted(set(r2.codes())),
                                                                       "genuine_accepted": v.verify(read_chain(writer.path)).valid}))
    return out


# --- 2. who can forge: a shared secret or a public key --------------------------------------------------------------------------------------
def hmac_chain(entries: list, key: bytes) -> list:
    out, prev = [], "0" * 64
    for e in entries:
        body = json.dumps({"entry": e, "prev": prev}, sort_keys=True)
        mac = hmac.new(key, body.encode(), hashlib.sha256).hexdigest()
        out.append({"entry": e, "prev": prev, "mac": mac})
        prev = mac
    return out


def hmac_verify(chain: list, key: bytes) -> bool:
    prev = "0" * 64
    for c in chain:
        body = json.dumps({"entry": c["entry"], "prev": c["prev"]}, sort_keys=True)
        if c["prev"] != prev or not hmac.compare_digest(c["mac"], hmac.new(key, body.encode(), hashlib.sha256).hexdigest()):
            return False
        prev = c["mac"]
    return True


def scenario_who_can_forge() -> list[Outcome]:
    key = os.urandom(32)
    genuine = hmac_chain([{"tool": "refund", "amount": 40}, {"tool": "refund", "amount": 25}], key)
    forged = hmac_chain([{"tool": "refund", "amount": 40}], key)  # a verifier holding the key quietly drops the second entry and re-seals
    out = [Outcome("who_can_forge", "hmac_chain (secret shared with the verifier)", {"genuine_verifies": hmac_verify(genuine, key), "insider_forgery_verifies": hmac_verify(forged, key)})]
    with tempfile.TemporaryDirectory() as tmp:
        agent, _, reg, _, writer, rec = setup(tmp)
        rec.call("refund", lambda **a: 1, {"amount": 40})
        # the verifier holds only `reg` (public keys). Everything it could do to forge:
        insider = Signer.generate("agent")
        attempt = ChainWriter(os.path.join(tmp, "x.jsonl"), insider, agent_id="support-bot", run_id="run-1", agent_version="v", model_id="m", clock=lambda: NOW, fsync=False)
        ToolRecorder(attempt, "p", "h").call("refund", lambda **a: 1, {"amount": 40})
        v = ChainVerifier(reg)
        out.append(Outcome("who_can_forge", "ed25519 (verifier holds public keys only)", {"genuine_verifies": v.verify(read_chain(writer.path)).valid, "insider_forgery_verifies": v.verify(read_chain(attempt.path)).valid}))
    return out


# --- 3. the end of the chain goes missing ---------------------------------------------------------------------------------------------
def scenario_truncation() -> list[Outcome]:
    with tempfile.TemporaryDirectory() as tmp:
        agent, ckpt, reg, creg, writer, rec = setup(tmp)
        for i in range(6):
            rec.call("refund", lambda **a: "ok", {"order": f"A{i}"})
        full = read_chain(writer.path)
        head = writer.head
        cp = make_checkpoint(ckpt, agent_id="support-bot", run_id="run-1", receipts=full, clock=lambda: NOW)
        cut = full[:8]  # two whole calls (4 receipts) removed from the end
        v = ChainVerifier(reg, creg)
        no_anchor = v.verify(cut)
        lesson = LessonReceiptedTool("refund", lambda **a: "ok", agent)
        for i in range(6):
            lesson(order=f"A{i}")
        return [Outcome("truncation", "lesson_verify_chain", {"cut_chain_verifies": lesson_verify_chain(lesson.receipts[:4], b64u(agent.public_bytes))}),
                Outcome("truncation", "chain_verifier_alone", {"valid": no_anchor.valid, "complete": no_anchor.complete, "truncation_detected": bool(set(no_anchor.codes()) & {"truncated", "head_mismatch"})}),
                Outcome("truncation", "with_a_pinned_head", {"truncation_detected": "head_mismatch" in v.verify(cut, expected_head=head).codes()}),
                Outcome("truncation", "with_a_signed_checkpoint", {"truncation_detected": "truncated_vs_checkpoint" in v.verify(cut, checkpoints=[cp]).codes(), "full_chain_valid": v.verify(full, checkpoints=[cp]).valid})]


# --- 4. the process dies after the effect --------------------------------------------------------------------------------------------
def scenario_crash() -> list[Outcome]:
    effects = []

    def effect(**a):
        effects.append(a)
        raise SimulatedCrash()
    signer = Signer.generate("agent")
    lesson = LessonReceiptedTool("book", effect, signer)
    try:
        lesson(flight="QF11")
    except SimulatedCrash:
        pass
    out = [Outcome("crash_after_effect", "lesson_wrapper (receipt written after the call)", {"effects_in_the_world": len(effects), "receipts": len(lesson.receipts)})]
    effects.clear()
    with tempfile.TemporaryDirectory() as tmp:
        agent, _, reg, _, writer, rec = setup(tmp)
        try:
            rec.call("book", effect, {"flight": "QF11"})
        except SimulatedCrash:
            pass
        rep = ChainVerifier(reg).verify(read_chain(writer.path))
        out.append(Outcome("crash_after_effect", "intent_then_result", {"effects_in_the_world": len(effects), "receipts": rep.length, "valid": rep.valid, "complete": rep.complete, "unresolved_intents": rep.unresolved}))
    return out


# --- 5. recovering the arguments from a hash ----------------------------------------------------------------------------------------------
def scenario_guessable_arguments() -> list[Outcome]:
    airports = [f"AP{i:02d}" for i in range(20)]
    dates = [f"2026-06-{d:02d}" for d in range(1, 31)]
    space = [{"origin": o, "destination": d, "date": dt} for o in airports for d in airports for dt in dates if o != d]
    secret = {"origin": "AP07", "destination": "AP13", "date": "2026-06-21"}
    plain = lesson_hash(secret)
    guesses = next((i + 1 for i, cand in enumerate(space) if lesson_hash(cand) == plain), None)
    salt = new_salt()
    committed = commit(secret, salt)
    attacker_hits = sum(1 for cand in space if commit(cand, new_salt()) == committed)  # without the salt, every guess uses a salt of its own
    return [Outcome("guessable_arguments", "plain_sha256_of_the_arguments", {"search_space": len(space), "arguments_recovered": guesses is not None, "guesses_needed": guesses}),
            Outcome("guessable_arguments", "salted_commitment", {"search_space": len(space), "arguments_recovered": attacker_hits > 0, "owner_can_open": open_commit(secret, salt, committed), "wrong_salt_opens": open_commit(secret, new_salt(), committed)})]


# --- 6. approvals ---------------------------------------------------------------------------------------------------------------------
def scenario_approvals() -> list[Outcome]:
    # the lesson-style alternative: a boolean the caller sets
    naive_effects = []
    naive_refund = lambda amount, approved=False: naive_effects.append(amount) if approved else None
    for amount in (4000, 400_000, 4000):  # approved once for 4000; here every call just sets the flag
        naive_refund(amount, approved=True)
    out = [Outcome("approvals", "an_approved_flag_on_the_call", {"refunds_executed": len(naive_effects), "amounts": naive_effects})]
    with tempfile.TemporaryDirectory() as tmp:
        agent, _, reg, _, writer, rec = setup(tmp)
        approver, attacker = Signer.generate("approver"), Signer.generate("approver")
        areg = KeyRegistry()
        areg.add_signer(approver)
        clock = [NOW]
        effects = []
        ledger = ApprovalLedger()
        policy = [1]
        ex = GuardedExecutor(rec, areg, ledger, {"refund"}, 1, clock=lambda: clock[0])

        def refund(**a):
            effects.append(a["amount"])
            return "refunded"
        args = {"order": "A1002", "amount": 4000}

        def approval(a=args, signer=approver, expires=NOW + 600, pv=1):
            return issue_approval(signer, approver_id="supervisor-7", tool="refund", args=a, policy_version=pv, expires_at=expires, now=NOW)

        def attempt(label, fn):
            try:
                fn()
                return "executed"
            except ApprovalRefused as exc:
                return exc.reason
        results = {}
        good = approval()
        results["no approval"] = attempt("a", lambda: ex.execute("refund", refund, args))
        results["valid approval"] = attempt("b", lambda: ex.execute("refund", refund, args, good))
        results["same approval again"] = attempt("c", lambda: ex.execute("refund", refund, args, good))
        results["bigger amount, same approval"] = attempt("d", lambda: ex.execute("refund", refund, {**args, "amount": 400_000}, approval()))
        results["approver key not registered"] = attempt("e", lambda: ex.execute("refund", refund, {**args, "order": "A9"}, approval({**args, "order": "A9"}, signer=attacker)))
        stale_policy = approval({**args, "order": "A7"}, pv=0)
        results["old policy version"] = attempt("f", lambda: ex.execute("refund", refund, {**args, "order": "A7"}, stale_policy))
        old = approval({**args, "order": "A8"}, expires=NOW + 60)
        clock[0] = NOW + 61
        results["approval expired"] = attempt("g", lambda: ex.execute("refund", refund, {**args, "order": "A8"}, old))
        clock[0] = NOW
        rotated = approval({**args, "order": "A6"})
        areg.revoke(approver.kid, NOW)
        results["approver key revoked since"] = attempt("h", lambda: ex.execute("refund", refund, {**args, "order": "A6"}, rotated))
        out.append(Outcome("approvals", "signed_approval_and_a_one_time_ledger", {"refunds_executed": len(effects), "results": results}))
    # two threads spend the same approval at once
    with tempfile.TemporaryDirectory() as tmp:
        agent, _, reg, _, writer, rec = setup(tmp)
        approver = Signer.generate("approver")
        areg = KeyRegistry()
        areg.add_signer(approver)
        effects = []
        ex = GuardedExecutor(rec, areg, ApprovalLedger(), {"refund"}, 1, clock=lambda: NOW)
        ap = issue_approval(approver, approver_id="s", tool="refund", args={"amount": 5}, policy_version=1, expires_at=NOW + 600, now=NOW)
        barrier, outcomes = threading.Barrier(8), []

        def spend():
            barrier.wait()
            try:
                ex.execute("refund", lambda **a: effects.append(1), {"amount": 5}, ap)
                outcomes.append("executed")
            except ApprovalRefused as exc:
                outcomes.append(exc.reason)
        threads = [threading.Thread(target=spend) for _ in range(8)]
        [t.start() for t in threads]
        [t.join() for t in threads]
        out.append(Outcome("approvals", "eight_threads_spend_one_approval", {"refunds_executed": len(effects), "outcomes": {k: outcomes.count(k) for k in set(outcomes)}}))
    return out


# --- 7. one chain for the whole agent --------------------------------------------------------------------------------------------------
def scenario_one_chain() -> list[Outcome]:
    signer = Signer.generate("agent")
    key = b64u(signer.public_bytes)
    tools = {n: LessonReceiptedTool(n, lambda **a: "ok", signer) for n in ("lookup_flights", "hold_seat", "confirm_booking")}
    for n, t in tools.items():
        t(x=1)
        t(x=2)
    survivors = {n: t.receipts for n, t in tools.items() if n != "hold_seat"}  # someone deletes the middle tool's whole chain
    out = [Outcome("one_chain", "a_chain_per_tool (the lesson's ReceiptedTool)", {"remaining_chains_verify": all(lesson_verify_chain(c, key) for c in survivors.values()), "deletion_detected": not all(lesson_verify_chain(c, key) for c in survivors.values()),
                                                                               "receipt_names_the_model": "model_id" in tools["hold_seat"].receipts[0]})]
    with tempfile.TemporaryDirectory() as tmp:
        agent, _, reg, _, writer, rec = setup(tmp)
        for n in ("lookup_flights", "hold_seat", "confirm_booking"):
            rec.call(n, lambda **a: "ok", {"x": 1})
        chain = read_chain(writer.path)
        without_middle = [r for r in chain if r.get("tool") != "hold_seat"]
        rep = ChainVerifier(reg).verify(without_middle)
        out.append(Outcome("one_chain", "one_chain_for_the_agent", {"deletion_detected": not rep.valid, "codes": sorted(set(rep.codes())),
                                                                   "receipt_names_the_model": chain[0]["model_id"] == "claude-sonnet-5-5" and chain[0]["agent_version"] == "v-3f9a2c"}))
    return out


def evaluate() -> list[Outcome]:
    return scenario_forgery() + scenario_who_can_forge() + scenario_truncation() + scenario_crash() + scenario_guessable_arguments() + scenario_approvals() + scenario_one_chain()


def print_table(outcomes: list[Outcome]) -> None:
    for o in outcomes:
        print(f"{o.scenario:<20} {o.strategy:<50} " + "  ".join(f"{k}={v}" for k, v in o.metrics.items()))


if __name__ == "__main__":
    print_table(evaluate())
