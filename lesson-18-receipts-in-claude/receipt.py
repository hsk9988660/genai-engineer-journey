"""The receipt: what is signed, how it is signed, and how it is checked.

Differences from the lesson's receipt, each for a reason:
  * the signed payload names the signing key (`kid`) and the algorithm (`alg`) INSIDE the signed bytes. The receipt never carries a public key
  * it records which MODEL and which AGENT VERSION produced the decision, and a policy hash, not just a policy name. Without those a receipt cannot answer "what made this decision?"
  * arguments and results are recorded as SALTED COMMITMENTS (keyed HMAC), not plain hashes. A plain hash of guessable arguments can be confirmed by anyone who tries the guesses.
    The salt is kept outside the chain and revealed only to someone who should be able to check the content
  * tool calls are recorded in two receipts: an INTENT written before the call and a RESULT after, so a crash between them is visible instead of silent
  * `seq` counts across the whole agent run, not per tool, and `prev` is the hash of the previous receipt including its signature
All times are integer seconds the SIGNER wrote. They are claims, not evidence.
"""

import base64
import hashlib
import hmac
import os
from decimal import Decimal

from canon import CanonError, canonicalize, digest
from keys import KeyProblem, KeyRegistry

ALG = "EdDSA"
INTENT, RESULT, APPROVAL, CHECKPOINT = "agent.tool_intent.v2", "agent.tool_result.v2", "human.approval.v2", "chain.checkpoint.v2"
ROLE_OF = {INTENT: "agent", RESULT: "agent", APPROVAL: "approver", CHECKPOINT: "checkpoint"}
OUTCOMES = ("success", "failure", "indeterminate")
_COMMON = {"type": str, "alg": str, "kid": str, "iat": int}
_CHAIN = {**_COMMON, "receipt_id": str, "agent_id": str, "run_id": str, "agent_version": str, "model_id": str, "seq": int, "prev": (str, type(None))}
SHAPES = {
    INTENT: {**_CHAIN, "tool": str, "args_commit": str, "policy_id": str, "policy_hash": str, "approval_ref": (str, type(None)), "action_digest": (str, type(None))},
    RESULT: {**_CHAIN, "intent_ref": str, "tool": str, "outcome": str, "result_commit": (str, type(None)), "error_type": (str, type(None))},
    APPROVAL: {**_COMMON, "approver_id": str, "tool": str, "args": dict, "action_digest": str, "policy_version": int, "expires_at": int, "nonce": str},
    CHECKPOINT: {**_COMMON, "agent_id": str, "run_id": str, "length": int, "head": (str, type(None)), "merkle_root": str, "ckpt_seq": int, "prev_checkpoint": (str, type(None))},
}


def b64u(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def b64u_decode(s: str) -> bytes:
    """Strict unpadded base64url. Python's decoder also accepts padding, the standard alphabet and stray characters, which would let one signature be written many ways."""
    if not isinstance(s, str):
        raise ValueError("not a string")
    data = base64.urlsafe_b64decode(s + "=" * ((4 - len(s) % 4) % 4))
    if b64u(data) != s:
        raise ValueError("not canonical unpadded base64url")
    return data


def to_tree(value, _depth: int = 0):
    """Turn an arbitrary tool argument or result into something canonicalisable WITHOUT silently losing information: floats and Decimals become tagged strings, bytes become base64."""
    if _depth > 30:
        raise TypeError("value is nested too deeply")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value if abs(value) <= 2 ** 53 - 1 else f"int:{value}"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise TypeError("NaN and infinity cannot be recorded")
        return f"float:{value!r}"
    if isinstance(value, Decimal):
        return f"decimal:{value}"
    if isinstance(value, (bytes, bytearray)):
        return f"bytes:{b64u(bytes(value))}"
    if isinstance(value, (list, tuple)):
        return [to_tree(v, _depth + 1) for v in value]
    if isinstance(value, dict):
        return {str(k): to_tree(v, _depth + 1) for k, v in value.items()}
    raise TypeError(f"cannot record a value of type {type(value).__name__}")


def new_salt() -> bytes:
    return os.urandom(16)


def commit(value, salt: bytes) -> str:
    """A keyed commitment: HMAC-SHA256 with the salt as key over the canonical form. Without the salt it cannot be checked against a guess."""
    if not isinstance(salt, (bytes, bytearray)) or len(salt) < 16:
        raise ValueError("a commitment salt must be at least 16 bytes")
    return "c1:" + hmac.new(bytes(salt), canonicalize(to_tree(value)), hashlib.sha256).hexdigest()


def open_commit(value, salt: bytes, commitment: str) -> bool:
    try:
        return hmac.compare_digest(commit(value, salt), commitment)
    except (ValueError, TypeError, CanonError):
        return False


def signed_bytes(receipt: dict) -> bytes:
    return canonicalize({k: v for k, v in receipt.items() if k != "sig"})


def sign(payload: dict, signer) -> dict:
    """Sign a payload. `alg` and `kid` are filled in and signed. The signature string sits next to the payload, outside the signed bytes."""
    if "sig" in payload:
        raise ValueError("a payload must not already contain a signature")
    if payload.get("type") not in ROLE_OF:
        raise ValueError(f"unknown receipt type {payload.get('type')!r}")
    if ROLE_OF[payload["type"]] != signer.role:
        raise ValueError(f"a {signer.role!r} key cannot sign a {payload['type']!r} receipt")
    body = {**payload, "alg": ALG, "kid": signer.kid}
    return {**body, "sig": b64u(signer.sign(canonicalize(body)))}


def receipt_hash(receipt: dict) -> str:
    """Hash of the WHOLE receipt, signature included. This is what the next receipt's `prev` and a result's `intent_ref` point at."""
    return digest(receipt)


def shape_problems(receipt) -> list[str]:
    if not isinstance(receipt, dict):
        return ["not an object"]
    shape = SHAPES.get(receipt.get("type"))
    if shape is None:
        return [f"unknown receipt type {receipt.get('type')!r}"]
    problems = []
    for name, typ in shape.items():
        if name not in receipt:
            problems.append(f"missing field {name!r}")
        elif not isinstance(receipt[name], typ) or (typ is int and isinstance(receipt[name], bool)):
            problems.append(f"field {name!r} has the wrong type")
    if not isinstance(receipt.get("sig"), str):
        problems.append("missing signature")
    return problems


def verify_signature(receipt: dict, registry: KeyRegistry, at: int | None = None):
    """Check the signature against the PINNED registry. Raises KeyProblem with a specific reason. Returns the key entry used."""
    problems = shape_problems(receipt)
    if problems:
        raise KeyProblem("malformed", "; ".join(problems))
    if receipt["alg"] != ALG:
        raise KeyProblem("algorithm", f"only {ALG} is accepted")
    try:
        sig = b64u_decode(receipt["sig"])
        message = signed_bytes(receipt)
    except (ValueError, CanonError) as exc:
        raise KeyProblem("malformed", str(exc)) from None
    return registry.verify(receipt["kid"], ROLE_OF[receipt["type"]], receipt["iat"] if at is None else at, message, sig)
