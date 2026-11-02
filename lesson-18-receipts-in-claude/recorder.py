"""Writing the chain: one ordered, durable, signed stream per agent run.

`ChainWriter` assigns the sequence number and the previous-receipt hash UNDER A LOCK, signs, and appends one line to a file with flush and fsync before returning. So two threads cannot
get the same sequence number, and a receipt that has been returned has reached the disk.

`ToolRecorder.call` is the heart of it. For every tool call it writes
    1. an INTENT receipt (before the tool runs), then runs the tool, then
    2. a RESULT receipt: success, failure, or indeterminate.
If the process dies between the two, the chain contains an intent with no result, which the verifier reports as UNRESOLVED. That is different from a missing receipt, which looks like nothing.
A tool that raises gets a failure receipt. A tool that raises `Indeterminate` (the outside world may or may not have acted) gets an indeterminate one. Neither is retried here.

Salts for the argument and result commitments are kept in a `SaltStore` OUTSIDE the chain. Reveal a salt only to someone who should be able to check that content.
"""

import json
import os
import threading
import time

from keys import Signer
from receipt import ALG, INTENT, RESULT, commit, new_salt, receipt_hash, sign


class ChainFileError(Exception):
    pass


class Indeterminate(Exception):
    """Raise from a tool when the outside world may or may not have acted (a timeout after sending a payment, say)."""


class ChainWriter:
    def __init__(self, path, signer: Signer, *, agent_id: str, run_id: str, agent_version: str, model_id: str, clock=lambda: int(time.time()), fsync: bool = True):
        if signer.role != "agent":
            raise ValueError("a chain is written with an 'agent' key")
        self.path, self.signer, self.clock, self.fsync = str(path), signer, clock, fsync
        self.agent_id, self.run_id, self.agent_version, self.model_id = agent_id, run_id, agent_version, model_id
        self._lock = threading.Lock()
        self._seq, self._head = 0, None
        if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
            last = self._read_last()
            self._seq, self._head = last["seq"] + 1, receipt_hash(last)

    def _read_last(self) -> dict:
        """Continue an existing file. The tail is read, not verified: a tampered file will fail verification later, but an unreadable one must not be silently overwritten."""
        with open(self.path, "rb") as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
        try:
            last = json.loads(lines[-1])
            if not isinstance(last, dict) or not isinstance(last.get("seq"), int):
                raise ValueError
            return last
        except (ValueError, IndexError):
            raise ChainFileError("the last line of the chain file is not a receipt: refusing to append") from None

    @property
    def length(self) -> int:
        return self._seq

    @property
    def head(self):
        return self._head

    def append(self, fields: dict) -> dict:
        with self._lock:
            payload = {"alg": ALG, "kid": self.signer.kid, "agent_id": self.agent_id, "run_id": self.run_id, "agent_version": self.agent_version, "model_id": self.model_id,
                       "seq": self._seq, "prev": self._head, "iat": int(self.clock()), "receipt_id": os.urandom(8).hex(), **fields}
            r = sign(payload, self.signer)
            line = json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n"
            with open(self.path, "ab") as f:
                f.write(line.encode("utf-8"))
                f.flush()
                if self.fsync:
                    os.fsync(f.fileno())
            self._seq += 1
            self._head = receipt_hash(r)
            return r


def read_chain(path) -> list[dict]:
    """Read a chain file. Lines that are not JSON objects are kept as they are, so the verifier can report them instead of hiding them."""
    out = []
    with open(path, "rb") as f:
        for raw in f.read().splitlines():
            if not raw.strip():
                continue
            try:
                out.append(json.loads(raw))
            except ValueError:
                out.append({"unreadable": raw[:80].decode("utf-8", "replace")})
    return out


class SaltStore:
    def __init__(self):
        self._salts: dict[tuple, bytes] = {}
        self._lock = threading.Lock()

    def put(self, receipt_id: str, field: str, salt: bytes) -> None:
        with self._lock:
            self._salts[(receipt_id, field)] = salt

    def get(self, receipt_id: str, field: str) -> bytes | None:
        with self._lock:
            return self._salts.get((receipt_id, field))


class ToolRecorder:
    def __init__(self, writer: ChainWriter, policy_id: str, policy_hash: str, salts: SaltStore | None = None):
        self.writer, self.policy_id, self.policy_hash, self.salts = writer, policy_id, policy_hash, salts or SaltStore()

    def call(self, tool: str, fn, args: dict, *, approval: str | None = None, action_digest: str | None = None):
        salt = new_salt()
        intent = self.writer.append({"type": INTENT, "tool": tool, "args_commit": commit(args, salt), "policy_id": self.policy_id, "policy_hash": self.policy_hash,
                                     "approval_ref": approval, "action_digest": action_digest})
        self.salts.put(intent["receipt_id"], "args", salt)
        ref = receipt_hash(intent)
        try:
            result = fn(**args)
        except Indeterminate as exc:
            self._result(ref, tool, "indeterminate", None, type(exc).__name__)
            raise
        except Exception as exc:
            self._result(ref, tool, "failure", None, type(exc).__name__)
            raise
        # BaseException (the process being killed, Ctrl-C) is NOT caught: no result is written, and the intent stays unresolved
        rsalt, rcommit, err = new_salt(), None, None
        try:
            rcommit = commit(result, rsalt)
        except (TypeError, ValueError):
            err = "UncommittableResult"  # the effect happened; say so even though the result cannot be committed to
        rec = self._result(ref, tool, "success", rcommit, err)
        if rcommit is not None:
            self.salts.put(rec["receipt_id"], "result", rsalt)
        return result

    def _result(self, ref: str, tool: str, outcome: str, result_commit, error_type) -> dict:
        return self.writer.append({"type": RESULT, "intent_ref": ref, "tool": tool, "outcome": outcome, "result_commit": result_commit, "error_type": error_type})
