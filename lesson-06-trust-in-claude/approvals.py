"""Human approval that is bound to the exact action, single-use, expiring, and fail-closed.

Approving "book a flight to Paris" must not authorize "book a flight to Rio". Each approval becomes a Grant that
records a hash of (tool name + exact arguments). Immediately before executing, the agent calls `redeem` with the
arguments it is ABOUT to run; any difference, a replay, or an expired grant is refused.
"""

import copy
import hashlib
import json
import secrets
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass


def args_hash(tool_name: str, args: dict) -> str:
    canonical = json.dumps({"tool": tool_name, "args": args}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass(frozen=True)
class Grant:
    tool: str
    args_hash: str
    expires_at: float
    nonce: str


class ApprovalGate:
    """`approver(summary: dict) -> str` shows a human the summary and returns the string "approve" to approve.

    Anything else (a different word, an exception, a timeout, no answer) is a denial.
    """

    def __init__(self, approver, ttl_s: float = 120.0, timeout_s: float = 60.0, clock=time.monotonic):
        self.approver, self.ttl_s, self.timeout_s, self.clock = approver, ttl_s, timeout_s, clock
        self._used: set[str] = set()

    def request(self, spec, args: dict, extra: dict | None = None):
        """Ask the human. Returns a Grant, or None if not approved.

        The hash is taken BEFORE the approver runs, and the approver gets a deep copy. Otherwise an approver (or any
        code holding the dict) could change the arguments and the grant would match the changed version.
        """
        fingerprint = args_hash(spec.name, args)
        summary = {"tool": spec.name, "args": copy.deepcopy(args), "irreversible": spec.irreversible,
                   "side_effects": spec.side_effects, **(extra or {})}
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            answer = pool.submit(self.approver, summary).result(timeout=self.timeout_s)
        except (FutureTimeout, Exception):
            return None  # fail closed: no answer, timeout, or a crash all mean no
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
        if answer != "approve":
            return None
        return Grant(spec.name, fingerprint, self.clock() + self.ttl_s, secrets.token_hex(8))

    def redeem(self, grant, tool_name: str, args: dict) -> tuple[bool, str]:
        """Check the grant against the action about to run. A grant is consumed by any attempt to use it."""
        if grant is None:
            return False, "no grant"
        if grant.nonce in self._used:
            return False, "grant already used"
        self._used.add(grant.nonce)
        if self.clock() > grant.expires_at:
            return False, "grant expired"
        if grant.tool != tool_name or grant.args_hash != args_hash(tool_name, args):
            return False, "action differs from what was approved"
        return True, "ok"
