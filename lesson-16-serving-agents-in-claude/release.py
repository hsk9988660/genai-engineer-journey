"""Releasing an agent version: an evaluation gate that knows how little it knows, and a canary rollout that can undo itself.

The lesson's gate runs FOUR test cases and ships if at least 80% pass. With four cases the pass rate can only be 0, 25, 50, 75 or 100%, so one flaky answer decides the release.
`naive_gate_error_rates` computes, exactly, how often that rule blocks a good agent and ships a bad one. The gate here uses a Wilson confidence interval and has THREE outcomes:
  pass            we are confident (95%) the true pass rate is at least the requirement
  fail            we are confident it is below the requirement
  inconclusive    not enough cases to tell. More testing is needed, so it does NOT ship
A canary sends a small, STABLE slice of users (by hash of the user id) to the new version, promotes through stages while it is not CONFIDENTLY worse than the old one, and rolls back when it is.
Promotion means "no evidence of harm yet", not "proven as good": a small difference needs far more traffic than a canary stage sees.
"""

import hashlib
import math
import threading
from dataclasses import dataclass


def wilson_interval(passes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return 0.0, 1.0
    p = passes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


@dataclass(frozen=True)
class GateResult:
    decision: str  # pass | fail | inconclusive
    passes: int
    n: int
    lower: float
    upper: float
    reason: str

    @property
    def ship(self) -> bool:
        return self.decision == "pass"


def gate(passes: int, n: int, required: float = 0.8, min_cases: int = 30) -> GateResult:
    if not 0 <= passes <= n:
        raise ValueError("passes must be between 0 and n")
    lo, hi = wilson_interval(passes, n)
    if n < min_cases:
        return GateResult("inconclusive", passes, n, lo, hi, f"only {n} cases; at least {min_cases} are needed")
    if lo >= required:
        return GateResult("pass", passes, n, lo, hi, f"95% sure the pass rate is at least {required:.0%}")
    if hi < required:
        return GateResult("fail", passes, n, lo, hi, f"95% sure the pass rate is below {required:.0%}")
    return GateResult("inconclusive", passes, n, lo, hi, f"the pass rate could be on either side of {required:.0%}: run more cases")


def evaluate(run_case, cases: list, required: float = 0.8, min_cases: int = 30) -> GateResult:
    """`run_case(case) -> bool` should exercise the WHOLE serving path (routing, cache, approval), not just one model."""
    passes = sum(1 for c in cases if run_case(c))
    return gate(passes, len(cases), required, min_cases)


def naive_gate_error_rates(true_rate_good: float, true_rate_bad: float, n: int = 4, need: float = 0.8) -> dict:
    """Exact probabilities for the lesson's rule (ship if passes/n >= need): how often it blocks a GOOD agent and ships a BAD one."""
    def p_at_least(p: float) -> float:
        k = math.ceil(need * n - 1e-9)
        return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))
    return {"good_agent_blocked": 1 - p_at_least(true_rate_good), "bad_agent_shipped": p_at_least(true_rate_bad)}


def bucket(user_id: str, salt: str) -> int:
    """A stable number 0..99 for this user. The same user always lands in the same bucket, so their experience does not flip between versions."""
    return int.from_bytes(hashlib.sha256(f"{salt}:{user_id}".encode()).digest()[:4], "big") % 100


class Rollout:
    STAGES = (1, 10, 50, 100)

    def __init__(self, stable: str, candidate: str, salt: str = "rollout", min_samples: int = 50, max_extra_error: float = 0.05, stages: tuple = STAGES):
        self.stable, self.candidate, self.salt, self.min_samples, self.max_extra_error, self.STAGES = stable, candidate, salt, min_samples, max_extra_error, tuple(stages)
        self._lock = threading.Lock()
        self.stage, self.state = 0, "running"  # running | promoted | rolled_back
        self._n = {stable: 0, candidate: 0}
        self._bad = {stable: 0, candidate: 0}
        self.history = []

    @property
    def percent(self) -> int:
        return 100 if self.state == "promoted" else 0 if self.state == "rolled_back" else self.STAGES[self.stage]

    def version_for(self, user_id: str) -> str:
        return self.candidate if bucket(user_id, self.salt) < self.percent else self.stable

    def observe(self, version_id: str, ok: bool) -> None:
        with self._lock:
            if version_id in self._n:
                self._n[version_id] += 1
                self._bad[version_id] += 0 if ok else 1

    def error_rate(self, version_id: str) -> float | None:
        with self._lock:
            return self._bad[version_id] / self._n[version_id] if self._n[version_id] else None

    def evaluate(self) -> str:
        """Call periodically. Returns waiting | advanced | promoted | rolled_back | done."""
        with self._lock:
            if self.state != "running":
                return "done"
            n_c = self._n[self.candidate]
            if n_c < self.min_samples:
                return "waiting"
            err_c = self._bad[self.candidate] / n_c
            n_s = self._n[self.stable]
            err_s = self._bad[self.stable] / n_s if n_s else 0.0
            lower_c, _ = wilson_interval(self._bad[self.candidate], n_c)
            # roll back only when we are 95% sure the candidate's TRUE error rate is above the stable one plus the margin. One bad answer in eight is not evidence
            if lower_c > err_s + self.max_extra_error:
                self.state = "rolled_back"
                self.history.append(("rolled_back", self.STAGES[self.stage], err_c, err_s))
                return "rolled_back"
            self.history.append(("healthy", self.STAGES[self.stage], err_c, err_s))
            if self.stage + 1 >= len(self.STAGES):
                self.state = "promoted"
                return "promoted"
            self.stage += 1
            self._n = {self.stable: 0, self.candidate: 0}  # each stage has to prove itself with fresh traffic
            self._bad = {self.stable: 0, self.candidate: 0}
            return "advanced"
