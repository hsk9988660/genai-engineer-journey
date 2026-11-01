"""Which traces to keep. Summaries are kept for every run; full traces are expensive and sensitive, so they are sampled.

Rules: ALWAYS keep the runs you will want to debug (errors, escalations, thumbs-down, fallbacks, expensive or slow runs),
and a fixed share of the rest. The share is decided by a hash of the run id, so the decision is deterministic: the whole
trace is kept or dropped together, and replaying the same id gives the same answer.
"""

import hashlib
from dataclasses import dataclass, field


@dataclass
class SamplingPolicy:
    rate: float = 0.1
    keep_cost_usd_above: float | None = None
    keep_latency_ms_above: float | None = None
    keep_outcomes: frozenset = field(default_factory=lambda: frozenset({"error", "abandoned", "escalated"}))
    keep_negative_feedback: bool = True
    keep_fallbacks: bool = True

    @staticmethod
    def bucket(run_id: str) -> float:
        """A stable number in [0, 1) derived from the run id."""
        return int(hashlib.sha256(run_id.encode()).hexdigest()[:8], 16) / 0x100000000

    def decide(self, rec) -> tuple[bool, str]:
        if rec.outcome in self.keep_outcomes:
            return True, f"outcome={rec.outcome}"
        if self.keep_negative_feedback and rec.feedback == "down":
            return True, "negative feedback"
        if self.keep_fallbacks and rec.fallbacks > 0:
            return True, "fallback used"
        if self.keep_cost_usd_above is not None and rec.cost_usd > self.keep_cost_usd_above:
            return True, "expensive run"
        if self.keep_latency_ms_above is not None and rec.latency_ms > self.keep_latency_ms_above:
            return True, "slow run"
        if self.bucket(rec.run_id) < self.rate:
            return True, f"random sample ({self.rate:.0%})"
        return False, "not sampled"
