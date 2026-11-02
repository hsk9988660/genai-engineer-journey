"""Latency as a distribution, not one number.

The lesson's assignment says to time each response once. One timing includes loading the model, filling caches and whatever else the machine was doing. This records many runs,
discards warm-up runs, and reports the median, the 95th percentile and the spread, plus tokens per second when the server reports completion tokens.
Time to first token needs a STREAMING response, which this client does not use, so it is not measured here.
"""

import math
import statistics
import time


class LatencyStats:
    def __init__(self):
        self.samples: list[tuple[float, int | None]] = []

    def record(self, seconds: float, completion_tokens: int | None = None) -> None:
        if seconds < 0:
            raise ValueError("a duration cannot be negative")
        self.samples.append((float(seconds), completion_tokens))

    def timed(self, fn, clock=time.monotonic):
        t0 = clock()
        out = fn()
        self.record(clock() - t0)
        return out

    def summary(self, discard_warmup: int = 1) -> dict:
        data = self.samples[discard_warmup:]
        if not data:
            return {"n": 0, "discarded": min(discard_warmup, len(self.samples))}
        secs = sorted(s for s, _ in data)
        rank = lambda p: secs[max(0, math.ceil(p / 100 * len(secs)) - 1)]
        out = {"n": len(secs), "discarded": discard_warmup, "median_s": round(statistics.median(secs), 4), "p95_s": round(rank(95), 4), "min_s": round(secs[0], 4), "max_s": round(secs[-1], 4)}
        if len(secs) >= 4:
            q = statistics.quantiles(secs, n=4)
            out["iqr_s"] = round(q[2] - q[0], 4)
        rates = [t / s for s, t in data if t and s > 0]
        if rates:
            out["tokens_per_s_median"] = round(statistics.median(rates), 2)
        return out
