"""The statistics that stop you shipping noise.

With 50 test cases and a true success rate near 80%, the 95% interval is about +/- 11 points, so "86% vs 80%" on 50
cases is inside the noise. These helpers make that visible.
"""

import math
import random


def normal_half_width(p: float, n: int, z: float = 1.96) -> float:
    """The quick back-of-envelope interval half-width. Fine for illustration, poor near 0% or 100% or with small n."""
    return z * math.sqrt(p * (1 - p) / n)


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A better interval for a pass rate than the normal approximation: stays inside [0, 1] and behaves with small n."""
    if n == 0:
        return 0.0, 1.0
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def paired_bootstrap(a: list[float], b: list[float], iters: int = 2000, seed: int = 0) -> dict:
    """Compare two versions scored on the SAME cases. Returns the mean difference (b - a) and a 95% interval.

    Pairing matters: it cancels out which cases happen to be easy, so a smaller real difference becomes detectable.
    """
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty lists of equal length (one score per case)")
    rng = random.Random(seed)
    diffs = [y - x for x, y in zip(a, b)]
    n = len(diffs)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(iters))
    return {"mean_diff": sum(diffs) / n, "ci_low": means[int(0.025 * iters)], "ci_high": means[int(0.975 * iters) - 1],
            "p_b_better": sum(m > 0 for m in means) / iters}


def cohens_kappa(labels_a: list, labels_b: list) -> float:
    """Agreement between two labelers beyond chance: 1 is perfect, 0 is what random guessing would give."""
    if len(labels_a) != len(labels_b) or not labels_a:
        raise ValueError("need two non-empty lists of equal length")
    n = len(labels_a)
    observed = sum(x == y for x, y in zip(labels_a, labels_b)) / n
    categories = set(labels_a) | set(labels_b)
    expected = sum((labels_a.count(c) / n) * (labels_b.count(c) / n) for c in categories)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)
