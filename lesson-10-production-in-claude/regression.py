"""The release gate: is the new version safe to ship, given noisy results?

Each case is run several times per version, so a case has a pass RATE, not just pass/fail. The gate then:
  * BLOCKS if any safety-tagged case got worse, no matter how much the average improved
  * BLOCKS if the new version is significantly worse overall
  * says SHIP only if, with 95% confidence, it is not worse than `tolerance` (a non-inferiority check)
  * otherwise says INCONCLUSIVE and asks for more cases or more runs, instead of guessing

A pass^k style consistency number is also provided: the share of cases that pass on ALL of their first k runs. An agent
that passes a case 8 times out of 10 is not reliable, whatever its average says.
"""

from dataclasses import dataclass, field

from stats import paired_bootstrap, wilson_interval


def per_case_pass_rates(results: dict[str, list[bool]]) -> dict[str, float]:
    return {cid: sum(runs) / len(runs) for cid, runs in results.items() if runs}


def consistency_pass_all_k(results: dict[str, list[bool]], k: int) -> float:
    eligible = [runs for runs in results.values() if len(runs) >= k]
    return sum(all(r[:k]) for r in eligible) / len(eligible) if eligible else 0.0


@dataclass
class Gate:
    tolerance: float = 0.02  # how much worse we are willing to be, in pass-rate points
    safety_tags: frozenset = frozenset({"safety"})
    min_cases: int = 30


@dataclass
class Decision:
    status: str  # ship | block | inconclusive
    reasons: list = field(default_factory=list)
    mean_diff: float = 0.0
    ci: tuple = (0.0, 0.0)
    cases: int = 0
    pass_rate_a: float = 0.0
    pass_rate_b: float = 0.0


def compare(results_a: dict[str, list[bool]], results_b: dict[str, list[bool]], tags: dict[str, tuple] | None = None,
            gate: Gate | None = None, seed: int = 0) -> Decision:
    """`a` is the current version and `b` the candidate, scored on the same cases."""
    gate, tags = gate or Gate(), tags or {}
    ra, rb = per_case_pass_rates(results_a), per_case_pass_rates(results_b)
    common = sorted(set(ra) & set(rb))
    if not common:
        return Decision("inconclusive", ["no cases in common between the two versions"])
    a, b = [ra[c] for c in common], [rb[c] for c in common]
    boot = paired_bootstrap(a, b, seed=seed)
    d = Decision("inconclusive", [], boot["mean_diff"], (boot["ci_low"], boot["ci_high"]), len(common),
                 sum(a) / len(a), sum(b) / len(b))

    regressed = [c for c in common if set(tags.get(c, ())) & gate.safety_tags and rb[c] < ra[c]]
    if regressed:
        d.status = "block"
        d.reasons.append(f"safety regression on {regressed}: a safety case got worse, regardless of the average")
    if boot["ci_high"] < 0 and boot["mean_diff"] < -gate.tolerance:
        d.status = "block"
        d.reasons.append(f"significantly worse overall: {boot['mean_diff']:+.1%} (95% CI {boot['ci_low']:+.1%} to {boot['ci_high']:+.1%})")
    if d.status == "block":
        return d
    if len(common) < gate.min_cases:
        d.reasons.append(f"only {len(common)} cases, fewer than the {gate.min_cases} needed to trust a comparison")
        return d
    if boot["ci_low"] >= -gate.tolerance:
        d.status = "ship"
        d.reasons.append(f"not worse than {gate.tolerance:.0%} with 95% confidence: {boot['mean_diff']:+.1%} "
                         f"(95% CI {boot['ci_low']:+.1%} to {boot['ci_high']:+.1%})")
    else:
        d.reasons.append(f"cannot rule out a drop larger than {gate.tolerance:.0%}: {boot['mean_diff']:+.1%} "
                         f"(95% CI {boot['ci_low']:+.1%} to {boot['ci_high']:+.1%}). Add cases or runs.")
    return d


def pass_rate_interval(results: dict[str, list[bool]]) -> tuple[float, float, float]:
    """Overall pass rate over every run, with a Wilson 95% interval."""
    flat = [r for runs in results.values() for r in runs]
    low, high = wilson_interval(sum(flat), len(flat))
    return (sum(flat) / len(flat) if flat else 0.0), low, high
