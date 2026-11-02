"""Cost accounting: what each request cost, who spent it, and a budget that stops spending.

Prices come from the agent definition (`Tier.in_price` / `out_price`, per 1,000 tokens). The numbers in this lesson are PLACEHOLDERS: they make the arithmetic testable,
they are not real prices. Money is `Decimal`, never float. A budget is checked BEFORE a request runs, and cache hits are recorded as cost 0 so the hit rate is visible.
"""

import threading
from collections import Counter, defaultdict
from datetime import date
from decimal import Decimal

from agentdef import Tier


class BudgetExceeded(Exception):
    def __init__(self, who: str):
        super().__init__(f"{who} budget used up for today")
        self.who = who


def request_cost(tier: Tier, input_tokens: int, output_tokens: int) -> Decimal:
    return (Decimal(input_tokens) * tier.in_price + Decimal(output_tokens) * tier.out_price) / Decimal(1000)


class CostLedger:
    def __init__(self, user_daily_budget: Decimal = Decimal("1.00"), global_daily_budget: Decimal = Decimal("100.00"), today=lambda: date.today().isoformat()):
        self.user_budget, self.global_budget, self.today = Decimal(user_daily_budget), Decimal(global_daily_budget), today
        self._lock = threading.Lock()
        self._user = defaultdict(Decimal)  # (day, user) -> spent
        self._total = defaultdict(Decimal)  # day -> spent
        self.by_tier = Counter()
        self.cost_by_tier = defaultdict(Decimal)
        self.cache_hits = 0

    def check(self, user: str) -> None:
        with self._lock:
            d = self.today()
            if self._total[d] >= self.global_budget:
                raise BudgetExceeded("global")
            if self._user[(d, user)] >= self.user_budget:
                raise BudgetExceeded("user")

    def record(self, user: str, tier: Tier, input_tokens: int, output_tokens: int) -> Decimal:
        cost = request_cost(tier, input_tokens, output_tokens)
        with self._lock:
            d = self.today()
            self._user[(d, user)] += cost
            self._total[d] += cost
            self.by_tier[tier.name] += 1
            self.cost_by_tier[tier.name] += cost
        return cost

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    def spent(self, user: str) -> Decimal:
        with self._lock:
            return self._user[(self.today(), user)]

    def spent_total(self) -> Decimal:
        with self._lock:
            return self._total[self.today()]

    def report(self) -> dict:
        with self._lock:
            return {"requests_by_tier": dict(self.by_tier), "served_from_cache": self.cache_hits, "cost_by_tier": {k: str(v) for k, v in self.cost_by_tier.items()},
                    "total_cost": str(sum(self.cost_by_tier.values(), Decimal(0)))}
