from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ..cost_estimator import CostEstimator


@dataclass
class BudgetManager:
    """Track test spending and enforce an optional budget."""

    cost_estimator: Optional[CostEstimator] = None
    budget: Optional[float] = None
    spent: float = 0.0

    def add_test(self, test_name: str) -> None:
        """Record the cost of ``test_name`` using ``cost_estimator``."""
        if self.cost_estimator:
            self.spent += self.cost_estimator.estimate_cost(test_name)

    def over_budget(self) -> bool:
        """Return ``True`` if the budget was exceeded."""
        return self.budget is not None and self.spent >= self.budget
