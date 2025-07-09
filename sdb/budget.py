from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .cost_estimator import CostEstimator


@dataclass
class BudgetTracker:
    """Accumulate spending on ordered tests and enforce a limit."""

    cost_estimator: Optional[CostEstimator] = None
    budget: Optional[float] = None
    spent: float = 0.0

    def add(self, test_name: str) -> None:
        """Add the cost of ``test_name`` to the total spent."""

        if self.cost_estimator:
            self.spent += self.cost_estimator.estimate_cost(test_name)

    def over_budget(self) -> bool:
        """Return ``True`` if ``spent`` exceeds ``budget``."""

        return self.budget is not None and self.spent >= self.budget
