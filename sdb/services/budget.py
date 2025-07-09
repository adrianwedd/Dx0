from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .budget_store import BudgetStore

from ..cost_estimator import CostEstimator


@dataclass
class BudgetManager:
    """Track test spending and enforce an optional budget."""

    cost_estimator: Optional[CostEstimator] = None
    budget: Optional[float] = None
    store: BudgetStore | None = None
    spent: float = 0.0

    def __post_init__(self) -> None:
        if self.store is not None:
            self.spent = self.store.total()

    def add_test(self, test_name: str) -> None:
        """Record the cost of ``test_name`` using ``cost_estimator``."""
        amount = 0.0
        if self.cost_estimator:
            amount = self.cost_estimator.estimate_cost(test_name)
        self.spent += amount
        if self.store is not None:
            self.store.record(test_name, amount)

    def over_budget(self) -> bool:
        """Return ``True`` if the budget was exceeded."""
        return self.budget is not None and self.spent >= self.budget
