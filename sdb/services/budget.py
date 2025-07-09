from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .budget_store import BudgetStore

from ..cost_estimator import CostEstimator


@dataclass
class BudgetManager:
    """Track test spending and enforce optional per-category budgets."""

    cost_estimator: Optional[CostEstimator] = None
    budget: Optional[float] = None
    category_limits: Dict[str, float] | None = None
    store: BudgetStore | None = None
    spent: float = 0.0
    spent_by_category: Dict[str, float] = field(default_factory=dict)
    test_categories: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.store is not None:
            self.spent = self.store.total()

    def add_test(self, test_name: str) -> None:
        """Record the cost of ``test_name`` using ``cost_estimator``."""
        amount = 0.0
        category = "unknown"
        if self.cost_estimator:
            amount, category = self.cost_estimator.estimate(test_name)
        self.spent += amount
        self.spent_by_category[category] = (
            self.spent_by_category.get(category, 0.0) + amount
        )
        self.test_categories[test_name] = category
        if self.store is not None:
            self.store.record(test_name, amount)

    def over_budget(self) -> bool:
        """Return ``True`` if any spending limit was exceeded."""

        if self.budget is not None and self.spent >= self.budget:
            return True
        if self.category_limits:
            for cat, limit in self.category_limits.items():
                if self.spent_by_category.get(cat, 0.0) >= limit:
                    return True
        return False
