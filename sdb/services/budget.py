from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

from .budget_store import BudgetStore
from ..ui.session_store import SessionStore

from ..cost_estimator import CostEstimator
from ..metrics import BUDGET_REMAINING, BUDGET_SPENT


@dataclass
class BudgetManager:
    """Track test spending and enforce optional per-category budgets."""

    cost_estimator: Optional[CostEstimator] = None
    budget: Optional[float] = None
    category_limits: Dict[str, float] | None = None
    store: BudgetStore | None = None
    session_db: SessionStore | None = None
    session_token: str | None = None
    spent: float = 0.0
    spent_by_category: Dict[str, float] = field(default_factory=dict)
    test_categories: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.store is not None:
            self.spent = self.store.total()
        if self.session_db is not None and self.session_token is not None:
            limit, spent = self.session_db.get_budget(self.session_token)
            if limit is not None:
                self.budget = limit
            self.spent = spent

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
        BUDGET_SPENT.inc(amount)
        if self.budget is not None:
            remaining = max(self.budget - self.spent, 0.0)
            BUDGET_REMAINING.set(remaining)
        if self.store is not None:
            self.store.record(test_name, amount)
        if self.session_db is not None and self.session_token is not None:
            self.session_db.update_spent(self.session_token, self.spent)

    def over_budget(self) -> bool:
        """Return ``True`` if any spending limit was exceeded."""

        if self.budget is not None and self.spent >= self.budget:
            return True
        if self.category_limits:
            for cat, limit in self.category_limits.items():
                if self.spent_by_category.get(cat, 0.0) >= limit:
                    return True
        return False
