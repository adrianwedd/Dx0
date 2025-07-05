from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TestCost:
    cpt_code: str
    price: float

class CostEstimator:
    """Map tests to CPT codes and prices."""

    def __init__(self, cost_table: Dict[str, TestCost]):
        self.cost_table = cost_table

    def lookup_cost(self, test_name: str) -> Optional[TestCost]:
        return self.cost_table.get(test_name)

    def estimate_cost(self, test_name: str) -> float:
        tc = self.lookup_cost(test_name)
        if tc:
            return tc.price
        # TODO: use language model to estimate missing costs
        return 0.0
