import csv
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TestCost:
    cpt_code: str
    price: float

class CostEstimator:
    """Map tests to CPT codes and prices."""

    def __init__(self, cost_table: Dict[str, TestCost]):
        self.cost_table = {k.lower(): v for k, v in cost_table.items()}
        self.aliases: Dict[str, str] = {}

    @staticmethod
    def load_from_csv(path: str) -> "CostEstimator":
        """Load CPT pricing table from CSV.

        The CSV file is expected to contain ``test_name``, ``cpt_code`` and
        ``price`` columns. Rows that are missing data are skipped.
        """
        table: Dict[str, TestCost] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    name = row["test_name"].strip().lower()
                    cpt = row["cpt_code"].strip()
                    price = float(row["price"])
                except Exception:
                    continue
                table[name] = TestCost(cpt_code=cpt, price=price)
        return CostEstimator(table)

    def lookup_cost(self, test_name: str) -> Optional[TestCost]:
        key = test_name.strip().lower()
        if key in self.aliases:
            key = self.aliases[key]
        return self.cost_table.get(key)

    def add_aliases(self, mapping: Dict[str, str]):
        """Register mapping of free text requests to canonical test names."""
        for k, v in mapping.items():
            self.aliases[k.lower()] = v.lower()

    def estimate_cost(self, test_name: str) -> float:
        tc = self.lookup_cost(test_name)
        if tc:
            return tc.price
        # Placeholder fallback cost estimation when CPT code is unknown.
        # In a full system this could call a language model. Here we return
        # an average price based on known tests if available.
        if self.cost_table:
            avg = sum(tc.price for tc in self.cost_table.values()) / len(self.cost_table)
            return avg
        return 0.0
