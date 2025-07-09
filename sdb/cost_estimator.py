import csv
from dataclasses import dataclass
from typing import Dict, Optional

from .cpt_lookup import lookup_cpt


@dataclass
class CptCost:
    """Mapping between a CPT code and its price."""

    cpt_code: str

    price: float


class CostEstimator:
    """Map tests to CPT codes and prices.

    Attributes
    ----------
    cost_table:
        Mapping of canonical test names to :class:`CptCost` entries.
    aliases:
        Mapping of alias names to canonical test names.
    unmatched_codes:
        List of CPT codes present in the CMS pricing table but not found in the
        provided mapping.
    match_rate:
        Fraction of CMS CPT codes matched by the provided table. ``1.0`` means
        100% coverage.
    """

    def __init__(self, cost_table: Dict[str, CptCost]):
        """Initialize estimator with a pricing table.

        Parameters
        ----------
        cost_table:
            Mapping of test names to :class:`CptCost` records.
        """

        self.cost_table = {k.lower(): v for k, v in cost_table.items()}
        self.aliases: Dict[str, str] = {}
        self.unmatched_codes: list[str] = []
        self.match_rate: float = 1.0

    @staticmethod
    def load_from_csv(
        path: str,
        cms_pricing_path: Optional[str] = None,
        coverage_threshold: float = 0.98,
        report_path: Optional[str] = None,
    ) -> "CostEstimator":
        """Load CPT pricing table from CSV and optionally verify coverage.

        Parameters
        ----------
        path:
            CSV mapping test names to CPT codes and prices.
        cms_pricing_path:
            Optional path to the CMS pricing table containing ``cpt_code``
            entries. When provided the coverage of this pricing table is
            checked against the loaded data.
        coverage_threshold:
            Minimum fraction of CMS codes that must be present in ``path``.
        report_path:
            Optional path to write a CSV report of unmatched CPT codes.
        """

        table: Dict[str, CptCost] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    name = row["test_name"].strip().lower()
                    cpt = row["cpt_code"].strip()
                    price = float(row["price"])
                except Exception:
                    continue
                if name and cpt:
                    table[name] = CptCost(cpt_code=cpt, price=price)

        estimator = CostEstimator(table)

        if cms_pricing_path:
            cms_codes: set[str] = set()
            with open(cms_pricing_path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    code = row.get("cpt_code", "").strip()
                    if code:
                        cms_codes.add(code)

            provided_codes = {c.cpt_code for c in table.values()}
            matched = cms_codes & provided_codes
            estimator.unmatched_codes = sorted(cms_codes - matched)
            if cms_codes:
                estimator.match_rate = len(matched) / len(cms_codes)
            else:
                estimator.match_rate = 1.0

            if report_path is not None:
                with open(
                    report_path,
                    "w",
                    newline="",
                    encoding="utf-8",
                ) as fh:
                    writer = csv.writer(fh)
                    writer.writerow(["cpt_code"])
                    for code in estimator.unmatched_codes:
                        writer.writerow([code])

            if estimator.match_rate < coverage_threshold:
                msg = (
                    f"Coverage {estimator.match_rate:.1%} below required "
                    f"{coverage_threshold:.1%}"
                )
                raise ValueError(msg)

        return estimator

    def lookup_cost(self, test_name: str) -> CptCost:
        """Return CPT pricing information for ``test_name``.

        Parameters
        ----------
        test_name:
            Name of the requested test.

        Returns
        -------
        CptCost
            Pricing entry for the test.

        Raises
        ------
        KeyError
            If ``test_name`` is not found in the pricing table or aliases.
        """

        key = test_name.strip().lower()
        if key in self.aliases:
            key = self.aliases[key]
        try:
            return self.cost_table[key]
        except KeyError as exc:
            raise KeyError(f"Unknown test name: {test_name}") from exc

    def add_aliases(self, mapping: Dict[str, str]):
        """Register mapping of free text requests to canonical test names."""
        for k, v in mapping.items():
            self.aliases[k.lower()] = v.lower()

    def load_aliases_from_csv(self, path: str) -> None:
        """Load alias mapping from CSV file.

        The CSV should provide ``alias`` and ``canonical`` columns. Rows
        missing these fields are skipped.
        """
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    alias = row["alias"].strip().lower()
                    canonical = row["canonical"].strip().lower()
                except Exception:
                    continue
                if not alias or not canonical:
                    continue
                self.aliases[alias] = canonical

    def estimate_cost(self, test_name: str) -> float:
        """Return the price for ``test_name`` or infer it via an LLM lookup."""

        try:
            tc = self.lookup_cost(test_name)
        except KeyError:
            tc = None
        if tc:
            return tc.price

        # Attempt LLM-based CPT lookup when a direct mapping is missing.
        code = lookup_cpt(test_name)
        if code:
            # Find any canonical entry with this CPT code.
            for canonical, entry in self.cost_table.items():
                if entry.cpt_code == code:
                    # Cache alias to avoid future LLM queries.
                    self.aliases[test_name.strip().lower()] = canonical
                    return entry.price

        # Fallback to an average of known prices when no mapping is found.
        if self.cost_table:
            avg = sum(e.price for e in self.cost_table.values()) / len(
                self.cost_table
            )
            return avg
        return 0.0
