from sdb.cost_estimator import CostEstimator, CptCost


def fixed_price_estimator(path: str) -> CostEstimator:
    """Return a simple estimator ignoring ``path``."""
    table = {"cbc": CptCost("100", 1.0)}
    return CostEstimator(table)
