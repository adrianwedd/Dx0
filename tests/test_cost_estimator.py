import tempfile
import csv
from sdb.cost_estimator import CostEstimator


def test_lookup_and_estimate():
    data = [
        {"test_name": "cbc", "cpt_code": "100", "price": "10"},
        {"test_name": "bmp", "cpt_code": "101", "price": "20"},
    ]
    with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
        writer.writeheader()
        writer.writerows(data)
        path = f.name
    ce = CostEstimator.load_from_csv(path)
    ce.add_aliases({"basic metabolic panel": "bmp"})

    assert ce.lookup_cost("cbc").price == 10.0
    assert ce.lookup_cost("basic metabolic panel").cpt_code == "101"
    # Unknown test uses average of known prices => (10+20)/2=15
    assert ce.estimate_cost("unknown") == 15.0
