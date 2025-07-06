import tempfile
import csv
import pytest
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


def test_load_aliases_from_csv(tmp_path):
    cost_rows = [
        {"test_name": "bmp", "cpt_code": "101", "price": "20"},
    ]
    cost_path = tmp_path / "cost.csv"
    with open(cost_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
        writer.writeheader()
        writer.writerows(cost_rows)

    alias_rows = [
        {"alias": "basic metabolic panel", "canonical": "bmp"},
    ]
    alias_path = tmp_path / "aliases.csv"
    with open(alias_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["alias", "canonical"])
        writer.writeheader()
        writer.writerows(alias_rows)

    ce = CostEstimator.load_from_csv(str(cost_path))
    ce.load_aliases_from_csv(str(alias_path))

    assert ce.lookup_cost("basic metabolic panel").cpt_code == "101"


def test_coverage_check(tmp_path):
    cost_rows = [
        {"test_name": "cbc", "cpt_code": "100", "price": "1"},
    ]
    cost_path = tmp_path / "cost.csv"
    with open(cost_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price"]
        )
        writer.writeheader()
        writer.writerows(cost_rows)

    cms_rows = [
        {"cpt_code": "100"},
        {"cpt_code": "101"},
    ]
    cms_path = tmp_path / "cms.csv"
    with open(cms_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["cpt_code"])
        writer.writeheader()
        writer.writerows(cms_rows)

    with pytest.raises(ValueError):
        CostEstimator.load_from_csv(
            str(cost_path),
            cms_pricing_path=str(cms_path),
            coverage_threshold=0.98,
        )
