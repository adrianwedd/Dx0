import csv
import tempfile
import pytest
from sdb.cost_estimator import CostEstimator
from sdb import cost_estimator as ce_mod


def test_lookup_and_estimate(monkeypatch):
    data = [
        {"test_name": "cbc", "cpt_code": "100", "price": "10", "category": "labs"},
        {"test_name": "bmp", "cpt_code": "101", "price": "20", "category": "imaging"},
    ]
    with tempfile.NamedTemporaryFile("w", newline="", delete=False) as f:
        writer = csv.DictWriter(
            f, fieldnames=["test_name", "cpt_code", "price", "category"]
        )
        writer.writeheader()
        writer.writerows(data)
        path = f.name
    ce = CostEstimator.load_from_csv(path)
    ce.add_aliases({"basic metabolic panel": "bmp"})

    monkeypatch.setattr(ce_mod, "lookup_cpt", lambda name: "101")

    assert ce.lookup_cost("cbc").price == 10.0
    assert ce.lookup_cost("cbc").category == "labs"
    assert ce.lookup_cost("basic metabolic panel").cpt_code == "101"
    assert ce.estimate_cost("unknown") == 20.0
    assert ce.estimate("unknown") == (20.0, "imaging")
    assert ce.aliases["unknown"] == "bmp"
    # Second call should use cached alias and avoid LLM lookup
    monkeypatch.setattr(
        ce_mod,
        "lookup_cpt",
        lambda name: (_ for _ in ()).throw(AssertionError),
    )
    assert ce.estimate_cost("unknown") == 20.0


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

    with pytest.raises(ValueError) as exc:
        CostEstimator.load_from_csv(
            str(cost_path),
            cms_pricing_path=str(cms_path),
            coverage_threshold=1.0,
        )
    assert "101" in str(exc.value)


def test_lookup_cost_missing():
    ce = CostEstimator({})
    with pytest.raises(KeyError):
        ce.lookup_cost("unknown")


def test_plugin_loader(monkeypatch):
    captured: dict[str, str] = {}

    def dummy_factory(path: str) -> CostEstimator:
        captured["path"] = path
        return CostEstimator({})

    class DummyEP:
        name = "dummy"
        value = "dummy:factory"
        group = "dx0.cost_estimators"

        def load(self):
            return dummy_factory

    monkeypatch.setattr(
        ce_mod.metadata,
        "entry_points",
        lambda group=None: [DummyEP()] if group == "dx0.cost_estimators" else [],
    )
    estimator = ce_mod.load_cost_estimator("table.csv", plugin_name="dummy")
    assert isinstance(estimator, CostEstimator)
    assert captured["path"] == "table.csv"
