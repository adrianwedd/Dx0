from sdb.services import BudgetManager, ResultAggregator
from sdb.metrics import BUDGET_REMAINING, BUDGET_SPENT


class DummyEstimator:
    def estimate_cost(self, _name: str) -> float:
        return 5.0

    def estimate(self, _name: str) -> tuple[float, str]:
        return 5.0, "labs"


def test_budget_manager_over_budget():
    bm = BudgetManager(DummyEstimator(), budget=9.0)
    bm.add_test("cbc")
    assert bm.spent == 5.0
    assert bm.spent_by_category["labs"] == 5.0
    assert not bm.over_budget()
    bm.add_test("bmp")
    assert bm.spent == 10.0
    assert bm.spent_by_category["labs"] == 10.0
    assert bm.over_budget()


def test_result_aggregator_records():
    agg = ResultAggregator()
    agg.record_test("cbc")
    agg.record_test("bmp")
    assert agg.ordered_tests == ["cbc", "bmp"]
    assert not agg.finished
    agg.record_diagnosis("flu")
    assert agg.final_diagnosis == "flu"
    assert agg.finished


def test_budget_metrics_update():
    BUDGET_SPENT.set(0)
    BUDGET_REMAINING.set(0)
    bm = BudgetManager(DummyEstimator(), budget=10.0)
    bm.add_test("cbc")
    assert BUDGET_SPENT._value.get() == 5.0
    assert BUDGET_REMAINING._value.get() == 5.0
