from dataclasses import dataclass

from sdb.evaluation import Evaluator
from sdb.judge import Judgement


@dataclass
class DummyJudge:
    score: int

    def evaluate(self, diagnosis: str, truth: str) -> Judgement:
        return Judgement(score=self.score, explanation="")


class DummyCostEstimator:
    def __init__(self, costs):
        self.costs = costs

    def estimate_cost(self, test_name: str) -> float:
        return self.costs.get(test_name, 0.0)


def test_evaluator_aggregates_score_and_cost():
    judge = DummyJudge(score=4)
    coster = DummyCostEstimator({"cbc": 10.0, "bmp": 20.0})
    ev = Evaluator(judge, coster)
    result = ev.evaluate("flu", "flu", ["cbc", "bmp"])
    assert result.score == 4
    assert result.total_cost == 30.0


def test_evaluator_zero_tests_cost():
    judge = DummyJudge(score=5)
    coster = DummyCostEstimator({})
    ev = Evaluator(judge, coster)
    result = ev.evaluate("x", "x", [])
    assert result.score == 5
    assert result.total_cost == 0.0

