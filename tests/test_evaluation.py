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
    result = ev.evaluate("flu", "flu", ["cbc", "bmp"], visits=2)
    assert result.score == 4
    assert result.correct
    assert result.total_cost == 630.0


def test_evaluator_zero_tests_cost():
    judge = DummyJudge(score=5)
    coster = DummyCostEstimator({})
    ev = Evaluator(judge, coster)
    result = ev.evaluate("x", "x", [], visits=3)
    assert result.score == 5
    assert result.correct
    assert result.total_cost == 900.0


def test_evaluator_correctness_threshold_default():
    judge = DummyJudge(score=3)
    coster = DummyCostEstimator({})
    ev = Evaluator(judge, coster)
    result = ev.evaluate("x", "x", [])
    assert not result.correct
    judge2 = DummyJudge(score=4)
    ev2 = Evaluator(judge2, coster)
    result2 = ev2.evaluate("x", "x", [])
    assert result2.correct


def test_evaluator_correctness_threshold_custom():
    judge = DummyJudge(score=3)
    coster = DummyCostEstimator({})
    ev = Evaluator(judge, coster, correct_threshold=3)
    result = ev.evaluate("x", "x", [])
    assert result.correct
    judge2 = DummyJudge(score=2)
    ev2 = Evaluator(judge2, coster, correct_threshold=3)
    result2 = ev2.evaluate("x", "x", [])
    assert not result2.correct
