from dataclasses import dataclass
from .cost_estimator import CostEstimator
from .judge import Judge


@dataclass
class SessionResult:
    """Outcome of a session evaluation.

    Attributes
    ----------
    total_cost:
        Sum of test costs incurred during the session.
    score:
        Judgement score for the final diagnosis.
    """

    total_cost: float
    score: int


class Evaluator:
    """Score diagnoses and tally the cost of ordered tests."""

    def __init__(self, judge: Judge, cost_estimator: CostEstimator):
        """Create an evaluator with a judge and cost estimator.

        Parameters
        ----------
        judge:
            :class:`Judge` instance used to grade diagnoses.
        cost_estimator:
            :class:`CostEstimator` used to compute the price of tests.
        """

        self.judge = judge
        self.cost_estimator = cost_estimator

    def evaluate(
        self, diagnosis: str, truth: str, tests: list[str]
    ) -> SessionResult:
        judgement = self.judge.evaluate(diagnosis, truth)
        total_cost = sum(
            self.cost_estimator.estimate_cost(t) for t in tests
        )
        return SessionResult(total_cost=total_cost, score=judgement.score)
