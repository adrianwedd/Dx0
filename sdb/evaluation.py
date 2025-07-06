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
    def __init__(self, judge: Judge, cost_estimator: CostEstimator):
        self.judge = judge
        self.cost_estimator = cost_estimator

    def evaluate(self, diagnosis: str, truth: str, tests: list) -> SessionResult:
        judgement = self.judge.evaluate(diagnosis, truth)
        total_cost = sum(self.cost_estimator.estimate_cost(t) for t in tests)
        return SessionResult(total_cost=total_cost, score=judgement.score)
