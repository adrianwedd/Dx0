from dataclasses import dataclass
from .cost_estimator import CostEstimator
from .judge import Judge

@dataclass
class SessionResult:
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
