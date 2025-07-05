from dataclasses import dataclass
from typing import Dict

@dataclass
class Judgement:
    score: int
    explanation: str

class Judge:
    """Evaluate diagnosis with physician-authored rubric."""

    def __init__(self, rubric: Dict[str, Any]):
        self.rubric = rubric

    def evaluate(self, diagnosis: str, truth: str) -> Judgement:
        # TODO: implement scoring logic
        return Judgement(score=0, explanation="Not implemented")
