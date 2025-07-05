import difflib
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class Judgement:
    score: int
    explanation: str

class Judge:
    """Evaluate diagnosis with physician-authored rubric."""

    def __init__(self, rubric: Dict[str, Any]):
        """Create a judge with a scoring rubric.

        The rubric may define ``exact_threshold`` and ``partial_threshold``
        similarity ratios. Defaults are 0.9 and 0.6.
        """
        self.rubric = rubric
        self.exact_threshold = float(rubric.get("exact_threshold", 0.9))
        self.partial_threshold = float(rubric.get("partial_threshold", 0.6))

    def evaluate(self, diagnosis: str, truth: str) -> Judgement:
        """Score the diagnosis against the truth and return judgement."""
        d = diagnosis.strip().lower()
        t = truth.strip().lower()
        ratio = difflib.SequenceMatcher(None, d, t).ratio()

        if ratio >= self.exact_threshold:
            score = 5
            explanation = "Exact or near exact match"
        elif ratio >= self.partial_threshold:
            score = 4
            explanation = "Reasonable partial match"
        elif d and t and (d in t or t in d):
            score = 3
            explanation = "Minor overlap"
        elif ratio > 0.3:
            score = 2
            explanation = "Poor match"
        else:
            score = 1
            explanation = "Incorrect diagnosis"

        return Judgement(score=score, explanation=explanation)
