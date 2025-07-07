from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence


@dataclass
class DiagnosisResult:
    """Result of a panel run with associated confidence and cost."""

    diagnosis: str
    confidence: float
    cost: float = 0.0
    run_id: str | None = None


class WeightedVoter:
    """Aggregate diagnoses using weighted voting."""

    def vote(
        self,
        results: Iterable[DiagnosisResult],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> str:
        """Return the diagnosis with the highest weighted score.

        Parameters
        ----------
        results:
            Iterable of ``DiagnosisResult`` objects.
        weights:
            Optional mapping from ``run_id`` to vote weight.

        Returns
        -------
        str
            The diagnosis with the highest weighted score. Returns an empty
            string if ``results`` is empty.
        """

        res_list = list(results)

        scores: dict[str, float] = defaultdict(float)
        for res in res_list:
            weight = 1.0
            if weights is not None and res.run_id is not None:
                weight = weights.get(res.run_id, 1.0)
            scores[res.diagnosis] += res.confidence * weight

        if not scores:
            return ""
        return max(scores.items(), key=lambda x: x[1])[0]


def cost_adjusted_selection(
    results: Iterable[DiagnosisResult], *, cost_weight: float = 0.001
) -> str:
    """Choose diagnosis maximizing confidence minus cost penalty."""
    best_diag = ""
    best_score = float("-inf")
    for res in results:
        score = res.confidence - cost_weight * res.cost
        if score > best_score:
            best_score = score
            best_diag = res.diagnosis
    return best_diag


class MetaPanel:
    """Synthesize a final diagnosis from multiple panel runs."""

    def __init__(self, voter: WeightedVoter | None = None):
        self.voter = voter or WeightedVoter()

    def synthesize(self, results: Sequence[DiagnosisResult]) -> str:
        """Return the weighted-vote winner from ``results``."""
        return self.voter.vote(results)
