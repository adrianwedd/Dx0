import difflib
import re
from dataclasses import dataclass
from typing import Dict, Any

from .prompt_loader import load_prompt
from .llm_client import LLMClient, OpenAIClient


@dataclass
class Judgement:
    score: int
    explanation: str


class Judge:
    """Evaluate diagnosis with physician-authored rubric via an LLM."""

    def __init__(
        self,
        rubric: Dict[str, Any],
        model: str = "gpt-4",
        client: LLMClient | None = None,
    ) -> None:
        """Create a judge with a scoring rubric and LLM configuration."""

        self.rubric = rubric
        self.model = model
        self.client = client or OpenAIClient()
        self.prompt = load_prompt("judge_system")
        self.exact_threshold = float(rubric.get("exact_threshold", 0.9))
        self.partial_threshold = float(rubric.get("partial_threshold", 0.6))

    def _llm_score(self, diagnosis: str, truth: str) -> int | None:
        """Return a Likert score from the LLM or ``None`` on failure."""

        messages = [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": f"Candidate: {diagnosis}\nTruth: {truth}",
            },
        ]
        reply = self.client.chat(messages, self.model)
        if reply is None:
            return None
        match = re.search(r"[1-5]", reply)
        if match:
            return int(match.group(0))
        return None

    def evaluate(self, diagnosis: str, truth: str) -> Judgement:
        """Score the diagnosis against the truth and return judgement."""
        d = diagnosis.strip()
        t = truth.strip()
        score = self._llm_score(d, t)

        if score is None:
            ratio = difflib.SequenceMatcher(None, d.lower(), t.lower()).ratio()
            if ratio >= self.exact_threshold:
                score = 5
            elif ratio >= self.partial_threshold:
                score = 4
            elif d and t and (
                d.lower() in t.lower() or t.lower() in d.lower()
            ):
                score = 3
            elif ratio > 0.3:
                score = 2
            else:
                score = 1

        explanations = {
            5: "Exact or near exact match",
            4: "Reasonable partial match",
            3: "Minor overlap",
            2: "Poor match",
            1: "Incorrect diagnosis",
        }

        return Judgement(score=score, explanation=explanations.get(score, ""))
