import re
from dataclasses import dataclass
from typing import Any, Dict

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
            score = 1

        explanations = {
            5: "Exact or near exact match",
            4: "Reasonable partial match",
            3: "Minor overlap",
            2: "Poor match",
            1: "Incorrect diagnosis",
        }

        return Judgement(score=score, explanation=explanations.get(score, ""))
