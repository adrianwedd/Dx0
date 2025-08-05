import re
from dataclasses import dataclass
from typing import Any, Dict

from .prompt_loader import load_prompt
from .llm_client import LLMClient, OpenAIClient
from .config import settings


@dataclass
class Judgement:
    score: int
    explanation: str


class Judge:
    """Evaluate diagnosis with physician-authored rubric via an LLM."""

    def __init__(
        self,
        rubric: Dict[str, Any],
        model: str | None = None,
        client: LLMClient | None = None,
    ) -> None:
        """Create a judge with a scoring rubric and LLM configuration."""

        self.rubric = rubric
        self.model = model or settings.openai_model
        self.client = client or OpenAIClient()
        self.prompt = load_prompt("judge_system")

    def _llm_score(self, diagnosis: str, truth: str) -> int:
        """Return a Likert score from the LLM."""

        messages = [
            {"role": "system", "content": self.prompt},
            {
                "role": "user",
                "content": f"Candidate: {diagnosis}\nTruth: {truth}",
            },
        ]
        reply = self.client.chat(messages, self.model)
        if reply is None:
            raise RuntimeError("LLM returned no reply")
        match = re.search(r"[1-5]", reply)
        if match is None:
            raise ValueError("LLM reply missing score")
        return int(match.group(0))

    def evaluate(self, diagnosis: str, truth: str) -> Judgement:
        """Score the diagnosis against the truth and return judgement."""
        d = diagnosis.strip()
        t = truth.strip()
        try:
            score = self._llm_score(d, t)
        except Exception:
            score = 1

        explanations = {
            5: "Exact or near exact match",
            4: "Reasonable partial match",
            3: "Minor overlap",
            2: "Poor match",
            1: "Incorrect diagnosis",
        }

        return Judgement(score=score, explanation=explanations.get(score, ""))
