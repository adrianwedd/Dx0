from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set
from abc import ABC, abstractmethod

from .actions import PanelAction
from .protocol import ActionType


@dataclass
class Context:
    """Information available to a decision engine."""

    turn: int
    past_infos: List[str]
    triggered_keywords: Set[str]


class DecisionEngine(ABC):
    """Interface for panel decision engines."""

    @abstractmethod
    def decide(self, context: Context) -> PanelAction:
        """Return the next panel action."""
        raise NotImplementedError


class RuleEngine(DecisionEngine):
    """Replicate the original keyword-based heuristics."""

    DEFAULT_KEYWORD_ACTIONS = {
        "cough": (ActionType.TEST, "chest x-ray"),
    }

    def __init__(self, keyword_actions: dict | None = None):
        self.keyword_actions = keyword_actions or self.DEFAULT_KEYWORD_ACTIONS

    def decide(self, context: Context) -> PanelAction:
        if context.turn == 1:
            return PanelAction(ActionType.QUESTION, "chief complaint")

        text = " ".join(context.past_infos).lower()
        for kw, (atype, content) in self.keyword_actions.items():
            if kw.lower() in text and kw not in context.triggered_keywords:
                context.triggered_keywords.add(kw)
                return PanelAction(atype, content)

        if context.turn == 2:
            return PanelAction(ActionType.TEST, "complete blood count")
        if context.turn == 3:
            return PanelAction(ActionType.QUESTION, "physical examination")
        return PanelAction(ActionType.DIAGNOSIS, "viral infection")


class LLMEngine(DecisionEngine):
    """Placeholder LLM-based engine using the rule engine for now."""

    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.fallback = RuleEngine()

    def decide(self, context: Context) -> PanelAction:
        # In this demo we do not integrate an actual LLM API.
        # The fallback rule engine is used instead.
        return self.fallback.decide(context)
