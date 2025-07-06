from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set
from abc import ABC, abstractmethod

from .actions import PanelAction, parse_panel_action
from .protocol import ActionType
from .prompt_loader import load_prompt
from .llm_client import LLMClient, OpenAIClient


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
    """LLM-driven decision engine following the Chain of Debate."""

    PERSONAS = [
        "hypothesis_system",
        "test_chooser_system",
        "challenger_system",
        "stewardship_system",
        "checklist_system",
    ]

    def __init__(self, model: str = "gpt-4", client: LLMClient | None = None):
        self.model = model
        self.client = client or OpenAIClient()
        self.fallback = RuleEngine()
        self.prompts = {name: load_prompt(name) for name in self.PERSONAS}
        for name, text in self.prompts.items():
            if not text.strip():
                raise ValueError(f"Prompt {name} is empty")

    def _chat(self, messages: list[dict]) -> str | None:
        return self.client.chat(messages, self.model)

    def decide(self, context: Context) -> PanelAction:
        conversation = "\n".join(context.past_infos)
        messages = []
        for name in self.PERSONAS:
            system = {"role": "system", "content": self.prompts[name]}
            user = {"role": "user", "content": conversation}
            messages.extend([system, user])
            reply = self._chat(messages)
            if reply is None:
                return self.fallback.decide(context)
            messages.append({"role": "assistant", "content": reply})
        action = parse_panel_action(messages[-1]["content"])
        if action is None:
            return self.fallback.decide(context)
        return action
