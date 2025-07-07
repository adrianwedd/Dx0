from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set
from abc import ABC, abstractmethod

import json
import logging

from .actions import PanelAction, parse_panel_action
from .protocol import ActionType
from .prompt_loader import load_prompt
from .llm_client import LLMClient, OpenAIClient

logger = logging.getLogger(__name__)


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
    """Rule-based decision engine using simple keyword heuristics.

    An analysis of the 304 sample cases revealed that ``fever``, ``abdominal
    pain`` and ``rash`` frequently appear early in the narratives but were not
    handled by the initial prototype. As a result, many sessions defaulted to a
    generic ``"viral infection"`` diagnosis. The rules below address these
    common presentations.
    """

    DEFAULT_KEYWORD_ACTIONS = {
        # Respiratory complaints
        "cough": (ActionType.TEST, "chest x-ray"),
        # New rules derived from dataset statistics
        "fever": (ActionType.TEST, "blood culture"),
        "abdominal pain": (ActionType.TEST, "abdominal ultrasound"),
        "chest pain": (ActionType.TEST, "electrocardiogram"),
        "shortness of breath": (ActionType.TEST, "pulse oximetry"),
        "rash": (ActionType.QUESTION, "rash appearance"),
    }

    # Multi-keyword rules evaluated before the single keyword lookup. Each set
    # of keywords maps to an action. Keys are stored as frozensets so they can
    # be used in dictionaries.
    DEFAULT_COMBO_ACTIONS = {
        frozenset({"fever", "rash"}): (
            ActionType.QUESTION,
            "recent travel history",
        ),
    }

    def __init__(
        self,
        keyword_actions: dict | None = None,
        combo_actions: (
            dict[frozenset[str], tuple[ActionType, str]] | None
        ) = None,
    ) -> None:
        self.keyword_actions = keyword_actions or self.DEFAULT_KEYWORD_ACTIONS
        self.combo_actions = combo_actions or self.DEFAULT_COMBO_ACTIONS

    def decide(self, context: Context) -> PanelAction:
        if context.turn == 1:
            return PanelAction(ActionType.QUESTION, "chief complaint")

        text = " ".join(context.past_infos).lower()
        # Check multi-keyword rules first so specific combinations take
        # precedence over single keywords.
        for keywords, (atype, content) in self.combo_actions.items():
            key = "+".join(sorted(keywords))
            if (
                all(kw in text for kw in keywords)
                and key not in context.triggered_keywords
            ):
                context.triggered_keywords.add(key)
                return PanelAction(atype, content)

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
        """Return the next panel action from the LLM persona chain.

        Falls back to the rule engine when the LLM output cannot be parsed.
        """

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
        logger.info(
            json.dumps(
                {
                    "event": "llm_decision",
                    "turn": context.turn,
                    "type": action.action_type.value,
                }
            )
        )
        return action
