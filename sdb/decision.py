from __future__ import annotations

from dataclasses import dataclass
from typing import List, Set
from abc import ABC, abstractmethod

import structlog
import asyncio

from .actions import PanelAction, parse_panel_action
from .protocol import ActionType
from .prompt_loader import load_prompt
from .llm_client import LLMClient, OpenAIClient, AsyncLLMClient
from .config import settings
from .exceptions import DecisionEngineError

logger = structlog.get_logger(__name__)


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
    common presentations and incorporate additional heuristics for complaints
    like ``headache`` or ``sore throat``. Combination rules such as ``fever``
    with ``neck stiffness`` help surface critical tests like lumbar puncture.
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
        # Additional rules improving diagnostic accuracy
        "headache": (ActionType.QUESTION, "headache duration"),
        "sore throat": (ActionType.TEST, "rapid strep test"),
        "dizziness": (ActionType.TEST, "blood pressure measurement"),
    }

    # Multi-keyword rules evaluated before the single keyword lookup. Each set
    # of keywords maps to an action. Keys are stored as frozensets so they can
    # be used in dictionaries.
    DEFAULT_COMBO_ACTIONS = {
        frozenset({"fever", "rash"}): (
            ActionType.QUESTION,
            "recent travel history",
        ),
        # New combos capturing more complex presentations
        frozenset({"fever", "neck stiffness"}): (
            ActionType.TEST,
            "lumbar puncture",
        ),
        frozenset({"weight loss", "night sweats"}): (
            ActionType.QUESTION,
            "tuberculosis exposure",
        ),
    }

    def __init__(
        self,
        keyword_actions: dict | None = None,
        combo_actions: dict[frozenset[str], tuple[ActionType, str]] | None = None,
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

    DEFAULT_PERSONAS = [
        "hypothesis_system",
        "test_chooser_system",
        "challenger_system",
        "stewardship_system",
        "checklist_system",
    ]

    def __init__(
        self,
        model: str = "gpt-4",
        client: LLMClient | AsyncLLMClient | None = None,
        personas: list[str] | None = None,
        *,
        persona_models: dict[str, str] | None = None,
        parallel_personas: bool | None = None,
    ) -> None:
        """Create an LLM engine with optional parallel persona execution.

        Parameters
        ----------
        model:
            Default model used when ``persona_models`` does not specify a
            persona-specific override.
        client:
            LLM client instance used to send chat messages.
        personas:
            Ordered list of persona prompt names.
        persona_models:
            Mapping from persona prompt name to model name.
        parallel_personas:
            Run personas concurrently when an :class:`AsyncLLMClient` is
            provided.
        """

        self.model = model
        self.client = client or OpenAIClient()
        self.fallback = RuleEngine()
        self.personas = personas or self.DEFAULT_PERSONAS
        self.persona_models = persona_models or {}
        self.parallel_personas = (
            settings.parallel_personas
            if parallel_personas is None
            else parallel_personas
        )
        self.prompts = {name: load_prompt(name) for name in self.personas}
        for name, text in self.prompts.items():
            if not text.strip():
                logger.error("prompt_empty", name=name)
                raise DecisionEngineError(f"Prompt {name} is empty")

    def _chat(self, messages: list[dict], model: str) -> str | None:
        """Return a chat completion for ``model`` via the configured client."""

        return self.client.chat(messages, model)

    async def _achat(self, messages: list[dict], model: str) -> str | None:
        """Asynchronous helper for chatting with ``model``."""

        if isinstance(self.client, AsyncLLMClient):
            return await self.client.chat(messages, model)
        return await asyncio.to_thread(self.client.chat, messages, model)

    def decide(self, context: Context) -> PanelAction:
        """Return the next panel action from the LLM persona chain.

        Falls back to the rule engine when the LLM output cannot be parsed.
        """

        conversation = "\n".join(context.past_infos)
        messages = []
        for name in self.personas:
            system = {"role": "system", "content": self.prompts[name]}
            user = {"role": "user", "content": conversation}
            messages.extend([system, user])
            model = self.persona_models.get(name, self.model)
            reply = self._chat(messages, model)
            if reply is None:
                logger.warning("llm_fallback", reason="no_reply")
                return self.fallback.decide(context)
            messages.append({"role": "assistant", "content": reply})
        action = parse_panel_action(messages[-1]["content"])
        if action is None:
            logger.warning("llm_fallback", reason="parse")
            return self.fallback.decide(context)
        logger.info(
            "llm_decision",
            turn=context.turn,
            type=action.action_type.value,
        )
        return action

    async def adecide(self, context: Context) -> PanelAction:
        """Asynchronous version of :meth:`decide`."""

        conversation = "\n".join(context.past_infos)
        messages = []
        if self.parallel_personas and isinstance(self.client, AsyncLLMClient):
            async def run_persona(name: str) -> tuple[dict, dict, str | None]:
                system = {"role": "system", "content": self.prompts[name]}
                user = {"role": "user", "content": conversation}
                model = self.persona_models.get(name, self.model)
                reply = await self._achat([system, user], model)
                return system, user, reply

            tasks = [run_persona(name) for name in self.personas]
            results = await asyncio.gather(*tasks)
            for system, user, reply in results:
                messages.extend([system, user])
                if reply is None:
                    logger.warning("llm_fallback", reason="no_reply")
                    return await asyncio.to_thread(self.fallback.decide, context)
                messages.append({"role": "assistant", "content": reply})
        else:
            for name in self.personas:
                system = {"role": "system", "content": self.prompts[name]}
                user = {"role": "user", "content": conversation}
                messages.extend([system, user])
                model = self.persona_models.get(name, self.model)
                reply = await self._achat(messages, model)
                if reply is None:
                    logger.warning("llm_fallback", reason="no_reply")
                    return await asyncio.to_thread(self.fallback.decide, context)
                messages.append({"role": "assistant", "content": reply})
        action = parse_panel_action(messages[-1]["content"])
        if action is None:
            logger.warning("llm_fallback", reason="parse")
            return await asyncio.to_thread(self.fallback.decide, context)
        logger.info(
            "llm_decision",
            turn=context.turn,
            type=action.action_type.value,
        )
        return action
