"""Virtual panel of doctors driven by a decision engine."""

from __future__ import annotations

import structlog
import asyncio
from typing import List, Set
from importlib import metadata

from .actions import PanelAction
from .decision import Context, DecisionEngine, RuleEngine, LLMEngine
from .metrics import PANEL_ACTIONS

logger = structlog.get_logger(__name__)


class VirtualPanel:
    """Simulate collaborative panel of doctors using pluggable engines."""

    def __init__(
        self,
        decision_engine: DecisionEngine | None = None,
        persona_chain: str | None = None,
    ):
        """Initialize the panel and underlying decision engine.

        Parameters
        ----------
        decision_engine:
            Optional engine used to select actions. If provided, it overrides
            ``persona_chain``.
        persona_chain:
            Name of an installed persona plugin to load. Ignored when
            ``decision_engine`` is supplied.

        The panel starts with no previous case information, zero turn count
        and an empty set of triggered keywords.
        """

        self.turn = 0
        self.last_case_info = ""
        self.past_infos: List[str] = []
        self.triggered_keywords: Set[str] = set()
        if decision_engine is not None:
            self.engine = decision_engine
        elif persona_chain is not None:
            chain = self._load_persona_chain(persona_chain)
            self.engine = LLMEngine(personas=chain)
        else:
            self.engine = RuleEngine()

    def _load_persona_chain(self, name: str) -> list[str]:
        """Return persona prompt names from the plugin ``name``."""

        for ep in metadata.entry_points(group="dx0.personas"):
            if ep.name == name:
                chain = ep.load()()
                if not isinstance(chain, list):
                    raise TypeError("Persona plugin must return a list of names")
                return chain
        raise ValueError(f"Persona chain '{name}' not found")

    def deliberate(self, case_info: str) -> PanelAction:
        """Run one deliberation step and return the chosen action."""

        self.last_case_info = case_info
        self.past_infos.append(case_info)
        self.turn += 1

        ctx = Context(
            turn=self.turn,
            past_infos=self.past_infos,
            triggered_keywords=self.triggered_keywords,
        )
        action = self.engine.decide(ctx)
        PANEL_ACTIONS.labels(action.action_type.value).inc()
        logger.info(
            "deliberate",
            turn=self.turn,
            type=action.action_type.value,
            content=action.content,
        )
        return action

    async def adeliberate(self, case_info: str) -> PanelAction:
        """Asynchronous version of :meth:`deliberate`."""

        self.last_case_info = case_info
        self.past_infos.append(case_info)
        self.turn += 1

        ctx = Context(
            turn=self.turn,
            past_infos=self.past_infos,
            triggered_keywords=self.triggered_keywords,
        )
        if hasattr(self.engine, "adecide"):
            action = await self.engine.adecide(ctx)  # type: ignore[attr-defined]
        else:
            action = await asyncio.to_thread(self.engine.decide, ctx)
        PANEL_ACTIONS.labels(action.action_type.value).inc()
        logger.info(
            "deliberate",
            turn=self.turn,
            type=action.action_type.value,
            content=action.content,
        )
        return action
