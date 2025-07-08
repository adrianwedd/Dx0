"""Virtual panel of doctors driven by a decision engine."""

from __future__ import annotations

import structlog
import asyncio
from typing import List, Set

from .actions import PanelAction
from .decision import Context, DecisionEngine, RuleEngine
from .metrics import PANEL_ACTIONS

logger = structlog.get_logger(__name__)


class VirtualPanel:
    """Simulate collaborative panel of doctors using pluggable engines."""

    def __init__(self, decision_engine: DecisionEngine | None = None):
        """Initialize the panel and underlying decision engine.

        Parameters
        ----------
        decision_engine:
            Optional engine used to select actions. If ``None``, a
            :class:`RuleEngine` is instantiated.

        The panel starts with no previous case information, zero turn count
        and an empty set of triggered keywords.
        """

        self.turn = 0
        self.last_case_info = ""
        self.past_infos: List[str] = []
        self.triggered_keywords: Set[str] = set()
        self.engine = decision_engine or RuleEngine()

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
