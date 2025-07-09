"""Coordinator driving panel actions and tracking cost."""

from __future__ import annotations

import structlog
import asyncio

from .panel import VirtualPanel, PanelAction
from .gatekeeper import Gatekeeper
from .protocol import build_action, ActionType
from .cost_estimator import CostEstimator
from .services import BudgetManager, ResultAggregator
import time
from .metrics import ORCHESTRATOR_TURNS, ORCHESTRATOR_LATENCY

logger = structlog.get_logger(__name__)


class Orchestrator:
    def __init__(
        self,
        panel: VirtualPanel,
        gatekeeper: Gatekeeper,
        cost_estimator: CostEstimator | None = None,
        budget: float | None = None,
        question_only: bool = False,
        *,
        budget_manager: BudgetManager | None = None,
        result_aggregator: ResultAggregator | None = None,
    ):
        """Coordinate panel actions and track test spending.

        Parameters
        ----------
        panel:
            :class:`VirtualPanel` generating actions.
        gatekeeper:
            Interface used to obtain answers from the case.
        cost_estimator:
            Optional :class:`CostEstimator` for test pricing.
        budget:
            Optional maximum spend allowed for tests.
        question_only:
            If ``True``, convert test requests into questions.
        """

        self.panel = panel
        self.gatekeeper = gatekeeper
        self.question_only = question_only
        self.budget_manager = budget_manager or BudgetManager(
            cost_estimator, budget
        )
        self.results = result_aggregator or ResultAggregator()
        self.total_time = 0.0

    @property
    def spent(self) -> float:
        """Total money spent on ordered tests."""

        return self.budget_manager.spent

    @property
    def ordered_tests(self) -> list[str]:
        """List of tests ordered so far."""

        return self.results.ordered_tests

    @property
    def final_diagnosis(self) -> str | None:
        """Diagnosis provided by the panel, if any."""

        return self.results.final_diagnosis

    @property
    def finished(self) -> bool:
        """Whether the session has concluded."""

        return self.results.finished or self.budget_manager.over_budget()

    def run_turn(self, case_info: str) -> str:
        """Process a single interaction turn with the panel."""

        start = time.perf_counter()
        action = self.panel.deliberate(case_info=case_info)
        ORCHESTRATOR_TURNS.inc()
        logger.info(
            "panel_action",
            turn=getattr(self.panel, "turn", 0),
            type=action.action_type.value,
            content=action.content,
        )
        if self.question_only and action.action_type == ActionType.TEST:
            action = PanelAction(ActionType.QUESTION, action.content)

        xml = build_action(action.action_type, action.content)
        result = self.gatekeeper.answer_question(xml)
        logger.info(
            "gatekeeper_response",
            synthetic=result.synthetic,
        )

        if action.action_type == ActionType.TEST:
            self.results.record_test(action.content)
            self.budget_manager.add_test(action.content)
            if self.budget_manager.over_budget():
                self.results.finished = True
        logger.info("spent", amount=self.spent)
        if action.action_type == ActionType.DIAGNOSIS:
            self.results.record_diagnosis(action.content)
            logger.info("final_diagnosis", diagnosis=action.content)

        duration = time.perf_counter() - start
        self.total_time += duration
        ORCHESTRATOR_LATENCY.observe(duration)
        return result.content

    async def run_turn_async(self, case_info: str) -> str:
        """Asynchronous version of :meth:`run_turn`."""

        start = time.perf_counter()
        if hasattr(self.panel, "adeliberate"):
            action = await self.panel.adeliberate(case_info=case_info)
        else:
            action = await asyncio.to_thread(self.panel.deliberate, case_info)
        ORCHESTRATOR_TURNS.inc()
        logger.info(
            "panel_action",
            turn=getattr(self.panel, "turn", 0),
            type=action.action_type.value,
            content=action.content,
        )
        if self.question_only and action.action_type == ActionType.TEST:
            action = PanelAction(ActionType.QUESTION, action.content)

        xml = build_action(action.action_type, action.content)
        result = await asyncio.to_thread(self.gatekeeper.answer_question, xml)
        logger.info(
            "gatekeeper_response",
            synthetic=result.synthetic,
        )

        if action.action_type == ActionType.TEST:
            self.results.record_test(action.content)
            self.budget_manager.add_test(action.content)
            if self.budget_manager.over_budget():
                self.results.finished = True
        logger.info("spent", amount=self.spent)
        if action.action_type == ActionType.DIAGNOSIS:
            self.results.record_diagnosis(action.content)
            logger.info("final_diagnosis", diagnosis=action.content)

        duration = time.perf_counter() - start
        self.total_time += duration
        ORCHESTRATOR_LATENCY.observe(duration)
        return result.content
