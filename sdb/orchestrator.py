"""Coordinator driving panel actions and tracking cost."""

from __future__ import annotations

import asyncio
import os
import time

import structlog
from opentelemetry import trace

from .gatekeeper import Gatekeeper
from .metrics import (
    ORCHESTRATOR_LATENCY,
    ORCHESTRATOR_TURNS,
    USER_MESSAGE_COST,
    USER_MESSAGE_TOKENS,
)
from .llm_client import LLMClient
from .panel import PanelAction, VirtualPanel
from .protocol import ActionType, build_action
from .services import BudgetManager, ResultAggregator

logger = structlog.get_logger(__name__)
tracer = trace.get_tracer(__name__)

SENTRY_ENABLED = False
if os.getenv("SENTRY_DSN"):
    try:
        import sentry_sdk

        sentry_sdk.init(os.environ["SENTRY_DSN"])
        SENTRY_ENABLED = True
    except Exception:
        pass


class Orchestrator:
    """Coordinate panel actions while enforcing test budgets."""

    def __init__(
        self,
        panel: VirtualPanel,
        gatekeeper: Gatekeeper,
        question_only: bool = False,
        *,
        budget_manager: BudgetManager | None = None,
        result_aggregator: ResultAggregator | None = None,
        session_id: str | None = None,
    ):
        """Coordinate panel actions and track test spending.

        Parameters
        ----------
        panel:
            :class:`VirtualPanel` generating actions.
        gatekeeper:
            Interface used to obtain answers from the case.
        question_only:
            If ``True``, convert test requests into questions.
        budget_manager:
            Optional :class:`BudgetManager` tracking test costs.
        session_id:
            Identifier used for error reporting context.
        """
        self.panel = panel
        self.gatekeeper = gatekeeper
        self.question_only = question_only
        self.budget_manager = budget_manager or BudgetManager()
        self.results = result_aggregator or ResultAggregator()
        self.session_id = session_id
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
        with tracer.start_as_current_span("orchestrator.run_turn"):
            start = time.perf_counter()
            prev_spent = self.budget_manager.spent
            tokens = LLMClient._count_tokens([{"role": "user", "content": case_info}])
            USER_MESSAGE_TOKENS.inc(tokens)
            try:
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
                return self._execute_turn(action, result, start, prev_spent)
            except Exception as exc:  # pragma: no cover - error path
                if SENTRY_ENABLED:
                    import sentry_sdk

                    with sentry_sdk.push_scope() as scope:
                        if self.session_id:
                            scope.set_tag("session_id", self.session_id)
                        sentry_sdk.capture_exception(exc)
                raise

    async def run_turn_async(self, case_info: str) -> str:
        """Asynchronous version of :meth:`run_turn`."""
        with tracer.start_as_current_span("orchestrator.run_turn_async"):
            start = time.perf_counter()
            prev_spent = self.budget_manager.spent
            tokens = LLMClient._count_tokens([{"role": "user", "content": case_info}])
            USER_MESSAGE_TOKENS.inc(tokens)
            try:
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
                if hasattr(self.gatekeeper, "aanswer_question"):
                    result = await self.gatekeeper.aanswer_question(xml)  # type: ignore[attr-defined]
                else:
                    result = await asyncio.to_thread(self.gatekeeper.answer_question, xml)
                return self._execute_turn(action, result, start, prev_spent)
            except Exception as exc:  # pragma: no cover - error path
                if SENTRY_ENABLED:
                    import sentry_sdk

                    with sentry_sdk.push_scope() as scope:
                        if self.session_id:
                            scope.set_tag("session_id", self.session_id)
                        sentry_sdk.capture_exception(exc)
                raise

    def _execute_turn(
        self,
        action: PanelAction,
        result: Gatekeeper.QueryResult | any,
        start: float,
        prev_spent: float,
    ) -> str:
        """Finalize bookkeeping after obtaining an action result."""
        logger.info(
            "gatekeeper_response",
            synthetic=getattr(result, "synthetic", False),
        )

        if action.action_type == ActionType.TEST:
            self.results.record_test(action.content)
            self.budget_manager.add_test(action.content)
            if self.budget_manager.over_budget():
                self.results.finished = True
        USER_MESSAGE_COST.inc(self.budget_manager.spent - prev_spent)
        logger.info("spent", amount=self.spent)
        if action.action_type == ActionType.DIAGNOSIS:
            self.results.record_diagnosis(action.content)
            logger.info("final_diagnosis", diagnosis=action.content)

        duration = time.perf_counter() - start
        self.total_time += duration
        ORCHESTRATOR_LATENCY.observe(duration)
        return result.content
