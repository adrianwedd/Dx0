"""Coordinator driving panel actions and tracking cost."""

from __future__ import annotations

import json
import logging

from .panel import VirtualPanel, PanelAction
from .gatekeeper import Gatekeeper
from .protocol import build_action, ActionType
from .cost_estimator import CostEstimator
from .metrics import ORCHESTRATOR_TURNS

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(
        self,
        panel: VirtualPanel,
        gatekeeper: Gatekeeper,
        cost_estimator: CostEstimator | None = None,
        budget: float | None = None,
        question_only: bool = False,
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
        self.cost_estimator = cost_estimator
        self.budget = budget
        self.question_only = question_only
        self.spent = 0.0
        self.finished = False
        self.ordered_tests: list[str] = []
        self.final_diagnosis: str | None = None

    def run_turn(self, case_info: str) -> str:
        """Process a single interaction turn with the panel."""

        action = self.panel.deliberate(case_info=case_info)
        ORCHESTRATOR_TURNS.inc()
        logger.info(
            json.dumps(
                {
                    "event": "panel_action",
                    "turn": getattr(self.panel, "turn", 0),
                    "type": action.action_type.value,
                    "content": action.content,
                }
            )
        )
        if self.question_only and action.action_type == ActionType.TEST:
            action = PanelAction(ActionType.QUESTION, action.content)

        xml = build_action(action.action_type, action.content)
        result = self.gatekeeper.answer_question(xml)
        logger.info(
            json.dumps(
                {"event": "gatekeeper_response", "synthetic": result.synthetic}
            )
        )

        if action.action_type == ActionType.TEST:
            self.ordered_tests.append(action.content)
            if self.cost_estimator:
                self.spent += self.cost_estimator.estimate_cost(action.content)
                if self.budget is not None and self.spent >= self.budget:
                    self.finished = True
        logger.info(json.dumps({"event": "spent", "amount": self.spent}))
        if action.action_type == ActionType.DIAGNOSIS:
            self.finished = True
            self.final_diagnosis = action.content
            logger.info(
                json.dumps(
                    {
                        "event": "final_diagnosis",
                        "diagnosis": action.content,
                    }
                )
            )

        return result.content
