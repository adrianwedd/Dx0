from .panel import VirtualPanel, PanelAction
from .gatekeeper import Gatekeeper
from .protocol import build_action, ActionType
from .cost_estimator import CostEstimator


class Orchestrator:
    def __init__(
        self,
        panel: VirtualPanel,
        gatekeeper: Gatekeeper,
        cost_estimator: CostEstimator | None = None,
        budget: float | None = None,
        question_only: bool = False,
    ):
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
        if self.question_only and action.action_type == ActionType.TEST:
            action = PanelAction(ActionType.QUESTION, action.content)

        xml = build_action(action.action_type, action.content)
        result = self.gatekeeper.answer_question(xml)

        if action.action_type == ActionType.TEST:
            self.ordered_tests.append(action.content)
            if self.cost_estimator:
                self.spent += self.cost_estimator.estimate_cost(action.content)
                if self.budget is not None and self.spent >= self.budget:
                    self.finished = True
        if action.action_type == ActionType.DIAGNOSIS:
            self.finished = True
            self.final_diagnosis = action.content

        return result.content
