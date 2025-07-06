from .panel import VirtualPanel
from .gatekeeper import Gatekeeper
from .protocol import build_action, ActionType

class Orchestrator:
    def __init__(self, panel: VirtualPanel, gatekeeper: Gatekeeper):
        self.panel = panel
        self.gatekeeper = gatekeeper
        self.finished = False
        self.ordered_tests = []
        self.final_diagnosis: str | None = None

    def run_turn(self, case_info: str) -> str:
        """Process a single interaction turn with the panel."""

        action = self.panel.deliberate(case_info=case_info)
        xml = build_action(action.action_type, action.content)
        result = self.gatekeeper.answer_question(xml)

        if action.action_type == ActionType.TEST:
            self.ordered_tests.append(action.content)
        if action.action_type == ActionType.DIAGNOSIS:
            self.finished = True
            self.final_diagnosis = action.content

        return result.content
