from .panel import VirtualPanel
from .gatekeeper import Gatekeeper
from .protocol import build_action

class Orchestrator:
    def __init__(self, panel: VirtualPanel, gatekeeper: Gatekeeper):
        self.panel = panel
        self.gatekeeper = gatekeeper
        self.finished = False

    def run_turn(self, case_info: str) -> str:
        action = self.panel.deliberate(case_info)
        xml = build_action(action.action_type, action.content)
        # TODO: send action to gatekeeper and get result
        result = self.gatekeeper.answer_question(xml)
        if action.action_type == action.action_type.DIAGNOSIS:
            self.finished = True
        return result.content
