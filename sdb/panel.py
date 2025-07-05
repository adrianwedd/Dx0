from dataclasses import dataclass
from .protocol import ActionType, build_action

@dataclass
class PanelAction:
    action_type: ActionType
    content: str

class VirtualPanel:
    """Simulate collaborative panel of doctors."""
    def __init__(self):
        self.turn = 0

    def deliberate(self, case_info: str) -> PanelAction:
        """Very small demo implementation of the Chain of Debate."""
        self.turn += 1

        if self.turn == 1:
            # Dr. Hypothesis asks for key symptom information
            return PanelAction(ActionType.QUESTION, "chief complaint")
        elif self.turn == 2:
            # Test-Chooser orders a basic test
            return PanelAction(ActionType.TEST, "complete blood count")
        elif self.turn == 3:
            # Challenger requests additional info
            return PanelAction(ActionType.QUESTION, "physical examination")
        else:
            # Stewardship/Checklist propose a diagnosis to finish
            return PanelAction(ActionType.DIAGNOSIS, "viral infection")
