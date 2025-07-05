from dataclasses import dataclass
from .protocol import ActionType, build_action

@dataclass
class PanelAction:
    action_type: ActionType
    content: str

class VirtualPanel:
    """Simulate collaborative panel of doctors."""

    def deliberate(self, case_info: str) -> PanelAction:
        # TODO: implement Chain of Debate between personas
        return PanelAction(ActionType.DIAGNOSIS, "Not implemented")
