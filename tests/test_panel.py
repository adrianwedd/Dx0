from sdb.panel import VirtualPanel
from sdb.protocol import ActionType


def test_panel_sequence():
    panel = VirtualPanel()
    actions = [panel.deliberate("").action_type for _ in range(4)]
    assert actions == [ActionType.QUESTION, ActionType.TEST, ActionType.QUESTION, ActionType.DIAGNOSIS]
