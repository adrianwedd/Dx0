from sdb.panel import VirtualPanel
from sdb.protocol import ActionType


def test_panel_sequence():
    panel = VirtualPanel()
    infos = ["info1", "info2", "info3", "info4"]
    actions = []

    for info in infos:
        actions.append(panel.deliberate(info).action_type)
        assert panel.last_case_info == info

    assert actions == [
        ActionType.QUESTION,
        ActionType.TEST,
        ActionType.QUESTION,
        ActionType.DIAGNOSIS,
    ]
