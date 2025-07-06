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


def test_keyword_triggers_chest_xray():
    panel = VirtualPanel()
    first = panel.deliberate("Patient complains of cough")
    assert first.action_type == ActionType.QUESTION

    second = panel.deliberate("additional info")
    assert panel.last_case_info == "additional info"
    assert second.action_type == ActionType.TEST
    assert second.content == "chest x-ray"
