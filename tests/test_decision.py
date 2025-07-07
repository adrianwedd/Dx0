from sdb.panel import VirtualPanel
from sdb.protocol import ActionType


def test_fever_triggers_blood_culture():
    panel = VirtualPanel()
    panel.deliberate("patient reports high fever")
    action = panel.deliberate("more info")
    assert action.action_type == ActionType.TEST
    assert action.content == "blood culture"


def test_combo_fever_rash_triggers_travel_question():
    panel = VirtualPanel()
    panel.deliberate("fever and myalgias")
    action = panel.deliberate("new rash on arms")
    assert action.action_type == ActionType.QUESTION
    assert action.content == "recent travel history"


def test_chest_pain_triggers_ecg():
    panel = VirtualPanel()
    panel.deliberate("patient with chest pain")
    action = panel.deliberate("symptoms persist")
    assert action.action_type == ActionType.TEST
    assert action.content == "electrocardiogram"
