from sdb.panel import VirtualPanel
from sdb.protocol import ActionType
from sdb.decision import LLMEngine, Context, RuleEngine
from sdb.llm_client import LLMClient
import pytest


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


def test_headache_triggers_question():
    panel = VirtualPanel()
    panel.deliberate("severe headache since morning")
    action = panel.deliberate("needs evaluation")
    assert action.action_type == ActionType.QUESTION
    assert action.content == "headache duration"


def test_sore_throat_triggers_strep_test():
    panel = VirtualPanel()
    panel.deliberate("complaining of sore throat")
    action = panel.deliberate("ongoing discomfort")
    assert action.action_type == ActionType.TEST
    assert action.content == "rapid strep test"


def test_combo_fever_neck_stiffness_triggers_lp():
    panel = VirtualPanel()
    panel.deliberate("high fever noted")
    action = panel.deliberate("neck stiffness present")
    assert action.action_type == ActionType.TEST
    assert action.content == "lumbar puncture"


def test_dizziness_triggers_bp_measurement():
    panel = VirtualPanel()
    panel.deliberate("reports episodes of dizziness")
    action = panel.deliberate("further details")
    assert action.action_type == ActionType.TEST
    assert action.content == "blood pressure measurement"


class NoneClient(LLMClient):
    def _chat(self, messages, model):
        return None


class BadXMLClient(LLMClient):
    def _chat(self, messages, model):
        return "<invalid>"


@pytest.mark.parametrize("client", [NoneClient(), BadXMLClient()])
def test_llm_fallback_matches_rule_engine(client):
    ctx1 = Context(turn=2, past_infos=["info"], triggered_keywords=set())
    expected = RuleEngine().decide(ctx1)

    ctx2 = Context(turn=2, past_infos=["info"], triggered_keywords=set())
    engine = LLMEngine(client=client)
    action = engine.decide(ctx2)

    assert action == expected
