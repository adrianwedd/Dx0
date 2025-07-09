from sdb.panel import VirtualPanel
from sdb.decision import LLMEngine, Context
from sdb.protocol import ActionType
from sdb.llm_client import LLMClient


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


def test_llm_engine_behaves_like_rule_engine():
    panel = VirtualPanel(decision_engine=LLMEngine())
    infos = ["info1", "info2", "info3", "info4"]
    actions = [panel.deliberate(info).action_type for info in infos]
    assert actions == [
        ActionType.QUESTION,
        ActionType.TEST,
        ActionType.QUESTION,
        ActionType.DIAGNOSIS,
    ]


def test_persona_plugin_loaded():
    panel = VirtualPanel(persona_chain="optimist")
    assert isinstance(panel.engine, LLMEngine)
    assert "optimist_system" in panel.engine.prompts


def test_llm_engine_persona_models():
    class CapturingClient(LLMClient):
        def __init__(self):
            super().__init__()
            self.models = []

        def _chat(self, messages, model):
            self.models.append(model)
            return "<test>done</test>"

    client = CapturingClient()
    pmodels = {"hypothesis_system": "m1", "test_chooser_system": "m2"}
    engine = LLMEngine(client=client, persona_models=pmodels, model="def")
    ctx = Context(turn=1, past_infos=["case"], triggered_keywords=set())
    engine.decide(ctx)
    assert client.models[0] == "m1"
    assert client.models[1] == "m2"
    assert all(m == "def" for m in client.models[2:])
