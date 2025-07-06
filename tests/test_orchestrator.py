from dataclasses import dataclass

from sdb.orchestrator import Orchestrator
from sdb.panel import VirtualPanel, PanelAction
from sdb.protocol import ActionType


@dataclass
class DummyResult:
    content: str
    synthetic: bool = False


class DummyGatekeeper:
    def answer_question(self, xml: str) -> DummyResult:
        return DummyResult("ack")


def test_run_turn_passes_case_info():
    panel = VirtualPanel()
    orch = Orchestrator(panel, DummyGatekeeper())
    orch.run_turn("snippet")
    assert panel.last_case_info == "snippet"


class StubPanel:
    """Panel returning a predefined sequence of actions."""

    def __init__(self, actions):
        self.actions = actions
        self.index = 0
        self.last_case_info = ""

    def deliberate(self, case_info: str) -> PanelAction:
        self.last_case_info = case_info
        action = self.actions[self.index]
        self.index += 1
        return action


def test_orchestrator_collects_tests_and_finishes():
    actions = [
        PanelAction(ActionType.TEST, "cbc"),
        PanelAction(ActionType.TEST, "bmp"),
        PanelAction(ActionType.DIAGNOSIS, "flu"),
    ]
    panel = StubPanel(actions)
    orch = Orchestrator(panel, DummyGatekeeper())

    orch.run_turn("step1")
    orch.run_turn("step2")
    assert orch.ordered_tests == ["cbc", "bmp"]
    assert orch.finished is False

    orch.run_turn("step3")
    assert orch.finished is True
    assert orch.ordered_tests == ["cbc", "bmp"]
    assert orch.final_diagnosis == "flu"


class DummyCostEstimator:
    def estimate_cost(self, test_name: str) -> float:
        return 5.0


def test_orchestrator_budget_stops_session():
    actions = [
        PanelAction(ActionType.TEST, "cbc"),
        PanelAction(ActionType.TEST, "bmp"),
        PanelAction(ActionType.DIAGNOSIS, "flu"),
    ]
    panel = StubPanel(actions)
    orch = Orchestrator(
        panel,
        DummyGatekeeper(),
        cost_estimator=DummyCostEstimator(),
        budget=7.0,
    )

    orch.run_turn("1")
    orch.run_turn("2")
    assert orch.finished is True
    assert orch.final_diagnosis is None
