from dataclasses import dataclass

import pytest

from sdb.orchestrator import Orchestrator
from sdb.services import BudgetManager, BudgetStore
from sdb.panel import VirtualPanel
from sdb.actions import PanelAction
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

    def estimate(self, test_name: str) -> tuple[float, str]:
        return 5.0, "labs"


def test_orchestrator_budget_stops_session():
    actions = [
        PanelAction(ActionType.TEST, "cbc"),
        PanelAction(ActionType.TEST, "bmp"),
        PanelAction(ActionType.DIAGNOSIS, "flu"),
    ]
    panel = StubPanel(actions)
    tracker = BudgetManager(DummyCostEstimator(), budget=7.0)
    orch = Orchestrator(
        panel,
        DummyGatekeeper(),
        budget_manager=tracker,
    )

    orch.run_turn("1")
    orch.run_turn("2")
    assert orch.finished is True
    assert orch.final_diagnosis is None


def test_category_limit_stops_session():
    actions = [
        PanelAction(ActionType.TEST, "cbc"),
        PanelAction(ActionType.TEST, "bmp"),
        PanelAction(ActionType.DIAGNOSIS, "flu"),
    ]
    panel = StubPanel(actions)
    tracker = BudgetManager(DummyCostEstimator(), category_limits={"labs": 7.0})
    orch = Orchestrator(panel, DummyGatekeeper(), budget_manager=tracker)

    orch.run_turn("1")
    assert orch.finished is False
    orch.run_turn("2")
    assert orch.finished is True
    assert orch.final_diagnosis is None


def test_budget_store_persists(tmp_path):
    store = BudgetStore(str(tmp_path / "budget.db"))
    estimator = DummyCostEstimator()

    panel1 = StubPanel([PanelAction(ActionType.TEST, "cbc")])
    bm1 = BudgetManager(estimator, store=store)
    orch1 = Orchestrator(panel1, DummyGatekeeper(), budget_manager=bm1)
    orch1.run_turn("step1")
    assert orch1.spent == 5.0

    panel2 = StubPanel([])
    bm2 = BudgetManager(estimator, store=store)
    orch2 = Orchestrator(panel2, DummyGatekeeper(), budget_manager=bm2)
    assert orch2.spent == 5.0


class TimeoutPanel:
    def deliberate(self, case_info: str) -> PanelAction:
        raise TimeoutError("panel timeout")


class TimeoutGatekeeper:
    def answer_question(self, xml: str) -> DummyResult:
        raise TimeoutError("gatekeeper timeout")


def test_panel_timeout_propagates():
    orch = Orchestrator(TimeoutPanel(), DummyGatekeeper())
    with pytest.raises(TimeoutError):
        orch.run_turn("info")


def test_gatekeeper_timeout_propagates():
    actions = [PanelAction(ActionType.QUESTION, "q1")]
    panel = StubPanel(actions)
    orch = Orchestrator(panel, TimeoutGatekeeper())
    with pytest.raises(TimeoutError):
        orch.run_turn("info")


class AsyncStubPanel:
    """Asynchronous stub panel returning predefined actions."""

    def __init__(self, actions):
        self.actions = actions
        self.index = 0
        self.last_case_info = ""

    async def adeliberate(self, case_info: str) -> PanelAction:
        self.last_case_info = case_info
        action = self.actions[self.index]
        self.index += 1
        return action


@pytest.mark.asyncio
async def test_run_turn_async_budget_stops_session():
    actions = [
        PanelAction(ActionType.TEST, "cbc"),
        PanelAction(ActionType.TEST, "bmp"),
        PanelAction(ActionType.DIAGNOSIS, "flu"),
    ]
    panel = AsyncStubPanel(actions)
    tracker = BudgetManager(DummyCostEstimator(), budget=7.0)
    orch = Orchestrator(panel, DummyGatekeeper(), budget_manager=tracker)

    await orch.run_turn_async("1")
    await orch.run_turn_async("2")
    assert orch.finished is True
    assert orch.final_diagnosis is None
