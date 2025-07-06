from dataclasses import dataclass

from sdb.orchestrator import Orchestrator
from sdb.panel import VirtualPanel


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
