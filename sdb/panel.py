"""Virtual panel of doctors driven by a decision engine."""

from __future__ import annotations

from typing import List, Set

from .actions import PanelAction
from .decision import Context, DecisionEngine, RuleEngine


class VirtualPanel:
    """Simulate collaborative panel of doctors using pluggable engines."""

    def __init__(self, decision_engine: DecisionEngine | None = None):

        self.turn = 0
        self.last_case_info = ""
        self.past_infos: List[str] = []
        self.triggered_keywords: Set[str] = set()
        self.engine = decision_engine or RuleEngine()

    def deliberate(self, case_info: str) -> PanelAction:
        """Run one deliberation step and return the chosen action."""

        self.last_case_info = case_info
        self.past_infos.append(case_info)
        self.turn += 1

        ctx = Context(
            turn=self.turn,
            past_infos=self.past_infos,
            triggered_keywords=self.triggered_keywords,
        )
        return self.engine.decide(ctx)
