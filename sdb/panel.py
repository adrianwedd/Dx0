"""Virtual panel of doctors using simple rule-based heuristics."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from .protocol import ActionType


@dataclass
class PanelAction:
    """Action proposed by the panel with its content."""

    action_type: ActionType
    content: str


class VirtualPanel:
    """Simulate collaborative panel of doctors with simple heuristics."""

    DEFAULT_KEYWORD_ACTIONS: Dict[str, Tuple[ActionType, str]] = {
        "cough": (ActionType.TEST, "chest x-ray"),
    }

    def __init__(
        self,
        keyword_actions: Optional[Dict[str, Tuple[ActionType, str]]] = None,
    ):
        self.turn = 0
        self.last_case_info = ""
        self.past_infos: List[str] = []
        self.keyword_actions = keyword_actions or self.DEFAULT_KEYWORD_ACTIONS
        self.triggered_keywords: Set[str] = set()

    def _check_keyword_rules(self) -> Optional[PanelAction]:
        """Return an action if any keyword rule matches accumulated info."""

        text = " ".join(self.past_infos).lower()
        for keyword, (atype, content) in self.keyword_actions.items():
            if (
                keyword.lower() in text
                and keyword not in self.triggered_keywords
            ):
                self.triggered_keywords.add(keyword)
                return PanelAction(atype, content)
        return None

    def deliberate(self, case_info: str) -> PanelAction:
        """Very small demo implementation of the Chain of Debate.

        Parameters
        ----------
        case_info:
            Latest information snippet from the gatekeeper.
        """

        self.last_case_info = case_info
        self.past_infos.append(case_info)
        self.turn += 1

        if self.turn == 1:
            # Dr. Hypothesis asks for key symptom information
            return PanelAction(ActionType.QUESTION, "chief complaint")

        action = self._check_keyword_rules()
        if action:
            return action

        if self.turn == 2:
            # Test-Chooser orders a basic test
            return PanelAction(ActionType.TEST, "complete blood count")
        elif self.turn == 3:
            # Challenger requests additional info
            return PanelAction(ActionType.QUESTION, "physical examination")
        else:
            # Stewardship/Checklist propose a diagnosis to finish
            return PanelAction(ActionType.DIAGNOSIS, "viral infection")
