from dataclasses import dataclass
from .protocol import ActionType


@dataclass
class PanelAction:
    """Action proposed by the panel with its content."""

    action_type: ActionType
    content: str
