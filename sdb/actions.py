from dataclasses import dataclass
import json
import re
import xml.etree.ElementTree as ET

from .protocol import ActionType


@dataclass
class PanelAction:
    """Action proposed by the panel with its content."""

    action_type: ActionType
    content: str


def parse_panel_action(text: str) -> PanelAction | None:
    """Parse XML or JSON text into a :class:`PanelAction`.

    Parameters
    ----------
    text:
        Raw model output expected to contain a single action.

    Returns
    -------
    PanelAction | None
        Parsed action or ``None`` if parsing failed.
    """

    text = text.strip()
    if not text:
        return None

    # Try XML-style parsing
    try:
        root = ET.fromstring(text)
        tag = root.tag.lower()
        if tag in {a.value for a in ActionType}:
            content = (root.text or "").strip()
            if content:
                return PanelAction(ActionType(tag), content)
    except ET.ParseError:
        pass

    m = re.search(r"<(question|test|diagnosis)>(.*?)</\\1>", text, re.I | re.S)
    if m:
        tag, content = m.group(1).lower(), m.group(2).strip()
        if content:
            return PanelAction(ActionType(tag), content)

    try:
        data = json.loads(text)
        action = data.get("action") or data.get("type")
        content = data.get("content")
        if action and content:
            action = str(action).lower()
            if action in {a.value for a in ActionType}:
                return PanelAction(ActionType(action), str(content).strip())
    except Exception:
        pass

    return None
