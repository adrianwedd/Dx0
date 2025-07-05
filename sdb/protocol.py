from enum import Enum

class ActionType(Enum):
    QUESTION = "question"
    TEST = "test"
    DIAGNOSIS = "diagnosis"

# Simple helpers for building XML tags

def build_action(action: ActionType, content: str) -> str:
    return f"<{action.value}>{content}</{action.value}>"
