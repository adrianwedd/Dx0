"""Example persona plugin providing an optimistic perspective."""

from __future__ import annotations

from typing import List


def optimistic_chain() -> List[str]:
    """Return a persona chain including the Optimist persona."""

    return [
        "hypothesis_system",
        "optimist_system",
        "test_chooser_system",
        "challenger_system",
        "stewardship_system",
        "checklist_system",
    ]
