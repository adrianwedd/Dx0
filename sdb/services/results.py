"""Helpers for tracking ordered tests and final diagnoses."""

from __future__ import annotations


class ResultAggregator:
    """Collect ordered tests and final diagnosis."""

    def __init__(self) -> None:
        """Initialize empty result tracking structures."""
        self.ordered_tests: list[str] = []
        self.final_diagnosis: str | None = None
        self.finished: bool = False

    def record_test(self, test_name: str) -> None:
        """Add ``test_name`` to the ordered test list."""
        self.ordered_tests.append(test_name)

    def record_diagnosis(self, diagnosis: str) -> None:
        """Store ``diagnosis`` and mark the session as finished."""
        self.final_diagnosis = diagnosis
        self.finished = True
