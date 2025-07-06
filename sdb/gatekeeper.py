from dataclasses import dataclass
from typing import Dict
import json
import os
import re
import xml.etree.ElementTree as ET

from .protocol import ActionType

from .case_database import CaseDatabase


@dataclass
class QueryResult:
    """Response content from the gatekeeper.

    Attributes
    ----------
    content:
        Text snippet answering the query.
    synthetic:
        Whether the response is generated rather than from the case.
    """

    content: str

    synthetic: bool = False


class Gatekeeper:
    """Information oracle mediating access to the case."""

    def __init__(self, db: CaseDatabase, case_id: str):
        self.case = db.get_case(case_id)
        self.known_tests: Dict[str, str] = {}

    def load_results_from_json(self, path: str) -> None:
        """Load test result fixtures from a JSON file."""

        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for name, result in data.items():
            self.register_test_result(name, str(result))

    def register_test_result(self, test_name: str, result: str):
        """Add known test result for the current case."""
        self.known_tests[test_name.lower()] = result

    def answer_question(self, query: str) -> QueryResult:
        """Return relevant snippet from case or synthetic result."""

        # Tiny XML parser for <question>, <test> or <diagnosis>
        try:
            root = ET.fromstring(query.strip())
        except ET.ParseError:
            return QueryResult("Invalid query", synthetic=True)

        tags = {el.tag for el in root.iter()}
        if (
            ActionType.QUESTION.value in tags
            and ActionType.TEST.value in tags
        ):
            return QueryResult(
                "Cannot mix questions and tests in one request",
                synthetic=True,
            )

        tag = root.tag
        text = (root.text or "").strip()

        if tag == ActionType.DIAGNOSIS.value:
            # We never reveal the diagnosis
            return QueryResult(
                "Diagnosis queries are not allowed",
                synthetic=True,
            )

        if tag == ActionType.QUESTION.value:
            # Refuse vague or diagnostic questions
            if any(
                word in text.lower()
                for word in ["diagnosis", "differential", "what is wrong"]
            ):
                return QueryResult(
                    "I can only answer explicit questions about findings.",
                    synthetic=True,
                )

            # Search summary and full text for the answer using
            # case-insensitive matching
            pattern = re.compile(re.escape(text), re.IGNORECASE | re.DOTALL)
            for section in [self.case.summary, self.case.full_text]:
                m = pattern.search(section)
                if m:
                    start = max(0, m.start() - 40)
                    end = min(len(section), m.end() + 40)
                    snippet = section[start:end]
                    return QueryResult(content=snippet, synthetic=False)
            return QueryResult("No information available", synthetic=True)

        if tag == ActionType.TEST.value:
            result = self.known_tests.get(text.lower())
            if result:
                return QueryResult(result, synthetic=False)
            return QueryResult("Synthetic result: normal", synthetic=True)

        return QueryResult("Unknown action", synthetic=True)
