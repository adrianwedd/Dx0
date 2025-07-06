from dataclasses import dataclass
from typing import Dict, Any
import re

from .protocol import ActionType

from .case_database import CaseDatabase, Case

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

    def register_test_result(self, test_name: str, result: str):
        """Add known test result for the current case."""
        self.known_tests[test_name.lower()] = result

    def answer_question(self, query: str) -> QueryResult:
        """Return relevant snippet from case or synthetic result."""

        # Very small XML parser for <question>, <test>, <diagnosis>
        m = re.match(r"<(?P<tag>\w+)>(?P<text>.*)</\w+>", query.strip(), re.S)
        if not m:
            return QueryResult("Invalid query", synthetic=True)

        tag = m.group("tag")
        text = m.group("text").strip()

        if tag == ActionType.DIAGNOSIS.value:
            # We never reveal the diagnosis
            return QueryResult("Diagnosis queries are not allowed", synthetic=True)

        if tag == ActionType.QUESTION.value:
            # Refuse vague or diagnostic questions
            if any(word in text.lower() for word in ["diagnosis", "differential", "what is wrong"]):
                return QueryResult("I can only answer explicit questions about findings.", synthetic=True)

            # Search summary and full text for the answer (naive search)
            for section in [self.case.summary, self.case.full_text]:
                idx = section.lower().find(text.lower())
                if idx != -1:
                    snippet = section[max(0, idx-40): idx+40]
                    return QueryResult(content=snippet, synthetic=False)
            return QueryResult("No information available", synthetic=True)

        if tag == ActionType.TEST.value:
            result = self.known_tests.get(text.lower())
            if result:
                return QueryResult(result, synthetic=False)
            return QueryResult("Synthetic result: normal", synthetic=True)

        return QueryResult("Unknown action", synthetic=True)
