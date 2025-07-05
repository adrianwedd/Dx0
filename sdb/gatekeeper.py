from dataclasses import dataclass
from typing import Dict, Any

from .case_database import CaseDatabase, Case

@dataclass
class QueryResult:
    content: str
    synthetic: bool = False

class Gatekeeper:
    """Information oracle mediating access to the case."""

    def __init__(self, db: CaseDatabase, case_id: str):
        self.case = db.get_case(case_id)

    def answer_question(self, query: str) -> QueryResult:
        """Return relevant snippet from case or synthetic result."""
        # TODO: restrict to explicit findings and synthesize unseen tests
        return QueryResult(content="Not implemented")
