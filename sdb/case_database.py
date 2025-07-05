from dataclasses import dataclass
from typing import List, Dict

@dataclass
class Case:
    id: str
    summary: str
    full_text: str

class CaseDatabase:
    """Stub for CPC case storage."""

    def __init__(self, cases: List[Case]):
        self.cases = {case.id: case for case in cases}

    def get_case(self, case_id: str) -> Case:
        return self.cases[case_id]
