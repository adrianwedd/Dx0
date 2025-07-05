import json
import os
from dataclasses import dataclass
from typing import List, Dict, Iterable

@dataclass
class Case:
    id: str
    summary: str
    full_text: str

class CaseDatabase:
    """Stub for CPC case storage."""

    def __init__(self, cases: Iterable[Case]):
        self.cases = {case.id: case for case in cases}

    def get_case(self, case_id: str) -> Case:
        return self.cases[case_id]

    @staticmethod
    def load_from_json(path: str) -> "CaseDatabase":
        """Load cases from a JSON file.

        The JSON file should contain a list of objects with ``id``,
        ``summary`` and ``full_text`` fields.
        """
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        cases = [Case(**item) for item in data]
        return CaseDatabase(cases)

    @staticmethod
    def load_from_directory(path: str) -> "CaseDatabase":
        """Load cases from a directory of text files.

        Each subdirectory should contain ``summary.txt`` and ``full.txt``
        files. The subdirectory name is used as the case ``id``.
        """
        cases = []
        for case_id in sorted(os.listdir(path)):
            case_dir = os.path.join(path, case_id)
            summary_file = os.path.join(case_dir, "summary.txt")
            full_file = os.path.join(case_dir, "full.txt")
            if not os.path.isfile(summary_file) or not os.path.isfile(full_file):
                continue
            with open(summary_file, "r", encoding="utf-8") as sf:
                summary = sf.read().strip()
            with open(full_file, "r", encoding="utf-8") as ff:
                full_text = ff.read().strip()
            cases.append(Case(id=case_id, summary=summary, full_text=full_text))
        return CaseDatabase(cases)
