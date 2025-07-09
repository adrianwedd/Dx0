import csv
import json
import os
import sqlite3
from dataclasses import dataclass
from typing import Iterable

from pydantic import BaseModel, ValidationError


@dataclass
class Case:
    """A clinical case with an identifier, summary and full text."""

    id: str
    summary: str
    full_text: str


class CaseModel(BaseModel):
    """Validation model for case records."""

    id: str
    summary: str
    full_text: str


class CaseDatabase:
    """Stub for CPC case storage."""

    def __init__(self, cases: Iterable[Case]):
        """Create a database from an iterable of cases.

        Parameters
        ----------
        cases:
            Iterable of :class:`Case` objects to index by their ``id``.
        """

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
        cases = []
        for item in data:
            try:
                model = CaseModel.model_validate(item)
            except ValidationError:
                continue
            cases.append(
                Case(
                    id=model.id.strip(),
                    summary=model.summary.strip(),
                    full_text=model.full_text.strip(),
                )
            )
        return CaseDatabase(cases)

    @staticmethod
    def load_from_csv(path: str) -> "CaseDatabase":
        """Load cases from a CSV file.

        The CSV file should contain ``id``, ``summary`` and ``full_text``
        columns. Rows missing these fields are skipped.
        """
        cases = []
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    model = CaseModel.model_validate(
                        {
                            "id": row.get("id"),
                            "summary": row.get("summary"),
                            "full_text": row.get("full_text"),
                        }
                    )
                except ValidationError:
                    continue
                case_id = model.id.strip()
                if not case_id:
                    continue
                cases.append(
                    Case(
                        id=case_id,
                        summary=model.summary.strip(),
                        full_text=model.full_text.strip(),
                    )
                )
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
            if (
                not os.path.isfile(summary_file)
                or not os.path.isfile(full_file)
            ):
                continue
            with open(summary_file, "r", encoding="utf-8") as sf:
                summary = sf.read().strip()
            with open(full_file, "r", encoding="utf-8") as ff:
                full_text = ff.read().strip()
            cases.append(
                Case(
                    id=case_id,
                    summary=summary,
                    full_text=full_text,
                )
            )
        return CaseDatabase(cases)


class SQLiteCaseDatabase:
    """Case storage backed by SQLite with lazy loading."""

    def __init__(self, path: str):
        """Initialize with a SQLite database ``path``."""

        self.path = path

    def get_case(self, case_id: str) -> Case:
        """Return the case with ``case_id`` from the SQLite file."""

        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "SELECT summary, full_text FROM cases WHERE id=?",
            (case_id,),
        )
        row = cur.fetchone()
        conn.close()
        if row is None:
            raise KeyError(case_id)
        return Case(id=case_id, summary=row[0], full_text=row[1])
