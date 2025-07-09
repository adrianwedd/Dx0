"""Convert case JSON, CSV, or directory to a SQLite database."""

from __future__ import annotations

import argparse
import os
from typing import Iterable

from sdb.case_database import CaseDatabase
from sdb.sqlite_db import save_to_sqlite


def _load_cases(path: str) -> Iterable[dict[str, str]]:
    if os.path.isdir(path):
        db = CaseDatabase.load_from_directory(path)
    elif path.endswith(".csv"):
        db = CaseDatabase.load_from_csv(path)
    else:
        db = CaseDatabase.load_from_json(path)
    return [
        {"id": c.id, "summary": c.summary, "full_text": c.full_text}
        for c in db.cases.values()
    ]


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Export cases to SQLite")
    parser.add_argument("input", help="Case directory, CSV, or JSON file")
    parser.add_argument("output", help="Destination SQLite file")
    parsed = parser.parse_args(args)

    cases = _load_cases(parsed.input)
    save_to_sqlite(parsed.output, cases)


if __name__ == "__main__":  # pragma: no cover
    main()
