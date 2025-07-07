"""Utilities for storing case data in SQLite."""

from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List

from .case_database import Case, CaseDatabase


def save_to_sqlite(path: str, cases: Iterable[Dict[str, object]]) -> None:
    """Save an iterable of case dicts to ``path``.

    Each case should have ``id``, ``summary`` and ``full_text`` fields.
    An existing database will be overwritten.
    """

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        (
            "CREATE TABLE IF NOT EXISTS cases ("
            "id TEXT PRIMARY KEY, summary TEXT, full_text TEXT)"
        )
    )
    cur.execute("DELETE FROM cases")
    for case in cases:
        cur.execute(
            "INSERT INTO cases (id, summary, full_text) VALUES (?, ?, ?)",
            (
                str(case["id"]),
                str(case["summary"]),
                str(case["full_text"]),
            ),
        )
    conn.commit()
    conn.close()


def load_from_sqlite(path: str) -> CaseDatabase:
    """Load cases from a SQLite database at ``path``."""

    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute("SELECT id, summary, full_text FROM cases")
    rows = cur.fetchall()
    conn.close()
    cases: List[Case] = [
        Case(id=row[0], summary=row[1], full_text=row[2]) for row in rows
    ]
    return CaseDatabase(cases)
