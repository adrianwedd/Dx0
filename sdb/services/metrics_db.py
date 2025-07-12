from __future__ import annotations

"""SQLite storage for session evaluation metrics."""

import sqlite3
from typing import Iterable

from ..evaluation import SessionResult


class MetricsDB:
    """Persist :class:`SessionResult` records in SQLite."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            (
                "CREATE TABLE IF NOT EXISTS results ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "case_id TEXT,"
                "total_cost REAL,"
                "score INTEGER,"
                "correct INTEGER,"
                "duration REAL,"
                "ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
        )
        conn.commit()
        conn.close()

    def record(self, case_id: str, result: SessionResult) -> None:
        """Insert a single session ``result`` for ``case_id``."""
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO results (case_id, total_cost, score, correct, duration) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                case_id,
                result.total_cost,
                result.score,
                int(result.correct),
                result.duration,
            ),
        )
        conn.commit()
        conn.close()

    def bulk_record(self, rows: Iterable[tuple[str, SessionResult]]) -> None:
        """Insert multiple ``(case_id, result)`` rows."""
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.executemany(
            "INSERT INTO results (case_id, total_cost, score, correct, duration) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (
                    case_id,
                    res.total_cost,
                    res.score,
                    int(res.correct),
                    res.duration,
                )
                for case_id, res in rows
            ],
        )
        conn.commit()
        conn.close()
