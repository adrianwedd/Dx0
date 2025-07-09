from __future__ import annotations

import sqlite3


class BudgetStore:
    """Persist test spending amounts using SQLite."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "CREATE TABLE IF NOT EXISTS spending (test_name TEXT, amount REAL)"
        )
        conn.commit()
        conn.close()

    def record(self, test_name: str, amount: float) -> None:
        """Add a spending record for ``test_name``."""
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO spending (test_name, amount) VALUES (?, ?)",
            (test_name, amount),
        )
        conn.commit()
        conn.close()

    def total(self) -> float:
        """Return the sum of recorded amounts."""
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute("SELECT SUM(amount) FROM spending")
        row = cur.fetchone()
        conn.close()
        return float(row[0]) if row and row[0] is not None else 0.0

    def clear(self) -> None:
        """Remove all spending records."""
        conn = sqlite3.connect(self.path)
        cur = conn.cursor()
        cur.execute("DELETE FROM spending")
        conn.commit()
        conn.close()
