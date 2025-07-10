import sqlite3
import time
from typing import Optional, Tuple


class SessionDB:
    """Store session tokens and budget info in a SQLite database."""

    def __init__(self, path: str = "sessions.db", ttl: int = 3600) -> None:
        self.path = path
        self.ttl = ttl
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS sessions "
                    "(token TEXT PRIMARY KEY, "
                    "username TEXT NOT NULL, "
                    "issue_time REAL NOT NULL)"
                )
            )
            cur = conn.execute("PRAGMA table_info(sessions)")
            cols = [row[1] for row in cur.fetchall()]
            if "budget_limit" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN budget_limit REAL")
            if "amount_spent" not in cols:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN amount_spent REAL DEFAULT 0"
                )
            conn.commit()

    def add(
        self,
        token: str,
        username: str,
        issue_time: Optional[float] = None,
        *,
        budget_limit: Optional[float] = None,
        amount_spent: float = 0.0,
    ) -> None:
        """Insert a session record."""

        ts = issue_time if issue_time is not None else time.time()
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT OR REPLACE INTO sessions "
                    "(token, username, issue_time, budget_limit, amount_spent) "
                    "VALUES (?, ?, ?, ?, ?)"
                ),
                (token, username, ts, budget_limit, amount_spent),
            )
            conn.commit()

    def remove(self, token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE token=?", (token,))
            conn.commit()

    def get(self, token: str) -> Optional[str]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT username, issue_time FROM sessions WHERE token=?", (token,)
            )
            row = cur.fetchone()
        if not row:
            return None
        username, issue_time = row
        if time.time() - issue_time > self.ttl:
            self.remove(token)
            return None
        return username

    def get_budget(self, token: str) -> Tuple[Optional[float], float]:
        """Return budget limit and amount spent for ``token``."""

        with self._connect() as conn:
            cur = conn.execute(
                "SELECT budget_limit, amount_spent FROM sessions WHERE token=?",
                (token,),
            )
            row = cur.fetchone()
        if not row:
            return None, 0.0
        return row[0], row[1] if row[1] is not None else 0.0

    def update_spent(self, token: str, spent: float) -> None:
        """Persist updated ``spent`` amount for ``token``."""

        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET amount_spent=? WHERE token=?",
                (spent, token),
            )
            conn.commit()

    def cleanup(self) -> None:
        cutoff = time.time() - self.ttl
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE issue_time < ?", (cutoff,))
            conn.commit()
