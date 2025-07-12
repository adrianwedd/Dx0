import sqlite3
import time
from typing import Optional, Tuple


class SessionStore:
    """Persist refresh tokens and budget info using SQLite."""

    def __init__(self, path: str = "sessions.db", ttl: int = 3600) -> None:
        self.path = path
        self.ttl = ttl
        self.migrate()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def migrate(self) -> None:
        """Create or update the sessions table."""
        with self._connect() as conn:
            conn.execute(
                (
                    "CREATE TABLE IF NOT EXISTS sessions "
                    "(session_id TEXT PRIMARY KEY, "
                    "refresh_token TEXT NOT NULL, "
                    "username TEXT NOT NULL, "
                    "issue_time REAL NOT NULL)"
                )
            )
            cur = conn.execute("PRAGMA table_info(sessions)")
            cols = [row[1] for row in cur.fetchall()]
            if "token" in cols and "session_id" not in cols:
                conn.execute("DROP TABLE sessions")
                conn.execute(
                    (
                        "CREATE TABLE sessions "
                        "(session_id TEXT PRIMARY KEY, "
                        "refresh_token TEXT NOT NULL, "
                        "username TEXT NOT NULL, "
                        "issue_time REAL NOT NULL, "
                        "budget_limit REAL, amount_spent REAL DEFAULT 0)"
                    )
                )
                cols = ["session_id", "refresh_token", "username", "issue_time", "budget_limit", "amount_spent"]
            if "budget_limit" not in cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN budget_limit REAL")
            if "amount_spent" not in cols:
                conn.execute(
                    "ALTER TABLE sessions ADD COLUMN amount_spent REAL DEFAULT 0"
                )
            conn.commit()

    def add(
        self,
        session_id: str,
        username: str,
        refresh_token: str,
        issue_time: Optional[float] = None,
        *,
        budget_limit: Optional[float] = None,
        amount_spent: float = 0.0,
    ) -> None:
        """Insert or update a session record."""

        ts = issue_time if issue_time is not None else time.time()
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT OR REPLACE INTO sessions "
                    "(session_id, refresh_token, username, issue_time, "
                    "budget_limit, amount_spent) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                ),
                (session_id, refresh_token, username, ts, budget_limit, amount_spent),
            )
            conn.commit()

    def remove(self, refresh_token: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE refresh_token=?", (refresh_token,))
            conn.commit()

    def get(self, session_id: str) -> Optional[str]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT username, issue_time FROM sessions WHERE session_id=?",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None
        username, issue_time = row
        if time.time() - issue_time > self.ttl:
            self.remove_by_session(session_id)
            return None
        return username

    def remove_by_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
            conn.commit()

    def get_refresh(self, session_id: str) -> Optional[str]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT refresh_token FROM sessions WHERE session_id=?",
                (session_id,),
            )
            row = cur.fetchone()
        return row[0] if row else None

    def find_by_refresh(self, refresh_token: str) -> Optional[tuple[str, str]]:
        with self._connect() as conn:
            cur = conn.execute(
                "SELECT session_id, username FROM sessions WHERE refresh_token=?",
                (refresh_token,),
            )
            row = cur.fetchone()
        return tuple(row) if row else None

    def update_refresh(self, session_id: str, refresh_token: str, issue_time: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET refresh_token=?, issue_time=? WHERE session_id=?",
                (refresh_token, issue_time, session_id),
            )
            conn.commit()

    def get_budget(self, session_id: str) -> Tuple[Optional[float], float]:
        """Return budget limit and amount spent for ``session_id``."""

        with self._connect() as conn:
            cur = conn.execute(
                "SELECT budget_limit, amount_spent FROM sessions WHERE session_id=?",
                (session_id,),
            )
            row = cur.fetchone()
        if not row:
            return None, 0.0
        return row[0], row[1] if row[1] is not None else 0.0

    def update_spent(self, session_id: str, spent: float) -> None:
        """Persist updated ``spent`` amount for ``session_id``."""

        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET amount_spent=? WHERE session_id=?",
                (spent, session_id),
            )
            conn.commit()

    def cleanup(self) -> None:
        cutoff = time.time() - self.ttl
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE issue_time < ?", (cutoff,))
            conn.commit()
