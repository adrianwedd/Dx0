import sqlite3
import time
from typing import Optional


class SessionDB:
    """Store session tokens in a SQLite database."""

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
            conn.commit()

    def add(
        self,
        token: str,
        username: str,
        issue_time: Optional[float] = None,
    ) -> None:
        ts = issue_time if issue_time is not None else time.time()
        with self._connect() as conn:
            conn.execute(
                (
                    "INSERT OR REPLACE INTO sessions "
                    "(token, username, issue_time) VALUES (?, ?, ?)"
                ),
                (token, username, ts),
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

    def cleanup(self) -> None:
        cutoff = time.time() - self.ttl
        with self._connect() as conn:
            conn.execute("DELETE FROM sessions WHERE issue_time < ?", (cutoff,))
            conn.commit()
