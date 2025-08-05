"""SQLite-based session storage backend for backward compatibility."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import time
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

from .session_backend import SessionBackend, SessionData


class SQLiteSessionBackend(SessionBackend):
    """SQLite-based session backend using existing SessionStore pattern."""
    
    def __init__(self, db_path: str = "sessions.db", max_workers: int = 4):
        self.db_path = db_path
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._setup_database()
    
    def _setup_database(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            # Main sessions table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    group_name TEXT DEFAULT 'default',
                    refresh_token TEXT NOT NULL,
                    issue_time REAL NOT NULL,
                    budget_limit REAL,
                    amount_spent REAL DEFAULT 0.0,
                    metadata TEXT DEFAULT '{}'
                )
            """)
            
            # Message timestamps for rate limiting
            conn.execute("""
                CREATE TABLE IF NOT EXISTS message_timestamps (
                    session_id TEXT,
                    timestamp REAL NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)
            
            # Failed login attempts
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failed_logins (
                    identifier TEXT,
                    timestamp REAL NOT NULL
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_refresh_token ON sessions (refresh_token)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_message_session ON message_timestamps (session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_failed_login ON failed_logins (identifier)")
            
            conn.commit()
    
    async def _run_in_executor(self, func, *args):
        """Run synchronous database operations in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, func, *args)
    
    def _get_session_sync(self, session_id: str) -> Optional[SessionData]:
        """Synchronous session retrieval."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            # Get session data
            cursor = conn.execute("""
                SELECT session_id, username, group_name, refresh_token, issue_time,
                       budget_limit, amount_spent, metadata
                FROM sessions WHERE session_id = ?
            """, (session_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Get message timestamps
            cursor = conn.execute("""
                SELECT timestamp FROM message_timestamps 
                WHERE session_id = ? ORDER BY timestamp DESC
            """, (session_id,))
            
            timestamps = [row[0] for row in cursor.fetchall()]
            
            # Create session data
            try:
                session_data = {
                    "session_id": row["session_id"],
                    "username": row["username"],
                    "group_name": row["group_name"] or "default",
                    "refresh_token": row["refresh_token"] or "",
                    "issue_time": row["issue_time"],
                    "budget_limit": row["budget_limit"],
                    "amount_spent": row["amount_spent"] or 0.0,
                    "message_timestamps": timestamps,
                    "failed_login_attempts": [],  # Not per-session
                    "metadata": json.loads(row["metadata"] or "{}"),
                }
                return SessionData.from_dict(session_data)
            except (json.JSONDecodeError, KeyError):
                return None
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session data by session ID."""
        return await self._run_in_executor(self._get_session_sync, session_id)
    
    def _set_session_sync(self, session_data: SessionData, ttl: int) -> None:
        """Synchronous session storage."""
        with sqlite3.connect(self.db_path) as conn:
            # Insert or update session
            conn.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, username, group_name, refresh_token, issue_time, 
                 budget_limit, amount_spent, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                session_data.session_id,
                session_data.username,
                session_data.group_name,
                session_data.refresh_token,
                session_data.issue_time,
                session_data.budget_limit,
                session_data.amount_spent,
                json.dumps(session_data.metadata),
            ))
            
            # Update message timestamps
            conn.execute("DELETE FROM message_timestamps WHERE session_id = ?", (session_data.session_id,))
            if session_data.message_timestamps:
                conn.executemany(
                    "INSERT INTO message_timestamps (session_id, timestamp) VALUES (?, ?)",
                    [(session_data.session_id, ts) for ts in session_data.message_timestamps]
                )
            
            conn.commit()
    
    async def set_session(self, session_data: SessionData, ttl: int) -> None:
        """Store or update session data with TTL."""
        await self._run_in_executor(self._set_session_sync, session_data, ttl)
    
    def _delete_session_sync(self, session_id: str) -> bool:
        """Synchronous session deletion."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM message_timestamps WHERE session_id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data. Returns True if session existed."""
        return await self._run_in_executor(self._delete_session_sync, session_id)
    
    def _find_by_refresh_token_sync(self, refresh_token: str) -> Optional[SessionData]:
        """Synchronous refresh token lookup."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT session_id FROM sessions WHERE refresh_token = ?", (refresh_token,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._get_session_sync(row[0])
    
    async def find_by_refresh_token(self, refresh_token: str) -> Optional[SessionData]:
        """Find session by refresh token."""
        return await self._run_in_executor(self._find_by_refresh_token_sync, refresh_token)
    
    def _update_refresh_token_sync(self, session_id: str, new_refresh_token: str) -> bool:
        """Synchronous refresh token update."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE sessions SET refresh_token = ?, issue_time = ? 
                WHERE session_id = ?
            """, (new_refresh_token, time.time(), session_id))
            conn.commit()
            return cursor.rowcount > 0
    
    async def update_refresh_token(self, session_id: str, new_refresh_token: str) -> bool:
        """Update refresh token for session. Returns True if successful."""
        return await self._run_in_executor(self._update_refresh_token_sync, session_id, new_refresh_token)
    
    def _update_budget_sync(self, session_id: str, amount_spent: float) -> bool:
        """Synchronous budget update."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                UPDATE sessions SET amount_spent = ? WHERE session_id = ?
            """, (amount_spent, session_id))
            conn.commit()
            return cursor.rowcount > 0
    
    async def update_budget(self, session_id: str, amount_spent: float) -> bool:
        """Update budget spent amount. Returns True if successful."""
        return await self._run_in_executor(self._update_budget_sync, session_id, amount_spent)
    
    def _add_message_timestamp_sync(self, session_id: str, timestamp: float) -> bool:
        """Synchronous message timestamp addition."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO message_timestamps (session_id, timestamp) VALUES (?, ?)
            """, (session_id, timestamp))
            
            # Keep only recent timestamps (last 100)
            conn.execute("""
                DELETE FROM message_timestamps 
                WHERE session_id = ? AND timestamp NOT IN (
                    SELECT timestamp FROM message_timestamps 
                    WHERE session_id = ? ORDER BY timestamp DESC LIMIT 100
                )
            """, (session_id, session_id))
            
            conn.commit()
            return True
    
    async def add_message_timestamp(self, session_id: str, timestamp: Optional[float] = None) -> bool:
        """Add message timestamp for rate limiting. Returns True if successful."""
        timestamp = timestamp or time.time()
        return await self._run_in_executor(self._add_message_timestamp_sync, session_id, timestamp)
    
    def _get_message_count_sync(self, session_id: str, window_seconds: int) -> int:
        """Synchronous message count retrieval."""
        cutoff = time.time() - window_seconds
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM message_timestamps 
                WHERE session_id = ? AND timestamp >= ?
            """, (session_id, cutoff))
            return cursor.fetchone()[0]
    
    async def get_message_count(self, session_id: str, window_seconds: int) -> int:
        """Get count of messages within the time window."""
        return await self._run_in_executor(self._get_message_count_sync, session_id, window_seconds)
    
    def _add_failed_login_sync(self, identifier: str, timestamp: float) -> bool:
        """Synchronous failed login addition."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO failed_logins (identifier, timestamp) VALUES (?, ?)
            """, (identifier, timestamp))
            
            # Keep only recent attempts (last 20 per identifier)
            conn.execute("""
                DELETE FROM failed_logins 
                WHERE identifier = ? AND timestamp NOT IN (
                    SELECT timestamp FROM failed_logins 
                    WHERE identifier = ? ORDER BY timestamp DESC LIMIT 20
                )
            """, (identifier, identifier))
            
            conn.commit()
            return True
    
    async def add_failed_login(self, identifier: str, timestamp: Optional[float] = None) -> bool:
        """Add failed login attempt for IP or session. Returns True if successful."""
        timestamp = timestamp or time.time()
        return await self._run_in_executor(self._add_failed_login_sync, identifier, timestamp)
    
    def _get_failed_login_count_sync(self, identifier: str, window_seconds: int) -> int:
        """Synchronous failed login count retrieval."""
        cutoff = time.time() - window_seconds
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT COUNT(*) FROM failed_logins 
                WHERE identifier = ? AND timestamp >= ?
            """, (identifier, cutoff))
            return cursor.fetchone()[0]
    
    async def get_failed_login_count(self, identifier: str, window_seconds: int) -> int:
        """Get count of failed login attempts within the time window."""
        return await self._run_in_executor(self._get_failed_login_count_sync, identifier, window_seconds)
    
    def _cleanup_expired_sessions_sync(self, ttl: int) -> int:
        """Synchronous session cleanup."""
        cutoff = time.time() - ttl
        with sqlite3.connect(self.db_path) as conn:
            # Clean expired sessions
            cursor = conn.execute("DELETE FROM sessions WHERE issue_time < ?", (cutoff,))
            expired_count = cursor.rowcount
            
            # Clean old message timestamps
            conn.execute("DELETE FROM message_timestamps WHERE timestamp < ?", (cutoff,))
            
            # Clean old failed login attempts
            conn.execute("DELETE FROM failed_logins WHERE timestamp < ?", (cutoff,))
            
            conn.commit()
            return expired_count
    
    async def cleanup_expired_sessions(self, ttl: int) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        return await self._run_in_executor(self._cleanup_expired_sessions_sync, ttl)
    
    def _get_active_session_count_sync(self) -> int:
        """Synchronous active session count."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            return cursor.fetchone()[0]
    
    async def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return await self._run_in_executor(self._get_active_session_count_sync)
    
    async def health_check(self) -> bool:
        """Check if backend is healthy and responsive."""
        try:
            await self._run_in_executor(lambda: sqlite3.connect(self.db_path).execute("SELECT 1").fetchone())
            return True
        except Exception:
            return False
    
    def close(self) -> None:
        """Close thread pool executor."""
        self.executor.shutdown(wait=True)