"""Abstract session storage backend interface."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from enum import Enum


class SessionStatus(Enum):
    """Session status enumeration."""
    ACTIVE = "active"
    EXPIRED = "expired"
    INVALID = "invalid"


@dataclass
class SessionData:
    """Session data model with all session-related information."""
    
    # Core session fields
    session_id: str
    username: str
    group_name: str = "default"
    refresh_token: str = ""
    issue_time: float = field(default_factory=time.time)
    
    # Budget tracking
    budget_limit: Optional[float] = None
    amount_spent: float = 0.0
    
    # Rate limiting data
    message_timestamps: List[float] = field(default_factory=list)
    
    # Failed login tracking (for IP-based sessions)
    failed_login_attempts: List[float] = field(default_factory=list)
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_expired(self, ttl: int) -> bool:
        """Check if session has expired based on TTL."""
        return time.time() - self.issue_time > ttl
    
    def add_message_timestamp(self, timestamp: Optional[float] = None) -> None:
        """Add a message timestamp for rate limiting."""
        timestamp = timestamp or time.time()
        self.message_timestamps.append(timestamp)
    
    def get_recent_messages(self, window_seconds: int) -> List[float]:
        """Get message timestamps within the specified window."""
        cutoff = time.time() - window_seconds
        return [ts for ts in self.message_timestamps if ts >= cutoff]
    
    def cleanup_old_timestamps(self, window_seconds: int) -> None:
        """Remove old timestamps outside the rate limiting window."""
        cutoff = time.time() - window_seconds
        self.message_timestamps = [ts for ts in self.message_timestamps if ts >= cutoff]
    
    def add_failed_login(self, timestamp: Optional[float] = None) -> None:
        """Add a failed login attempt timestamp."""
        timestamp = timestamp or time.time()
        self.failed_login_attempts.append(timestamp)
    
    def get_recent_failed_logins(self, window_seconds: int) -> List[float]:
        """Get failed login attempts within the specified window."""
        cutoff = time.time() - window_seconds
        return [ts for ts in self.failed_login_attempts if ts >= cutoff]
    
    def cleanup_old_failed_logins(self, window_seconds: int) -> None:
        """Remove old failed login attempts outside the tracking window."""
        cutoff = time.time() - window_seconds
        self.failed_login_attempts = [ts for ts in self.failed_login_attempts if ts >= cutoff]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert session data to dictionary for serialization."""
        return {
            "session_id": self.session_id,
            "username": self.username,
            "group_name": self.group_name,
            "refresh_token": self.refresh_token,
            "issue_time": self.issue_time,
            "budget_limit": self.budget_limit,
            "amount_spent": self.amount_spent,
            "message_timestamps": self.message_timestamps,
            "failed_login_attempts": self.failed_login_attempts,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SessionData:
        """Create SessionData from dictionary."""
        return cls(
            session_id=data["session_id"],
            username=data["username"],
            group_name=data.get("group_name", "default"),
            refresh_token=data.get("refresh_token", ""),
            issue_time=data.get("issue_time", time.time()),
            budget_limit=data.get("budget_limit"),
            amount_spent=data.get("amount_spent", 0.0),
            message_timestamps=data.get("message_timestamps", []),
            failed_login_attempts=data.get("failed_login_attempts", []),
            metadata=data.get("metadata", {}),
        )


class SessionBackend(ABC):
    """Abstract interface for session storage backends."""
    
    @abstractmethod
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session data by session ID."""
        raise NotImplementedError
    
    @abstractmethod
    async def set_session(self, session_data: SessionData, ttl: int) -> None:
        """Store or update session data with TTL."""
        raise NotImplementedError
    
    @abstractmethod
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data. Returns True if session existed."""
        raise NotImplementedError
    
    @abstractmethod
    async def find_by_refresh_token(self, refresh_token: str) -> Optional[SessionData]:
        """Find session by refresh token."""
        raise NotImplementedError
    
    @abstractmethod
    async def update_refresh_token(self, session_id: str, new_refresh_token: str) -> bool:
        """Update refresh token for session. Returns True if successful."""
        raise NotImplementedError
    
    @abstractmethod
    async def update_budget(self, session_id: str, amount_spent: float) -> bool:
        """Update budget spent amount. Returns True if successful."""
        raise NotImplementedError
    
    @abstractmethod
    async def add_message_timestamp(self, session_id: str, timestamp: Optional[float] = None) -> bool:
        """Add message timestamp for rate limiting. Returns True if successful."""
        raise NotImplementedError
    
    @abstractmethod
    async def get_message_count(self, session_id: str, window_seconds: int) -> int:
        """Get count of messages within the time window."""
        raise NotImplementedError
    
    @abstractmethod
    async def add_failed_login(self, identifier: str, timestamp: Optional[float] = None) -> bool:
        """Add failed login attempt for IP or session. Returns True if successful."""
        raise NotImplementedError
    
    @abstractmethod
    async def get_failed_login_count(self, identifier: str, window_seconds: int) -> int:
        """Get count of failed login attempts within the time window."""
        raise NotImplementedError
    
    @abstractmethod
    async def cleanup_expired_sessions(self, ttl: int) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        raise NotImplementedError
    
    @abstractmethod
    async def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        raise NotImplementedError
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if backend is healthy and responsive."""
        raise NotImplementedError
    
    async def clear_failed_logins(self) -> None:
        """Clear all failed login attempts (for testing). Optional method."""
        pass
    
    async def clear_all_session_data(self) -> None:
        """Clear all session data including message history (for testing). Optional method."""
        pass


class InMemorySessionBackend(SessionBackend):
    """In-memory session backend for development and testing."""
    
    def __init__(self):
        self._sessions: Dict[str, SessionData] = {}
        self._refresh_tokens: Dict[str, str] = {}  # refresh_token -> session_id
        self._failed_logins: Dict[str, List[float]] = {}  # identifier -> timestamps
    
    async def get_session(self, session_id: str) -> Optional[SessionData]:
        """Retrieve session data by session ID."""
        return self._sessions.get(session_id)
    
    async def set_session(self, session_data: SessionData, ttl: int) -> None:
        """Store or update session data with TTL."""
        # TTL is not enforced in memory backend - sessions persist until explicitly cleaned
        self._sessions[session_data.session_id] = session_data
        if session_data.refresh_token:
            self._refresh_tokens[session_data.refresh_token] = session_data.session_id
    
    async def delete_session(self, session_id: str) -> bool:
        """Delete session data. Returns True if session existed."""
        session = self._sessions.pop(session_id, None)
        if session and session.refresh_token:
            self._refresh_tokens.pop(session.refresh_token, None)
        return session is not None
    
    async def find_by_refresh_token(self, refresh_token: str) -> Optional[SessionData]:
        """Find session by refresh token."""
        session_id = self._refresh_tokens.get(refresh_token)
        if session_id:
            return self._sessions.get(session_id)
        return None
    
    async def update_refresh_token(self, session_id: str, new_refresh_token: str) -> bool:
        """Update refresh token for session. Returns True if successful."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        # Remove old refresh token mapping
        if session.refresh_token:
            self._refresh_tokens.pop(session.refresh_token, None)
        
        # Update session and add new mapping
        session.refresh_token = new_refresh_token
        session.issue_time = time.time()
        self._refresh_tokens[new_refresh_token] = session_id
        return True
    
    async def update_budget(self, session_id: str, amount_spent: float) -> bool:
        """Update budget spent amount. Returns True if successful."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.amount_spent = amount_spent
        return True
    
    async def add_message_timestamp(self, session_id: str, timestamp: Optional[float] = None) -> bool:
        """Add message timestamp for rate limiting. Returns True if successful."""
        session = self._sessions.get(session_id)
        if not session:
            return False
        
        session.add_message_timestamp(timestamp)
        return True
    
    async def get_message_count(self, session_id: str, window_seconds: int) -> int:
        """Get count of messages within the time window."""
        session = self._sessions.get(session_id)
        if not session:
            return 0
        
        recent_messages = session.get_recent_messages(window_seconds)
        return len(recent_messages)
    
    async def add_failed_login(self, identifier: str, timestamp: Optional[float] = None) -> bool:
        """Add failed login attempt for IP or session. Returns True if successful."""
        timestamp = timestamp or time.time()
        if identifier not in self._failed_logins:
            self._failed_logins[identifier] = []
        
        self._failed_logins[identifier].append(timestamp)
        return True
    
    async def get_failed_login_count(self, identifier: str, window_seconds: int) -> int:
        """Get count of failed login attempts within the time window."""
        attempts = self._failed_logins.get(identifier, [])
        cutoff = time.time() - window_seconds
        recent_attempts = [ts for ts in attempts if ts >= cutoff]
        return len(recent_attempts)
    
    async def cleanup_expired_sessions(self, ttl: int) -> int:
        """Remove expired sessions. Returns count of removed sessions."""
        cutoff = time.time() - ttl
        expired_sessions = [
            session_id for session_id, session in self._sessions.items()
            if session.issue_time < cutoff
        ]
        
        for session_id in expired_sessions:
            await self.delete_session(session_id)
        
        # Also cleanup old failed login attempts
        for identifier in list(self._failed_logins.keys()):
            attempts = self._failed_logins[identifier]
            recent_attempts = [ts for ts in attempts if ts >= cutoff]
            if recent_attempts:
                self._failed_logins[identifier] = recent_attempts
            else:
                del self._failed_logins[identifier]
        
        return len(expired_sessions)
    
    async def get_active_session_count(self) -> int:
        """Get count of active sessions."""
        return len(self._sessions)
    
    async def health_check(self) -> bool:
        """Check if backend is healthy and responsive."""
        return True
    
    async def clear_failed_logins(self) -> None:
        """Clear all failed login attempts (for testing)."""
        self._failed_logins.clear()
    
    async def clear_all_session_data(self) -> None:
        """Clear all session data including message history (for testing)."""
        self._sessions.clear()
        self._refresh_tokens.clear()
        self._failed_logins.clear()


class SessionBackendFactory:
    """Factory for creating session backends based on configuration."""
    
    @staticmethod
    def create_backend(backend_type: str, **kwargs) -> SessionBackend:
        """Create session backend based on type and configuration."""
        if backend_type == "memory":
            return InMemorySessionBackend()
        elif backend_type == "redis":
            # Import here to avoid dependency issues if Redis not available
            try:
                from .redis_session_backend import RedisSessionBackend
                return RedisSessionBackend(**kwargs)
            except ImportError:
                raise RuntimeError("Redis backend requested but aioredis not available")
        elif backend_type == "sqlite":
            # Import here for lazy loading
            from .sqlite_session_backend import SQLiteSessionBackend
            return SQLiteSessionBackend(**kwargs)
        else:
            raise ValueError(f"Unknown session backend type: {backend_type}")