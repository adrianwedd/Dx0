"""Backward compatibility module for session storage."""

from .session_store import SessionStore


class SessionDB(SessionStore):
    """Deprecated alias for :class:`SessionStore`."""

    pass
