"""Shared HTTP client utilities using httpx."""

from __future__ import annotations

import httpx

_client: httpx.Client | None = None


def get_client() -> httpx.Client:
    """Return a global ``httpx.Client`` instance."""
    global _client
    if _client is None:
        _client = httpx.Client(timeout=30)
    return _client


def close_client() -> None:
    """Close the global ``httpx.Client`` instance if open."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
