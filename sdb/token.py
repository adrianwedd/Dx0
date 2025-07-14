"""Utility functions for CLI session tokens."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
import jwt

TOKEN_PATH = Path.home() / ".dx0" / "token.json"


def _save_tokens(access: str, refresh: str, path: Path | None = None) -> None:
    """Persist ``access`` and ``refresh`` tokens to ``path`` with expiry."""

    path = path or TOKEN_PATH
    payload = jwt.decode(access, options={"verify_signature": False})
    expires = int(payload.get("exp", 0))
    os.makedirs(path.parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "access_token": access,
                "refresh_token": refresh,
                "expires": expires,
            },
            fh,
        )
    os.chmod(path, 0o600)


def load_tokens(path: Path | None = None) -> Optional[dict]:
    """Return token data loaded from ``path`` if it exists."""

    path = path or TOKEN_PATH
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def login(
    api_url: str, username: str, password: str, *, path: Path | None = None
) -> dict:
    """Authenticate with ``api_url`` and store the returned tokens."""

    path = path or TOKEN_PATH
    res = httpx.post(
        f"{api_url.rstrip('/')}/login",
        json={"username": username, "password": password},
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    _save_tokens(data["access_token"], data["refresh_token"], path)
    return data


def refresh(api_url: str, refresh_token: str, *, path: Path | None = None) -> dict:
    """Refresh ``refresh_token`` via ``api_url`` and persist new tokens."""

    path = path or TOKEN_PATH
    res = httpx.post(
        f"{api_url.rstrip('/')}/refresh",
        json={"refresh_token": refresh_token},
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    _save_tokens(data["access_token"], data["refresh_token"], path)
    return data


def get_access_token(api_url: str, *, path: Path | None = None) -> str:
    """Return a valid access token, refreshing if necessary."""

    path = path or TOKEN_PATH
    data = load_tokens(path)
    if not data:
        raise RuntimeError("No saved tokens. Run 'dx0 login' first.")
    if int(data.get("expires", 0)) <= int(time.time()):
        data = refresh(api_url, data["refresh_token"], path=path)
    return data["access_token"]
