from __future__ import annotations

"""Utility functions for CLI session tokens."""

import json
import os
import time
from pathlib import Path
from typing import Optional

import httpx
import jwt

TOKEN_PATH = Path.home() / ".dx0" / "token.json"


def _save_tokens(access: str, refresh: str, path: Path = TOKEN_PATH) -> None:
    """Persist ``access`` and ``refresh`` tokens to ``path`` with expiry."""

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


def load_tokens(path: Path = TOKEN_PATH) -> Optional[dict]:
    """Return token data loaded from ``path`` if it exists."""

    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def login(
    api_url: str, username: str, password: str, *, path: Path = TOKEN_PATH
) -> dict:
    """Authenticate with ``api_url`` and store the returned tokens."""

    res = httpx.post(
        f"{api_url.rstrip('/')}/login",
        json={"username": username, "password": password},
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    _save_tokens(data["access_token"], data["refresh_token"], path)
    return data


def refresh(api_url: str, refresh_token: str, *, path: Path = TOKEN_PATH) -> dict:
    """Refresh ``refresh_token`` via ``api_url`` and persist new tokens."""

    res = httpx.post(
        f"{api_url.rstrip('/')}/refresh",
        json={"refresh_token": refresh_token},
        timeout=30,
    )
    res.raise_for_status()
    data = res.json()
    _save_tokens(data["access_token"], data["refresh_token"], path)
    return data


def get_access_token(api_url: str, *, path: Path = TOKEN_PATH) -> str:
    """Return a valid access token, refreshing if necessary."""

    data = load_tokens(path)
    if not data:
        raise RuntimeError("No saved tokens. Run 'dx0 login' first.")
    if int(data.get("expires", 0)) <= int(time.time()):
        data = refresh(api_url, data["refresh_token"], path=path)
    return data["access_token"]
