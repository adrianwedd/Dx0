"""Abstractions for communicating with different LLM providers."""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import List

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None

import requests


class LLMClient(ABC):
    """Generic interface for chat-based language models."""

    @abstractmethod
    def chat(self, messages: List[dict], model: str) -> str | None:
        """Return the assistant reply for the given messages."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Client for the OpenAI chat completion API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def chat(self, messages: List[dict], model: str) -> str | None:
        if openai is None or not self.api_key:
            return None

        openai.api_key = self.api_key
        for _ in range(3):
            try:
                resp = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    max_tokens=64,
                )
                return resp.choices[0].message["content"]
            except Exception:  # pragma: no cover - network issues
                time.sleep(1)
        return None


class OllamaClient(LLMClient):
    """Client for a local Ollama server."""

    def __init__(self, base_url: str = "http://localhost:11434") -> None:
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[dict], model: str) -> str | None:
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content")
        except Exception:  # pragma: no cover - network or server issues
            return None
