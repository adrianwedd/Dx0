"""Abstractions for communicating with different LLM providers."""

from __future__ import annotations

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import List

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None

import requests
from .metrics import LLM_LATENCY, LLM_TOKENS

logger = logging.getLogger(__name__)


class LLMClient(ABC):
    """Generic interface for chat-based language models with metrics."""

    def chat(self, messages: List[dict], model: str) -> str | None:
        """Return the assistant reply and record latency and token metrics."""

        start = time.perf_counter()
        reply = self._chat(messages, model)
        duration = time.perf_counter() - start
        LLM_LATENCY.observe(duration)
        tokens = self._count_tokens(messages)
        if reply is not None:
            tokens += self._count_tokens(
                [{"role": "assistant", "content": reply}]
            )
        LLM_TOKENS.inc(tokens)
        logger.info(
            json.dumps(
                {
                    "event": "llm_chat",
                    "model": model,
                    "latency": duration,
                    "tokens": tokens,
                }
            )
        )
        return reply

    @abstractmethod
    def _chat(self, messages: List[dict], model: str) -> str | None:
        """Implement provider-specific chat call."""
        raise NotImplementedError

    @staticmethod
    def _count_tokens(messages: List[dict]) -> int:
        """Approximate the number of tokens in a list of messages."""

        return sum(len(m.get("content", "").split()) for m in messages)


class OpenAIClient(LLMClient):
    """Client for the OpenAI chat completion API."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

    def _chat(self, messages: List[dict], model: str) -> str | None:
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

    def _chat(self, messages: List[dict], model: str) -> str | None:
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages}
        try:
            resp = requests.post(url, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content")
        except Exception:  # pragma: no cover - network or server issues
            return None
