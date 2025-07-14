"""Abstractions for communicating with different LLM providers."""

from __future__ import annotations

import os
import time
import json
import structlog
from abc import ABC, abstractmethod
from typing import List, OrderedDict
from filelock import FileLock
from .config import settings

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    tiktoken = None

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None

from .http_utils import get_client
from .metrics import LLM_LATENCY, LLM_TOKENS

logger = structlog.get_logger(__name__)


class FileCache:
    """Simple JSONL-based LRU cache for LLM responses."""

    def __init__(self, path: str, max_size: int = 128) -> None:
        self.path = path
        self.max_size = max_size
        self.lock = FileLock(self.path + ".lock")
        self.data: OrderedDict[str, str] = OrderedDict()
        if os.path.exists(path):
            try:
                with self.lock:
                    with open(path, "r", encoding="utf-8") as fh:
                        for line in fh:
                            item = json.loads(line)
                            self.data[item["key"]] = item["value"]
            except Exception:  # pragma: no cover - corrupt cache
                self.data.clear()

    def get(self, key: str) -> str | None:
        with self.lock:
            value = self.data.get(key)
            if value is not None:
                # refresh position for LRU
                self.data.pop(key)
                self.data[key] = value
            return value

    def set(self, key: str, value: str) -> None:
        if key in self.data:
            self.data.pop(key)
        elif len(self.data) >= self.max_size:
            self.data.pop(next(iter(self.data)))
        self.data[key] = value
        self._write()

    def _write(self) -> None:
        tmp = self.path + ".tmp"
        with self.lock:
            with open(tmp, "w", encoding="utf-8") as fh:
                for k, v in self.data.items():
                    fh.write(json.dumps({"key": k, "value": v}) + "\n")
            os.replace(tmp, self.path)


class LLMClient(ABC):
    """Generic interface for chat-based language models with metrics."""

    def __init__(self, cache_path: str | None = None, cache_size: int = 128) -> None:
        self.cache = FileCache(cache_path, cache_size) if cache_path else None

    def chat(self, messages: List[dict], model: str) -> str | None:
        """Return the assistant reply and record latency and token metrics."""

        key = json.dumps(
            {"model": model, "messages": messages},
            sort_keys=True,
        )
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        start = time.perf_counter()
        reply = self._chat(messages, model)
        duration = time.perf_counter() - start
        LLM_LATENCY.observe(duration)
        tokens = self._count_tokens(messages)
        if reply is not None:
            tokens += self._count_tokens([{"role": "assistant", "content": reply}])
        LLM_TOKENS.inc(tokens)
        if self.cache and reply is not None:
            self.cache.set(key, reply)
        logger.info(
            "llm_chat",
            model=model,
            latency=duration,
            tokens=tokens,
        )
        return reply

    @abstractmethod
    def _chat(self, messages: List[dict], model: str) -> str | None:
        """Implement provider-specific chat call."""
        raise NotImplementedError

    @staticmethod
    def _count_tokens(messages: List[dict]) -> int:
        """Approximate the number of tokens in a list of messages."""

        if tiktoken is not None:
            enc = tiktoken.get_encoding("cl100k_base")
            return sum(len(enc.encode(m.get("content", ""))) for m in messages)
        return sum(len(m.get("content", "").split()) for m in messages)


class AsyncLLMClient(ABC):
    """Asynchronous interface for chat-based language models with metrics."""

    def __init__(self, cache_path: str | None = None, cache_size: int = 128) -> None:
        self.cache = FileCache(cache_path, cache_size) if cache_path else None

    async def chat(self, messages: List[dict], model: str) -> str | None:
        """Return the assistant reply and record latency and token metrics."""

        key = json.dumps({"model": model, "messages": messages}, sort_keys=True)
        if self.cache:
            cached = self.cache.get(key)
            if cached is not None:
                return cached

        start = time.perf_counter()
        reply = await self._chat(messages, model)
        duration = time.perf_counter() - start
        LLM_LATENCY.observe(duration)
        tokens = LLMClient._count_tokens(messages)
        if reply is not None:
            tokens += LLMClient._count_tokens([{"role": "assistant", "content": reply}])
        LLM_TOKENS.inc(tokens)
        if self.cache and reply is not None:
            self.cache.set(key, reply)
        logger.info(
            "llm_chat",
            model=model,
            latency=duration,
            tokens=tokens,
        )
        return reply

    @abstractmethod
    async def _chat(self, messages: List[dict], model: str) -> str | None:
        """Implement provider-specific async chat call."""
        raise NotImplementedError


class OpenAIClient(LLMClient):
    """Client for the OpenAI chat completion API."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: str | None = None,
        cache_size: int = 128,
    ) -> None:
        super().__init__(cache_path=cache_path, cache_size=cache_size)
        self.api_key = api_key or settings.openai_api_key or os.getenv("OPENAI_API_KEY")

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

    def __init__(
        self,
        base_url: str | None = None,
        cache_path: str | None = None,
        cache_size: int = 128,
    ) -> None:
        super().__init__(cache_path=cache_path, cache_size=cache_size)
        base_url = base_url or settings.ollama_base_url
        self.base_url = base_url.rstrip("/")

    def _chat(self, messages: List[dict], model: str) -> str | None:
        url = f"{self.base_url}/api/chat"
        payload = {"model": model, "messages": messages}
        try:
            client = get_client()
            resp = client.post(url, json=payload)
            resp.raise_for_status()
            data = resp.json()
            return data.get("message", {}).get("content")
        except Exception:  # pragma: no cover - network or server issues
            return None


class HFLocalClient(LLMClient):
    """Client for Hugging Face models loaded locally."""

    def __init__(self, model_path: str, cache_path: str | None = None, cache_size: int = 128) -> None:
        super().__init__(cache_path=cache_path, cache_size=cache_size)
        from transformers import pipeline  # type: ignore

        self.generator = pipeline("text-generation", model=model_path)

    def _chat(self, messages: List[dict], model: str) -> str | None:
        prompt = "\n".join(m.get("content", "") for m in messages)
        try:
            out = self.generator(prompt, max_new_tokens=64)
        except Exception:  # pragma: no cover - model errors
            return None
        if not out:
            return None
        text = out[0].get("generated_text", "")
        return text[len(prompt) :].strip()
