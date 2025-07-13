"""Utility functions for performance benchmarks."""
from __future__ import annotations

import time
from typing import List, Tuple

from .case_database import CaseDatabase
from .retrieval import load_retrieval_index
from .llm_client import HFLocalClient, OllamaClient, OpenAIClient


def _load_documents(db: CaseDatabase) -> List[str]:
    """Return a list of paragraphs from ``db``."""
    docs: List[str] = []
    for case in db.cases.values():
        for para in case.full_text.split("\n"):
            text = para.strip()
            if text:
                docs.append(text)
    return docs


def measure_retrieval_latency(
    cases_path: str,
    *,
    queries: int = 100,
    top_k: int = 5,
    backend: str | None = None,
) -> float:
    """Return average retrieval latency in seconds."""
    if cases_path.endswith(".json"):
        db = CaseDatabase.load_from_json(cases_path)
    elif cases_path.endswith(".csv"):
        db = CaseDatabase.load_from_csv(cases_path)
    else:
        raise ValueError("cases must be JSON or CSV")

    docs = _load_documents(db)
    index = load_retrieval_index(docs, plugin_name=backend, cache_ttl=None)
    queries_list = [case.summary.split(".")[0] for case in db.cases.values()]
    queries_list = queries_list[:queries]

    start = time.perf_counter()
    for q in queries_list:
        index.query(q, top_k=top_k)
    duration = time.perf_counter() - start
    return duration / len(queries_list)


def measure_llm_latency(
    provider: str,
    *,
    model: str,
    runs: int = 5,
    model_path: str | None = None,
) -> float:
    """Return average LLM request latency in seconds."""
    if provider == "hf-local":
        if not model_path:
            raise ValueError("model_path required for hf-local provider")
        client = HFLocalClient(model_path)
    elif provider == "ollama":
        client = OllamaClient()
    else:
        client = OpenAIClient()

    messages = [{"role": "user", "content": "Hello"}]
    start = time.perf_counter()
    for _ in range(runs):
        client.chat(messages, model=model)
    duration = time.perf_counter() - start
    return duration / runs


def measure_cache_latency(
    cases_path: str,
    *,
    queries: int = 50,
    backend: str | None = None,
    ttl: float = 300.0,
) -> Tuple[float, float]:
    """Return (cold_avg, warm_avg) retrieval latency in seconds."""
    if cases_path.endswith(".json"):
        db = CaseDatabase.load_from_json(cases_path)
    elif cases_path.endswith(".csv"):
        db = CaseDatabase.load_from_csv(cases_path)
    else:
        raise ValueError("cases must be JSON or CSV")

    docs = _load_documents(db)
    index = load_retrieval_index(docs, plugin_name=backend, cache_ttl=ttl)
    queries_list = [case.summary.split(".")[0] for case in db.cases.values()]
    queries_list = queries_list[:queries]

    start = time.perf_counter()
    for q in queries_list:
        index.query(q)
    cold_duration = time.perf_counter() - start

    start = time.perf_counter()
    for q in queries_list:
        index.query(q)
    warm_duration = time.perf_counter() - start

    return cold_duration / len(queries_list), warm_duration / len(queries_list)

