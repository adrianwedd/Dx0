"""Measure HTTP request latency with and without connection pooling."""

from __future__ import annotations

import time
import httpx

URL = "https://httpbin.org/get"


def measure_without_pooling(n: int = 10) -> float:
    start = time.perf_counter()
    for _ in range(n):
        with httpx.Client() as client:
            resp = client.get(URL)
            resp.raise_for_status()
    duration = time.perf_counter() - start
    return duration / n


def measure_with_pooling(n: int = 10) -> float:
    client = httpx.Client()
    start = time.perf_counter()
    for _ in range(n):
        resp = client.get(URL)
        resp.raise_for_status()
    duration = time.perf_counter() - start
    client.close()
    return duration / n


if __name__ == "__main__":  # pragma: no cover - manual diagnostic script
    avg_no_pool = measure_without_pooling()
    avg_pool = measure_with_pooling()
    print(f"Avg latency without pooling: {avg_no_pool:.4f}s")
    print(f"Avg latency with pooling: {avg_pool:.4f}s")
