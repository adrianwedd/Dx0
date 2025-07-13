"""Run performance benchmarks and store baseline metrics."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List

from sdb.benchmarks import (
    measure_cache_latency,
    measure_llm_latency,
    measure_retrieval_latency,
)


PROVIDERS = ["openai", "ollama", "hf-local"]


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run Dx0 performance benchmarks")
    parser.add_argument("cases", help="Path to case JSON or CSV")
    parser.add_argument("--provider", choices=PROVIDERS, default="openai")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model")
    parser.add_argument("--model-path", help="Local model path for hf-local")
    parser.add_argument("--backend", default=None, help="Retrieval backend")
    parser.add_argument("--output", default="results/performance_baseline.csv")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--runs", type=int, default=5, help="Number of LLM runs")
    parser.add_argument("--ttl", type=float, default=300.0, help="Cache TTL")
    args = parser.parse_args(argv)

    retrieval_avg = measure_retrieval_latency(
        args.cases, queries=args.queries, backend=args.backend
    )
    llm_avg = measure_llm_latency(
        args.provider, model=args.model, runs=args.runs, model_path=args.model_path
    )
    cold_avg, warm_avg = measure_cache_latency(
        args.cases, queries=args.queries, backend=args.backend, ttl=args.ttl
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "retrieval_avg_s",
                "llm_avg_s",
                "cache_cold_avg_s",
                "cache_warm_avg_s",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "retrieval_avg_s": f"{retrieval_avg:.4f}",
                "llm_avg_s": f"{llm_avg:.4f}",
                "cache_cold_avg_s": f"{cold_avg:.4f}",
                "cache_warm_avg_s": f"{warm_avg:.4f}",
            }
        )
    print(f"Baseline metrics written to {args.output}")


if __name__ == "__main__":  # pragma: no cover - manual script
    main()
