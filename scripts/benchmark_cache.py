"""Benchmark retrieval cache performance."""
from __future__ import annotations

import argparse
import time
from typing import List

from sdb.case_database import CaseDatabase
from sdb.retrieval import load_retrieval_index


def load_documents(db: CaseDatabase) -> List[str]:
    docs: List[str] = []
    for case in db.cases.values():
        for para in case.full_text.split("\n"):
            text = para.strip()
            if text:
                docs.append(text)
    return docs


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Benchmark caching impact")
    parser.add_argument("cases", help="Path to case JSON or CSV")
    parser.add_argument("--backend", default=None, help="Retrieval backend")
    parser.add_argument("--queries", type=int, default=50, help="Number of queries")
    parser.add_argument("--ttl", type=float, default=300.0, help="Cache TTL")
    args = parser.parse_args(argv)

    if args.cases.endswith(".json"):
        db = CaseDatabase.load_from_json(args.cases)
    elif args.cases.endswith(".csv"):
        db = CaseDatabase.load_from_csv(args.cases)
    else:
        raise ValueError("cases must be JSON or CSV")

    docs = load_documents(db)
    index = load_retrieval_index(docs, plugin_name=args.backend, cache_ttl=args.ttl)

    queries = [case.summary.split(".")[0] for case in db.cases.values()]
    queries = queries[: args.queries]

    # first run (cold cache)
    start = time.perf_counter()
    for q in queries:
        index.query(q)
    cold_duration = time.perf_counter() - start

    # second run (warm cache)
    start = time.perf_counter()
    for q in queries:
        index.query(q)
    warm_duration = time.perf_counter() - start

    print(
        f"cold_avg={cold_duration/len(queries):.4f}s warm_avg={warm_duration/len(queries):.4f}s"
    )


if __name__ == "__main__":  # pragma: no cover - manual script
    main()
