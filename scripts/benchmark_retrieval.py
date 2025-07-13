"""Benchmark retrieval index latency."""
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
    parser = argparse.ArgumentParser(description="Benchmark retrieval latency")
    parser.add_argument("cases", help="Path to case JSON or CSV")
    parser.add_argument("--backend", default=None, help="Retrieval backend")
    parser.add_argument("--queries", type=int, default=100, help="Number of queries")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results")
    args = parser.parse_args(argv)

    if args.cases.endswith(".json"):
        db = CaseDatabase.load_from_json(args.cases)
    elif args.cases.endswith(".csv"):
        db = CaseDatabase.load_from_csv(args.cases)
    else:
        raise ValueError("cases must be JSON or CSV")

    docs = load_documents(db)
    index = load_retrieval_index(docs, plugin_name=args.backend, cache_ttl=None)

    queries = [case.summary.split(".")[0] for case in db.cases.values()]
    queries = queries[: args.queries]

    start = time.perf_counter()
    for q in queries:
        index.query(q, top_k=args.top_k)
    duration = time.perf_counter() - start

    print(f"avg_query_latency={duration/len(queries):.4f}s over {len(queries)} queries")


if __name__ == "__main__":  # pragma: no cover - manual script
    main()
