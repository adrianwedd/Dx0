from __future__ import annotations

"""Evaluate retrieval accuracy for case documents."""

import argparse
from typing import Tuple

from sdb.case_database import CaseDatabase
from sdb.retrieval import load_retrieval_index


def evaluate_retrieval(
    db: CaseDatabase,
    *,
    top_k: int = 5,
    retrieval_backend: str | None = None,
) -> Tuple[float, float]:
    """Compute recall@k and mean reciprocal rank for the database."""
    docs: list[str] = []
    doc_to_case: dict[str, str] = {}
    for case in db.cases.values():
        for para in case.full_text.split("\n"):
            text = para.strip()
            if text:
                docs.append(text)
                doc_to_case[text] = case.id

    index = load_retrieval_index(docs, plugin_name=retrieval_backend)

    hits = 0
    rr_total = 0.0
    total = 0
    for case in db.cases.values():
        query = case.summary.split(".")[0]
        results = index.query(query, top_k=top_k)
        total += 1
        for rank, (doc, _score) in enumerate(results, start=1):
            if doc_to_case.get(doc) == case.id:
                hits += 1
                rr_total += 1.0 / rank
                break
        else:
            rr_total += 0.0
    recall = hits / total if total else 0.0
    mrr = rr_total / total if total else 0.0
    return recall, mrr


def main(argv: list[str] | None = None) -> None:
    """Entry point for command line invocation."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval accuracy")
    parser.add_argument("cases", help="Path to cases JSON or CSV")
    parser.add_argument("--top-k", type=int, default=5, help="Rank cutoff")
    parser.add_argument("--backend", help="Retrieval backend", default=None)
    args = parser.parse_args(argv)

    if args.cases.endswith(".json"):
        db = CaseDatabase.load_from_json(args.cases)
    elif args.cases.endswith(".csv"):
        db = CaseDatabase.load_from_csv(args.cases)
    else:
        raise ValueError("cases must be JSON or CSV")

    recall, mrr = evaluate_retrieval(
        db, top_k=args.top_k, retrieval_backend=args.backend
    )
    print(f"recall@{args.top_k}: {recall:.3f}")
    print(f"mrr: {mrr:.3f}")


if __name__ == "__main__":  # pragma: no cover
    main()
