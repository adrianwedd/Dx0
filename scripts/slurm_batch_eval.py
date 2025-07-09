"""Submit batch_evaluate jobs as a Slurm array."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

from sdb.evaluation import batch_evaluate


def run_case(case_id: str) -> dict[str, str]:
    """Example case evaluation function."""
    return {"id": case_id, "score": "0"}


def main(argv: list[str] | None = None) -> None:
    """Run ``batch_evaluate`` for a Slurm task."""
    parser = argparse.ArgumentParser(description="Run batch evaluations")
    parser.add_argument("--start", type=int, required=True, help="First case id")
    parser.add_argument("--end", type=int, required=True, help="Last case id")
    parser.add_argument(
        "--concurrency", type=int, default=2, help="Concurrent sessions"
    )
    parser.add_argument("--output", required=True, help="CSV result file")
    args = parser.parse_args(argv)

    case_ids = [str(i) for i in range(args.start, args.end + 1)]
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id:
        idx = int(task_id) - 1
        if 0 <= idx < len(case_ids):
            case_ids = [case_ids[idx]]

    results = batch_evaluate(case_ids, run_case, concurrency=args.concurrency)

    out_path = Path(args.output)
    if task_id:
        out_path = out_path.with_name(f"{out_path.stem}_{task_id}{out_path.suffix}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = sorted(results[0].keys()) if results else []
    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
            writer.writerows(results)


if __name__ == "__main__":  # pragma: no cover
    main()
