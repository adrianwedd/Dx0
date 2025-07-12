#!/usr/bin/env python
"""Prepare NEJM case corpus for language model fine-tuning."""
from __future__ import annotations

import argparse
import json
import os
from typing import Iterable


def load_cases(src_dir: str) -> Iterable[dict]:
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(src_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            yield json.load(fh)


def build_record(case: dict) -> dict[str, str]:
    steps = case.get("steps", [])
    text = "\n\n".join(step.get("text", "") for step in steps)
    prompt = f"Case ID: {case.get('id')}\n{text}\n\nDiagnosis:".strip()
    diagnosis = steps[-1].get("text", "").strip() if steps else ""
    return {"prompt": prompt, "completion": " " + diagnosis}


def prepare_dataset(src_dir: str, dest_file: str) -> None:
    records = [build_record(case) for case in load_cases(src_dir)]
    with open(dest_file, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create JSONL file for fine-tuning")
    parser.add_argument("src", help="Directory with case JSON files")
    parser.add_argument("dest", help="Output JSONL file")
    parsed = parser.parse_args(args)
    prepare_dataset(parsed.src, parsed.dest)


if __name__ == "__main__":  # pragma: no cover
    main()
