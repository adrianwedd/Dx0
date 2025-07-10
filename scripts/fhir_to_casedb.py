"""Convert FHIR bundles or DiagnosticReports to case database format."""

from __future__ import annotations

import argparse
import json
import os
from typing import Iterable, List

from sdb.fhir_import import bundle_to_case, diagnostic_report_to_case
from sdb.case_database import Case
from sdb.sqlite_db import save_to_sqlite


def _load_case(path: str, case_id: str) -> Case:
    """Return a :class:`Case` loaded from ``path``."""

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    if data.get("resourceType") == "Bundle":
        case_data = bundle_to_case(data, case_id=case_id)
    else:
        case_data = diagnostic_report_to_case(data, case_id=case_id)

    text = "\n\n".join(step["text"] for step in case_data.get("steps", []))
    return Case(id=case_data["id"], summary=case_data["summary"], full_text=text)


def _collect_paths(inputs: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for inp in inputs:
        if os.path.isdir(inp):
            for name in sorted(os.listdir(inp)):
                if name.endswith(".json"):
                    paths.append(os.path.join(inp, name))
        else:
            paths.append(inp)
    return paths


def main(argv: List[str] | None = None) -> None:
    """CLI entry point for FHIR bundle conversion."""

    parser = argparse.ArgumentParser(
        description="Convert FHIR bundles to case database format"
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="One or more bundle JSON files or directories",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .json or .db file",
    )
    parser.add_argument(
        "--sqlite",
        action="store_true",
        help="Write SQLite database instead of JSON",
    )
    args = parser.parse_args(argv)

    file_paths = _collect_paths(args.inputs)

    cases = [_load_case(p, os.path.splitext(os.path.basename(p))[0]) for p in file_paths]
    case_dicts = [
        {"id": c.id, "summary": c.summary, "full_text": c.full_text} for c in cases
    ]

    if args.sqlite or args.output.endswith(".db"):
        save_to_sqlite(args.output, case_dicts)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            json.dump(case_dicts, fh, indent=2)


if __name__ == "__main__":  # pragma: no cover
    main()
