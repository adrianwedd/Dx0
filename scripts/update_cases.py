"""Fetch newly published NEJM CPC cases and append them to the dataset."""

from __future__ import annotations

import argparse

from sdb.ingest.pipeline import update_dataset


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Update CPC dataset")
    parser.add_argument(
        "--raw-dir",
        default="data/raw_cases",
        help="Directory for raw case text",
    )
    parser.add_argument(
        "--output-dir",
        default="data/sdbench/cases",
        help="Destination directory for JSON cases",
    )
    parser.add_argument(
        "--hidden-dir",
        default="data/sdbench/hidden_cases",
        help="Directory for held-out cases",
    )
    parsed = parser.parse_args(args)
    update_dataset(
        raw_dir=parsed.raw_dir,
        output_dir=parsed.output_dir,
        hidden_dir=parsed.hidden_dir,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
