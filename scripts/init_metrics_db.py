"""Initialize an evaluation metrics SQLite database."""

from __future__ import annotations

import argparse

from sdb.services import MetricsDB


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create metrics SQLite file")
    parser.add_argument("path", help="Destination database file")
    args = parser.parse_args(argv)
    MetricsDB(args.path)  # initialize schema


if __name__ == "__main__":  # pragma: no cover
    main()
