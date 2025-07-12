"""Scrape NEJM CPC case articles and store raw text files.

This script queries PubMed for Case Records of the Massachusetts General
Hospital articles, downloads the article abstract or full text when
available, and stores each case in ``data/raw_cases/`` as
``case_<id>.txt``. The script is intentionally simple and relies on
``httpx`` for HTTP access.

Examples
--------
Run the script from the repository root::

    python scripts/collect_cases.py --dest data/raw_cases
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Iterable, List

from sdb.http_utils import get_client


PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)


def fetch_case_pmids(count: int = 304) -> List[str]:
    """Return a list of PubMed IDs for the latest CPC cases."""

    params = {
        "db": "pubmed",
        "term": "Case Records of the Massachusetts General Hospital[Title]",
        "retmax": str(count),
        "sort": "pub+date",
    }
    client = get_client()
    resp = client.get(PUBMED_SEARCH_URL, params=params)
    resp.raise_for_status()
    pmids = re.findall(r"<Id>(\d+)</Id>", resp.text)
    return pmids[:count]


def fetch_case_text(pmid: str) -> str:
    """Download case abstract from PubMed."""

    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "text",
        "rettype": "abstract",
    }
    client = get_client()
    resp = client.get(PUBMED_FETCH_URL, params=params)
    resp.raise_for_status()
    return resp.text.strip()


def save_case_text(case_id: int, text: str, dest_dir: str) -> str:
    """Write ``text`` to ``case_<id>.txt`` inside ``dest_dir``.

    Parameters
    ----------
    case_id:
        Sequential identifier starting at 1.
    text:
        Raw text to store.
    dest_dir:
        Directory that will contain the output files.

    Returns
    -------
    str
        Path of the file written.
    """

    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.join(dest_dir, f"case_{case_id:03d}.txt")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(text.strip() + "\n")
    return filename


def collect_cases(dest_dir: str = "data/raw_cases") -> None:
    """Download CPC cases and store them in ``dest_dir``."""

    pmids = fetch_case_pmids()
    for idx, pmid in enumerate(pmids, 1):
        try:
            text = fetch_case_text(pmid)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"Failed to fetch PMID {pmid}: {exc}")
            continue
        save_case_text(idx, text, dest_dir)
        if idx % 10 == 0:
            print(f"Saved {idx} cases")


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Collect NEJM CPC cases")
    parser.add_argument(
        "--dest",
        default="data/raw_cases",
        help="output directory",
    )
    parsed = parser.parse_args(args)
    collect_cases(parsed.dest)


if __name__ == "__main__":  # pragma: no cover
    main()
