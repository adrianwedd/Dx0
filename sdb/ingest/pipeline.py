"""End-to-end case ingestion pipeline for SDBench."""

from __future__ import annotations

import os
import re
from typing import List, Set

try:
    import requests
except Exception:  # pragma: no cover - optional
    requests = None

from .convert import convert_directory

PUBMED_SEARCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
)
PUBMED_FETCH_URL = (
    "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
)


def fetch_case_pmids(count: int = 304) -> List[str]:
    """Return PubMed IDs for the latest CPC cases."""
    if requests is None:  # pragma: no cover - optional dependency
        raise RuntimeError("requests package is required for collection")
    params = {
        "db": "pubmed",
        "term": "Case Records of the Massachusetts General Hospital[Title]",
        "retmax": str(count),
        "sort": "pub+date",
    }
    resp = requests.get(PUBMED_SEARCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    pmids = re.findall(r"<Id>(\d+)</Id>", resp.text)
    return pmids[:count]


def fetch_case_text(pmid: str) -> str:
    """Download case abstract text from PubMed."""
    if requests is None:  # pragma: no cover - optional dependency
        raise RuntimeError("requests package is required for collection")
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "text",
        "rettype": "abstract",
    }
    resp = requests.get(PUBMED_FETCH_URL, params=params, timeout=30)
    resp.raise_for_status()
    return resp.text.strip()


def save_case_text(case_id: int, text: str, dest_dir: str) -> str:
    """Write ``text`` to ``case_<id>.txt`` in ``dest_dir``."""
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.join(dest_dir, f"case_{case_id:03d}.txt")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(text.strip() + "\n")
    return filename


def collect_cases(
    dest_dir: str = "data/raw_cases", count: int = 304
) -> List[str]:
    """Download CPC cases and store them in ``dest_dir``."""
    pmids = fetch_case_pmids(count)
    paths = []
    for idx, pmid in enumerate(pmids, 1):
        try:
            text = fetch_case_text(pmid)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"Failed to fetch PMID {pmid}: {exc}")
            continue
        path = save_case_text(idx, text, dest_dir)
        paths.append(path)
    return paths


PMID_RE = re.compile(r"PMID:\s*(\d+)")


def _existing_pmids(raw_dir: str) -> Set[str]:
    """Return PMIDs already present in ``raw_dir``."""

    pmids: Set[str] = set()
    if not os.path.isdir(raw_dir):
        return pmids
    for name in os.listdir(raw_dir):
        if not name.startswith("case_") or not name.endswith(".txt"):
            continue
        path = os.path.join(raw_dir, name)
        try:
            with open(path, "r", encoding="utf-8") as fh:
                match = PMID_RE.search(fh.read())
                if match:
                    pmids.add(match.group(1))
        except OSError:  # pragma: no cover - file access issues
            continue
    return pmids


def _next_case_id(raw_dir: str) -> int:
    """Return the next sequential case identifier."""

    highest = 0
    if not os.path.isdir(raw_dir):
        return 1
    for name in os.listdir(raw_dir):
        if name.startswith("case_") and name.endswith(".txt"):
            try:
                num = int(name[5:-4])
            except ValueError:
                continue
            highest = max(highest, num)
    return highest + 1


def collect_new_cases(dest_dir: str = "data/raw_cases") -> List[str]:
    """Download only new CPC cases into ``dest_dir``."""

    existing = _existing_pmids(dest_dir)
    next_id = _next_case_id(dest_dir)
    pmids = fetch_case_pmids(count=1000)
    paths = []
    for pmid in pmids:
        if pmid in existing:
            continue
        try:
            text = fetch_case_text(pmid)
        except Exception as exc:  # pragma: no cover - network errors
            print(f"Failed to fetch PMID {pmid}: {exc}")
            continue
        path = save_case_text(next_id, text, dest_dir)
        next_id += 1
        paths.append(path)
    return paths


def run_pipeline(
    *,
    raw_dir: str = "data/raw_cases",
    output_dir: str = "data/sdbench/cases",
    hidden_dir: str | None = "data/sdbench/hidden_cases",
    fetch: bool = True,
) -> List[str]:
    """Run the full ingestion pipeline."""
    if fetch:
        collect_cases(raw_dir)
    return convert_directory(raw_dir, output_dir, hidden_dir)


def update_dataset(
    *,
    raw_dir: str = "data/raw_cases",
    output_dir: str = "data/sdbench/cases",
    hidden_dir: str | None = "data/sdbench/hidden_cases",
) -> List[str]:
    """Fetch newly released cases and convert the entire dataset."""

    collect_new_cases(raw_dir)
    return convert_directory(raw_dir, output_dir, hidden_dir)


if __name__ == "__main__":  # pragma: no cover
    run_pipeline()
