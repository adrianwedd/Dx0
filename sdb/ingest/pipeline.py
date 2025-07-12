"""End-to-end case ingestion pipeline for SDBench."""

from __future__ import annotations

import asyncio
import os
import re
import json
from typing import List, Set

from tqdm import tqdm

try:
    import aiohttp
except Exception:  # pragma: no cover - optional
    aiohttp = None

try:
    import requests
except Exception:  # pragma: no cover - optional
    requests = None

from .convert import convert_directory
from ..sqlite_db import save_to_sqlite

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


async def fetch_case_text_async(
    session: "aiohttp.ClientSession", pmid: str
) -> str:
    """Download case abstract text asynchronously."""
    if aiohttp is None:  # pragma: no cover - optional dependency
        raise RuntimeError("aiohttp package is required for collection")
    params = {
        "db": "pubmed",
        "id": pmid,
        "retmode": "text",
        "rettype": "abstract",
    }
    async with session.get(PUBMED_FETCH_URL, params=params, timeout=30) as resp:
        resp.raise_for_status()
        text = await resp.text()
    return text.strip()


def save_case_text(case_id: int, text: str, dest_dir: str) -> str:
    """Write ``text`` to ``case_<id>.txt`` in ``dest_dir``."""
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.join(dest_dir, f"case_{case_id:03d}.txt")
    with open(filename, "w", encoding="utf-8") as fh:
        fh.write(text.strip() + "\n")
    return filename


async def _download_cases_async(
    pmids: List[str], dest_dir: str, start_id: int, concurrency: int
) -> List[str]:
    """Download ``pmids`` concurrently and write them to ``dest_dir``."""
    if aiohttp is None:  # pragma: no cover - optional dependency
        raise RuntimeError("aiohttp package is required for collection")

    semaphore = asyncio.Semaphore(concurrency)
    paths: List[str] = []
    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def fetch_and_save(idx: int, pmid: str) -> str | None:
            async with semaphore:
                try:
                    text = await fetch_case_text_async(session, pmid)
                except Exception as exc:  # pragma: no cover - network errors
                    print(f"Failed to fetch PMID {pmid}: {exc}")
                    return None
            return save_case_text(start_id + idx, text, dest_dir)
        tasks = [
            asyncio.create_task(fetch_and_save(i, pmid))
            for i, pmid in enumerate(pmids)
        ]
        for coro in tqdm(
            asyncio.as_completed(tasks), total=len(tasks), desc="Fetching cases"
        ):
            result = await coro
            if result:
                paths.append(result)
    return paths


def collect_cases(
    dest_dir: str = "data/raw_cases", count: int = 304, concurrency: int = 3
) -> List[str]:
    """Download CPC cases and store them in ``dest_dir``.

    Parameters
    ----------
    dest_dir:
        Destination directory for raw case text files.
    count:
        Number of cases to retrieve.
    concurrency:
        Maximum number of concurrent downloads.
    """
    pmids = fetch_case_pmids(count)
    return asyncio.run(
        _download_cases_async(pmids, dest_dir, start_id=1, concurrency=concurrency)
    )


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


def collect_new_cases(
    dest_dir: str = "data/raw_cases", concurrency: int = 3
) -> List[str]:
    """Download only new CPC cases into ``dest_dir``.

    Parameters
    ----------
    dest_dir:
        Directory that will receive new case files.
    concurrency:
        Maximum number of concurrent downloads.
    """

    existing = _existing_pmids(dest_dir)
    next_id = _next_case_id(dest_dir)
    pmids = [pmid for pmid in fetch_case_pmids(count=1000) if pmid not in existing]
    if not pmids:
        return []
    return asyncio.run(
        _download_cases_async(pmids, dest_dir, start_id=next_id, concurrency=concurrency)
    )


def run_pipeline(
    *,
    raw_dir: str = "data/raw_cases",
    output_dir: str = "data/sdbench/cases",
    hidden_dir: str | None = "data/sdbench/hidden_cases",
    fetch: bool = True,
    concurrency: int = 3,
    sqlite_path: str | None = None,
) -> List[str]:
    """Run the full ingestion pipeline.

    Parameters
    ----------
    raw_dir:
        Directory containing raw case text files.
    output_dir:
        Directory that will receive converted JSON cases.
    hidden_dir:
        Directory for held-out cases from 2024–2025.
    fetch:
        Whether to download cases before conversion.
    concurrency:
        Maximum number of concurrent downloads when ``fetch`` is ``True``.
    sqlite_path:
        Optional path to a SQLite database where case summaries will be stored.
    """
    if fetch:
        collect_cases(raw_dir, concurrency=concurrency)
    paths = convert_directory(raw_dir, output_dir, hidden_dir)
    if sqlite_path:
        json_dirs = [output_dir]
        if hidden_dir:
            json_dirs.append(hidden_dir)
        cases = []
        for d in json_dirs:
            for name in sorted(os.listdir(d)):
                if not name.endswith(".json"):
                    continue
                with open(os.path.join(d, name), "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                step_texts = [s["text"] for s in data.get("steps", [])]
                full_text = "\n\n".join(step_texts)
                cases.append(
                    {
                        "id": data["id"],
                        "summary": data["summary"],
                        "full_text": full_text,
                    }
                )
        save_to_sqlite(sqlite_path, cases)
    return paths


def update_dataset(
    *,
    raw_dir: str = "data/raw_cases",
    output_dir: str = "data/sdbench/cases",
    hidden_dir: str | None = "data/sdbench/hidden_cases",
    concurrency: int = 3,
    sqlite_path: str | None = None,
) -> List[str]:
    """Fetch newly released cases and convert the entire dataset.

    Parameters
    ----------
    raw_dir:
        Directory for raw case text files.
    output_dir:
        Directory for converted JSON cases.
    hidden_dir:
        Directory for held-out cases from 2024–2025.
    concurrency:
        Maximum number of concurrent downloads when fetching new cases.
    sqlite_path:
        Optional path to a SQLite database where case summaries will be stored.
    """

    collect_new_cases(raw_dir, concurrency=concurrency)
    return run_pipeline(
        raw_dir=raw_dir,
        output_dir=output_dir,
        hidden_dir=hidden_dir,
        fetch=False,
        concurrency=concurrency,
        sqlite_path=sqlite_path,
    )


if __name__ == "__main__":  # pragma: no cover
    run_pipeline()
