"""Utilities for mapping free-text test names to CPT codes via LLM."""

from __future__ import annotations

import csv
import os
import time
from typing import Dict, Optional

from .prompt_loader import load_prompt
from .metrics import CPT_CACHE_HITS, CPT_LLM_LOOKUPS

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - openai not required for tests
    openai = None


DEFAULT_CACHE = os.path.join("data", "cpt_lookup.csv")

# Load the system prompt used for LLM CPT lookups
CPT_LOOKUP_PROMPT = load_prompt("cpt_lookup_system")


def _load_cache(path: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    if not os.path.exists(path):
        return mapping
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                name = row["test_name"].strip().lower()
                code = row["cpt_code"].strip()
            except Exception:
                continue
            if name and code:
                mapping[name] = code
    return mapping


def _append_cache(path: str, test_name: str, cpt_code: str) -> None:
    exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["test_name", "cpt_code"])
        if not exists:
            writer.writeheader()
        writer.writerow({"test_name": test_name, "cpt_code": cpt_code})


def _query_llm(test_name: str, retries: int = 3) -> Optional[str]:
    if openai is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    openai.api_key = api_key
    model = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    messages = [
        {"role": "system", "content": CPT_LOOKUP_PROMPT},
        {"role": "user", "content": test_name},
    ]
    for _ in range(retries):
        try:
            resp = openai.ChatCompletion.create(
                model=model,
                messages=messages,
                max_tokens=10,
            )
            return resp.choices[0].message["content"].strip().split()[0]
        except Exception:  # pragma: no cover - network issues
            time.sleep(1)
    return None


def lookup_cpt(
    test_name: str,
    cache_path: str = DEFAULT_CACHE,
) -> Optional[str]:
    """Return the CPT code for ``test_name`` using cache or LLM lookup."""

    key = test_name.strip().lower()
    cache = _load_cache(cache_path)
    if key in cache:
        CPT_CACHE_HITS.inc()
        return cache[key]

    CPT_LLM_LOOKUPS.inc()
    code = _query_llm(test_name)
    if code:
        _append_cache(cache_path, key, code)
    return code
