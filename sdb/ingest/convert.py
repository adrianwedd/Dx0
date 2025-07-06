"""Utilities for converting raw case text into SDBench JSON format."""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional

from ..prompt_loader import load_prompt

try:  # Optional dependency
    import openai  # type: ignore
except Exception:  # pragma: no cover - openai not required for tests
    openai = None


SUMMARY_PROMPT = load_prompt("case_summary_system")


def split_steps(text: str) -> List[str]:
    """Split raw text into stepwise sections.

    Parameters
    ----------
    text:
        Raw case text.

    Returns
    -------
    list of str
        List of non-empty paragraphs representing sequential steps.
    """

    parts = re.split(r"\n\s*\n", text.strip())
    paragraphs = [p.strip() for p in parts if p.strip()]
    return paragraphs


def _extract_abstract(text: str) -> Optional[str]:
    """Return the abstract section if present in ``text``."""

    match = re.search(r"(?im)^abstract[:\s]*\n(.+?)(?:\n\s*\n|$)", text)
    if match:
        abstract = re.sub(r"\s+", " ", match.group(1)).strip()
        if abstract:
            return abstract
    return None


def _llm_summarize(text: str) -> Optional[str]:
    """Summarize ``text`` using an LLM if credentials are configured."""

    if openai is None:
        return None
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model=os.getenv("OPENAI_MODEL", "gpt-3.5-turbo"),
            messages=[
                {"role": "system", "content": SUMMARY_PROMPT},
                {"role": "user", "content": text[:4000]},
            ],
            max_tokens=100,
        )
        return resp.choices[0].message["content"].strip()
    except Exception:  # pragma: no cover - network issues
        return None


def summarize(text: str) -> str:
    """Produce a short summary for a case."""

    summary = _extract_abstract(text)
    if not summary:
        summary = _llm_summarize(text)
    if not summary:
        steps = split_steps(text)
        summary = " ".join(steps[:2]) if steps else ""
    return summary.strip()


def extract_year(text: str) -> Optional[int]:
    """Return the first four-digit year found in ``text``."""

    match = re.search(r"\b(19|20)\d{2}\b", text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:  # pragma: no cover - regex ensures digits
            return None
    return None


def convert_text(text: str, case_id: int) -> Dict[str, object]:
    """Convert raw case text to SDBench JSON structure.

    Parameters
    ----------
    text:
        Raw case text.
    case_id:
        Sequential case identifier starting at 1.

    Returns
    -------
    dict
        Dictionary ready to be serialized as JSON.
    """

    steps = split_steps(text)
    summary = summarize(text)
    steps_list = []
    for idx, step in enumerate(steps):
        steps_list.append({"id": idx + 1, "text": step})
    data: dict[str, object] = {
        "id": f"case_{case_id:03d}",
        "summary": summary,
        "steps": steps_list,
    }
    return data


def convert_directory(
    src_dir: str, dest_dir: str, hidden_dir: str | None = None
) -> List[str]:
    """Convert all ``case_*.txt`` files in ``src_dir`` to JSON files.

    Parameters
    ----------
    src_dir:
        Directory containing raw ``case_###.txt`` files.
    dest_dir:
        Output directory for ``case_###.json`` files.
    hidden_dir:
        Directory for cases from 2024–2025 that should be held out.

    Returns
    -------
    list of str
        Paths of files written.
    """

    os.makedirs(dest_dir, exist_ok=True)
    if hidden_dir:
        os.makedirs(hidden_dir, exist_ok=True)
    written: List[str] = []
    for name in sorted(os.listdir(src_dir)):
        if not name.startswith("case_") or not name.endswith(".txt"):
            continue
        num_part = name[5:-4]
        try:
            case_num = int(num_part)
        except ValueError:
            continue
        path = os.path.join(src_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            text = fh.read()
        year = extract_year(text)
        data = convert_text(text, case_num)
        target_dir = dest_dir
        if hidden_dir and year and 2024 <= year <= 2025:
            target_dir = hidden_dir
        out_path = os.path.join(target_dir, f"case_{case_num:03d}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        summary_file = f"case_{case_num:03d}_summary.txt"
        summary_path = os.path.join(target_dir, summary_file)
        with open(summary_path, "w", encoding="utf-8") as sf:
            sf.write(str(data["summary"]))
        if target_dir == dest_dir:
            written.append(out_path)
    return written


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Convert raw cases to JSON")
    parser.add_argument("src", help="Directory with raw case text files")
    parser.add_argument("dest", help="Output directory for JSON cases")
    parser.add_argument(
        "--hidden",
        help="Directory for held-out cases",
        default=None,
    )
    args = parser.parse_args()
    convert_directory(args.src, args.dest, args.hidden)
