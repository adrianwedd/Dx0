"""Utilities for converting raw case text into SDBench JSON format."""

from __future__ import annotations

import json
import os
import re
from typing import List, Dict


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
    summary = steps[0] if steps else ""
    data = {
        "id": f"case_{case_id:03d}",
        "summary": summary,
        "steps": [
            {"id": idx + 1, "text": step} for idx, step in enumerate(steps)
        ],
    }
    return data


def convert_directory(src_dir: str, dest_dir: str) -> List[str]:
    """Convert all ``case_*.txt`` files in ``src_dir`` to JSON files.

    Parameters
    ----------
    src_dir:
        Directory containing raw ``case_###.txt`` files.
    dest_dir:
        Output directory for ``case_###.json`` files.

    Returns
    -------
    list of str
        Paths of files written.
    """

    os.makedirs(dest_dir, exist_ok=True)
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
        data = convert_text(text, case_num)
        out_path = os.path.join(dest_dir, f"case_{case_num:03d}.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
        written.append(out_path)
    return written


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Convert raw cases to JSON")
    parser.add_argument("src", help="Directory with raw case text files")
    parser.add_argument("dest", help="Output directory for JSON cases")
    args = parser.parse_args()
    convert_directory(args.src, args.dest)
