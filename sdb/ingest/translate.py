"""Utilities for translating case text into other languages."""

from __future__ import annotations

import json
import os
from typing import List, Dict, Any

from ..llm_client import OpenAIClient
from ..config import settings


TRANSLATE_PROMPT = (
    "Translate the following medical text into {lang}. Only output the"
    " translated text."
)


def translate_text(text: str, lang: str, client: OpenAIClient | None = None) -> str:
    """Return ``text`` translated into ``lang`` using an LLM if available."""
    if not text.strip():
        return ""
    client = client or OpenAIClient(api_key=settings.openai_api_key)
    prompt = TRANSLATE_PROMPT.format(lang=lang)
    reply = client.chat(
        [{"role": "user", "content": f"{prompt}\n\n{text}"}],
        model=os.getenv("OPENAI_MODEL", settings.openai_model),
    )
    return reply.strip() if reply else text


def translate_case(
    data: Dict[str, Any], lang: str, client: OpenAIClient | None = None
) -> Dict[str, Any]:
    """Return a translated copy of ``data`` for the given language code."""
    return {
        "id": data["id"],
        "summary": translate_text(data.get("summary", ""), lang, client),
        "steps": [
            {
                "id": step["id"],
                "text": translate_text(step["text"], lang, client),
            }
            for step in data.get("steps", [])
        ],
    }


def translate_directory(
    src_dir: str,
    lang: str,
    dest_dir: str | None = None,
    client: OpenAIClient | None = None,
) -> List[str]:
    """Translate all JSON cases in ``src_dir`` into ``lang``."""
    dest_dir = dest_dir or src_dir
    os.makedirs(dest_dir, exist_ok=True)
    written: List[str] = []
    for name in sorted(os.listdir(src_dir)):
        if not name.endswith(".json") or name.endswith(f"_{lang}.json"):
            continue
        path = os.path.join(src_dir, name)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        out_data = translate_case(data, lang, client)
        out_name = name[:-5] + f"_{lang}.json"
        out_path = os.path.join(dest_dir, out_name)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(out_data, fh, ensure_ascii=False, indent=2)
        written.append(out_path)
    return written


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Translate case files")
    parser.add_argument("src", help="Directory with JSON case files")
    parser.add_argument("lang", help="Target language code, e.g. 'es'")
    parser.add_argument("--dest", help="Output directory", default=None)
    args = parser.parse_args()
    translate_directory(args.src, args.lang, args.dest)
