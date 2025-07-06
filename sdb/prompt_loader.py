"""Utilities for loading prompt templates by name."""

from __future__ import annotations

import os

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
PROMPT_DIR = os.path.join(ROOT_DIR, "prompts")


def load_prompt(name: str) -> str:
    """Return the contents of ``prompts/<name>.txt``.

    Parameters
    ----------
    name:
        Base filename of the prompt to load, without extension.

    Returns
    -------
    str
        Prompt text with surrounding whitespace stripped.
    """

    path = os.path.join(PROMPT_DIR, f"{name}.txt")
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read().strip()
