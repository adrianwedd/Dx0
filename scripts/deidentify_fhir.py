"""Remove personal identifiers from FHIR Bundles."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Iterable

# Keys matching this regex will be removed unless whitelisted.
PHI_KEY_PATTERN = re.compile(
    r"name|telecom|phone|email|address|birthdate|given|family|city|state|postalcode",
    re.IGNORECASE,
)

# Patterns within string values to redact.
PHI_VALUE_PATTERNS: Iterable[re.Pattern[str]] = [
    re.compile(r"\b\d{3}-\d{3}-\d{4}\b"),
    re.compile(r"\b[\w.-]+@[\w.-]+\.[A-Za-z]{2,}\b"),
]

# Allowed fields which should not be stripped even if they match the PHI regex.
WHITELIST = {
    "resourceType",
    "id",
    "status",
    "code",
    "text",
    "valueString",
    "valueCodeableConcept",
    "valueQuantity",
    "result",
    "contained",
    "entry",
    "conclusion",
}


def _clean_value(value: Any) -> Any:
    """Return ``value`` with PHI patterns redacted."""
    if isinstance(value, str):
        for pat in PHI_VALUE_PATTERNS:
            value = pat.sub("[REDACTED]", value)
    return value


def strip_identifiers(data: Any, whitelist: Iterable[str] | None = None) -> Any:
    """Return ``data`` with personal identifiers removed."""
    if whitelist is None:
        whitelist = WHITELIST
    if isinstance(data, dict):
        cleaned = {}
        for key, val in data.items():
            if key not in whitelist and PHI_KEY_PATTERN.search(key):
                continue
            cleaned[key] = strip_identifiers(val, whitelist)
        return cleaned
    if isinstance(data, list):
        return [strip_identifiers(v, whitelist) for v in data]
    return _clean_value(data)


def main(args: list[str] | None = None) -> None:
    """CLI entry point for FHIR de-identification."""
    parser = argparse.ArgumentParser(description="Remove PHI from a FHIR Bundle")
    parser.add_argument("input", help="Path to FHIR Bundle JSON")
    parser.add_argument(
        "output",
        nargs="?",
        help="Destination path (writes to stdout if omitted)",
    )
    parsed = parser.parse_args(args)

    data = json.loads(Path(parsed.input).read_text(encoding="utf-8"))
    cleaned = strip_identifiers(data)
    text = json.dumps(cleaned, indent=2)

    if parsed.output:
        Path(parsed.output).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":  # pragma: no cover
    main()
