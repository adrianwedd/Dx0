"""Update local CPT pricing using data from CMS."""

from __future__ import annotations

import argparse
import csv
import datetime
import io
import os
import zipfile
from typing import Dict

import requests


def default_cms_url() -> str:
    """Return the CMS pricing ZIP URL for the current year."""
    year = datetime.date.today().year
    return (
        "https://download.cms.gov/MedicareClinicalLabFeeSched/"
        f"{str(year)[-2:]}CLAB.zip"
    )


DEFAULT_URL = os.getenv("CMS_PRICING_URL", default_cms_url())
DEFAULT_PATH = os.path.join("data", "cpt_lookup.csv")


def _fetch_cms_prices(url: str = DEFAULT_URL) -> Dict[str, float]:
    """Return mapping of CPT/HCPCS codes to prices from ``url``."""

    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    content = resp.content
    if url.lower().endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(content)) as zf:
            name = next(
                (n for n in zf.namelist() if n.lower().endswith(".csv")),
                None,
            )
            if name is None:
                raise ValueError("ZIP archive contains no CSV file")
            data = io.TextIOWrapper(zf.open(name), encoding="utf-8")
            reader = csv.DictReader(data)
            rows = list(reader)
    else:
        text = content.decode("utf-8")
        rows = list(csv.DictReader(io.StringIO(text)))

    mapping: Dict[str, float] = {}
    for row in rows:
        lower = {k.lower(): v for k, v in row.items()}
        code = (
            lower.get("cpt_code")
            or lower.get("hcpcs")
            or lower.get("code")
            or lower.get("hcpcs code")
        )
        price = (
            lower.get("price")
            or lower.get("rate")
            or lower.get("payment_rate")
            or lower.get("payment rate")
            or lower.get("2024 payment rate")
        )
        if not code or not price:
            continue
        try:
            mapping[code.strip()] = float(str(price).strip().lstrip("$"))
        except ValueError:
            continue
    return mapping


def refresh_pricing(path: str = DEFAULT_PATH, url: str = DEFAULT_URL) -> None:
    """Update ``path`` with prices fetched from ``url``."""

    prices = _fetch_cms_prices(url)
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            code = row.get("cpt_code", "").strip()
            if code in prices:
                row["price"] = str(prices[code])
            rows.append(row)

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["test_name", "cpt_code", "price"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Update CPT pricing from CMS")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help="CMS CSV or ZIP URL",
    )
    parser.add_argument(
        "--path",
        default=DEFAULT_PATH,
        help="Path to cpt_lookup.csv",
    )
    parsed = parser.parse_args(args)
    refresh_pricing(parsed.path, parsed.url)


if __name__ == "__main__":  # pragma: no cover
    main()
