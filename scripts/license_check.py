"""Verify NEJM dataset license compliance.

Run this script before packaging or releasing the project. It ensures the
repository does not accidentally include the NEJM case text and that required
license notices are present.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_readme() -> bool:
    """Return ``True`` if README contains required license statements."""
    text = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    required = ["research use only", "may not be redistributed"]
    missing = [phrase for phrase in required if phrase not in text]
    if missing:
        print(f"README missing phrases: {missing}")
        return False
    return True


def check_license() -> bool:
    """Return ``True`` if LICENSE mentions dataset restrictions."""
    text = (ROOT / "LICENSE").read_text(encoding="utf-8").lower()
    return "dataset licensing" in text and "non-commercial use" in text


def check_cases() -> bool:
    """Ensure NEJM case JSON files are not bundled."""
    case_dir = ROOT / "data" / "sdbench" / "cases"
    if case_dir.exists() and any(case_dir.glob("*.json")):
        print("Error: NEJM case JSON files found in repository.")
        return False
    return True


def main() -> None:
    ok = all([check_readme(), check_license(), check_cases()])
    if not ok:
        sys.exit(1)
    print("License checks passed.")


if __name__ == "__main__":
    main()
