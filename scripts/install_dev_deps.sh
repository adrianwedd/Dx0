#!/usr/bin/env bash
# Install development dependencies listed in requirements.lock
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

cd "$repo_root"

pip install -r requirements.lock

# Install browsers for the Playwright test suite
# Install all supported browsers (chromium, firefox, and webkit)
playwright install
