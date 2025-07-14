"""Generate release notes from Git commit history."""
from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path


def git(*args: str) -> str:
    """Run a git command and return its output."""
    result = subprocess.run(
        ["git", *args], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    return result.stdout.strip()


def gather_messages(since_tag: str) -> list[str]:
    log_range = f"{since_tag}..HEAD" if since_tag else "HEAD"
    output = git("log", log_range, "--pretty=format:%s")
    messages = []
    for line in output.splitlines():
        if line and not line.startswith("Merge"):
            messages.append(line)
    return messages


def update_file(path: Path, header: str, notes: list[str]) -> None:
    if path.exists():
        existing = path.read_text()
    else:
        existing = ""

    lines = existing.splitlines()
    if lines and lines[0].startswith("#"):
        content = "\n".join(lines[1:])
        new_text = f"{lines[0]}\n\n{header}\n" + "\n".join(f"* {m}" for m in notes) + "\n\n" + content
    else:
        new_text = f"# Release Notes\n\n{header}\n" + "\n".join(f"* {m}" for m in notes) + "\n\n" + existing
    path.write_text(new_text)


def main(args: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Assemble release notes")
    parser.add_argument("--since-tag", help="Tag to start from. Defaults to latest tag")
    parser.add_argument("--output", default="docs/release_notes.md", help="File to update")
    ns = parser.parse_args(args)

    if ns.since_tag:
        tag = ns.since_tag
    else:
        try:
            tag = git("describe", "--tags", "--abbrev=0")
        except subprocess.CalledProcessError:
            tag = ""
    messages = gather_messages(tag)
    today = date.today().isoformat()
    header = f"## Unreleased - {today}"
    update_file(Path(ns.output), header, messages)


if __name__ == "__main__":
    main()
