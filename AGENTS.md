# Contribution Guidelines for Dx0 Repository

This repository contains a minimal demonstration of the SDBench framework and the MAI‑DxO diagnostic agent.

## Coding Style
- Use standard Python 3 style as defined in [PEP 8](https://peps.python.org/pep-0008/).
- Provide docstrings for public classes and functions.

## Testing
- Run the unit tests with `pytest` before committing any changes.
- All tests should pass: `pytest -q`.

## Project Management
- The high‑level roadmap lives in `tasks.yml`. Keep it updated when adding major features.
- When adding or modifying tasks, summarize the changes in the commit title/body.
  Example: "Add Phase 1–4 backlog tasks to tasks.yml."
- Mark completed tasks as `done` in `tasks.yml` and mention the task IDs in the commit message when closing them.

## Commit Messages
- Summarize the purpose of the change in the commit title (e.g., "Fix retrieval cache timing for tests").
- Reference the relevant task ID from `tasks.yml` in the commit body when applicable.
