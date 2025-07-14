# Changelog

All notable changes to this project will be documented in this file.

## [0.4.0] - 2024-??-??
### Added
- `BudgetManager` service to track session spending and enforce limits.
- `--budget-limit` CLI flag and `UI_BUDGET_LIMIT` environment variable.
### Fixed
- Exposed `HFLocalClient` at the package root.
- Added side-effect import for the `metrics` module.
- Adjusted retrieval latency measurement timing.

## [0.3.0] - 2024-??-??
### Added
- `WeightedVoter` combining panel diagnoses using confidence scores.
- `--vote-weights` CLI flag for ensemble mode.

## [0.2.0] - 2024-??-??
### Added
- `/case` endpoint for delivering case summaries in the UI.
- Revised web interface with summary panel, ordered-tests list and flow diagram.

## [0.1.0] - 2023-??-??
### Added
- Initial public release of SDB and Dx0 demo with dataset tools, rule-based and LLM-driven panels, cost estimation, permutation tests and CLI.
