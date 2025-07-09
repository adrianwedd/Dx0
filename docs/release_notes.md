# Release Notes

## v0.3.0

* Added a `WeightedVoter` that combines panel diagnoses using confidence
  scores. Use the new `--vote-weights` CLI flag to supply run-specific
  weights for ensemble mode.

## v0.2.0

* Added `/case` endpoint delivering the case summary to the UI.
* Web interface now displays a summary panel, ordered-tests list, and diagnostic
  flow diagram using a simple grid layout.

## v0.1.0

Initial public release of the SDB package and Dx0 demo. Highlights:

- Dataset ingestion tools for NEJM CPC cases.
- Rule-based and LLM-driven diagnostic panel.
- Cost estimation and CMS CPT mapping.
- Permutation test statistics and evaluation utilities.
- CLI for running sessions and significance tests.
- Simple FastAPI UI with Prometheus metrics and Grafana dashboard.

## Release Procedure

1. Update the version and changelog sections above.
2. Ensure all tests pass and the `dist/` directory is clean.
3. Build the wheel and source archives:

   ```bash
   python -m build
   ```

4. Tag the commit and push the archives to your package index.


