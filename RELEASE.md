# Release Procedure

Follow these steps to cut a new release of the project.

1. Update `pyproject.toml` with the new version and add an entry to `CHANGELOG.md`.
2. Ensure all tests pass and the `dist/` directory is clean:
   ```bash
   pytest -q
   ```
3. Build the wheel and source archives:
   ```bash
   python -m build
   ```
4. Commit the changes, tag the commit with the version number and push the tag.
5. Upload the built archives to your package index.
