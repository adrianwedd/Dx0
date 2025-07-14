# Release Workflow

This project publishes a new version whenever a tag matching `v*.*.*` is pushed.
The GitHub Actions workflow builds the Docker image and Python wheel,
uploads them, and creates a GitHub release.

Release notes are assembled automatically from commit messages using the
`scripts/generate_release_notes.py` helper.

## Steps to cut a release

1. Run `python scripts/generate_release_notes.py` to update
   `docs/release_notes.md` with changes since the last tag.
2. Commit the updated notes and push them to the `main` branch.
3. Create a signed tag for the new version, e.g. `git tag -s v1.2.0 -m "v1.2.0"`.
4. Push the tag with `git push origin v1.2.0`.
5. The workflow will publish artifacts and use the generated notes as the
   body of the GitHub release.
