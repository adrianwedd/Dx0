# Contributing to Dx0

Thank you for wanting to contribute! This project uses several development tools managed via `pre-commit`.

## Setup

1. Create a Python virtual environment and activate it.
2. Install development dependencies:

   ```bash
   pip install -r requirements-dev.txt
   ```

3. Install the pre-commit hooks so they run automatically on each commit:

   ```bash
   pre-commit install
   ```

## Development workflow

- Run `pre-commit run --all-files` to format and lint the code.
- Execute the unit tests with:

  ```bash
  pytest -q
  ```

All tests should pass before submitting a pull request.
