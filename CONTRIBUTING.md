# Contributing to Dx0

Thank you for wanting to contribute! This project uses several development tools managed via `pre-commit`.

## Setup

1. Create a Python virtual environment and activate it.
2. Install the pinned dependencies and development tools:

   ```bash
   pip install -r requirements.lock
   make install-dev
   ```

3. Install the pre-commit hooks so they run automatically on each commit:

   ```bash
   pre-commit install
   ```

## Testing requirements

Install **all** packages listed in `requirements.lock` before running the
test suite. The `make install-dev` target installs additional developer
tools, including `pytest`.

## Development workflow

- Run `pre-commit run --all-files` to format and lint the code.
- Execute the unit tests with:

  ```bash
  pytest -q
  ```

All tests should pass before submitting a pull request.
