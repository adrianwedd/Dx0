# Getting Started

This short guide walks through setting up a development environment for Dx0 and SDBench. It covers creating a Python virtual environment, installing dependencies and running the test suite.

## Environment Setup

Create and activate a virtual environment with Python 3.12 or later:

```bash
python3 -m venv venv
source venv/bin/activate
```

Set `OPENAI_API_KEY` if you intend to query the OpenAI API:

```bash
export OPENAI_API_KEY="sk-..."
```

If you prefer to run models locally, start an [Ollama](https://github.com/jmorganca/ollama) server (``ollama serve``) and pass ``--llm-provider ollama`` to the CLI.

## Installing Dependencies

Install the development requirements to get `pytest` and other tools:

```bash
pip install -r requirements-dev.txt
```

For an editable installation of the library itself run:

```bash
pip install -e .
```

This provides the `dx0` and `sdbench` modules on your `PYTHONPATH` and installs additional developer tools such as `pytest`.

## Running Tests

Execute the unit tests from the repository root:

```bash
pytest -q
```

The suite relies on packages listed in `requirements.lock`. All tests should pass before submitting a pull request.

