# Installation

The SDB package can be installed from PyPI or any compliant registry.

```bash
pip install sdb
```

To install from a private registry, specify an extra index URL:

```bash
pip install --extra-index-url <URL> sdb
```

Developers working from a cloned repository can install in editable mode:

```bash
pip install -e .
```

## Environment Setup

### Using the OpenAI API

Set the `OPENAI_API_KEY` environment variable so the orchestration engine can
access the OpenAI chat completion endpoint:

```bash
export OPENAI_API_KEY="sk-..."
```

### Using a Local Ollama Server

If you prefer to run models locally with
[Ollama](https://github.com/jmorganca/ollama), ensure the server is running
(for example via `ollama serve`) and install the package dependencies. The
client defaults to `http://localhost:11434` but you can specify a different
address with the `--llm-provider ollama` and `--llm-model` CLI options.

## Building a Distribution

To create a wheel and source distribution for publishing, install the
`build` tool and run:

```bash
pip install build
python -m build
```

The archives will be written to the `dist/` directory as
`sdb-<version>.tar.gz` and `sdb-<version>-py3-none-any.whl`.
