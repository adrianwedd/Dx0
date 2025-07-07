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

## Development Dependencies

To run the unit tests and other developer tools, install the additional
requirements listed in `requirements-dev.txt`:

```bash
pip install -r requirements-dev.txt
```

The `pytest` suite relies on packages such as `numpy`, `httpx`,
`starlette`, and `pydantic`.

## Data Versioning with DVC

The case data and computed embeddings are tracked with [DVC](https://dvc.org/).
Install the tool and pull the datasets after cloning:

```bash
pip install "dvc[s3]"
dvc pull
```

If you modify files under `data/`, rerun `dvc add` on the directories and push the
changes with `dvc push` so other contributors can reproduce your results.

## Running the Demo UI

Install the development dependencies to get `uvicorn` and the packages used by
the small FastAPI demo:

```bash
pip install -r requirements-dev.txt
```

Start the server and open `http://localhost:8000`:

```bash
uvicorn sdb.ui.app:app --reload
```

Log in with username `physician` and password `secret`. After authentication the
interface calls the `/case` endpoint to display the vignette in the **Case
Summary** panel. The **Ordered Tests** list and **Diagnostic Flow** log update as
you interact with the chat panel.

![Screenshot of UI](images/ui.png.b64)
The screenshot file is base64 encoded. Decode it with:

```bash
base64 -d docs/images/ui.png.b64 > ui.png
```
