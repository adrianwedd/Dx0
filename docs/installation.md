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

## Usage Examples

Run the CLI on a single case in budgeted mode:

```bash
python -m dx0.cli \
  --mode budgeted \
  --case-file data/sdbench/cases/case_001.json \
  --budget 1000 \
  --output results/case_001.json
```

The CLI instantiates :class:`sdb.services.BudgetManager` to track the cumulative
cost of ordered tests. Once the limit is reached the session stops.

To start the demo web UI locally:

```bash
uvicorn sdb.ui.app:app --reload
```

Export a saved transcript to a FHIR bundle:

```bash
python cli.py export-fhir --input session.json --output fhir/bundle.json
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
requirements listed in `requirements.lock`:

```bash
pip install -r requirements.lock
```

The `pytest` suite relies on packages such as `numpy`, `httpx`,
`starlette`, and `pydantic`.

## Retrieval Plugins

Third-party retrieval backends can be integrated through Python entry points.
Expose your index class in a small package and declare it under the
`sdb.retrieval_plugins` group in `pyproject.toml`:

```toml
[project.entry-points."sdb.retrieval_plugins"]
my-index = "my_package.module:MyIndex"
```

After installing the package with `pip`, the gatekeeper can load your plugin
instead of the built-in backends. Set `retrieval_backend` in `settings.yaml`
or the `SDB_RETRIEVAL_BACKEND` environment variable to select the plugin.

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
pip install -r requirements.lock
```

Start the server and open `http://localhost:8000`:

```bash
uvicorn sdb.ui.app:app --reload
```

Set `UI_BUDGET_LIMIT` to change the default spending cap for new sessions:

```bash
export UI_BUDGET_LIMIT=750
uvicorn sdb.ui.app:app --reload
```

You can also override the limit for a specific session by including a
`budget` query parameter when the client opens the WebSocket connection:

```
ws://localhost:8000/api/v1/ws?token=<ACCESS_TOKEN>&budget=500
```

The server tracks failed login attempts per IP address. Adjust the limit and
cooldown with environment variables:

```bash
export FAILED_LOGIN_LIMIT=5       # attempts before blocking
export FAILED_LOGIN_COOLDOWN=300  # seconds to wait before retry
```

Message rates are also throttled per session. Configure the sliding window
limits with:

```bash
export MESSAGE_RATE_LIMIT=30      # messages per window
export MESSAGE_RATE_WINDOW=60     # window size in seconds
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

### Managing UI Users

The web interface reads credentials from `sdb/ui/users.yml`. Use the CLI to add
or remove accounts without editing the file directly:

```bash
python cli.py manage-users add alice --password secret
python cli.py manage-users list
python cli.py manage-users remove alice
```

### Exporting Traces to Jaeger

Tracing is disabled by default. To collect spans and send them to a running
Jaeger agent, set the following environment variables before starting the
server:

```bash
export SDB_TRACING_ENABLED=true
export SDB_TRACING_HOST=localhost  # Jaeger agent host
export SDB_TRACING_PORT=6831       # Jaeger UDP port
```

Then run the FastAPI application as usual. Jaeger can be started locally with:

```bash
docker run -p 6831:6831/udp -p 16686:16686 jaegertracing/all-in-one
```

Browse to `http://localhost:16686` to view the collected traces.

## Docker Compose Setup

Use the provided `docker-compose.yml` to start the demo alongside Prometheus and Grafana:

```bash
docker compose up
```

The Dx0 API is available on [http://localhost:8000](http://localhost:8000), Prometheus on port `9090`, and Grafana on `3000`.

## Kubernetes Setup

If you prefer Kubernetes, apply the manifests in the `k8s` directory:

```bash
kubectl apply -f k8s/
```

Port-forward the API service with:

```bash
kubectl port-forward service/dx0 8000:8000
```


## Testing

Install the development dependencies and run the test suite:

```bash
./scripts/install_dev_deps.sh
pytest -q
```

