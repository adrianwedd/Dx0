# Dx0 Documentation

Welcome to the Dx0 documentation! This directory contains comprehensive documentation for the Dx0 diagnostic orchestrator system.

## API Documentation

### Quick Start
- **[API Quick Start Guide](quickstart.md)** - Get started with the Dx0 API quickly
- **[API Reference](api-reference.md)** - Complete API documentation with examples
- **[Client Examples](api-client-examples.md)** - Production-ready code examples in Python, JavaScript, and more

### OpenAPI Specification
- **[OpenAPI JSON](openapi.json)** - Machine-readable API specification
- **[OpenAPI YAML](openapi.yaml)** - Human-friendly API specification
- **[Swagger UI](swagger-ui.html)** - Interactive API documentation (open in browser)

## System Documentation

### Getting Started
- **[Installation](installation.md)** - Installation and setup instructions
- **[Getting Started](getting_started.md)** - Basic usage and configuration

### Core Features
- **[Budget Manager](budget_manager.md)** - Cost tracking and budget management
- **[FHIR Export](fhir_export.md)** - Healthcare data interoperability
- **[Monitoring Setup](monitoring_setup.md)** - System monitoring with Prometheus and Grafana
- **[Security Report](security_report.md)** - Security features and best practices

### Development
- **[CLI Reference](sphinx/cli_reference.rst)** - Command-line interface documentation
- **[Plugin Development](persona_plugins.md)** - Creating custom persona plugins
- **[Retrieval Plugins](retrieval_plugins.md)** - Custom retrieval implementations
- **[Cost Estimator Plugins](cost_estimator_plugins.md)** - Custom cost estimation
- **[Async Testing Guide](async_testing_guide.md)** - Async test configuration and best practices

### Evaluation and Testing
- **[Batch Evaluation](batch_eval.md)** - Running large-scale evaluations
- **[Performance Benchmarks](performance_benchmark.md)** - System performance metrics
- **[Weighted Voting](weighted_voter.md)** - Multi-model decision making

## Web Interface

### Running the Physician UI

Launch the FastAPI server with:

```bash
uvicorn sdb.ui.app:app --reload
```

Visit `http://localhost:8000` to open the chat interface.

See `dvc_setup.md` for dataset versioning instructions.

## Case Summary Endpoint

The API exposes a small `/case` endpoint that returns the current case summary
as JSON. The UI calls this endpoint immediately after a successful login to
display the vignette before any questions are asked.

## Web Interface Layout

The React demo arranges four panels in a two-column grid:

* **Case Summary** – shows the summary text returned by `/case`.
* **Ordered Tests** – lists completed labs or imaging studies.
* **Chat Panel** – spans both columns and displays the running conversation and
  live cost summary from :class:`sdb.services.BudgetManager` (which
  replaced the previous `BudgetTracker`).
* **Diagnostic Flow** – captures a step-by-step log of the debate.

Together these panels provide an overview of the ordered tests and reasoning
flow while you chat with the Gatekeeper.

## Weighted Voting

See `weighted_voter.md` for an overview of how the `WeightedVoter` class
combines multiple diagnoses using confidence scores and optional run-specific
weights.

## Persona Models

Each persona in the virtual panel can use a different model. Pass a JSON
mapping via the `--persona-models` CLI option or define `persona_models` in the
YAML settings file:

```yaml
persona_models:
  hypothesis_system: gpt-4
  challenger_system: gpt-3.5-turbo
```

Unspecified personas fall back to the model selected by `--llm-model`.

## Structured Logging

Logs are emitted in JSON format. Initialize logging with
`sdb.configure_logging()` and refer to `logging.md` for examples of consuming
the output.

## Evaluation

Dx0 relies on an LLM-based judge to grade diagnoses on a five-point rubric.
The system prompt in `prompts/judge_system.txt` explains the scale from 1
(completely incorrect) to 5 (clinically equivalent). During evaluation the
model's numeric score is used directly. If no score can be parsed, a default
rating of 1 is assigned.

## Building CLI Documentation

A Sphinx configuration lives in `docs/sphinx`. Install the development
requirements and run `sphinx-build docs/sphinx docs/_build` to generate
HTML docs for the `dx0` command line interface.

## API Documentation Tools

### Exporting OpenAPI Specification

Use the included script to export and validate the OpenAPI specification:

```bash
# Export both JSON and YAML formats
python scripts/export_openapi.py

# Export specific format with info
python scripts/export_openapi.py --format json --info

# Validate the specification
python scripts/export_openapi.py --validate

# Custom output directory
python scripts/export_openapi.py --output-dir /path/to/output
```

### Interactive API Documentation

After exporting the OpenAPI specification, you can use the generated Swagger UI:

1. Open `docs/swagger-ui.html` in your web browser
2. The interface will load the OpenAPI specification from `docs/openapi.json`
3. Explore endpoints, try API calls, and view response schemas

### External Tool Integration

The exported OpenAPI specifications can be used with various tools:

- **Postman**: Import `openapi.json` to generate a Postman collection
- **Insomnia**: Import the specification for API testing
- **Code Generators**: Use with openapi-generator to create client SDKs
- **Documentation Tools**: Generate static documentation with Redoc or similar tools

## Grafana Dashboard

The `grafana-dashboard.json` file contains a minimal dashboard displaying
metrics exported by the Dx0 system. Import it into Grafana to visualize
`panel_actions_total`, `orchestrator_turns_total`, `llm_request_seconds`, and
`llm_tokens_total` counters collected by Prometheus. The latency panel uses the
`llm_request_seconds` histogram to compute the average response time over a
five-minute window.
