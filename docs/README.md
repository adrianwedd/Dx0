# Grafana Dashboard

The `grafana-dashboard.json` file contains a minimal dashboard displaying
metrics exported by the MAI-DxO demo. Import it into Grafana to visualize
`panel_actions_total`, `orchestrator_turns_total`, `llm_request_seconds`, and
`llm_tokens_total` counters collected by Prometheus. The latency panel uses the
`llm_request_seconds` histogram to compute the average response time over a
five-minute window.

## Running the Physician UI

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
