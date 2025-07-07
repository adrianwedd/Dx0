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
