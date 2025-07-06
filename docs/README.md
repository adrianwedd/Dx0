# Grafana Dashboard

The `grafana-dashboard.json` file contains a minimal dashboard displaying
metrics exported by the MAI-DxO demo. Import it into Grafana to visualize
`panel_actions_total` and `orchestrator_turns_total` counters collected by
Prometheus.

## Running the Physician UI

Launch the FastAPI server with:

```bash
uvicorn sdb.ui.app:app --reload
```

Visit `http://localhost:8000` to open the chat interface.
