# Structured Logging

Dx0 now emits logs in JSON format using [structlog](https://www.structlog.org/).
Initialize logging at application start with `sdb.configure_logging()` which
sets up a JSON renderer and standard timestamp/level fields.

Each log line contains the event name plus keyword attributes. Example:

```json
{"timestamp": "2024-01-01T00:00:00Z", "level": "info", "event": "panel_action", "turn": 1, "type": "question", "content": "chief complaint"}
```

These logs can be piped to tools like `jq` or shipped to your log aggregation
system for analysis.

## Viewing Metrics in Grafana

The application exposes Prometheus counters via `start_metrics_server()`. Import
`docs/grafana-dashboard.json` into Grafana to visualize these metrics. In
addition to the existing LLM and panel counters, two new metrics track resource
usage per user message:

- `user_message_tokens_total` – number of tokens contained in user prompts.
- `user_message_cost_total` – cumulative cost of tests ordered by the user.

Add these series to a dashboard panel to monitor token consumption and spending
over time.

