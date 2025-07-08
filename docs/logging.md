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

