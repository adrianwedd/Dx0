"""Prometheus metrics helpers for MAI-DxO."""

from prometheus_client import Counter, start_http_server

ORCHESTRATOR_TURNS = Counter(
    "orchestrator_turns_total", "Number of orchestrator turns executed."
)
PANEL_ACTIONS = Counter(
    "panel_actions_total",
    "Count of panel actions by type.",
    ["action_type"],
)


def start_metrics_server(port: int = 8000) -> None:
    """Start a Prometheus metrics HTTP server."""
    start_http_server(port)
