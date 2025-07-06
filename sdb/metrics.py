"""Prometheus metrics helpers for MAI-DxO."""

from prometheus_client import Counter, Histogram, start_http_server

ORCHESTRATOR_TURNS = Counter(
    "orchestrator_turns_total", "Number of orchestrator turns executed."
)
PANEL_ACTIONS = Counter(
    "panel_actions_total",
    "Count of panel actions by type.",
    ["action_type"],
)

LLM_LATENCY = Histogram(
    "llm_request_seconds",
    "Latency of LLM requests in seconds.",
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total number of tokens processed by the LLM.",
)


def start_metrics_server(port: int = 8000) -> None:
    """Start a Prometheus metrics HTTP server."""
    start_http_server(port)
