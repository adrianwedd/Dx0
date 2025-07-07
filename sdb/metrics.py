"""Prometheus metrics helpers for MAI-DxO."""

import os
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

# Time spent processing each orchestrator turn.
ORCHESTRATOR_LATENCY = Histogram(
    "orchestrator_turn_seconds", "Time spent in each orchestrator turn.",
)

LLM_TOKENS = Counter(
    "llm_tokens_total",
    "Total number of tokens processed by the LLM.",
)


def start_metrics_server(port: int | None = None) -> None:
    """Start a Prometheus metrics HTTP server.

    Parameters
    ----------
    port:
        Port for the HTTP server. If ``None`` the value from the
        ``SDB_METRICS_PORT`` environment variable is used when set,
        otherwise ``8000``.
    """

    if port is None:
        env = os.getenv("SDB_METRICS_PORT")
        port = int(env) if env else 8000
    start_http_server(port)
