"""Prometheus metrics helpers for MAI-DxO."""

import os
from prometheus_client import Counter, Histogram, start_http_server
from .config import settings

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

# Tokens present in each user message sent to the orchestrator.
USER_MESSAGE_TOKENS = Counter(
    "user_message_tokens_total",
    "Total number of tokens contained in user messages.",
)

# Cost accrued from user actions such as ordering tests.
USER_MESSAGE_COST = Counter(
    "user_message_cost_total",
    "Total cost incurred as a result of user messages.",
)

# Count of CPT lookups served from the local cache.
CPT_CACHE_HITS = Counter(
    "cpt_cache_hits_total", "Number of CPT lookups served from cache."
)

# Count of CPT lookups that required an LLM call.
CPT_LLM_LOOKUPS = Counter(
    "cpt_llm_lookups_total", "Number of CPT lookups resolved via LLM."
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
        if env:
            port = int(env)
        else:
            port = settings.metrics_port
    start_http_server(port)
