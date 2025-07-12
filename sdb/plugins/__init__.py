"""Plugin utilities and built-in persona plugins."""

from __future__ import annotations

from importlib import metadata
from typing import Iterable, List

import structlog
from pydantic import BaseModel, ValidationError, field_validator

logger = structlog.get_logger(__name__)

# Entry point groups exposing pluggable components
PLUGIN_GROUPS = (
    "dx0.personas",
    "sdb.retrieval_plugins",
    "dx0.cost_estimators",
)


class PluginInfo(BaseModel):
    """Metadata describing an installed plugin."""

    name: str
    version: str
    entry_point: str

    @field_validator("name", "version", "entry_point")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("must not be empty")
        return value


def validate_plugins(groups: Iterable[str] = PLUGIN_GROUPS) -> List[PluginInfo]:
    """Return metadata for plugins registered under ``groups``.

    Raises
    ------
    RuntimeError
        If a plugin is missing required metadata.
    """

    infos: List[PluginInfo] = []
    for group in groups:
        for ep in metadata.entry_points(group=group):
            dist = ep.dist
            data = {
                "name": getattr(dist, "name", ""),
                "version": getattr(dist, "version", ""),
                "entry_point": ep.value,
            }
            try:
                info = PluginInfo.model_validate(data)
            except ValidationError as exc:  # pragma: no cover - sanity check
                raise RuntimeError(
                    f"Invalid plugin metadata for '{ep.value}': {exc}"
                ) from exc
            infos.append(info)
    return infos


# Validate plugins on import to fail fast when misconfigured
validate_plugins()
