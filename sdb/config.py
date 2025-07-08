from dataclasses import dataclass
from typing import Optional

import os
import yaml


@dataclass
class Settings:
    """Configuration options loaded from YAML or environment variables."""

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    ollama_base_url: str = "http://localhost:11434"
    metrics_port: int = 8000
    semantic_retrieval: bool = False
    cross_encoder_model: Optional[str] = None


def load_settings(path: str | None = None) -> Settings:
    """Return :class:`Settings` from ``path`` and environment variables."""

    data: dict[str, object] = {}
    if path:
        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    env = os.getenv
    if "openai_api_key" not in data and env("OPENAI_API_KEY"):
        data["openai_api_key"] = env("OPENAI_API_KEY")
    if "openai_model" not in data and env("OPENAI_MODEL"):
        data["openai_model"] = env("OPENAI_MODEL")
    if "ollama_base_url" not in data and env("OLLAMA_BASE_URL"):
        data["ollama_base_url"] = env("OLLAMA_BASE_URL")
    if "metrics_port" not in data and env("SDB_METRICS_PORT"):
        try:
            data["metrics_port"] = int(env("SDB_METRICS_PORT"))
        except ValueError:
            pass
    return Settings(**data)


# Global settings instance used by the package
settings = load_settings()
