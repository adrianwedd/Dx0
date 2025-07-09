from typing import Optional

from pydantic import BaseModel, ValidationError, HttpUrl, field_validator

import os
import yaml


class Settings(BaseModel):
    """Configuration options loaded from YAML or environment variables."""

    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    ollama_base_url: HttpUrl = "http://localhost:11434"
    metrics_port: int = 8000
    semantic_retrieval: bool = False
    cross_encoder_model: Optional[str] = None
    case_db: Optional[str] = None
    case_db_sqlite: Optional[str] = None
    parallel_personas: bool = False

    @field_validator("metrics_port")
    @classmethod
    def _check_port(cls, value: int) -> int:
        """Ensure the metrics port is within the valid TCP range."""
        if not (1 <= value <= 65535):
            raise ValueError("metrics_port must be between 1 and 65535")
        return value


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
    if "case_db" not in data and env("SDB_CASE_DB"):
        data["case_db"] = env("SDB_CASE_DB")
    if "case_db_sqlite" not in data and env("SDB_CASE_DB_SQLITE"):
        data["case_db_sqlite"] = env("SDB_CASE_DB_SQLITE")
    if "parallel_personas" not in data and env("SDB_PARALLEL_PERSONAS"):
        data["parallel_personas"] = env("SDB_PARALLEL_PERSONAS").lower() == "true"
    try:
        settings_obj = Settings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    globals()["settings"] = settings_obj
    return settings_obj


# Global settings instance used by the package
settings = load_settings()
