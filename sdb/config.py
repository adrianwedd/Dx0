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
    persona_models: dict[str, str] = {}
    tracing: bool = False
    tracing_host: str = "localhost"
    tracing_port: int = 6831

    @field_validator("metrics_port")
    @classmethod
    def _check_port(cls, value: int) -> int:
        """Ensure the metrics port is within the valid TCP range."""
        if not (1 <= value <= 65535):
            raise ValueError("metrics_port must be between 1 and 65535")
        return value

    @field_validator("tracing_port")
    @classmethod
    def _check_tracing_port(cls, value: int) -> int:
        """Validate that the Jaeger port is in the valid range."""
        if not (1 <= value <= 65535):
            raise ValueError("tracing_port must be between 1 and 65535")
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
    if "tracing" not in data and env("SDB_TRACING_ENABLED"):
        data["tracing"] = env("SDB_TRACING_ENABLED").lower() == "true"
    if "tracing_host" not in data and env("SDB_TRACING_HOST"):
        data["tracing_host"] = env("SDB_TRACING_HOST")
    if "tracing_port" not in data and env("SDB_TRACING_PORT"):
        try:
            data["tracing_port"] = int(env("SDB_TRACING_PORT"))
        except ValueError:
            pass
    try:
        settings_obj = Settings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration: {exc}") from exc
    globals()["settings"] = settings_obj
    return settings_obj


# Global settings instance used by the package
settings = load_settings()


def configure_tracing() -> None:
    """Initialize Jaeger tracing if enabled."""

    if not settings.tracing:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    except Exception:  # pragma: no cover - optional dependency
        return

    resource = Resource.create({"service.name": "sdb"})
    provider = TracerProvider(resource=resource)
    exporter = JaegerExporter(
        agent_host_name=settings.tracing_host,
        agent_port=settings.tracing_port,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)


configure_tracing()
