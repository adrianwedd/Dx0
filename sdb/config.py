from typing import Optional

from pydantic import BaseModel, ValidationError, HttpUrl, field_validator

import os
import yaml


class Settings(BaseModel):
    """Configuration options loaded from YAML or environment variables."""

    # LLM Configuration
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4"
    hf_model: Optional[str] = None
    ollama_base_url: HttpUrl = "http://localhost:11434"
    
    # Core Application Settings
    metrics_port: int = 8000
    semantic_retrieval: bool = False
    cross_encoder_model: Optional[str] = None
    retrieval_backend: Optional[str] = None
    cost_estimator_plugin: Optional[str] = None
    retrieval_cache_ttl: int = 300
    case_db: Optional[str] = None
    case_db_sqlite: Optional[str] = None
    parallel_personas: bool = False
    persona_models: dict[str, str] = {}
    
    # UI and Session Configuration
    ui_budget_limit: Optional[float] = None
    ui_secret_key: str = "change-me"
    ui_token_ttl: int = 3600
    sessions_db: str = "sessions.db"
    ui_users_file: Optional[str] = None
    failed_login_limit: int = 5
    failed_login_cooldown: int = 300
    message_rate_limit: int = 30
    message_rate_window: int = 60
    
    # External Services
    sentry_dsn: Optional[str] = None
    cms_pricing_url: Optional[str] = None
    
    # Tracing Configuration
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

    @field_validator("retrieval_cache_ttl")
    @classmethod
    def _check_cache_ttl(cls, value: int) -> int:
        """Ensure the retrieval cache TTL is positive."""
        if value <= 0:
            raise ValueError("retrieval_cache_ttl must be positive")
        return value

    @field_validator("ui_token_ttl")
    @classmethod
    def _check_token_ttl(cls, value: int) -> int:
        """Ensure the UI token TTL is positive."""
        if value <= 0:
            raise ValueError("ui_token_ttl must be positive")
        return value

    @field_validator("failed_login_limit")
    @classmethod
    def _check_failed_login_limit(cls, value: int) -> int:
        """Ensure failed login limit is positive."""
        if value <= 0:
            raise ValueError("failed_login_limit must be positive")
        return value

    @field_validator("failed_login_cooldown")
    @classmethod
    def _check_failed_login_cooldown(cls, value: int) -> int:
        """Ensure failed login cooldown is positive."""
        if value <= 0:
            raise ValueError("failed_login_cooldown must be positive")
        return value

    @field_validator("message_rate_limit")
    @classmethod
    def _check_message_rate_limit(cls, value: int) -> int:
        """Ensure message rate limit is positive."""
        if value <= 0:
            raise ValueError("message_rate_limit must be positive")
        return value

    @field_validator("message_rate_window")
    @classmethod
    def _check_message_rate_window(cls, value: int) -> int:
        """Ensure message rate window is positive."""
        if value <= 0:
            raise ValueError("message_rate_window must be positive")
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
    if "hf_model" not in data and env("HF_MODEL"):
        data["hf_model"] = env("HF_MODEL")
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
    if "retrieval_backend" not in data and env("SDB_RETRIEVAL_BACKEND"):
        data["retrieval_backend"] = env("SDB_RETRIEVAL_BACKEND")
    if "cost_estimator_plugin" not in data and env("SDB_COST_ESTIMATOR"):
        data["cost_estimator_plugin"] = env("SDB_COST_ESTIMATOR")
    if "retrieval_cache_ttl" not in data and env("SDB_RETRIEVAL_CACHE_TTL"):
        try:
            data["retrieval_cache_ttl"] = int(env("SDB_RETRIEVAL_CACHE_TTL"))
        except ValueError:
            pass
    if "tracing" not in data and env("SDB_TRACING_ENABLED"):
        data["tracing"] = env("SDB_TRACING_ENABLED").lower() == "true"
    if "tracing_host" not in data and env("SDB_TRACING_HOST"):
        data["tracing_host"] = env("SDB_TRACING_HOST")
    if "tracing_port" not in data and env("SDB_TRACING_PORT"):
        try:
            data["tracing_port"] = int(env("SDB_TRACING_PORT"))
        except ValueError:
            pass
    
    # UI and Session Configuration Environment Variables
    if "ui_budget_limit" not in data and env("UI_BUDGET_LIMIT"):
        try:
            data["ui_budget_limit"] = float(env("UI_BUDGET_LIMIT"))
        except ValueError:
            pass
    if "ui_secret_key" not in data and env("UI_SECRET_KEY"):
        data["ui_secret_key"] = env("UI_SECRET_KEY")
    if "ui_token_ttl" not in data and env("UI_TOKEN_TTL"):
        try:
            data["ui_token_ttl"] = int(env("UI_TOKEN_TTL"))
        except ValueError:
            pass
    if "sessions_db" not in data and env("SESSIONS_DB"):
        data["sessions_db"] = env("SESSIONS_DB")
    if "ui_users_file" not in data and env("UI_USERS_FILE"):
        data["ui_users_file"] = env("UI_USERS_FILE")
    if "failed_login_limit" not in data and env("FAILED_LOGIN_LIMIT"):
        try:
            data["failed_login_limit"] = int(env("FAILED_LOGIN_LIMIT"))
        except ValueError:
            pass
    if "failed_login_cooldown" not in data and env("FAILED_LOGIN_COOLDOWN"):
        try:
            data["failed_login_cooldown"] = int(env("FAILED_LOGIN_COOLDOWN"))
        except ValueError:
            pass
    if "message_rate_limit" not in data and env("MESSAGE_RATE_LIMIT"):
        try:
            data["message_rate_limit"] = int(env("MESSAGE_RATE_LIMIT"))
        except ValueError:
            pass
    if "message_rate_window" not in data and env("MESSAGE_RATE_WINDOW"):
        try:
            data["message_rate_window"] = int(env("MESSAGE_RATE_WINDOW"))
        except ValueError:
            pass
    
    # External Services Environment Variables
    if "sentry_dsn" not in data and env("SENTRY_DSN"):
        data["sentry_dsn"] = env("SENTRY_DSN")
    if "cms_pricing_url" not in data and env("CMS_PRICING_URL"):
        data["cms_pricing_url"] = env("CMS_PRICING_URL")
    
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
