"""
Application configuration — Milestone 8 (production hardening).

All configuration is sourced from environment variables with the ``ETI_`` prefix.
No credentials, API keys, or passwords are ever hardcoded. Missing required
secrets raise a clear configuration error at startup.

Uses pydantic-settings for validation and type coercion.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """ETI application settings.

    Environment variables are read with the ``ETI_`` prefix. Example::

        ETI_APP_NAME="Egreed Trade Intelligence"
        ETI_APP_ENV=production
        ETI_API_KEY=supersecret123
        ETI_SECRET_KEY=supersecret123
        ETI_LOG_LEVEL=INFO
        ETI_RATE_LIMIT_REQUESTS_PER_MIN=60
    """

    model_config = SettingsConfigDict(
        env_prefix="ETI_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "Egreed Trade Intelligence"
    app_env: str = Field(
        default="development",
        pattern="^(development|staging|production)$",
        description="Deployment environment.",
    )
    app_version: str = "0.1.0"
    debug: bool = Field(default=False, description="Enable debug mode.")

    # Paths (relative to project root)
    data_dir: Path = Field(default=Path("data"), description="Base data directory.")
    raw_data_dir: Path = Field(
        default=Path("data/raw"), description="Raw CSV source directory."
    )
    quality_dir: Path = Field(
        default=Path("data/quality"), description="Derived quality reports."
    )
    logs_dir: Path = Field(
        default=Path("logs"), description="Application log directory."
    )

    # Security
    api_key: str | None = Field(
        default=None,
        description=(
            "Optional API key. When set, clients must send it via the "
            "X-API-Key header. Must not be a placeholder value."
        ),
    )
    api_key_env_name: str = Field(
        default="ETI_API_KEY",
        description="Environment variable name holding the API key.",
    )
    secret_key: str | None = Field(
        default=None,
        description="Secret key for signing/token operations. Required in production.",
    )
    allowed_origins: list[str] = Field(
        default=["*"],
        description="CORS allowed origins. Restrict in production.",
    )

    # Rate limiting
    rate_limit_requests_per_min: int = Field(
        default=60, ge=1, le=5000, description="Per-client rate limit."
    )
    rate_limit_enabled: bool = Field(
        default=True, description="Enable rate limiting."
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        pattern="^(DEBUG|INFO|WARNING|ERROR|CRITICAL)$",
        description="Root logger level.",
    )
    log_format: str = Field(
        default="json",
        pattern="^(json|text)$",
        description="Log output format.",
    )

    # Data quality / backtest defaults
    default_cost_bps: float = Field(
        default=0.2, ge=0, le=1000, description="Default transaction cost in bps."
    )
    default_train_frac: float = Field(
        default=0.6, ge=0.01, le=0.99, description="Default train fraction."
    )
    default_val_frac: float = Field(
        default=0.2, ge=0.01, le=0.99, description="Default validation fraction."
    )

    @validator("secret_key", always=True)
    @classmethod
    def _secret_key_required_in_production(cls, v: str | None, values) -> str | None:
        if values.data.get("app_env") == "production" and not v:
            raise ValueError("ETI_SECRET_KEY is required when ETI_APP_ENV=production")
        return v

    @validator("api_key")
    @classmethod
    def _api_key_not_placeholder(cls, v: str | None) -> str | None:
        if v and v.lower() in {"change-me", "changeme", "default", "secret", "password"}:
            raise ValueError("ETI_API_KEY must not be a placeholder value")
        return v

    @validator("allowed_origins", always=True)
    @classmethod
    def _restrict_cors_in_production(cls, v: list[str], values) -> list[str]:
        if values.data.get("app_env") == "production" and "*" in v:
            # Keep "*" but flag it; operator should replace with specific origins
            import warnings
            warnings.warn(
                "CORS allowed_origins=['*'] in production — restrict for deployment."
            )
        return v

    def is_production(self) -> bool:
        return self.app_env == "production"

    def is_development(self) -> bool:
        return self.app_env == "development"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance. Call once at startup to validate config early."""
    return Settings()


def validate_startup_config() -> list[str]:
    """Validate configuration and return a list of human-readable warnings.

    Call at application startup. Does not raise; callers can log warnings
    and continue in development, but should treat production warnings seriously.
    """
    settings = get_settings()
    warnings: list[str] = []
    if settings.is_production():
        if settings.debug:
            warnings.append("DEBUG is enabled in production — disable it.")
        if not settings.secret_key:
            warnings.append("SECRET_KEY is not set in production.")
        if not settings.api_key:
            warnings.append("API_KEY is not set in production — API is unauthenticated.")
        if settings.allowed_origins == ["*"]:
            warnings.append(
                "CORS allows all origins (*) — restrict for production deployment."
            )
    if settings.rate_limit_requests_per_min > 1000:
        warnings.append(
            "Rate limit is very high (>{} req/min) — consider lowering.".format(
                settings.rate_limit_requests_per_min
            )
        )
    if not settings.data_dir.exists():
        warnings.append("DATA_DIR does not exist: {}".format(settings.data_dir))
    if not settings.raw_data_dir.exists():
        warnings.append("RAW_DATA_DIR does not exist: {}".format(settings.raw_data_dir))
    return warnings