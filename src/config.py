"""
pydantic-settings configuration for ml-service2.0.

All settings are read from environment variables and/or a ``.env`` file.
Variables are case-insensitive by default.

Usage::

    from src.config import settings
    print(settings.port)          # 8100
    print(settings.deployment_mode)   # DeploymentMode.RESEARCH
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.schemas.base import DeploymentMode


class Settings(BaseSettings):
    """
    Centralised configuration for ml-service2.0.

    Every field maps 1-to-1 with an environment variable (case-insensitive).
    Fields without a default are **required** — the service will refuse to
    start if they are absent from the environment or ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow extra env vars without raising a validation error
        extra="ignore",
    )

    # ── Server ────────────────────────────────────────────────────────────────
    port: int = Field(default=8100, description="Port ml-service2.0 listens on.")
    log_level: str = Field(
        default="INFO",
        description="Minimum structlog log level: DEBUG | INFO | WARNING | ERROR | CRITICAL.",
    )
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.RESEARCH,
        description=(
            "Controls heuristic fallback and live-trading eligibility. "
            "RESEARCH | PAPER | SHADOW | VALIDATED_ML_ONLY"
        ),
    )
    # pydantic-settings will try to JSON-decode `list[str]` fields when they
    # come from env vars.  Declaring the type as `str | list[str]` prevents
    # that pre-parse step; the `_parse_allowed_origins` validator below
    # normalises the value to `list[str]` regardless of how it was supplied.
    allowed_origins: str | list[str] = Field(
        default=["http://localhost:3000"],
        description=(
            "CORS allowed origins for alpha-forge. "
            "Accepts a JSON array or a comma-separated string (env var friendly)."
        ),
    )

    # ── Authentication ────────────────────────────────────────────────────────
    ml_service_api_key: str = Field(
        description="API key required in X-API-KEY header for all prediction endpoints.",
    )

    # ── Upstream: data-service2.0 ─────────────────────────────────────────────
    data_service_2_url: str = Field(
        default="http://localhost:8200",
        description="Base URL for data-service2.0.",
    )
    data_service_api_key: str = Field(
        description="API key or JWT used to authenticate with data-service2.0.",
    )

    # ── Upstream: SentinelPulse ───────────────────────────────────────────────
    sentinel_pulse_url: str = Field(
        default="http://localhost:3001",
        description="Base URL for SentinelPulse NLP/news intelligence service.",
    )
    sentinel_pulse_api_key: str = Field(
        default="",
        description="API key for SentinelPulse (required in production, can be empty for dev).",
    )

    # ── Storage: Redis ────────────────────────────────────────────────────────
    redis_url: str = Field(
        default="redis://localhost:6379",
        description="Redis connection URL (feature cache TTL=60s, news cache TTL=90s).",
    )

    # ── Storage: MLflow ───────────────────────────────────────────────────────
    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI for experiment logging and model registry.",
    )

    # ── Storage: Filesystem ───────────────────────────────────────────────────
    model_artifacts_path: Path = Field(
        default=Path("./artifacts"),
        description=(
            "Root path for model artifact storage "
            "(must be a persistent volume in production)."
        ),
    )
    audit_log_path: Path = Field(
        default=Path("./audit.jsonl"),
        description="Path to the append-only audit log (JSON Lines format).",
    )

    # ── Feature Pipeline ──────────────────────────────────────────────────────
    feature_cache_ttl: int = Field(
        default=60,
        ge=10,
        description="TTL in seconds for feature vectors cached in Redis.",
    )
    news_context_cache_ttl: int = Field(
        default=90,
        ge=10,
        description="TTL in seconds for SentinelPulse news context in the LRU cache.",
    )
    min_confidence_score: int = Field(
        default=70,
        ge=0,
        le=100,
        description=(
            "Minimum DataConfidenceScore from data-service2.0 (0–100). "
            "Responses below this threshold return INSUFFICIENT_EVIDENCE."
        ),
    )
    alpha360_enabled: bool = Field(
        default=False,
        description="Set to true to enable Qlib Alpha360 360-dimensional factor computation (slower).",
    )

    # ── Training Pipeline ─────────────────────────────────────────────────────
    embargo_period_days: int = Field(
        default=10,
        ge=5,
        description="Embargo period in trading days for purged K-fold CV.",
    )
    shadow_trading_min_days: int = Field(
        default=20,
        ge=1,
        description=(
            "Minimum trading days a model must spend in SHADOW "
            "before LIVE promotion eligibility."
        ),
    )
    optuna_n_trials: int = Field(
        default=50,
        ge=50,
        description="Number of Optuna HPO trials per model.",
    )
    cpcv_n_paths: int = Field(
        default=10,
        ge=10,
        description="Number of overlapping test paths for CPCV.",
    )
    model_acceptance_min_ic: float = Field(
        default=0.02,
        gt=0.0,
        description="Minimum mean IC for the model acceptance gate.",
    )
    model_acceptance_max_pbo: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Maximum backtest overfitting probability for the acceptance gate.",
    )

    # ── Online Learning ───────────────────────────────────────────────────────
    online_learning_enabled: bool = Field(
        default=False,
        description="Set to true to enable incremental model updates on IC degradation.",
    )
    max_consecutive_online_updates: int = Field(
        default=5,
        ge=1,
        description=(
            "Maximum consecutive incremental updates before a full retraining "
            "cycle is required."
        ),
    )
    online_learning_ic_degradation_threshold: float = Field(
        default=0.20,
        gt=0.0,
        le=1.0,
        description=(
            "IC degradation fraction from 90-day baseline that triggers "
            "online learning."
        ),
    )

    # ── gRPC ──────────────────────────────────────────────────────────────────
    grpc_connection_timeout: float = Field(
        default=5.0,
        gt=0.0,
        description="gRPC connection establishment timeout in seconds.",
    )
    grpc_per_call_deadline: float = Field(
        default=2.0,
        gt=0.0,
        description="gRPC per-call deadline in seconds.",
    )
    grpc_max_reconnect_attempts: int = Field(
        default=3,
        ge=1,
        description=(
            "Maximum reconnect attempts for an interrupted gRPC stream "
            "before returning UNAVAILABLE."
        ),
    )

    # ── LLM / Meta-Decision Engine ────────────────────────────────────────────
    llm_inference_timeout_ms: int = Field(
        default=120,
        ge=50,
        description=(
            "Maximum time in milliseconds for FinGPT/FinBERT LLM inference "
            "before falling back to cache."
        ),
    )
    llm_model_name: str = Field(
        default="ProsusAI/finbert",
        description=(
            "HuggingFace model identifier for LLM news reasoning. "
            "Default: ProsusAI/finbert | Alternative: THUDM/FinGPT-v3.3"
        ),
    )

    # ── Promotion Gates ───────────────────────────────────────────────────────
    predictive_gate_margin: float = Field(
        default=0.005,
        ge=0.001,
        le=0.05,
        description=(
            "Minimum IC improvement over champion required for the PREDICTIVE gate "
            "(range 0.001–0.05)."
        ),
    )
    approval_token_expiry_hours: int = Field(
        default=24,
        ge=1,
        description=(
            "Number of hours before an approvalToken expires "
            "for human-approved promotions."
        ),
    )

    # ── Performance ───────────────────────────────────────────────────────────
    uvicorn_workers: int = Field(
        default=4,
        ge=1,
        description="Number of Uvicorn worker processes.",
    )

    # ── Validators ───────────────────────────────────────────────────────────

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_allowed_origins(cls, value: Any) -> list[str]:
        """
        Handle comma-separated env var strings transparently.

        pydantic-settings will attempt JSON-decode on complex fields first.
        If the env var is a plain comma-separated string (not JSON), the
        JSON decode fails and pydantic-settings raises before this validator
        runs.  We work around that by accepting ``str`` here and splitting
        manually; when the value arrives already as a list (e.g. from a
        JSON-array env var or from code) we pass it through unchanged.

        ``ALLOWED_ORIGINS=http://localhost:3000,https://prod.example.com``
        is equivalent to
        ``ALLOWED_ORIGINS=["http://localhost:3000","https://prod.example.com"]``.
        """
        if isinstance(value, str):
            stripped = value.strip()
            # If it looks like a JSON array, let Pydantic handle it normally
            # (this branch is reached only if pydantic-settings passes the
            # raw string through without pre-parsing, which can happen in
            # some versions).
            if stripped.startswith("["):
                import json as _json
                return _json.loads(stripped)
            return [origin.strip() for origin in stripped.split(",") if origin.strip()]
        return list(value)


# ── Module-level singleton ─────────────────────────────────────────────────────
#
# Imported by every module that needs configuration.  The singleton is created
# once at import time so the .env file is read only once per process.
settings = Settings()
