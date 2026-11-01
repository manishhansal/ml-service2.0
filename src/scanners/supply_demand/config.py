"""
SDConfig — all tunable parameters for the Supply & Demand Zone Scanner pipeline.

Values are read from environment variables with the ``SD_`` prefix, allowing
per-environment overrides without code changes.  Every field carries a
``description`` for self-documentation and IDE introspection.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SDConfig(BaseSettings):
    """
    All scanner parameters.  Reads from env vars (SD_* prefix) or .env file.
    Passed as a single config object to every sub-component so tests can
    override any parameter without monkey-patching globals.
    """

    model_config = SettingsConfigDict(env_prefix="SD_", extra="ignore")

    # ── ATR ───────────────────────────────────────────────────────────────────

    atr_window: int = Field(
        default=14,
        description="ATR rolling window used throughout the pipeline",
    )

    # ── Base candle detection ─────────────────────────────────────────────────

    base_body_threshold: float = Field(
        default=0.5,
        description="Max base candle body as fraction of ATR14",
    )

    # ── Departure candle detection ────────────────────────────────────────────

    departure_body_threshold: float = Field(
        default=1.5,
        description="Min departure candle body as multiple of ATR14",
    )

    # ── Zone width ────────────────────────────────────────────────────────────

    min_zone_width_multiplier: float = Field(
        default=0.3,
        description="Min zone height as multiple of ATR14 at departure bar",
    )

    # ── Base candle count bounds ──────────────────────────────────────────────

    min_base_candles: int = Field(
        default=2,
        description="Minimum number of base candles required in a valid zone",
    )
    max_base_candles: int = Field(
        default=5,
        description="Maximum number of base candles allowed in a valid zone",
    )

    # ── Multi-TF confluence ───────────────────────────────────────────────────

    confluence_price_pct: float = Field(
        default=0.01,
        description="Price proximity threshold (as fraction) for multi-timeframe confluence",
    )

    # ── Signal quality filters ────────────────────────────────────────────────

    require_trend_alignment: bool = Field(
        default=True,
        description="Only emit zones aligned with symbol trend",
    )
    min_daily_value_crore: float = Field(
        default=50.0,
        description="Skip illiquid symbols",
    )
    min_rr_ratio: float = Field(
        default=1.0,
        description="Minimum intraday R:R ratio",
    )
    fresh_zones_only: bool = Field(
        default=True,
        description="Only surface FRESH zones (never tested)",
    )

    # ── API defaults ──────────────────────────────────────────────────────────

    default_limit: int = Field(
        default=50,
        description="Default API result count",
    )
    max_limit: int = Field(
        default=200,
        description="Max API result count",
    )
