"""
FVGConfig — all tunable parameters for the Fair Value Gap Scanner pipeline.

Values are read from environment variables with the ``FVG_`` prefix, allowing
per-environment overrides without code changes.  Every field carries a
``description`` for self-documentation and IDE introspection.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class FVGConfig(BaseSettings):
    """
    All scanner parameters.  Reads from env vars (FVG_* prefix) or .env file.
    Passed as a single config object to every sub-component so tests can
    override any parameter without monkey-patching globals.
    """

    model_config = SettingsConfigDict(env_prefix="FVG_", extra="ignore")

    # ── Detection ─────────────────────────────────────────────────────────────

    atr_window: int = Field(
        default=14,
        description="ATR rolling window for impulse and gap size measurement",
    )
    impulse_atr_multiplier: float = Field(
        default=1.0,
        description="Candle 2 body must exceed N×ATR to qualify as impulse candle",
    )
    min_gap_atr_multiplier: float = Field(
        default=0.5,
        description="Gap between candle 1 high and candle 3 low must be at least N×ATR",
    )

    # ── Zone lifecycle ────────────────────────────────────────────────────────

    fvg_expiry_bars: int = Field(
        default=100,
        description="Default number of bars after which an UNFILLED FVG expires",
    )

    # Per-timeframe expiry overrides
    tf_5m_expiry_bars: int = Field(
        default=288,
        description="Per-timeframe expiry override for 5m — number of bars before expiry",
    )
    tf_15m_expiry_bars: int = Field(
        default=96,
        description="Per-timeframe expiry override for 15m — number of bars before expiry",
    )
    tf_1h_expiry_bars: int = Field(
        default=24,
        description="Per-timeframe expiry override for 1h — number of bars before expiry",
    )
    tf_1d_expiry_bars: int = Field(
        default=20,
        description="Per-timeframe expiry override for 1d — number of bars before expiry",
    )

    # ── Signal quality filters ────────────────────────────────────────────────

    require_trend_alignment: bool = Field(
        default=True,
        description="Only emit FVGs aligned with symbol SMA50 trend",
    )
    min_daily_value_crore: float = Field(
        default=50.0,
        description="Skip illiquid symbols",
    )
    min_rr_ratio: float = Field(
        default=1.0,
        description="Minimum intraday R:R ratio",
    )
    min_combined_score: float = Field(
        default=0.8,
        description="Minimum gap_size * volume_score threshold",
    )

    # ── Multi-TF confluence ───────────────────────────────────────────────────

    confluence_price_pct: float = Field(
        default=0.01,
        description="1% proximity threshold for multi-timeframe confluence detection",
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
