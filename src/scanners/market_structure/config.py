"""
MSConfig — all tunable parameters for the Market Structure Scanner pipeline.

Values are read from environment variables with the ``MS_`` prefix, allowing
per-environment overrides without code changes.  Every field carries a
``description`` for self-documentation and IDE introspection.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MSConfig(BaseSettings):
    """
    All scanner parameters.  Reads from env vars (MS_* prefix) or .env file.
    Passed as a single config object to every sub-component so tests can
    override any parameter without monkey-patching globals.
    """

    model_config = SettingsConfigDict(env_prefix="MS_", extra="ignore")

    # ── Detection ─────────────────────────────────────────────────────────────

    atr_window: int = Field(
        default=14,
        description="ATR rolling window for impulse detection",
    )

    # ── Swing identification ───────────────────────────────────────────────────

    swing_lookback: int = Field(
        default=10,
        description="Trailing window for swing high/low identification",
    )

    # Per-timeframe swing lookback overrides
    tf_5m_swing_lookback: int = Field(
        default=5,
        description="Per-timeframe override for 5m — swing high/low lookback window",
    )
    tf_15m_swing_lookback: int = Field(
        default=7,
        description="Per-timeframe override for 15m — swing high/low lookback window",
    )
    tf_1h_swing_lookback: int = Field(
        default=10,
        description="Per-timeframe override for 1h — swing high/low lookback window",
    )
    tf_1d_swing_lookback: int = Field(
        default=5,
        description="Per-timeframe override for 1d — swing high/low lookback window",
    )

    # ── Multi-TF confluence ───────────────────────────────────────────────────

    confluence_bars_window: int = Field(
        default=3,
        description="Max bar distance for multi-TF confluence",
    )

    # ── Signal quality filters ────────────────────────────────────────────────

    require_trend_alignment: bool = Field(
        default=True,
        description="Only emit BoS/ChoCh aligned with market direction",
    )
    min_daily_value_crore: float = Field(
        default=50.0,
        description="Skip illiquid symbols",
    )
    min_rr_ratio: float = Field(
        default=1.5,
        description="Minimum intraday R:R ratio",
    )
    min_break_magnitude: float = Field(
        default=0.1,
        description="Minimum break_magnitude to avoid hairline breaks",
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
