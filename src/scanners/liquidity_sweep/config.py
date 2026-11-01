"""
SweepConfig — all tunable parameters for the Liquidity Sweep Scanner pipeline.

Values are read from environment variables with the ``SWEEP_`` prefix, allowing
per-environment overrides without code changes.  Every field carries a
``description`` for self-documentation and IDE introspection.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SweepConfig(BaseSettings):
    """
    All scanner parameters.  Reads from env vars (SWEEP_* prefix) or .env file.
    Passed as a single config object to every sub-component so tests can
    override any parameter without monkey-patching globals.
    """

    model_config = SettingsConfigDict(env_prefix="SWEEP_", extra="ignore")

    # ── Detection ─────────────────────────────────────────────────────────────

    atr_window: int = Field(
        default=14,
        description="ATR rolling window for sweep magnitude normalisation",
    )
    swing_lookback: int = Field(
        default=10,
        description="Trailing window (bars) for identifying swing highs/lows as liquidity pools",
    )

    # ── Volume confirmation ───────────────────────────────────────────────────

    volume_confirmation_multiplier: float = Field(
        default=1.5,
        description="Sweep bar volume must exceed N x avg_volume to confirm institutional activity",
    )
    avg_volume_window: int = Field(
        default=20,
        description="Rolling window for average volume baseline",
    )

    # ── Sweep depth filter ────────────────────────────────────────────────────

    min_sweep_magnitude_atr_multiplier: float = Field(
        default=0.25,
        description=(
            "Minimum wick extension beyond the swept level as a multiple of ATR14. "
            "Sweeps whose wick is shallower than this threshold are treated as noise "
            "and excluded (e.g. 0.25 means wick must exceed 0.25 x ATR)."
        ),
    )

    # ── Reversal confirmation ─────────────────────────────────────────────────

    require_reversal_confirmation: bool = Field(
        default=True,
        description="Require k+1 candle to close further in reversal direction",
    )

    # ── Liquidity pool clustering ─────────────────────────────────────────────

    equal_level_tolerance_pct: float = Field(
        default=0.001,
        description="0.1% proximity for liquidity pool clustering",
    )

    # ── Multi-TF confluence ───────────────────────────────────────────────────

    confluence_atr_multiplier: float = Field(
        default=0.5,
        description="Max ATR-distance between swept levels for multi-TF confluence",
    )

    # ── Signal quality filters ────────────────────────────────────────────────

    require_trend_alignment: bool = Field(
        default=True,
        description="Only emit sweeps aligned with market trend",
    )
    min_daily_value_crore: float = Field(
        default=50.0,
        description="Skip illiquid symbols",
    )
    min_rr_ratio: float = Field(
        default=1.5,
        description="Minimum intraday R:R ratio",
    )
    min_equal_levels_count: int = Field(
        default=2,
        description="Require at least N equal highs/lows at swept level for a genuine pool",
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
