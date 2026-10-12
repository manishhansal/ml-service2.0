"""
OrderBlockConfig — all tunable parameters for the Order Block Scanner pipeline.

Values are read from environment variables with the ``OB_`` prefix, allowing
per-environment overrides without code changes.  Every field carries a
``description`` for self-documentation and IDE introspection.
"""
from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class OrderBlockConfig(BaseSettings):
    """
    All scanner parameters.  Reads from env vars (OB_* prefix) or .env file.
    Passed as a single config object to every sub-component so tests can
    override any parameter without monkey-patching globals.
    """

    model_config = SettingsConfigDict(env_prefix="OB_", extra="ignore")

    # ── Detection ─────────────────────────────────────────────────────────────

    atr_window: int = Field(
        default=14,
        description="ATR rolling window for impulse detection",
    )
    impulse_multiplier: float = Field(
        default=1.5,
        description="Body must exceed N×ATR to qualify as impulse",
    )
    lookback_bars: int = Field(
        default=5,
        description="Max bars to look back from impulse for opposing candle",
    )

    # ── Structure validation ──────────────────────────────────────────────────

    swing_lookback: int = Field(
        default=10,
        description="Trailing window for swing high/low",
    )
    max_confirmation_bars: int = Field(
        default=20,
        description="BoS must be confirmed within N bars",
    )

    # Per-timeframe overrides (applied by ObScanner before passing config to StructureValidator)
    tf_5m_confirmation_bars: int = Field(
        default=30,
        description="Per-timeframe override for 5m — BoS confirmation window",
    )
    tf_15m_confirmation_bars: int = Field(
        default=24,
        description="Per-timeframe override for 15m — BoS confirmation window",
    )
    tf_1h_confirmation_bars: int = Field(
        default=15,
        description="Per-timeframe override for 1h — BoS confirmation window",
    )
    tf_1d_confirmation_bars: int = Field(
        default=10,
        description="Per-timeframe override for 1d — BoS confirmation window",
    )

    # ── Strength filter ───────────────────────────────────────────────────────

    strength_multiplier: float = Field(
        default=1.5,
        description="move_ratio >= this to pass filter",
    )
    atr_strength_window: int = Field(
        default=50,
        description="ATR window for strength baseline",
    )

    # ── Volume ranking ────────────────────────────────────────────────────────

    avg_volume_window: int = Field(
        default=20,
        description="Rolling window for average volume",
    )

    # ── Zone lifecycle ────────────────────────────────────────────────────────

    zone_expiry_bars: int = Field(
        default=200,
        description="ACTIVE zones expire after N bars without test",
    )

    # ── Signal quality filters ────────────────────────────────────────────────

    require_trend_alignment: bool = Field(
        default=True,
        description="Only emit signals aligned with symbol SMA50 trend and market direction",
    )
    min_daily_value_crore: float = Field(
        default=50.0,
        description="Skip symbols with avg daily turnover below this crore INR threshold",
    )
    min_rr_ratio: float = Field(
        default=1.0,
        description="Minimum intraday R:R ratio to surface a signal",
    )
    min_volume_score: float = Field(
        default=1.2,
        description="Minimum relative volume score at OB candle",
    )

    # ── Multi-TF confluence ───────────────────────────────────────────────────

    confluence_price_pct: float = Field(
        default=0.02,
        description="2% proximity threshold for confluence",
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
