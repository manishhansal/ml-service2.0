"""
Feature schemas for ml-service2.0.

FeatureVector      — PIT-correct feature payload for a single symbol at a single timestamp.
FeatureQualityReport — Quality summary emitted alongside every feature vector batch.

All fields are Optional[float] (or equivalent) to allow partial feature availability
without hard failures; provenance tracking communicates data quality downstream.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import Field

from src.schemas.base import BaseSchema, ImpactDirection, PredictionProvenance


class FeatureVector(BaseSchema):
    """PIT-correct feature payload for a single symbol at a single timestamp.

    Every constituent feature's source timestamp must be strictly less than
    ``timestamp`` (the PIT boundary).  The ``provenance`` field communicates
    overall data quality to downstream consumers so they can gate live
    capital deployment appropriately.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    symbol: str
    timestamp: datetime  # UTC — the PIT boundary: all source data must be < this
    pit_validated: bool = False  # True only when LeakageValidator has signed off

    # ── PIT provenance chain (P0-005) ─────────────────────────────────────────
    data_as_of: datetime | None = None  # latest market-data source timestamp used
    news_as_of: datetime | None = None  # latest news source timestamp used

    # ── Data quality gates from data-service2.0 ───────────────────────────────
    data_confidence_score: int = Field(default=0, ge=0, le=100)
    signal_engine_allowed: bool = False

    # ── OHLCV (daily) ─────────────────────────────────────────────────────────
    open: float | None = None
    high: float | None = None
    low: float | None = None
    close: float | None = None
    volume: float | None = None
    vwap: float | None = None

    # ── Intraday derived ──────────────────────────────────────────────────────
    atr: float | None = None
    adx: float | None = None
    rsi_14: float | None = None
    macd_histogram: float | None = None
    momentum_5d: float | None = None
    momentum_10d: float | None = None
    relative_volume: float | None = None
    gap_pct: float | None = None
    ema_stack_score: float | None = None
    bollinger_position: float | None = None

    # ── India VIX ─────────────────────────────────────────────────────────────
    india_vix: float | None = None
    vix_change_pct: float | None = None

    # ── Market breadth ────────────────────────────────────────────────────────
    advance_decline_ratio: float | None = None
    market_breadth_pct: float | None = None  # % F&O stocks above 20 SMA
    sector_momentum: float | None = None
    relative_strength_vs_nifty: float | None = None

    # ── F&O / options ─────────────────────────────────────────────────────────
    put_call_ratio: float | None = None
    iv_rank: float | None = None
    oi_buildup_score: float | None = None
    delivery_pct: float | None = None
    options_oi_score: float | None = None

    # ── FII/DII flows (crores) ────────────────────────────────────────────────
    fii_net_cr: float | None = None
    dii_net_cr: float | None = None

    # ── SentinelPulse news features ───────────────────────────────────────────
    news_impact_score: float = 0.0  # [0,1], 0.0 when unavailable
    impact_direction: ImpactDirection = ImpactDirection.NEUTRAL
    impact_confidence: float = 0.0  # [0,1]
    sentiment_overall: float = 0.0
    sentiment_market: float = 0.0
    sentiment_company: float = 0.0
    sentiment_macro: float = 0.0
    sentiment_risk: float = 0.0
    market_regime_nlp: str = "NEUTRAL"  # from SentinelPulse regime endpoint
    sentinel_available: bool = True  # False when SentinelPulse was unreachable

    # ── Qlib Alpha factors ────────────────────────────────────────────────────
    alpha158_factors: dict[str, float] = Field(default_factory=dict)
    alpha360_factors: dict[str, float] = Field(default_factory=dict)

    # ── Provenance tracking ───────────────────────────────────────────────────
    provenance: PredictionProvenance = PredictionProvenance.UNAVAILABLE


class FeatureQualityReport(BaseSchema):
    """Quality summary for a feature vector batch.

    Produced by FeaturePipeline alongside every call to ``build_vector`` or
    ``build_batch``.  Exposed via ``GET /features/quality`` (latest batch).
    """

    batch_id: str
    timestamp: datetime
    total_features_requested: int = 0
    missing_count: int = 0
    imputed_count: int = 0
    rejected_count: int = 0
    unavailable_families: list[str] = Field(default_factory=list)
    pit_violations_count: int = 0
    discarded_backtest_records: int = 0  # records discarded due to pit_date in backtest mode
    processing_time_ms: float = 0.0
    sentinel_pulse_available: bool = True
    data_service_available: bool = True
