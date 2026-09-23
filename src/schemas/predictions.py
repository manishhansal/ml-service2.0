"""
Prediction request/response schemas for ml-service2.0.

Migrated and upgraded from alpha-forge/ml-service/src/schemas.py.
All schemas extend BaseSchema (strict=True, frozen=True) instead of BaseModel.
All enums are imported from src.schemas.base — no local enum definitions.
Every response schema carries a `provenance: PredictionProvenance` field.
"""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from src.schemas.base import (
    BaseSchema,
    ExecutionAction,
    IVRegime,
    MarketRegime,
    PredictionProvenance,
    TradingStrategy,
)


# ─── Market Regime ────────────────────────────────────────────────────────────


class RegimePredictionRequest(BaseSchema):
    """Input features for market regime classification."""

    nifty_change_pct: Optional[float] = Field(
        default=None, description="NIFTY 50 intraday % change"
    )
    banknifty_change_pct: Optional[float] = Field(
        default=None, description="BANKNIFTY intraday % change"
    )
    india_vix: Optional[float] = Field(default=None, description="India VIX level")
    nifty_atr_pct: Optional[float] = Field(
        default=None, description="NIFTY ATR(14) as % of price"
    )
    nifty_adx: Optional[float] = Field(default=None, description="NIFTY ADX(14)")
    advance_decline_ratio: Optional[float] = Field(
        default=None, description="NSE advance/decline ratio"
    )
    market_breadth: Optional[float] = Field(
        default=None, description="% of F&O stocks above 20 SMA"
    )
    sector_strength: Optional[float] = Field(
        default=None, description="Avg sector % change"
    )
    volume_ratio: Optional[float] = Field(
        default=None, description="Market volume vs 20-day avg"
    )
    gap_pct: Optional[float] = Field(
        default=None, description="Opening gap % from previous close"
    )
    # Additional engineered features
    vix_change_pct: Optional[float] = Field(default=None)
    nifty_rsi: Optional[float] = Field(default=None)
    nifty_macd_hist: Optional[float] = Field(default=None)
    fii_net_cr: Optional[float] = Field(default=None)
    put_call_ratio: Optional[float] = Field(default=None)


class RegimePredictionResponse(BaseSchema):
    """Market regime prediction with probabilities and SHAP attributions."""

    regime: MarketRegime
    confidence: float = Field(ge=0, le=1)
    probabilities: dict[str, float] = Field(
        description="Probability for each regime class"
    )
    features_used: int
    model_version: str
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC
    shap_top10: list[dict[str, float]] = Field(
        default_factory=list,
        description="Top-10 SHAP feature contributions",
    )


# ─── Stock Ranking ────────────────────────────────────────────────────────────


class StockFeatures(BaseSchema):
    """Per-stock feature vector for the ranking model."""

    symbol: str
    relative_volume: float = Field(description="Volume vs 20-day avg")
    atr_expansion: float = Field(description="ATR expansion rate (today vs 20-day)")
    momentum_5d: float = Field(description="5-day return %")
    momentum_10d: float = Field(description="10-day return %")
    vwap_distance_pct: float = Field(description="% distance from VWAP")
    ema_stack_score: float = Field(description="EMA alignment score [-1,1]")
    rsi_14: float = Field(description="RSI(14)")
    macd_histogram: float = Field(description="MACD histogram value")
    adx_14: float = Field(description="ADX(14)")
    delivery_pct: Optional[float] = Field(default=None, description="Delivery %")
    sector_momentum: float = Field(description="Sector avg momentum")
    relative_strength_vs_nifty: float = Field(description="RS ratio vs NIFTY")
    options_oi_score: Optional[float] = Field(
        default=None, description="OI build-up score"
    )
    pcr: Optional[float] = Field(default=None, description="Put-Call Ratio")
    iv_rank: Optional[float] = Field(default=None, description="IV percentile rank")
    market_breadth: float = Field(description="Market breadth score")
    volume_profile_score: Optional[float] = Field(default=None)
    gap_pct: float = Field(description="Gap % from prev close")
    # Additional features
    bollinger_position: Optional[float] = Field(default=None)
    atr_pct: Optional[float] = Field(default=None)
    obv_trend: Optional[float] = Field(default=None)
    stoch_rsi: Optional[float] = Field(default=None)
    williams_r: Optional[float] = Field(default=None)
    cci: Optional[float] = Field(default=None)
    mfi: Optional[float] = Field(default=None)
    cmf: Optional[float] = Field(default=None)


class RankingRequest(BaseSchema):
    """Batch ranking request for the full F&O universe."""

    stocks: list[StockFeatures]
    regime: MarketRegime = Field(
        description="Current market regime (conditions the ranking)"
    )
    top_n: int = Field(default=20, ge=1, le=200, description="Number of top stocks to return")


class StockRank(BaseSchema):
    """A single stock's ranking result."""

    symbol: str
    score: float = Field(ge=0, le=100, description="Outperformance score 0-100")
    rank: int
    factors: dict[str, float] = Field(description="SHAP-based factor contributions")


class RankingResponse(BaseSchema):
    """Ranked stock universe with explanations."""

    rankings: list[StockRank]
    model_version: str
    regime_used: MarketRegime
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC


# ─── Strategy Selection ───────────────────────────────────────────────────────


class StrategyRequest(BaseSchema):
    """Context for strategy selection."""

    regime: MarketRegime
    symbol: str
    rsi: float
    adx: float
    atr_pct: float
    volume_ratio: float
    vwap_distance_pct: float
    bollinger_position: float
    trend_strength: float = Field(description="[-1, 1] from SMA stack")
    volatility_rank: float = Field(
        description="Percentile of current vol vs history"
    )
    time_of_day_minutes: int = Field(
        description="Minutes since market open (0-375)"
    )
    iv_regime: IVRegime = Field(
        description="IV regime classification — mandatory input for StrategySelector"
    )


class StrategyResponse(BaseSchema):
    """Selected strategy with confidence breakdown."""

    strategy: TradingStrategy
    confidence: float = Field(ge=0, le=1)
    alternatives: list[dict[str, float]] = Field(
        description="Other strategies with their probabilities"
    )
    rationale: str
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC


# ─── Risk Prediction ──────────────────────────────────────────────────────────


class RiskRequest(BaseSchema):
    """Input for per-trade risk estimation."""

    symbol: str
    direction: str = Field(description="LONG or SHORT")
    entry: float
    stop_loss: float
    target: float
    atr: float
    regime: MarketRegime
    rsi: float
    adx: float
    volume_ratio: float
    vix: float
    time_to_expiry_minutes: Optional[int] = Field(default=None)
    pcr: Optional[float] = Field(default=None)
    oi_buildup_score: Optional[float] = Field(default=None)
    news_impact_score: float = Field(
        default=0.0, description="SentinelPulse news impact score"
    )
    sentiment_risk: float = Field(
        default=0.0, description="SentinelPulse sentiment risk dimension"
    )


class RiskResponse(BaseSchema):
    """Risk prediction output."""

    prob_stop_hit: float = Field(ge=0, le=1, description="P(stop loss hit)")
    prob_target_hit: float = Field(ge=0, le=1, description="P(target hit)")
    expected_drawdown_pct: float = Field(description="Expected max drawdown %")
    suggested_position_size_pct: float = Field(
        description="Optimal position size %"
    )
    risk_score: float = Field(ge=0, le=10, description="Overall risk score 0-10")
    factors: dict[str, float] = Field(description="Risk factor contributions")
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC
    reason_codes: list[str] = Field(
        default_factory=list,
        description="Diagnostic codes (e.g. HIGH_RISK_BLOCKED, CALIBRATION_VIOLATION)",
    )


# ─── Portfolio Optimization ───────────────────────────────────────────────────


class PortfolioAsset(BaseSchema):
    """Single asset for portfolio optimization."""

    symbol: str
    expected_return: float
    risk_score: float
    sector: str
    rank_score: float


class PortfolioRequest(BaseSchema):
    """Portfolio optimization request (legacy v1 endpoint)."""

    assets: list[PortfolioAsset]
    max_positions: int = Field(default=10)
    max_sector_weight: float = Field(
        default=0.4,
        ge=0,
        le=1,
        description="Max weight per sector [0, 1]",
    )
    risk_budget_pct: float = Field(
        default=2.0, description="Total portfolio risk %"
    )


class PortfolioAllocation(BaseSchema):
    """Single asset allocation in the optimized portfolio."""

    symbol: str
    weight: float = Field(ge=0, le=1)
    sector: str
    rationale: str


class PortfolioResponse(BaseSchema):
    """Optimized portfolio with allocations (legacy v1 endpoint)."""

    allocations: list[PortfolioAllocation]
    expected_return: float
    portfolio_risk: float
    sharpe_ratio: float
    diversification_ratio: float
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC


class PortfolioV2Request(BaseSchema):
    """
    Portfolio optimization request for the v2 Riskfolio-Lib endpoint.

    Supports HRP, CVaR-minimized MVO, ERC, and Maximum Diversification methods.
    """

    symbols: list[str] = Field(description="Asset symbols to include in the portfolio")
    method: str = Field(
        default="hrp",
        description="Optimization method: hrp | cvar | erc | max_div",
    )
    returns: Optional[dict[str, list[float]]] = Field(
        default=None,
        description="Historical return series per symbol (key=symbol, value=return list). "
        "If omitted, data-service2.0 is queried.",
    )
    alpha: float = Field(
        default=0.05,
        description="Tail probability for CVaR computation (default 5%)",
    )


class PortfolioV2Response(BaseSchema):
    """Optimized portfolio response for the v2 Riskfolio-Lib endpoint."""

    method: str
    weights: dict[str, float] = Field(
        description="Symbol → portfolio weight mapping (sum to 1.0)"
    )
    risk_metrics: dict[str, float] = Field(
        description="Portfolio risk statistics (volatility, sharpe, cvar, max_drawdown, "
        "diversification_ratio, expected_return)",
    )
    available: bool = Field(
        description="False when there is insufficient return history (< 20 observations per asset)"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Reason code when available=False (e.g. INSUFFICIENT_RETURN_HISTORY)",
    )
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC


# ─── RL Execution ─────────────────────────────────────────────────────────────


class ExecutionState(BaseSchema):
    """Current trade state for the RL execution agent."""

    symbol: str
    direction: str
    entry: float
    current_price: float
    stop_loss: float
    target: float
    unrealized_pnl_pct: float
    time_in_trade_minutes: int
    regime: MarketRegime
    volume_ratio: float
    price_vs_vwap: float
    atr: float
    momentum: float
    iv_regime: Optional[IVRegime] = Field(
        default=None,
        description="IV regime classification (CRUSH | STABLE | SPIKE)",
    )
    news_impact_score: float = Field(
        default=0.0, description="SentinelPulse news impact score"
    )
    current_risk_score: float = Field(
        default=0.0, description="Current risk score from RiskPredictor [0, 10]"
    )


class ExecutionDecision(BaseSchema):
    """RL agent's execution recommendation."""

    action: ExecutionAction
    confidence: float = Field(ge=0, le=1)
    new_stop_loss: Optional[float] = Field(default=None)
    exit_pct: Optional[float] = Field(
        default=None, description="% of position to exit"
    )
    rationale: str
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC


# ─── Deep Learning Models ─────────────────────────────────────────────────────


class PriceRegimeRequest(BaseSchema):
    """
    Input for the Temporal Fusion Transformer price-regime forecaster.

    ``last_60_bars`` must be a 60 × 9 matrix (60 candles, 9 OHLCV-derived features):
    [open, high, low, close, volume, atr, rsi, macd_hist, vwap_distance].
    """

    last_60_bars: list[list[float]] = Field(
        description="60 × 9 matrix of OHLCV-derived features (shape [60, 9])"
    )


class PriceRegimeResponse(BaseSchema):
    """Price-regime forecast output from the TFT model."""

    regime: str = Field(description="Predicted price regime label")
    probability: float = Field(ge=0, le=1, description="Regime probability")
    q10: float = Field(description="10th-percentile price forecast")
    q90: float = Field(description="90th-percentile price forecast")
    available: bool = Field(description="False if no trained artifact is loaded")
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC


class IVRegimeRequest(BaseSchema):
    """
    Input for the PatchTST implied-volatility regime classifier.

    ``data`` must be a 20 × 5 matrix (20 time steps, 5 IV surface features):
    [iv_atm, skew, term_spread, pcr, vix_level].
    """

    data: list[list[float]] = Field(
        description="20 × 5 matrix of IV surface features (shape [20, 5])"
    )


class IVRegimeResponse(BaseSchema):
    """IV regime classification output from the PatchTST model."""

    iv_regime: IVRegime
    confidence: float = Field(ge=0, le=1)
    available: bool = Field(description="False if no trained artifact is loaded")
    provenance: PredictionProvenance = PredictionProvenance.HEURISTIC


# ─── SHAP Explainability ──────────────────────────────────────────────────────


class ExplainRequest(BaseSchema):
    """Request explanation for a specific prediction."""

    model: str = Field(
        description="Model name: regime | ranker | strategy | risk | rl_agent"
    )
    features: dict[str, float] = Field(
        description="Input features used for prediction"
    )
    prediction: str = Field(description="The prediction that was made")


class FeatureContribution(BaseSchema):
    """A single feature's SHAP contribution to the prediction."""

    feature: str
    value: float = Field(description="Raw feature value")
    contribution: float = Field(description="SHAP contribution score")
    direction: str = Field(description="positive or negative")


class ExplainResponse(BaseSchema):
    """SHAP-based explanation for a prediction."""

    model: str
    prediction: str
    base_value: float
    contributions: list[FeatureContribution]
    total_positive: float
    total_negative: float
    top_drivers: list[str] = Field(description="Top 5 most influential features")
