"""
StockRanker — LightGBM-based outperformance scorer for the F&O universe.

Provides:
- LightGBM inference when a trained model artifact is available
- Score-based heuristic fallback with regime-conditioned weights
- Outperformance score in [0, 100] and a rank for each symbol
- SHAP-like top-5 factor decomposition per ranked symbol
- Idempotence: same inputs + same model version → same outputs
- Graceful degradation when SentinelPulse news context is unavailable

Requirements: Req 5.1, Req 5.2, Req 5.3, Req 5.6, Req 5.7
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.schemas.base import MarketRegime, PredictionProvenance
from src.schemas.predictions import RankingResponse, StockRank

# Ordered feature names expected by the LightGBM model artifact.
# The order must match the column order used during training.
STOCK_RANKER_FEATURE_NAMES: list[str] = [
    "relative_volume",
    "atr_expansion",
    "momentum_5d",
    "momentum_10d",
    "vwap_distance_pct",
    "ema_stack_score",
    "rsi_14",
    "macd_histogram",
    "adx_14",
    "delivery_pct",
    "sector_momentum",
    "relative_strength_vs_nifty",
    "options_oi_score",
    "pcr",
    "iv_rank",
    "market_breadth",
    "volume_profile_score",
    "gap_pct",
    "bollinger_position",
    "atr_pct",
    "obv_trend",
    "stoch_rsi",
    "williams_r",
    "cci",
    "mfi",
    "cmf",
]

# Regime-specific feature weight tables.
# Weights are applied in the score-based heuristic fallback when no trained
# model artifact is available. Keys must be a subset of STOCK_RANKER_FEATURE_NAMES.
_REGIME_WEIGHTS: dict[MarketRegime, dict[str, float]] = {
    MarketRegime.STRONG_BULL: {
        "relative_volume": 2.0,
        "momentum_5d": 4.0,
        "momentum_10d": 3.0,
        "rsi_14": 1.0,
        "relative_strength_vs_nifty": 4.5,
        "ema_stack_score": 2.5,
        "adx_14": 1.5,
        "sector_momentum": 2.0,
        "market_breadth": 1.5,
    },
    MarketRegime.BULL: {
        "relative_volume": 2.0,
        "momentum_5d": 3.0,
        "momentum_10d": 2.5,
        "rsi_14": 1.0,
        "relative_strength_vs_nifty": 3.5,
        "ema_stack_score": 2.0,
        "adx_14": 1.5,
        "sector_momentum": 1.5,
        "market_breadth": 1.0,
    },
    MarketRegime.SIDEWAYS: {
        "relative_volume": 1.5,
        "momentum_5d": 1.0,
        "momentum_10d": 1.0,
        "rsi_14": 2.0,
        "relative_strength_vs_nifty": 2.0,
        "ema_stack_score": 1.5,
        "adx_14": 2.0,
        "vwap_distance_pct": 2.5,
        "bollinger_position": 2.0,
        "mean_reversion_signal": 1.5,  # will gracefully zero-out if absent
    },
    MarketRegime.VOLATILE: {
        "relative_volume": 3.0,
        "momentum_5d": 1.5,
        "momentum_10d": 1.0,
        "rsi_14": 1.5,
        "relative_strength_vs_nifty": 2.0,
        "ema_stack_score": 1.0,
        "adx_14": 2.5,
        "atr_expansion": 2.5,
        "iv_rank": 2.0,
    },
    MarketRegime.BEAR: {
        "relative_volume": 2.0,
        # Invert momentum signals for short-biased ranking
        "momentum_5d": -3.0,
        "momentum_10d": -2.5,
        "rsi_14": -1.5,
        "relative_strength_vs_nifty": -3.5,
        "ema_stack_score": -2.0,
        "adx_14": 1.5,
        "sector_momentum": -2.0,
    },
    MarketRegime.CRASH: {
        "relative_volume": 3.0,
        "momentum_5d": -4.0,
        "momentum_10d": -3.5,
        "rsi_14": -2.0,
        "relative_strength_vs_nifty": -4.5,
        "ema_stack_score": -3.0,
        "adx_14": 2.0,
        "atr_expansion": 3.0,
        "sector_momentum": -3.0,
    },
}

# Fallback to BULL weights for any regime not explicitly listed above.
_DEFAULT_WEIGHTS: dict[str, float] = _REGIME_WEIGHTS[MarketRegime.BULL]

# Top-5 features used for SHAP-like factor decomposition (in priority order).
_FACTOR_FEATURES: list[str] = [
    "relative_strength_vs_nifty",
    "momentum_5d",
    "relative_volume",
    "rsi_14",
    "ema_stack_score",
]


class StockRanker:
    """LightGBM-backed outperformance scorer for the NSE F&O universe.

    When a trained LightGBM model artifact is available at ``model_path``,
    inference uses the trained model and sets ``PredictionProvenance.TRAINED_MODEL``.
    Otherwise a regime-conditioned score-based heuristic is used and
    ``PredictionProvenance.HEURISTIC`` is set (Req 5.2).

    Idempotence (Req 5.7 / Property 11): all scoring paths are deterministic —
    no random state or mutable side effects occur between calls with the same
    input vectors and loaded model version.

    Usage::

        ranker = StockRanker(model_path=Path("artifacts/stock_ranker.lgb"))
        response = ranker.rank(
            stocks=[{"relative_volume": 1.2, "momentum_5d": 0.5, ...}],
            symbols=["RELIANCE"],
            regime=MarketRegime.BULL,
            top_n=20,
        )
    """

    def __init__(self, model_path: Path | None = None) -> None:
        """Initialise the ranker, loading a trained artifact when available.

        Args:
            model_path: Optional path to a LightGBM model file
                (``*.lgb`` or ``*.txt``). If the path does not exist or the
                load fails, the heuristic fallback is used silently.
        """
        self.model: Any = None
        self.model_version: str = "1.0.0-heuristic"
        self.feature_names: list[str] = STOCK_RANKER_FEATURE_NAMES

        if model_path and model_path.exists():
            try:
                import lightgbm as lgb  # type: ignore[import-untyped]

                self.model = lgb.Booster(model_file=str(model_path))
                self.model_version = "1.0.0-trained"
            except Exception:
                self.model = None

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def has_trained_model(self) -> bool:
        """Return ``True`` when a trained LightGBM model artifact is loaded."""
        return self.model is not None

    def rank(
        self,
        stocks: list[dict[str, Any]],
        symbols: list[str],
        regime: MarketRegime,
        top_n: int = 20,
    ) -> RankingResponse:
        """Rank stocks by outperformance probability.

        Scores are normalised to [0, 100] and sorted descending.  When all
        raw scores are identical the normalised score is fixed at 50.0 for
        every symbol (avoids division-by-zero).

        SHAP-like factor decomposition is attached to each returned
        :class:`~src.schemas.predictions.StockRank` via the ``factors`` field
        (top-5 features, Req 5.5).

        SentinelPulse news context fields (``news_impact_score``, etc.) are
        treated as optional — when absent for a symbol the corresponding
        values default to ``0.0`` and the symbol is still ranked (Req 5.6).

        Args:
            stocks:  List of feature dicts, one per symbol.  Missing features
                     default to ``0.0``.
            symbols: Ordered list of symbol strings that maps 1:1 to ``stocks``.
            regime:  Current market regime — used to select heuristic weight
                     table (Req 5.8) or condition model inference.
            top_n:   Maximum number of ranked results to return (≥ 1).

        Returns:
            :class:`~src.schemas.predictions.RankingResponse` — never raises.
        """
        from src.logging_config import get_logger

        logger = get_logger(__name__)

        if not stocks:
            return RankingResponse(
                rankings=[],
                model_version=self.model_version,
                regime_used=regime,
                provenance=PredictionProvenance.HEURISTIC,
            )

        import numpy as np  # type: ignore[import-untyped]

        raw_scores: list[float]
        provenance: PredictionProvenance

        if self.model is not None:
            # ── LightGBM inference path ───────────────────────────────────────
            try:
                X = np.array(
                    [
                        [float(s.get(f, 0.0) or 0.0) for f in self.feature_names]
                        for s in stocks
                    ],
                    dtype=float,
                )
                raw_scores = self.model.predict(X).tolist()
                provenance = PredictionProvenance.TRAINED_MODEL
            except Exception as exc:
                logger.warning("stock_ranker_model_failed", error=str(exc))
                raw_scores = self._heuristic_scores(stocks, regime)
                provenance = PredictionProvenance.HEURISTIC
        else:
            # ── Score-based heuristic fallback ────────────────────────────────
            raw_scores = self._heuristic_scores(stocks, regime)
            provenance = PredictionProvenance.HEURISTIC

        # ── Normalise to [0, 100] ─────────────────────────────────────────────
        # Using explicit min/max to keep the computation deterministic (no
        # library random state involved — Req 5.7).
        raw_array = np.array(raw_scores, dtype=float)
        raw_min = float(raw_array.min())
        raw_max = float(raw_array.max())
        score_range = raw_max - raw_min
        if score_range > 0:
            normalized: list[float] = (
                (raw_array - raw_min) / score_range * 100.0
            ).tolist()
        else:
            normalized = [50.0] * len(raw_scores)

        # ── Stable sort → idempotent ranking (Req 5.7) ───────────────────────
        # Python's sort is stable, so ties are broken by the original insertion
        # order which is deterministic for the same input list.
        indexed = sorted(
            enumerate(normalized), key=lambda x: x[1], reverse=True
        )

        # ── Build StockRank list ──────────────────────────────────────────────
        rankings: list[StockRank] = []
        for rank_pos, (orig_idx, score) in enumerate(indexed[:top_n], start=1):
            stock = stocks[orig_idx]
            factors = self._compute_factors(stock)

            rankings.append(
                StockRank(
                    symbol=symbols[orig_idx],
                    score=round(score, 2),
                    rank=rank_pos,
                    factors=factors,
                )
            )

        return RankingResponse(
            rankings=rankings,
            model_version=self.model_version,
            regime_used=regime,
            provenance=provenance,
        )

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _heuristic_scores(
        self,
        stocks: list[dict[str, Any]],
        regime: MarketRegime,
    ) -> list[float]:
        """Compute raw scores using regime-conditioned weights (Req 5.8).

        Features absent from a stock dict are treated as ``0.0`` so that
        symbols with partial feature coverage (e.g. no SentinelPulse data)
        are still ranked rather than excluded (Req 5.6).
        """
        weights = _REGIME_WEIGHTS.get(regime, _DEFAULT_WEIGHTS)
        scores: list[float] = []
        for s in stocks:
            score = 0.0
            for feature, weight in weights.items():
                val = float(s.get(feature, 0.0) or 0.0)
                score += val * weight
            scores.append(score)
        return scores

    def _compute_factors(
        self,
        stock: dict[str, Any],
    ) -> dict[str, float]:
        """Return top-5 factor contributions for SHAP-like attribution (Req 5.5).

        Uses the raw feature values as proxy contributions — a lightweight
        approximation that avoids a full SHAP pass at ranking time.  The
        :class:`~src.explainability.explainer.ModelExplainer` can be invoked
        separately for exact SHAP values when needed.
        """
        factors: dict[str, float] = {}
        for feature in _FACTOR_FEATURES:
            val = float(stock.get(feature, 0.0) or 0.0)
            factors[feature] = round(val, 4)
        return factors
