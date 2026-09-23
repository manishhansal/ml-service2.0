"""
StrategySelector — rule-based and CatBoost-backed strategy selection.

Selects the optimal trading strategy given market features, regime, and IV regime.
Falls back to heuristic rules when no trained model artifact is available.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.schemas.base import IVRegime, MarketRegime, PredictionProvenance, TradingStrategy
from src.schemas.predictions import StrategyResponse

STRATEGY_FEATURE_NAMES = [
    "rsi",
    "adx",
    "atr_pct",
    "volume_ratio",
    "vwap_distance_pct",
    "bollinger_position",
    "trend_strength",
    "volatility_rank",
    "time_of_day_minutes",
    # Encoded categoricals:
    "regime_encoded",    # MarketRegime → int
    "iv_regime_encoded", # IVRegime → int
]

REGIME_ENCODING = {
    MarketRegime.STRONG_BULL: 5,
    MarketRegime.BULL: 4,
    MarketRegime.SIDEWAYS: 3,
    MarketRegime.VOLATILE: 2,
    MarketRegime.BEAR: 1,
    MarketRegime.CRASH: 0,
}

IV_REGIME_ENCODING = {
    IVRegime.CRUSH: 0,
    IVRegime.STABLE: 1,
    IVRegime.SPIKE: 2,
}


class StrategySelector:
    """
    Selects the optimal trading strategy for a given market context.

    When a trained CatBoost artifact is present the model path is loaded and
    used for inference.  Otherwise the selector falls back to deterministic
    heuristic rules so the service is always functional.

    Requirements covered: 6.1, 6.2, 6.3, 6.5, 6.6.
    """

    def __init__(self, model_path: Path | None = None) -> None:
        self.model = None
        self.model_version = "1.0.0-heuristic"
        self.feature_names = STRATEGY_FEATURE_NAMES

        if model_path and model_path.exists():
            try:
                from catboost import CatBoostClassifier  # type: ignore[import]

                self.model = CatBoostClassifier()
                self.model.load_model(str(model_path))
                self.model_version = "1.0.0-trained"
            except Exception:
                self.model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def has_trained_model(self) -> bool:
        """True when a CatBoost artifact was successfully loaded."""
        return self.model is not None

    def select(
        self,
        features: dict[str, Any],
        regime: MarketRegime,
    ) -> StrategyResponse:
        """
        Select the optimal trading strategy with calibrated probabilities.

        ``iv_regime`` must be present in *features* (Req 6.3).  If it is
        missing or invalid it defaults to ``IVRegime.STABLE``.

        When the top-strategy confidence is < 0.40 at least 2 alternatives
        are returned (Req 6.6).
        """
        iv_regime = self._resolve_iv_regime(features)

        if self.model is not None:
            try:
                strategy, confidence, probas, provenance = self._model_select(
                    features, regime, iv_regime
                )
            except Exception:
                strategy, confidence, probas, provenance = self._heuristic_select(
                    features, regime, iv_regime
                )
        else:
            strategy, confidence, probas, provenance = self._heuristic_select(
                features, regime, iv_regime
            )

        # Build alternatives list sorted by probability (excluding winner).
        all_sorted = sorted(probas.items(), key=lambda x: x[1], reverse=True)
        alternatives = [
            {strat: prob}
            for strat, prob in all_sorted
            if strat != strategy.value
        ][:5]  # cap at top-5

        # Req 6.6: when confidence < 0.40 ensure at least 2 alternatives.
        if confidence < 0.40 and len(alternatives) < 2:
            alternatives = [
                {strat: prob}
                for strat, prob in all_sorted
                if strat != strategy.value
            ]

        return StrategyResponse(
            strategy=strategy,
            confidence=confidence,
            alternatives=alternatives,
            rationale=self._build_rationale(strategy, regime, iv_regime, features),
            provenance=provenance,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_iv_regime(features: dict[str, Any]) -> IVRegime:
        """Coerce the raw iv_regime value from *features* to an IVRegime."""
        raw = features.get("iv_regime")
        if isinstance(raw, IVRegime):
            return raw
        if raw is not None:
            try:
                return IVRegime(str(raw))
            except ValueError:
                pass
        return IVRegime.STABLE

    def _model_select(
        self,
        features: dict[str, Any],
        regime: MarketRegime,
        iv_regime: IVRegime,
    ) -> tuple[TradingStrategy, float, dict[str, float], PredictionProvenance]:
        """CatBoost-based strategy selection (Req 6.1)."""
        import numpy as np  # type: ignore[import]

        feature_vector = np.array(
            [
                [
                    features.get("rsi", 50.0),
                    features.get("adx", 20.0),
                    features.get("atr_pct", 1.0),
                    features.get("volume_ratio", 1.0),
                    features.get("vwap_distance_pct", 0.0),
                    features.get("bollinger_position", 0.5),
                    features.get("trend_strength", 0.0),
                    features.get("volatility_rank", 0.5),
                    features.get("time_of_day_minutes", 120),
                    REGIME_ENCODING.get(regime, 3),
                    IV_REGIME_ENCODING.get(iv_regime, 1),
                ]
            ]
        )

        probas_raw = self.model.predict_proba(feature_vector)[0]
        idx = int(probas_raw.argmax())
        all_strats = list(TradingStrategy)
        strategy = all_strats[idx % len(all_strats)]
        confidence = float(probas_raw[idx])
        probas = {s.value: float(p) for s, p in zip(all_strats, probas_raw)}
        return strategy, confidence, probas, PredictionProvenance.TRAINED_MODEL

    def _heuristic_select(
        self,
        features: dict[str, Any],
        regime: MarketRegime,
        iv_regime: IVRegime,
    ) -> tuple[TradingStrategy, float, dict[str, float], PredictionProvenance]:
        """
        Deterministic rule-based strategy selection (Req 6.2).

        Rules are applied in priority order:
        1. IV regime (SPIKE / CRUSH) overrides regime-level logic.
        2. Regime-based rules apply when IV is STABLE.
        """
        rsi = float(features.get("rsi", 50))
        adx = float(features.get("adx", 20))
        trend_strength = float(features.get("trend_strength", 0))
        time_min = int(features.get("time_of_day_minutes", 120))

        strategy: TradingStrategy
        confidence: float

        if iv_regime == IVRegime.SPIKE:
            # High IV favours volatility-exploiting strategies (Req 6.3).
            if adx > 25:
                strategy = TradingStrategy.VOLATILITY_BREAKOUT
                confidence = 0.70
            else:
                strategy = TradingStrategy.SCALPING
                confidence = 0.65

        elif iv_regime == IVRegime.CRUSH:
            # Low IV environment: mean-reversion / range strategies win (Req 6.3).
            strategy = TradingStrategy.RANGE_TRADING
            confidence = 0.68

        else:  # IVRegime.STABLE — defer to market regime (Req 6.2, 6.5)
            if regime in (MarketRegime.STRONG_BULL, MarketRegime.BULL):
                if adx > 25 and trend_strength > 0.3:
                    strategy = TradingStrategy.TREND_FOLLOWING
                    confidence = 0.75
                elif time_min < 60:
                    strategy = TradingStrategy.BREAKOUT
                    confidence = 0.70
                else:
                    strategy = TradingStrategy.MOMENTUM
                    confidence = 0.68

            elif regime in (MarketRegime.BEAR, MarketRegime.CRASH):
                strategy = TradingStrategy.MEAN_REVERSION
                confidence = 0.65

            elif regime == MarketRegime.VOLATILE:
                strategy = TradingStrategy.SCALPING
                confidence = 0.62

            else:  # SIDEWAYS
                if rsi < 40:
                    strategy = TradingStrategy.VWAP_BOUNCE
                    confidence = 0.65
                else:
                    strategy = TradingStrategy.RANGE_TRADING
                    confidence = 0.63

        # Spread remaining probability uniformly across all other strategies.
        all_strats = list(TradingStrategy)
        remainder = (1.0 - confidence) / max(len(all_strats) - 1, 1)
        probas: dict[str, float] = {s.value: remainder for s in all_strats}
        probas[strategy.value] = confidence

        return strategy, confidence, probas, PredictionProvenance.HEURISTIC

    @staticmethod
    def _build_rationale(
        strategy: TradingStrategy,
        regime: MarketRegime,
        iv_regime: IVRegime,
        features: dict[str, Any],
    ) -> str:
        """Human-readable explanation of the selection decision (Req 6.5)."""
        return (
            f"Selected {strategy.value} for {regime.value} regime "
            f"with IV regime {iv_regime.value}. "
            f"ADX={features.get('adx', 'N/A')}, RSI={features.get('rsi', 'N/A')}."
        )
