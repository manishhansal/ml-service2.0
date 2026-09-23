"""
RegimeClassifier — market regime prediction with XGBoost and heuristic fallback.

Design guarantees:
- Uses a trained XGBoost champion artifact from ModelRegistry when available.
- Falls back to rule-based heuristic (PredictionProvenance.HEURISTIC) when no
  trained model is loaded.
- Returns PredictionProvenance.INSUFFICIENT_EVIDENCE with reason code
  LOW_REGIME_CONFIDENCE when confidence < 0.35.
- Attaches SHAP top-10 attributions to every response via ModelExplainer.
- Emits a `regime_transition_event` log entry whenever the predicted regime
  changes from the previous prediction.
- Thread-safe: all mutable state is guarded by a threading.Lock.

Requirements: Req 4.1, Req 4.2, Req 4.3, Req 4.5, Req 4.6, Req 4.7, Req 4.8, Req 11.1, Req 11.1
"""
from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from src.explainability.explainer import ModelExplainer
from src.logging_config import get_logger
from src.registry.registry import ModelRegistry
from src.schemas.base import MarketRegime, ModelLifecycleStage, PredictionProvenance
from src.schemas.predictions import RegimePredictionResponse

logger = get_logger(__name__)

# ── Feature names (must match training column order) ───────────────────────────
REGIME_FEATURE_NAMES: list[str] = [
    "nifty_change_pct",
    "banknifty_change_pct",
    "india_vix",
    "nifty_atr_pct",
    "nifty_adx",
    "advance_decline_ratio",
    "market_breadth",
    "sector_strength",
    "volume_ratio",
    "gap_pct",
    "vix_change_pct",
    "nifty_rsi",
    "nifty_macd_hist",
    "fii_net_cr",
    "put_call_ratio",
]

# Confidence threshold below which provenance is downgraded to
# INSUFFICIENT_EVIDENCE with reason code LOW_REGIME_CONFIDENCE.
_LOW_CONFIDENCE_THRESHOLD: float = 0.35

# Ordered list of MarketRegime members — must align with XGBoost label encoding.
_REGIME_ORDER: list[MarketRegime] = [
    MarketRegime.STRONG_BULL,
    MarketRegime.BULL,
    MarketRegime.SIDEWAYS,
    MarketRegime.VOLATILE,
    MarketRegime.BEAR,
    MarketRegime.CRASH,
]


class RegimeClassifier:
    """
    Market regime classifier with XGBoost / heuristic fallback.

    Usage::

        classifier = RegimeClassifier()
        response = classifier.predict({"india_vix": 18.5, "nifty_change_pct": 0.5})
        print(response.regime, response.provenance)
    """

    def __init__(self, model_path: Path | None = None) -> None:
        """
        Initialise the classifier.

        Args:
            model_path: Optional explicit path to an XGBoost artifact file.
                        When provided, the registry lookup is skipped and this
                        path is used directly.  Primarily for testing.
        """
        self._lock = Lock()
        self._model: Any | None = None
        self._model_version: str = "heuristic-v1"
        self._last_regime: MarketRegime | None = None
        self._last_confidence: float = 0.0
        self._explainer: ModelExplainer = ModelExplainer()
        self.feature_names: list[str] = REGIME_FEATURE_NAMES

        if model_path is not None:
            self._load_from_path(model_path)
        else:
            self._load_from_registry()

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def model_version(self) -> str:
        """Return the version string of the loaded artifact, or 'heuristic-v1'."""
        return self._model_version

    @property
    def has_trained_model(self) -> bool:
        """True when an XGBoost model artifact is loaded and ready."""
        return self._model is not None

    # ── Public interface ───────────────────────────────────────────────────────

    def predict(self, features: dict[str, float]) -> RegimePredictionResponse:
        """
        Predict the current market regime from a feature dict.

        Feature keys correspond to ``REGIME_FEATURE_NAMES``; missing keys
        default to 0.0.

        Args:
            features: Dict mapping feature name → float value.

        Returns:
            :class:`~src.schemas.predictions.RegimePredictionResponse`
        """
        with self._lock:
            return self._predict_locked(features)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _predict_locked(self, features: dict[str, float]) -> RegimePredictionResponse:
        """Core prediction logic (must be called with self._lock held)."""
        if self._model is not None:
            regime, confidence, proba_dict = self._model_predict(features)
            provenance = PredictionProvenance.TRAINED_MODEL
        else:
            regime, confidence = self._heuristic_predict(features)
            proba_dict = {r.value: 0.0 for r in MarketRegime}
            proba_dict[regime.value] = confidence
            provenance = PredictionProvenance.HEURISTIC

        # Downgrade provenance when confidence is too low
        reason_codes: list[str] = []
        if confidence < _LOW_CONFIDENCE_THRESHOLD:
            provenance = PredictionProvenance.INSUFFICIENT_EVIDENCE
            reason_codes.append("LOW_REGIME_CONFIDENCE")

        # SHAP top-10 attributions
        shap_response = self._explainer.explain(
            model_name="market_regime",
            features=features,
            prediction=regime.value,
            top_k=10,
            provenance=provenance,
        )
        # shap_top10 schema is list[dict[str, float]]: each dict maps the
        # feature name to its SHAP contribution value.
        shap_top10: list[dict[str, float]] = [
            {c.feature: c.contribution}
            for c in shap_response.contributions
        ]

        # Regime transition logging
        if self._last_regime is not None and regime != self._last_regime:
            logger.info(
                "regime_transition_event",
                previous_regime=self._last_regime.value,
                new_regime=regime.value,
                confidence_delta=confidence - self._last_confidence,
                top_3_features=[c.feature for c in shap_response.contributions[:3]],
            )

        self._last_regime = regime
        self._last_confidence = confidence

        return RegimePredictionResponse(
            regime=regime,
            confidence=confidence,
            probabilities=proba_dict,
            features_used=sum(1 for k in self.feature_names if k in features),
            model_version=self._model_version,
            provenance=provenance,
            shap_top10=shap_top10,
        )

    # ── Heuristic fallback ─────────────────────────────────────────────────────

    def _heuristic_predict(
        self, features: dict[str, float]
    ) -> tuple[MarketRegime, float]:
        """Rule-based regime classification."""
        vix = features.get("india_vix", 0)
        nifty_change = features.get("nifty_change_pct", 0)
        adx = features.get("nifty_adx", 0)

        # Crash: VIX > 35 or dramatic decline
        if vix > 35 or nifty_change < -3.0:
            return MarketRegime.CRASH, 0.8
        # Strong bull: VIX low + strong upside
        if vix < 15 and nifty_change > 1.0:
            return MarketRegime.STRONG_BULL, 0.75
        # Bear: VIX > 25 + negative
        if vix > 25 and nifty_change < -0.5:
            return MarketRegime.BEAR, 0.7
        # Volatile: high VIX with mixed signals
        if vix > 22 or adx > 30:
            return MarketRegime.VOLATILE, 0.65
        # Bull: positive momentum + moderate VIX
        if nifty_change > 0.3 and vix < 20:
            return MarketRegime.BULL, 0.7
        # Default: SIDEWAYS
        return MarketRegime.SIDEWAYS, 0.6

    # ── XGBoost model prediction ───────────────────────────────────────────────

    def _model_predict(
        self, features: dict[str, float]
    ) -> tuple[MarketRegime, float, dict[str, float]]:
        """XGBoost-based prediction returning (regime, confidence, probabilities)."""
        import numpy as np  # type: ignore[import-untyped]

        feature_vector = np.array(
            [[features.get(f, 0.0) for f in self.feature_names]]
        )
        probas = self._model.predict_proba(feature_vector)[0]
        class_idx = int(np.argmax(probas))
        confidence = float(probas[class_idx])
        regime = self._idx_to_regime(class_idx)
        proba_dict = {r.value: float(p) for r, p in zip(MarketRegime, probas)}
        return regime, confidence, proba_dict

    @staticmethod
    def _idx_to_regime(idx: int) -> MarketRegime:
        """Map a class index to a :class:`MarketRegime` enum member."""
        if 0 <= idx < len(_REGIME_ORDER):
            return _REGIME_ORDER[idx]
        return MarketRegime.SIDEWAYS

    # ── Online learning trigger ────────────────────────────────────────────────

    def trigger_online_learning_check(
        self, current_ic: float, baseline_ic: float
    ) -> bool:
        """
        Check whether online learning should be triggered due to IC degradation.

        Returns True when the 30-day rolling IC has degraded more than 20 %
        relative to the 90-day baseline IC, which satisfies Req 4.8:

            degradation > 20 %  ↔  current_ic < baseline_ic * 0.80

        When triggered, callers should invoke
        ``OnlineLearner.initiate_update("market_regime", last_60_trading_days)``
        while continuing to serve predictions from the existing champion model.

        Args:
            current_ic:  30-day rolling information coefficient.
            baseline_ic: 90-day baseline information coefficient.

        Returns:
            True if online learning should be initiated, False otherwise.
        """
        if baseline_ic > 0 and current_ic < baseline_ic * 0.8:
            logger.info(
                "regime_classifier_ic_degradation_detected",
                current_ic=current_ic,
                baseline_ic=baseline_ic,
                degradation_pct=(baseline_ic - current_ic) / baseline_ic * 100,
            )
            return True
        return False

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_from_registry(self) -> None:
        """Attempt to load the champion 'market_regime' artifact from the registry."""
        try:
            registry = ModelRegistry()
            artifact = registry.get_champion("market_regime")
            if artifact is None:
                logger.info(
                    "regime_classifier_no_champion",
                    message="No champion artifact found — using heuristic fallback.",
                )
                return

            artifact_path = registry.load_artifact(artifact)
            self._load_model_file(artifact_path, version=artifact.version)

        except Exception as exc:
            logger.warning(
                "regime_classifier_registry_load_failed",
                error=str(exc),
                message="Falling back to heuristic.",
            )

    def _load_from_path(self, model_path: Path) -> None:
        """Load an XGBoost artifact from an explicit file path."""
        try:
            self._load_model_file(model_path, version=model_path.stem)
        except Exception as exc:
            logger.warning(
                "regime_classifier_path_load_failed",
                path=str(model_path),
                error=str(exc),
                message="Falling back to heuristic.",
            )

    def _load_model_file(self, path: Path, version: str) -> None:
        """Deserialise a joblib/pickle artifact and register it with the explainer."""
        import joblib  # type: ignore[import-untyped]

        model = joblib.load(path)
        self._model = model
        self._model_version = version

        # Register the model with the SHAP explainer so subsequent calls
        # can compute tree-based attributions.
        self._explainer.register_model(
            model_name="market_regime",
            model=model,
            feature_names=self.feature_names,
        )

        logger.info(
            "regime_classifier_model_loaded",
            version=version,
            path=str(path),
        )
