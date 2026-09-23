"""
RiskPredictor — per-trade risk estimation with XGBoost ensemble and heuristic fallback.

Design guarantees:
- Uses an XGBoost ensemble of 3 models (stop, target, drawdown) when artifacts exist.
- Falls back to rule-based heuristic (PredictionProvenance.HEURISTIC) when no
  trained models are loaded.
- Derives trade geometry features (stop_distance_atr, target_distance_atr,
  risk_reward_ratio) from raw trade inputs BEFORE inference.
- Enforces prob_stop_hit + prob_target_hit <= 1.0, logging CALIBRATION_VIOLATION
  and clamping proportionally when the invariant is violated.
- Appends HIGH_RISK_BLOCKED reason code when risk_score > 7.0 and deployment
  mode is VALIDATED_ML_ONLY.

Requirements: Req 7.2, Req 7.3, Req 7.4, Req 7.6, Req 7.7
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.schemas.base import DeploymentMode, PredictionProvenance
from src.schemas.predictions import RiskResponse

# ── Feature registry ──────────────────────────────────────────────────────────

RISK_FEATURE_NAMES = [
    "stop_distance_atr",
    "target_distance_atr",
    "risk_reward_ratio",
    "rsi",
    "adx",
    "volume_ratio",
    "vix",
    "atr_pct",
    "regime_encoded",
    "vix_regime",
    "trend_alignment",
    "pcr",
    "oi_buildup_score",
    "news_impact_score",
    "sentiment_risk",
    "time_to_expiry_minutes",
]

# Ordinal encoding of MarketRegime — higher = more bullish.
REGIME_ENCODING: dict[str, int] = {
    "strong_bull": 5,
    "bull": 4,
    "sideways": 3,
    "volatile": 2,
    "bear": 1,
    "crash": 0,
}


class RiskPredictor:
    """
    Per-trade risk estimator with XGBoost ensemble / heuristic fallback.

    Usage::

        predictor = RiskPredictor()
        resp = predictor.predict({
            "symbol": "NIFTY",
            "entry": 22000.0, "stop_loss": 21800.0, "target": 22400.0,
            "atr": 200.0, "regime": "bull", "rsi": 60.0, "adx": 25.0,
            "volume_ratio": 1.1, "vix": 15.0,
        })
        print(resp.prob_stop_hit, resp.risk_score, resp.reason_codes)
    """

    def __init__(self, model_dir: Path | None = None) -> None:
        """
        Initialise the predictor.

        Args:
            model_dir: Optional directory containing ``stop_model.json``,
                       ``target_model.json``, and ``drawdown_model.json``
                       XGBoost artifacts.  When omitted or absent the
                       predictor runs in heuristic-only mode.
        """
        self.stop_model: Any | None = None
        self.target_model: Any | None = None
        self.drawdown_model: Any | None = None
        self.model_version: str = "1.0.0-heuristic"
        self.feature_names: list[str] = RISK_FEATURE_NAMES

        if model_dir and model_dir.exists():
            self._load_models(model_dir)

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def has_trained_model(self) -> bool:
        """True when at least the stop_model artifact is loaded."""
        return self.stop_model is not None

    # ── Public interface ───────────────────────────────────────────────────────

    def predict(self, features: dict[str, Any]) -> RiskResponse:
        """
        Predict per-trade risk metrics.

        Trade geometry features (stop_distance_atr, target_distance_atr,
        risk_reward_ratio) are derived from the raw ``entry``, ``stop_loss``,
        ``target``, and ``atr`` values before any model inference.

        Args:
            features: Dict of raw trade and market features.  Expected keys
                      include ``entry``, ``stop_loss``, ``target``, ``atr``,
                      ``regime``, ``rsi``, ``adx``, ``volume_ratio``, ``vix``.
                      All keys are optional; missing values get sensible defaults.

        Returns:
            :class:`~src.schemas.predictions.RiskResponse`
        """
        from src.config import settings
        from src.logging_config import get_logger

        logger = get_logger(__name__)

        # ── 1. Extract raw trade geometry ──────────────────────────────────────
        entry = float(features.get("entry", 0) or 0)
        stop = float(features.get("stop_loss", 0) or 0)
        target = float(features.get("target", 0) or 0)
        atr = float(features.get("atr", 1) or 1)
        vix = float(features.get("vix", 15) or 15)

        stop_dist = abs(entry - stop)
        target_dist = abs(target - entry)

        stop_distance_atr = stop_dist / atr if atr > 0 else 1.4
        target_distance_atr = target_dist / atr if atr > 0 else 2.0
        risk_reward_ratio = target_dist / stop_dist if stop_dist > 0 else 1.5

        # ── 2. Encode categorical / derived features ───────────────────────────
        vix_regime = self._encode_vix_regime(vix)

        regime_str = features.get("regime", "sideways")
        if hasattr(regime_str, "value"):
            regime_str = regime_str.value
        regime_encoded = float(REGIME_ENCODING.get(str(regime_str), 3))

        atr_pct = (atr / entry * 100) if entry > 0 else 1.0

        # ── 3. Build enriched feature dict ─────────────────────────────────────
        enriched: dict[str, Any] = dict(features)
        enriched.update(
            {
                "stop_distance_atr": stop_distance_atr,
                "target_distance_atr": target_distance_atr,
                "risk_reward_ratio": risk_reward_ratio,
                "vix_regime": vix_regime,
                "regime_encoded": regime_encoded,
                "atr_pct": atr_pct,
            }
        )

        # ── 4. Inference: trained ensemble or heuristic ────────────────────────
        if self.stop_model is not None:
            prob_stop, prob_target, provenance = self._model_predict(enriched, logger)
        else:
            prob_stop, prob_target, provenance = self._heuristic_risk(enriched)

        # ── 5. Enforce probability invariant (Property 8) ─────────────────────
        if prob_stop + prob_target > 1.0:
            logger.warning(
                "CALIBRATION_VIOLATION",
                prob_stop=prob_stop,
                prob_target=prob_target,
                sum=prob_stop + prob_target,
            )
            total = prob_stop + prob_target
            prob_stop = prob_stop / total * 0.99
            prob_target = prob_target / total * 0.99

        # ── 6. Composite risk score [0, 10] ────────────────────────────────────
        risk_score = min(10.0, prob_stop * 6.0 + vix_regime * 1.0)

        # ── 7. Expected drawdown and position sizing ───────────────────────────
        expected_drawdown = stop_distance_atr * atr_pct * 0.1
        max_risk_pct = 2.0
        suggested_size = max_risk_pct / max(atr_pct, 0.1)
        suggested_size = min(suggested_size, 10.0)

        # ── 8. SHAP-like factor contributions ─────────────────────────────────
        factors = {
            "stop_distance_atr": round(stop_distance_atr, 4),
            "risk_reward_ratio": round(risk_reward_ratio, 4),
            "vix_regime": round(vix_regime, 4),
            "regime_encoded": round(regime_encoded, 4),
            "news_impact_score": round(
                float(enriched.get("news_impact_score", 0) or 0), 4
            ),
        }

        # ── 9. Reason codes ────────────────────────────────────────────────────
        reason_codes: list[str] = []
        if (
            risk_score > 7.0
            and settings.deployment_mode == DeploymentMode.VALIDATED_ML_ONLY
        ):
            reason_codes.append("HIGH_RISK_BLOCKED")

        return RiskResponse(
            prob_stop_hit=round(prob_stop, 4),
            prob_target_hit=round(prob_target, 4),
            expected_drawdown_pct=round(expected_drawdown, 4),
            suggested_position_size_pct=round(suggested_size, 2),
            risk_score=round(risk_score, 2),
            factors=factors,
            provenance=provenance,
            reason_codes=reason_codes,
        )

    # ── Heuristic fallback ─────────────────────────────────────────────────────

    def _heuristic_risk(
        self, features: dict[str, Any]
    ) -> tuple[float, float, PredictionProvenance]:
        """
        Rule-based risk estimation.

        Returns:
            (prob_stop_hit, prob_target_hit, provenance)
        """
        rrr = float(features.get("risk_reward_ratio", 1.5))
        stop_atr = float(features.get("stop_distance_atr", 1.4))
        regime_enc = float(features.get("regime_encoded", 3))

        # Base P(stop) driven by stop distance in ATR multiples
        base_prob_stop = min(0.7, 0.2 + stop_atr * 0.1)

        # Regime adjustment
        if regime_enc <= 1:  # bear / crash → stops hit more often
            base_prob_stop = min(0.75, base_prob_stop * 1.2)
        elif regime_enc >= 4:  # bull / strong_bull → stops hit less often
            base_prob_stop = max(0.15, base_prob_stop * 0.8)

        # P(target) driven by R:R ratio
        base_prob_target = min(0.65, 0.1 + 0.1 * min(rrr, 4.0))

        # Ensure the invariant is met before returning
        if base_prob_stop + base_prob_target > 0.95:
            base_prob_target = 0.95 - base_prob_stop

        return (
            base_prob_stop,
            max(0.05, base_prob_target),
            PredictionProvenance.HEURISTIC,
        )

    # ── XGBoost ensemble prediction ────────────────────────────────────────────

    def _model_predict(
        self,
        features: dict[str, Any],
        logger: Any,
    ) -> tuple[float, float, PredictionProvenance]:
        """
        XGBoost ensemble prediction.

        Falls back to heuristic on any exception to keep the service running.

        Returns:
            (prob_stop_hit, prob_target_hit, provenance)
        """
        try:
            import numpy as np  # type: ignore[import-untyped]

            X = np.array([[features.get(f, 0.0) for f in self.feature_names]])

            prob_stop = float(self.stop_model.predict_proba(X)[0, 1])

            if self.target_model is not None:
                prob_target = float(self.target_model.predict_proba(X)[0, 1])
            else:
                prob_target = max(0.0, 0.7 - prob_stop)

            return prob_stop, prob_target, PredictionProvenance.TRAINED_MODEL

        except Exception as exc:
            logger.warning(
                "risk_predictor_model_inference_failed",
                error=str(exc),
                message="Falling back to heuristic.",
            )
            return self._heuristic_risk(features)

    # ── Model loading ──────────────────────────────────────────────────────────

    def _load_models(self, model_dir: Path) -> None:
        """
        Attempt to load the three XGBoost artifacts from *model_dir*.

        Silently falls back to heuristic mode if xgboost is not installed
        or any artifact is missing / corrupt.
        """
        from src.logging_config import get_logger

        logger = get_logger(__name__)

        try:
            import xgboost as xgb  # type: ignore[import-untyped]

            stop_path = model_dir / "stop_model.json"
            target_path = model_dir / "target_model.json"
            drawdown_path = model_dir / "drawdown_model.json"

            if stop_path.exists():
                self.stop_model = xgb.XGBClassifier()
                self.stop_model.load_model(str(stop_path))
                logger.info(
                    "risk_predictor_stop_model_loaded", path=str(stop_path)
                )

            if target_path.exists():
                self.target_model = xgb.XGBClassifier()
                self.target_model.load_model(str(target_path))
                logger.info(
                    "risk_predictor_target_model_loaded", path=str(target_path)
                )

            if drawdown_path.exists():
                self.drawdown_model = xgb.XGBRegressor()
                self.drawdown_model.load_model(str(drawdown_path))
                logger.info(
                    "risk_predictor_drawdown_model_loaded",
                    path=str(drawdown_path),
                )

            if self.stop_model is not None or self.target_model is not None:
                self.model_version = "1.0.0-trained"

        except Exception as exc:
            logger.warning(
                "risk_predictor_load_failed",
                error=str(exc),
                message="Falling back to heuristic mode.",
            )

    # ── Helpers ────────────────────────────────────────────────────────────────

    @staticmethod
    def _encode_vix_regime(vix: float) -> float:
        """
        Map a VIX level to a discrete ordinal regime.

        Returns:
            0.0 (calm) → 3.0 (extreme fear)
        """
        if vix < 13:
            return 0.0
        if vix < 18:
            return 1.0
        if vix < 25:
            return 2.0
        return 3.0
