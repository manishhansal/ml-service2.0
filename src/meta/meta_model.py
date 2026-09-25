"""
MetaDecisionEngine — AlphaForge Meta Decision Engine Orchestrator.

Combines predictions from all base models into a single, calibrated,
explainable trading decision.

Architecture
------------

    ┌─────────────────────────────────────────────────────────┐
    │                    MetaDecisionEngine                   │
    │                                                         │
    │  MetaInput (raw base-model predictions)                 │
    │       │                                                 │
    │       ▼                                                 │
    │  CalibrationStore  ──► calibrate each model's score     │
    │       │                                                 │
    │       ▼                                                 │
    │  EnsembleWeighter  ──► regime-aware weights +           │
    │                        disagreement metrics             │
    │       │                                                 │
    │       ▼                                                 │
    │  AbstentionPolicy  ──► WAIT / NO_TRADE gate             │
    │       │                                                 │
    │       ▼                                                 │
    │  DecisionPolicy    ──► confidence decomposition +       │
    │                        reason codes + explainability    │
    │       │                                                 │
    │       ▼                                                 │
    │  MetaOutput  (action, confidence, decomposition, ...)   │
    └─────────────────────────────────────────────────────────┘

OOS Training
------------
``MetaDecisionEngine.fit_meta_layer()`` accepts a list of
``OOSPredictionRecord`` objects — each record must come from a fold where
the base model was evaluated on data it was **not** trained on.

This is critical to prevent stacking leakage:

  - Base models are trained on folds 1-4
  - Their predictions on fold 5 (held-out) are collected
  - Those OOS predictions feed the meta-layer calibrators
  - The meta-layer never sees in-sample base-model predictions

"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from ..schemas import MarketRegime
from .abstention import (
    AbstentionDecision,
    AbstentionInputs,
    AbstentionKind,
    AbstentionPolicy,
    AbstentionThresholds,
)
from .calibration import CalibrationStore
from .decision_policy import (
    ConfidenceDecomposition,
    DecisionOutput,
    DecisionPolicy,
    DecisionPolicyConfig,
    ExplainabilityReport,
    TradeAction,
)
from .ensemble import EnsembleResult, EnsembleWeighter, ModelSignal

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Input / Output data structures
# ---------------------------------------------------------------------------

@dataclass
class RegimePrediction:
    """Snapshot of the regime classifier's output."""

    regime: MarketRegime
    confidence: float                       # ∈ [0, 1]
    probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class RankerPrediction:
    """Stock ranker output for the symbol under evaluation."""

    score: float                            # 0–100
    rank: int = 0
    factors: dict[str, float] = field(default_factory=dict)


@dataclass
class StrategyPrediction:
    """Strategy selector output."""

    strategy: str                           # TradingStrategy value
    confidence: float                       # ∈ [0, 1]
    alternatives: list[dict[str, float]] = field(default_factory=list)


@dataclass
class RiskPrediction:
    """Risk predictor output."""

    prob_stop_hit: float                    # ∈ [0, 1]
    prob_target_hit: float                  # ∈ [0, 1]
    expected_drawdown_pct: float
    suggested_position_size_pct: float
    risk_score: float                       # 0–10
    factors: dict[str, float] = field(default_factory=dict)


@dataclass
class PriceForecast:
    """Price forecaster output."""

    regime: str                             # "bull" | "bear" | "flat"
    probability: float                      # ∈ [0, 1]
    q10: float
    q90: float


@dataclass
class IVPrediction:
    """IV regime classifier output."""

    regime: str                             # "CRUSH" | "STABLE" | "SPIKE"
    # Phase 3F fix (BUG #8): was hardcoded 0.7 — fabricated confidence.
    # Now Optional[float] = None so absence is explicit.
    confidence: float | None = None         # ∈ [0, 1]; None = unavailable


@dataclass
class QuantEngineSignal:
    """Rule-based quant engine output."""

    direction: int                          # +1 | 0 | -1
    confidence: float                       # ∈ [0, 1]
    rules_fired: list[str] = field(default_factory=list)


@dataclass
class MetaInput:
    """
    Complete set of base-model predictions for a single evaluation.

    All fields are optional so the engine degrades gracefully when a
    model is unavailable.  Data quality is automatically computed from
    how many models provided signals.

    Parameters
    ----------
    symbol          : trading symbol being evaluated
    regime          : market regime prediction (required for weighting)
    ranker          : stock ranker prediction
    strategy        : strategy selector prediction
    risk            : risk predictor output
    price_forecast  : price forecaster output
    iv              : IV regime classifier output
    quant           : rule-based quant engine signal
    data_quality    : explicit data quality override [0,1]; computed
                      automatically from available models if None
    time_of_day_minutes : minutes since market open (0–375)
    """

    symbol: str
    regime: RegimePrediction | None = None
    ranker: RankerPrediction | None = None
    strategy: StrategyPrediction | None = None
    risk: RiskPrediction | None = None
    price_forecast: PriceForecast | None = None
    iv: IVPrediction | None = None
    quant: QuantEngineSignal | None = None
    data_quality: float | None = None      # override; auto-computed if None
    time_of_day_minutes: int | None = None

    def computed_data_quality(self) -> float:
        """Fraction of expected models that provided a prediction."""
        sources = [
            self.regime, self.ranker, self.strategy,
            self.risk, self.price_forecast, self.iv, self.quant,
        ]
        available = sum(1 for s in sources if s is not None)
        return self.data_quality if self.data_quality is not None \
            else available / len(sources)


@dataclass
class MetaOutput:
    """
    Complete output of the Meta Decision Engine.

    Attributes
    ----------
    symbol             : trading symbol
    action             : BUY | SELL | WAIT | NO_TRADE
    confidence         : composite final confidence ∈ [0, 1]
    uncertainty        : complementary uncertainty ∈ [0, 1]
    agreement          : model agreement ratio ∈ [0, 1]
    reason_codes       : machine-readable list explaining the decision
    contributing_models: models that had an active signal
    ensemble_score     : raw weighted meta-score ∈ [-1, 1]
    decomposition      : five-component confidence breakdown
    abstention         : full abstention decision (kind + reasons)
    explainability     : top factors + disagreeing models
    regime             : regime used for weighting
    """

    symbol: str
    action: TradeAction
    confidence: float
    uncertainty: float
    agreement: float
    reason_codes: list[str]
    contributing_models: list[str]
    ensemble_score: float
    decomposition: dict[str, float]
    abstention: AbstentionDecision
    explainability: ExplainabilityReport
    regime: MarketRegime | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol":             self.symbol,
            "action":             self.action.value,
            "confidence":         round(self.confidence, 4),
            "uncertainty":        round(self.uncertainty, 4),
            "agreement":          round(self.agreement, 4),
            "reasonCodes":        self.reason_codes,
            "contributingModels": self.contributing_models,
            "ensembleScore":      round(self.ensemble_score, 4),
            "decomposition":      {k: round(v, 4) for k, v in self.decomposition.items()},
            "abstention": {
                "kind":    self.abstention.kind.value,
                "reasons": [r.value for r in self.abstention.reasons],
                "details": self.abstention.reason_details,
                "severity": round(self.abstention.severity, 4),
            },
            "explainability": self.explainability.to_dict(),
            "regime": self.regime.value if self.regime else None,
        }


# ---------------------------------------------------------------------------
# OOS training record
# ---------------------------------------------------------------------------

@dataclass
class OOSPredictionRecord:
    """
    One out-of-sample prediction from a base model — used to train
    the meta-layer calibrators.

    **Never include in-sample predictions here** — stacking leakage will
    inflate calibration quality and cause the meta layer to be overconfident.

    Parameters
    ----------
    model_name   : one of calibration.MODEL_NAMES
    raw_score    : the uncalibrated score produced by the base model
    label        : ground-truth binary outcome {0, 1}
                   (e.g. 1 = trade was profitable, 0 = was not)
    fold_id      : cross-validation fold this prediction came from
    """

    model_name: str
    raw_score: float
    label: float                   # 0.0 or 1.0
    fold_id: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# MetaDecisionEngine
# ---------------------------------------------------------------------------

class MetaDecisionEngine:
    """
    AlphaForge Meta Decision Engine.

    Combines predictions from all base models into a single, calibrated,
    explainable trading decision using contextual ensemble weighting,
    probability calibration, confidence decomposition, and abstention logic.

    Parameters
    ----------
    calibration_store    : pre-loaded CalibrationStore; created fresh if None
    abstention_thresholds: custom AbstentionThresholds; defaults used if None
    decision_config      : custom DecisionPolicyConfig; defaults used if None
    disagreement_penalty : how much to down-weight dissenting models (0–1)
    """

    def __init__(
        self,
        calibration_store: CalibrationStore | None = None,
        abstention_thresholds: AbstentionThresholds | None = None,
        decision_config: DecisionPolicyConfig | None = None,
        disagreement_penalty: float = 0.5,
    ) -> None:
        self.calibration_store = calibration_store or CalibrationStore()
        self._abstention_policy = AbstentionPolicy(thresholds=abstention_thresholds)
        self._decision_policy = DecisionPolicy(config=decision_config)
        self._disagreement_penalty = disagreement_penalty
        self._weighter: EnsembleWeighter | None = None   # built lazily with cal quality
        logger.info("meta_decision_engine_initialised")

    # ------------------------------------------------------------------
    # Primary inference entry point
    # ------------------------------------------------------------------

    def decide(self, meta_input: MetaInput) -> MetaOutput:
        """
        Produce a trading decision from a complete set of base-model predictions.

        Parameters
        ----------
        meta_input : MetaInput with all available base-model signals

        Returns
        -------
        MetaOutput — action, confidence, uncertainty, agreement,
                     reason codes, explainability, abstention
        """
        inp = meta_input

        # ── 1. Build calibration-quality-aware weighter ──────────────────
        weighter = self._get_weighter()

        # ── 2. Translate base-model predictions → ModelSignals ───────────
        signals = self._build_signals(inp)

        # ── 3. Regime-aware ensemble ─────────────────────────────────────
        regime = inp.regime.regime if inp.regime else None
        ensemble: EnsembleResult = weighter.compute(signals, regime)

        # ── 4. Compute abstention inputs ─────────────────────────────────
        data_quality = inp.computed_data_quality()
        regime_confidence = inp.regime.confidence if inp.regime else 0.5
        prob_stop_hit = inp.risk.prob_stop_hit if inp.risk else 0.5
        # Phase 3F fix (BUG #7): use ensemble.weighted_confidence (calibration-
        # quality-weighted mean of calibrated confidences) rather than computing
        # an independent unweighted mean.  This is consistent with the ensemble
        # computation and respects calibration quality differences.
        mean_conf = float(np.clip(ensemble.weighted_confidence, 0.0, 1.0)) \
            if ensemble.active_models else 0.5

        abstention_inputs = AbstentionInputs(
            agreement_ratio=ensemble.disagreement.agreement_ratio,
            direction_disagreement=ensemble.disagreement.direction_disagreement,
            data_quality=data_quality,
            mean_confidence=mean_conf,
            regime_confidence=regime_confidence,
            prob_stop_hit=prob_stop_hit,
            confidence_spread=ensemble.disagreement.confidence_spread,
            ensemble_score=ensemble.weighted_score,
            n_available_models=len(ensemble.active_models),
            prediction_variance=ensemble.disagreement.prediction_variance,
        )

        # ── 5. Abstention check ──────────────────────────────────────────
        abstention: AbstentionDecision = self._abstention_policy.evaluate(
            abstention_inputs
        )

        # ── 6. Decision policy ───────────────────────────────────────────
        iv_regime = inp.iv.regime if inp.iv else None
        output, decomp, explain = self._decision_policy.build_decision(
            ensemble_result=ensemble,
            signals=signals,
            data_quality=data_quality,
            regime_confidence=regime_confidence,
            prob_stop_hit=prob_stop_hit,
            time_of_day_minutes=inp.time_of_day_minutes,
            iv_regime=iv_regime,
            abstention_kind=abstention.kind.value,
        )

        return MetaOutput(
            symbol=inp.symbol,
            action=output.action,
            confidence=output.confidence,
            uncertainty=output.uncertainty,
            agreement=output.agreement,
            reason_codes=output.reason_codes,
            contributing_models=output.contributing_models,
            ensemble_score=output.ensemble_score,
            decomposition=output.decomposition,
            abstention=abstention,
            explainability=explain,
            regime=regime,
        )

    # ------------------------------------------------------------------
    # OOS meta-layer training
    # ------------------------------------------------------------------

    def fit_meta_layer(
        self,
        oos_records: list[OOSPredictionRecord],
        min_samples_per_model: int = 50,
    ) -> dict[str, Any]:
        """
        Train the meta-layer calibrators from out-of-sample predictions.

        **Critical**: every record in *oos_records* must be a prediction
        made by a base model on data it was **not** trained on.  Passing
        in-sample predictions will cause stacking leakage and make the
        calibrators over-confident.

        Parameters
        ----------
        oos_records          : list of OOSPredictionRecord
        min_samples_per_model: minimum OOS samples required to fit a
                               calibrator (models below this threshold
                               are skipped — raw scores used instead)

        Returns
        -------
        dict mapping model_name → CalibrationQuality summary
        """
        # Group records by model
        grouped: dict[str, list[OOSPredictionRecord]] = {}
        for record in oos_records:
            grouped.setdefault(record.model_name, []).append(record)

        results: dict[str, Any] = {}

        for model_name, records in grouped.items():
            n = len(records)
            if n < min_samples_per_model:
                logger.warning(
                    "skipping_calibration_too_few_samples",
                    model=model_name,
                    n=n,
                    min_required=min_samples_per_model,
                )
                results[model_name] = {
                    "status": "skipped",
                    "n_samples": n,
                    "reason": f"fewer than {min_samples_per_model} OOS samples",
                }
                continue

            scores = np.array([r.raw_score for r in records])
            labels = np.array([r.label for r in records])

            # Validate: labels must be binary
            unique_labels = np.unique(labels)
            if not set(unique_labels).issubset({0.0, 1.0}):
                logger.warning(
                    "non_binary_labels_skipping_calibration",
                    model=model_name,
                    unique_labels=unique_labels.tolist(),
                )
                results[model_name] = {
                    "status": "skipped",
                    "reason": "labels are not binary {0, 1}",
                }
                continue

            # Fit both calibrators; store selects the better one automatically
            q_platt, q_iso = self.calibration_store.fit_both(
                model_name, scores, labels
            )

            results[model_name] = {
                "status": "fitted",
                "n_samples": n,
                "platt": {
                    "ece":         round(q_platt.ece, 4),
                    "mce":         round(q_platt.mce, 4),
                    "brier_score": round(q_platt.brier_score, 4),
                    "quality_score": round(q_platt.quality_score, 4),
                },
                "isotonic": {
                    "ece":         round(q_iso.ece, 4),
                    "mce":         round(q_iso.mce, 4),
                    "brier_score": round(q_iso.brier_score, 4),
                    "quality_score": round(q_iso.quality_score, 4),
                },
            }

            logger.info(
                "meta_layer_calibration_fitted",
                model=model_name,
                n=n,
                platt_ece=round(q_platt.ece, 4),
                iso_ece=round(q_iso.ece, 4),
            )

        # Rebuild the weighter with updated calibration quality
        self._weighter = None  # force rebuild on next decide() call

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, directory: Path) -> None:
        """
        Persist the meta engine's calibration store to *directory*.
        Base models are not serialised here — only the meta-layer state.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.calibration_store.save(directory)
        logger.info("meta_engine_saved", path=str(directory))

    def load(self, directory: Path) -> None:
        """Reload calibration state from a previously saved directory."""
        directory = Path(directory)
        self.calibration_store.load(directory)
        self._weighter = None   # invalidate cached weighter
        logger.info("meta_engine_loaded", path=str(directory))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_weighter(self) -> EnsembleWeighter:
        """
        Return (or lazily build) the EnsembleWeighter with up-to-date
        calibration quality scores.
        """
        if self._weighter is None:
            cal_quality = self.calibration_store.all_quality_scores()
            self._weighter = EnsembleWeighter(
                disagreement_penalty=self._disagreement_penalty,
                calibration_quality_scores=cal_quality,
            )
        return self._weighter

    # ------------------------------------------------------------------

    def _build_signals(self, inp: MetaInput) -> list[ModelSignal]:
        """
        Convert MetaInput base-model predictions into calibrated ModelSignals.

        Each model contributes a direction (+1 / 0 / -1) and a calibrated
        confidence score ∈ [0, 1].

        Mapping rules
        -------------
        regime       : STRONG_BULL/BULL → +1, BEAR/CRASH → -1, others → 0
                       confidence = regime.confidence
        ranker       : score > 60 → +1, score < 40 → -1, else → 0
                       confidence = score / 100
        strategy     : bullish strategies → +1, bearish/defensive → -1
        risk         : direction derived from prob_target_hit vs prob_stop_hit
                       confidence = |prob_target_hit - prob_stop_hit|
        price_forecast: "bull" → +1, "bear" → -1, "flat" → 0
        iv_classifier: "SPIKE" → down-signal (-1); "CRUSH"/"STABLE" → neutral/up
        quant_engine : direction and confidence passed through directly
        """
        signals: list[ModelSignal] = []

        # ── Regime ───────────────────────────────────────────────────────
        if inp.regime is not None:
            raw = inp.regime.confidence
            cal = self.calibration_store.calibrate("regime", raw)
            direction = _regime_to_direction(inp.regime.regime)
            signals.append(ModelSignal(
                model_name="regime",
                direction=direction,
                confidence=cal,
                raw_score=raw,
                is_available=True,
            ))
        else:
            signals.append(ModelSignal("regime", 0, 0.5, is_available=False))

        # ── Stock Ranker ─────────────────────────────────────────────────
        if inp.ranker is not None:
            raw = inp.ranker.score / 100.0    # normalise to [0, 1]
            cal = self.calibration_store.calibrate("ranker", raw)
            direction = 1 if raw > 0.60 else (-1 if raw < 0.40 else 0)
            signals.append(ModelSignal(
                model_name="ranker",
                direction=direction,
                confidence=cal,
                raw_score=raw,
                is_available=True,
            ))
        else:
            signals.append(ModelSignal("ranker", 0, 0.5, is_available=False))

        # ── Strategy Selector ────────────────────────────────────────────
        if inp.strategy is not None:
            raw = inp.strategy.confidence
            cal = self.calibration_store.calibrate("strategy", raw)
            direction = _strategy_to_direction(inp.strategy.strategy)
            signals.append(ModelSignal(
                model_name="strategy",
                direction=direction,
                confidence=cal,
                raw_score=raw,
                is_available=True,
            ))
        else:
            signals.append(ModelSignal("strategy", 0, 0.5, is_available=False))

        # ── Risk Predictor ───────────────────────────────────────────────
        if inp.risk is not None:
            # Positive direction when target is more likely to hit than stop
            diff = inp.risk.prob_target_hit - inp.risk.prob_stop_hit
            direction = 1 if diff > 0.10 else (-1 if diff < -0.10 else 0)
            # Phase 3F fix (BUG #5): do NOT calibrate abs(diff) through the
            # risk calibrator — that calibrator is trained on raw logit-scale
            # scores, but abs(diff) is already a probability-difference ∈ [0,1].
            # Running it through Platt/isotonic produces a meaningless transform.
            # Instead use abs(diff) directly as confidence — it is already
            # bounded [0,1] and represents directional conviction.
            raw = float(abs(diff))
            # Clip to valid confidence range; do NOT call calibrate() here.
            cal = float(np.clip(raw, 0.0, 1.0))
            signals.append(ModelSignal(
                model_name="risk",
                direction=direction,
                confidence=cal,
                raw_score=raw,
                is_available=True,
                metadata={"risk_score": inp.risk.risk_score},
            ))
        else:
            signals.append(ModelSignal("risk", 0, 0.5, is_available=False))

        # ── Price Forecaster ─────────────────────────────────────────────
        if inp.price_forecast is not None:
            raw = inp.price_forecast.probability
            cal = self.calibration_store.calibrate("price_forecaster", raw)
            direction = {
                "bull": 1, "bear": -1, "flat": 0
            }.get(inp.price_forecast.regime.lower(), 0)
            signals.append(ModelSignal(
                model_name="price_forecaster",
                direction=direction,
                confidence=cal,
                raw_score=raw,
                is_available=True,
            ))
        else:
            signals.append(ModelSignal("price_forecaster", 0, 0.5, is_available=False))

        # ── IV Classifier ─────────────────────────────────────────────────
        if inp.iv is not None:
            # Phase 3F fix (BUG #8): iv.confidence may now be None.
            # When absent, use 0.5 (neutral/uninformative) and mark explicitly.
            raw_iv = inp.iv.confidence
            iv_available = raw_iv is not None
            raw = float(raw_iv) if iv_available else 0.5
            cal = self.calibration_store.calibrate("iv_classifier", raw)
            # SPIKE → unfavourable for entry (-1)
            # CRUSH → slightly positive (+1, options are cheap)
            # STABLE → neutral (0)
            direction = {"SPIKE": -1, "CRUSH": 1, "STABLE": 0}.get(
                inp.iv.regime.upper(), 0
            )
            signals.append(ModelSignal(
                model_name="iv_classifier",
                direction=direction,
                confidence=cal,
                raw_score=raw,
                is_available=True,
            ))
        else:
            signals.append(ModelSignal("iv_classifier", 0, 0.5, is_available=False))

        # ── Quant Engine ─────────────────────────────────────────────────
        if inp.quant is not None:
            raw = inp.quant.confidence
            cal = self.calibration_store.calibrate("quant_engine", raw)
            signals.append(ModelSignal(
                model_name="quant_engine",
                direction=inp.quant.direction,
                confidence=cal,
                raw_score=raw,
                is_available=True,
                metadata={"rules_fired": inp.quant.rules_fired},
            ))
        else:
            signals.append(ModelSignal("quant_engine", 0, 0.5, is_available=False))

        return signals


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _regime_to_direction(regime: MarketRegime) -> int:
    """Map a MarketRegime to a directional signal."""
    bullish = {MarketRegime.STRONG_BULL, MarketRegime.BULL}
    bearish = {MarketRegime.BEAR, MarketRegime.CRASH}
    if regime in bullish:
        return 1
    if regime in bearish:
        return -1
    return 0  # SIDEWAYS, VOLATILE


# Strategies that suggest a long bias
_BULLISH_STRATEGIES = {
    "breakout", "momentum", "trend_following", "volatility_breakout",
}
# Mean-reversion is direction-AGNOSTIC: it fades extended moves in either
# direction.  Mapping it to -1 (bearish) was incorrect because a mean-
# reversion signal on an oversold stock generates a LONG, not a SHORT.
# It is moved to _NEUTRAL_STRATEGIES so the ensemble treats the strategy
# selector as providing zero directional opinion when mean_reversion is
# selected — the direction comes from other models (regime, ranker, risk).
_NEUTRAL_STRATEGIES = {
    "mean_reversion",  # direction-agnostic: long when oversold, short when overbought
    "vwap_bounce",     # execution-timing strategy; direction from broader signal
    "range_trading",   # symmetric; no inherent directional bias
    "scalping",        # execution-timing; no strategic directional bias
}


def _strategy_to_direction(strategy: str) -> int:
    """
    Map a TradingStrategy name to a directional signal (+1 / 0 / -1).

    Mean-reversion is intentionally 0 (neutral).
    A strategy family is NOT a trade direction — the trade direction for
    a mean-reversion strategy is determined by the ranker/regime/risk
    models, not by the strategy name alone.  Mapping mean_reversion to
    -1 was a bug that systematically penalised mean-reversion signals in
    the ensemble.
    """
    s = strategy.lower()
    if s in _BULLISH_STRATEGIES:
        return 1
    # All remaining strategies (mean_reversion, vwap_bounce, range_trading,
    # scalping) are neutral — they say HOW to trade, not WHICH direction.
    return 0
