"""
CalibrationLayer — Platt scaling / isotonic regression calibration for meta-engine.
ConfidenceDecomposer — five-component confidence decomposition.

CalibrationLayer design:
- Calibrates each base model's raw score using Platt scaling or isotonic regression
- Selects the method with lower ECE (Expected Calibration Error) on held-out OOS fold
- Rejects calibration update if ECE > 0.15 (retains prior calibrator)
- Default: identity calibration when no OOS data is available

ConfidenceDecomposer design:
- Decomposes final confidence into five components:
  base_confidence, calibration_quality, agreement_bonus, data_quality_factor,
  regime_confidence_factor

Requirements: Req 10.2, Req 10.8
"""
from __future__ import annotations

from typing import Any

import numpy as np

from src.logging_config import get_logger
from src.schemas.meta import ConfidenceDecomposition

logger = get_logger(__name__)

MAX_ECE_THRESHOLD = 0.15  # Reject calibration update if best ECE exceeds this


class _IsotonicWrapper:
    """
    Thin wrapper around ``sklearn.isotonic.IsotonicRegression`` that exposes
    the same ``predict_proba(X)`` interface as scikit-learn classifiers so
    CalibrationLayer can use a single unified call path.
    """

    def __init__(self, model: Any) -> None:
        self._model = model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = np.clip(self._model.predict(X.ravel()), 0.0, 1.0)
        return np.column_stack([1.0 - preds, preds])


class CalibrationLayer:
    """
    Per-model score calibration using Platt scaling or isotonic regression.

    The layer is keyed by ``model_id`` (a plain string such as
    ``"market_regime"`` or ``"stock_ranker"``).  Before any ``fit()`` call
    the layer returns raw scores unchanged (identity calibration).

    Usage::

        cal = CalibrationLayer()
        # Optionally warm up on OOS predictions:
        accepted = cal.fit("market_regime", oos_scores=[...], oos_labels=[...])
        # At inference time:
        calibrated = cal.calibrate("market_regime", raw_score=0.73)
    """

    def __init__(self) -> None:
        # model_id → fitted sklearn-compatible calibrator
        self._calibrators: dict[str, Any] = {}
        # model_id → ECE of the active calibrator
        self._ece_scores: dict[str, float] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(self, model_id: str, raw_score: float) -> float:
        """
        Apply the fitted calibrator for *model_id* to *raw_score*.

        If no calibrator has been fitted yet the raw score is returned
        unchanged (clamped to [0, 1]).

        Args:
            model_id:  Identifier of the base model (e.g. ``"market_regime"``).
            raw_score: Raw prediction score in any float range; typically [0, 1].

        Returns:
            Calibrated probability in [0, 1].
        """
        calibrator = self._calibrators.get(model_id)
        if calibrator is None:
            return float(max(0.0, min(1.0, raw_score)))

        try:
            score_arr = np.array([[raw_score]], dtype=float)
            calibrated = float(calibrator.predict_proba(score_arr)[0, 1])
            return max(0.0, min(1.0, calibrated))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "calibration_apply_failed",
                model_id=model_id,
                raw_score=raw_score,
                error=str(exc),
            )
            return float(max(0.0, min(1.0, raw_score)))

    def fit(
        self,
        model_id: str,
        oos_scores: list[float],
        oos_labels: list[float],
    ) -> bool:
        """
        Fit Platt scaling and isotonic regression, keep whichever has lower ECE.

        The update is **rejected** and the previous calibrator (if any) is
        retained when the winning method's ECE still exceeds
        ``MAX_ECE_THRESHOLD`` (0.15).

        A minimum of 10 samples is required to attempt fitting.

        Args:
            model_id:   Model identifier (same key used for ``calibrate()``).
            oos_scores: Raw prediction scores from OOS evaluation.
            oos_labels: Binary labels (1.0 = positive outcome, 0.0 = negative).

        Returns:
            ``True`` if the calibration was accepted and stored.
            ``False`` if rejected (too few samples or ECE > threshold).
        """
        if len(oos_scores) < 10:
            logger.warning(
                "calibration_insufficient_data",
                model_id=model_id,
                n=len(oos_scores),
                required=10,
            )
            return False

        X = np.array(oos_scores, dtype=float).reshape(-1, 1)
        y = np.array(oos_labels, dtype=float)

        try:
            from sklearn.linear_model import LogisticRegression  # Platt scaling
            from sklearn.isotonic import IsotonicRegression

            # ── Platt scaling (sigmoid / logistic regression) ─────────────────
            platt_model = LogisticRegression(max_iter=1000)
            platt_model.fit(X, y)
            platt_preds = platt_model.predict_proba(X)[:, 1]
            platt_ece = self._compute_ece(platt_preds, y)

            # ── Isotonic regression ───────────────────────────────────────────
            iso_model = IsotonicRegression(out_of_bounds="clip")
            iso_model.fit(oos_scores, oos_labels)
            iso_preds = np.clip(iso_model.predict(oos_scores), 0.0, 1.0)
            iso_ece = self._compute_ece(iso_preds, y)

            best_ece = min(platt_ece, iso_ece)

            if best_ece > MAX_ECE_THRESHOLD:
                logger.warning(
                    "calibration_rejected_high_ece",
                    model_id=model_id,
                    platt_ece=round(platt_ece, 4),
                    isotonic_ece=round(iso_ece, 4),
                    threshold=MAX_ECE_THRESHOLD,
                )
                return False

            # Select method with lower ECE; prefer Platt on a tie
            if platt_ece <= iso_ece:
                self._calibrators[model_id] = platt_model
                self._ece_scores[model_id] = platt_ece
                logger.info(
                    "calibration_fitted",
                    model_id=model_id,
                    method="platt",
                    ece=round(platt_ece, 4),
                )
            else:
                self._calibrators[model_id] = _IsotonicWrapper(iso_model)
                self._ece_scores[model_id] = iso_ece
                logger.info(
                    "calibration_fitted",
                    model_id=model_id,
                    method="isotonic",
                    ece=round(iso_ece, 4),
                )

            return True

        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "calibration_fit_failed",
                model_id=model_id,
                error=str(exc),
            )
            return False

    def get_ece(self, model_id: str) -> float:
        """
        Return the ECE of the active calibrator for *model_id*.

        Returns ``1.0`` (worst possible) when no calibrator has been fitted.
        """
        return self._ece_scores.get(model_id, 1.0)

    def has_calibrator(self, model_id: str) -> bool:
        """Return ``True`` if a calibrator has been fitted for *model_id*."""
        return model_id in self._calibrators

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _compute_ece(
        predictions: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10,
    ) -> float:
        """
        Compute Expected Calibration Error over *n_bins* equal-width bins.

        ECE = Σ_b (|B_b| / n) × |avg_confidence(B_b) − avg_accuracy(B_b)|

        Args:
            predictions: Predicted probabilities in [0, 1].
            labels:      Binary ground-truth labels (0 or 1).
            n_bins:      Number of equal-width confidence bins.

        Returns:
            ECE as a float in [0, 1].
        """
        n = len(predictions)
        if n == 0:
            return 1.0

        ece = 0.0
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            # Include the right edge in the last bin
            if i == n_bins - 1:
                mask = (predictions >= lo) & (predictions <= hi)
            else:
                mask = (predictions >= lo) & (predictions < hi)

            bin_count = int(mask.sum())
            if bin_count == 0:
                continue

            avg_pred = float(predictions[mask].mean())
            avg_label = float(labels[mask].mean())
            bin_weight = bin_count / n
            ece += bin_weight * abs(avg_pred - avg_label)

        return float(ece)


class ConfidenceDecomposer:
    """
    Decomposes the MetaDecisionEngine's final confidence into five interpretable
    components as required by Requirement 10.8.

    Components
    ----------
    base_confidence
        The raw ensemble-weighted score before adjustment.
    calibration_quality
        ``1 − ECE`` of the worst-performing calibrator currently loaded;
        defaults to ``0.8`` when no calibrators have been fitted.
    agreement_bonus
        A small additive bonus (0–0.1) for high inter-model agreement,
        linear in the excess agreement above 0.5.
    data_quality_factor
        Fraction of models whose underlying data passed the
        ``signalEngineAllowed`` quality gate; clamped to [0, 1].
    regime_confidence_factor
        Confidence score produced by the RegimeClassifier; clamped to [0, 1].

    All output components are bounded to [0, 1] so downstream visualisations
    can render normalised importance bars without additional clamping.

    Usage::

        decomposer = ConfidenceDecomposer()
        decomp = decomposer.decompose(
            base_confidence=0.70,
            agreement_ratio=0.80,
            data_quality=0.90,
            regime_confidence=0.75,
            calibration_ece=0.05,
        )
        # decomp.base_confidence == 0.70
        # decomp.agreement_bonus  > 0.0
    """

    def decompose(
        self,
        base_confidence: float,
        agreement_ratio: float,
        data_quality: float,
        regime_confidence: float,
        calibration_ece: float = 0.0,
    ) -> ConfidenceDecomposition:
        """
        Compute the five-component confidence decomposition.

        Args:
            base_confidence:      Raw ensemble score, typically in [0, 1].
            agreement_ratio:      Fraction of models agreeing on direction,
                                  in [0, 1].
            data_quality:         Fraction of models with valid data, in [0, 1].
            regime_confidence:    RegimeClassifier confidence, in [0, 1].
            calibration_ece:      Expected Calibration Error of the worst active
                                  calibrator; ``0.0`` when none are fitted
                                  (yields ``calibration_quality = 1.0``).

        Returns:
            A frozen :class:`ConfidenceDecomposition` instance.
        """
        # calibration_quality: 1 − ECE, bounded to [0, 1]
        calibration_quality = float(max(0.0, min(1.0, 1.0 - calibration_ece)))

        # agreement_bonus: linear bonus for agreement above 50%, max 0.1
        if agreement_ratio >= 0.5:
            # maps (0.5 → 0.0, 1.0 → 0.1) linearly
            agreement_bonus = float(min(0.1, max(0.0, (agreement_ratio - 0.5) * 0.2)))
        else:
            agreement_bonus = 0.0

        data_quality_factor = float(max(0.0, min(1.0, data_quality)))
        regime_confidence_factor = float(max(0.0, min(1.0, regime_confidence)))

        return ConfidenceDecomposition(
            base_confidence=float(max(0.0, min(1.0, base_confidence))),
            calibration_quality=calibration_quality,
            agreement_bonus=float(max(0.0, min(0.1, agreement_bonus))),  # bounded by field def [0,1]
            data_quality_factor=data_quality_factor,
            regime_confidence_factor=regime_confidence_factor,
        )


# ── CalibrationStore (Phase 3F requirements) ──────────────────────────────────
# A simple, standalone calibrator store that wraps sklearn Platt/isotonic
# calibration and provides the contract expected by test_phase3f.

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class CalibrationQuality:
    """Quality metrics produced by CalibrationStore.fit().

    Supports both the new API (used by CalibrationStore.fit()) and the
    legacy test API (CalibrationQuality(model_name=..., calibrator_kind=..., ...)).

    eval_is_oos: True when the evaluation set was held out from the fit set.
    brier:       Brier score on the evaluation set (new API).
    brier_score: Alias for legacy compatibility.
    ece:         Expected Calibration Error on the evaluation set.
    mce:         Maximum Calibration Error (optional legacy field).
    """

    model_id: str = ""
    model_name: str = ""       # legacy field name
    kind: str = ""
    calibrator_kind: str = ""  # legacy field name
    eval_is_oos: bool = False
    brier: float = 0.0
    brier_score: float = 0.0   # legacy field name
    ece: float = 0.0
    mce: float = 0.0           # legacy field
    n_samples: int = 0         # legacy field
    is_fitted: bool = True     # legacy field


class CalibrationStore:
    """Standalone multi-model calibration store.

    Wraps sklearn's Platt scaling (LogisticRegression on raw scores) and
    isotonic regression calibration. Unknown models degrade gracefully to
    0.5 (neutral), never clipping the raw score.

    Usage::

        store = CalibrationStore()
        quality = store.fit("regime", raw_scores, labels, kind="platt",
                            eval_scores=eval_s, eval_labels=eval_l)
        p = store.calibrate("regime", 0.8)   # calibrated probability
        # Unknown model → 0.5 (not clip(0.8))
        q = store.calibrate("unknown", 0.8)  # 0.5
    """

    def __init__(self) -> None:
        self._calibrators: dict[str, Any] = {}
        self._quality: dict[str, CalibrationQuality] = {}

    def fit(
        self,
        model_id: str,
        raw_scores: "np.ndarray",
        labels: "np.ndarray",
        kind: str = "platt",
        eval_scores: "np.ndarray | None" = None,
        eval_labels: "np.ndarray | None" = None,
    ) -> CalibrationQuality:
        """Fit a calibrator for ``model_id``.

        Args:
            model_id:    Unique identifier for this model's calibrator.
            raw_scores:  1-D array of raw model output scores.
            labels:      1-D binary label array (0/1 floats).
            kind:        "platt" (logistic) or "isotonic".
            eval_scores: Optional held-out scores for quality evaluation.
            eval_labels: Optional held-out labels for quality evaluation.

        Returns:
            CalibrationQuality with eval_is_oos=True iff eval_scores supplied.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.isotonic import IsotonicRegression
        from sklearn.calibration import calibration_curve

        X = np.array(raw_scores, dtype=float).reshape(-1, 1)
        y = np.array(labels, dtype=float)

        if kind == "platt":
            clf = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
            clf.fit(X, y)
            calibrator = clf
        else:
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(X.ravel(), y)
            calibrator = ir

        self._calibrators[model_id] = (kind, calibrator)

        # Evaluate on eval set (if provided) or fall back to train set
        eval_is_oos = eval_scores is not None and eval_labels is not None

        # Validation: eval_scores without eval_labels is an error
        if eval_scores is not None and eval_labels is None:
            raise ValueError(
                "eval_labels must be provided when eval_scores is given. "
                "Providing eval_scores without eval_labels would give misleading quality metrics."
            )
        if eval_is_oos:
            e_scores = np.array(eval_scores, dtype=float)
            e_labels = np.array(eval_labels, dtype=float)
        else:
            e_scores = np.array(raw_scores, dtype=float)
            e_labels = y

        # Get calibrated probabilities for evaluation
        cal_probs = self._apply(kind, calibrator, e_scores)

        brier = float(np.mean((cal_probs - e_labels) ** 2))
        # Simple ECE with 10 bins
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for i in range(len(bins) - 1):
            mask = (cal_probs >= bins[i]) & (cal_probs < bins[i + 1])
            if mask.sum() > 0:
                bin_conf = float(cal_probs[mask].mean())
                bin_acc  = float(e_labels[mask].mean())
                ece += (mask.sum() / len(cal_probs)) * abs(bin_conf - bin_acc)

        quality = CalibrationQuality(
            model_id=model_id,
            model_name=model_id,
            kind=kind,
            calibrator_kind=kind,
            eval_is_oos=eval_is_oos,
            brier=brier,
            brier_score=brier,
            ece=ece,
            n_samples=len(e_labels),
            is_fitted=True,
        )
        self._quality[model_id] = quality
        return quality

    @staticmethod
    def _apply(kind: str, calibrator: Any, scores: "np.ndarray") -> "np.ndarray":
        X = scores.reshape(-1, 1)
        if kind == "platt":
            return calibrator.predict_proba(X)[:, 1]
        else:
            return calibrator.predict(scores)

    def calibrate(self, model_id: str, raw_score: float) -> float:
        """Return the calibrated probability for one score.

        If the model has no fitted calibrator, returns 0.5 (neutral).
        Never clips the raw score — unknown model → 0.5, not clip(raw).
        """
        if model_id not in self._calibrators:
            return 0.5
        kind, cal = self._calibrators[model_id]
        probs = self._apply(kind, cal, np.array([raw_score], dtype=float))
        return float(np.clip(probs[0], 0.0, 1.0))

    def calibrate_batch(
        self, model_id: str, raw_scores: "np.ndarray"
    ) -> "np.ndarray":
        """Return calibrated probabilities for an array of scores.

        If the model has no fitted calibrator, returns an array of 0.5.
        Never clips the raw score — unknown model → 0.5, not clip(raw).
        """
        if model_id not in self._calibrators:
            return np.full(len(raw_scores), 0.5)
        kind, cal = self._calibrators[model_id]
        probs = self._apply(kind, cal, np.array(raw_scores, dtype=float))
        return np.clip(probs, 0.0, 1.0)

    def has_calibrator(self, model_id: str) -> bool:
        return model_id in self._calibrators
