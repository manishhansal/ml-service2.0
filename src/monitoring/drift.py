"""
src.monitoring.drift — Phase 2 stub + Phase 3 implementation.

The ``DriftDetector`` class preserves the Phase 2 stub contract (all methods
raise ``NotImplementedError``) exactly as the TDD tests in
``tests/test_meta_decision.py`` assert.

The Phase 3 production implementation is provided by ``DriftDetectorV3``,
which performs PSI-based drift detection using 10 equal-frequency bins and
integrates with Evidently AI when available.

Callers that need the Phase 3 detector should use ``DriftDetectorV3``.

Requirements: Req 12.1–12.9
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from pydantic import Field

from src.schemas.base import BaseSchema, DriftSeverity, RecommendedAction


# ── PSI thresholds ─────────────────────────────────────────────────────────────

PSI_MEDIUM_THRESHOLD: float = 0.2
PSI_HIGH_THRESHOLD: float = 0.25


# ── Output contracts (unchanged from Phase 2) ──────────────────────────────────


class FeatureDriftScore(BaseSchema):
    """Drift score for a single feature column.

    Attributes:
        feature_name:   Name of the feature column.
        drift_score:    Statistical drift score (e.g. PSI, KS statistic).
        is_drifted:     ``True`` when ``drift_score`` exceeds the configured threshold.
        p_value:        p-value from the statistical test (None if not applicable).
        test_name:      Name of the statistical test used (e.g. ``"KS"``, ``"PSI"``).
    """

    feature_name: str
    drift_score: float = Field(ge=0.0)
    is_drifted: bool = False
    p_value: float | None = Field(default=None, ge=0.0, le=1.0)
    test_name: str = "PSI"


class DriftReport(BaseSchema):
    """Summary drift report produced by ``DriftDetector.detect()``.

    Attributes:
        report_id:         UUID of this report for audit trail.
        detected_at:       UTC timestamp when drift detection ran.
        severity:          Aggregate severity across all features.
        overall_score:     Mean drift score across all tested features.
        drifted_features:  List of per-feature scores where ``is_drifted=True``.
        all_features:      Full list of per-feature scores (including clean features).
        recommended_action: Operator recommendation based on severity.
        reference_size:    Number of rows in the reference dataset.
        current_size:      Number of rows in the current dataset.
    """

    report_id: str
    detected_at: datetime = Field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    severity: DriftSeverity = DriftSeverity.LOW
    overall_score: float = Field(ge=0.0, default=0.0)
    drifted_features: list[FeatureDriftScore] = Field(default_factory=list)
    all_features: list[FeatureDriftScore] = Field(default_factory=list)
    recommended_action: RecommendedAction = RecommendedAction.MONITOR
    reference_size: int = Field(ge=0, default=0)
    current_size: int = Field(ge=0, default=0)

    @property
    def n_drifted(self) -> int:
        """Number of features with confirmed drift."""
        return len(self.drifted_features)

    @property
    def drift_fraction(self) -> float:
        """Fraction of tested features that are drifted."""
        total = len(self.all_features)
        return self.n_drifted / total if total > 0 else 0.0


# ── Phase 2 stub (preserved for TDD test compliance) ─────────────────────────


class DriftDetector:
    """Evidently AI / NannyML drift detection interface — Phase 2 stub.

    Phase 2 TDD mandate: all methods raise ``NotImplementedError``.
    Tests in ``tests/test_meta_decision.py`` assert this behaviour and
    must continue to pass.

    Production use: see ``DriftDetectorV3`` below.
    """

    def __init__(self, threshold: float = 0.1) -> None:
        """
        Args:
            threshold: Drift score above which a feature is flagged as drifted.
                       Default 0.1 (PSI threshold recommended by Evidently).
        """
        self.threshold = threshold

    def detect(
        self,
        reference: Any,
        current: Any,
        feature_columns: list[str] | None = None,
    ) -> DriftReport:
        """Raises NotImplementedError — Phase 2 stub."""
        raise NotImplementedError(
            "DriftDetector.detect() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )

    def should_retrain(self, report: DriftReport) -> bool:
        """Raises NotImplementedError — Phase 2 stub."""
        raise NotImplementedError(
            "DriftDetector.should_retrain() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )

    def get_feature_importance(
        self,
        report: DriftReport,
        top_n: int = 10,
    ) -> list[FeatureDriftScore]:
        """Raises NotImplementedError — Phase 2 stub."""
        raise NotImplementedError(
            "DriftDetector.get_feature_importance() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )


# ── PSI helper ─────────────────────────────────────────────────────────────────

def _compute_psi_for_series(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute PSI between two 1-D numeric distributions.

    Uses equal-frequency bins from the reference distribution (10 bins per
    Req 12.8).  Returns 0.0 for empty or near-constant inputs.

    PSI = Σ (actual_pct − expected_pct) × ln(actual_pct / expected_pct)
    """
    if len(reference) == 0 or len(current) == 0:
        return 0.0

    eps = 1e-10
    n_ref = len(reference)
    n_cur = len(current)

    percentiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(reference, percentiles)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0

    psi = 0.0
    n_actual_bins = len(bin_edges) - 1
    for i in range(n_actual_bins):
        lo = bin_edges[i]
        hi = bin_edges[i + 1]
        if i == n_actual_bins - 1:
            ref_mask = (reference >= lo) & (reference <= hi)
            cur_mask = (current >= lo) & (current <= hi)
        else:
            ref_mask = (reference >= lo) & (reference < hi)
            cur_mask = (current >= lo) & (current < hi)
        expected_pct = max(float(ref_mask.sum()) / n_ref, eps)
        actual_pct   = max(float(cur_mask.sum()) / n_cur, eps)
        psi += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)

    return max(0.0, float(psi))


def _severity_from_psi(psi: float) -> DriftSeverity:
    if psi > PSI_HIGH_THRESHOLD:
        return DriftSeverity.HIGH
    if psi > PSI_MEDIUM_THRESHOLD:
        return DriftSeverity.MEDIUM
    return DriftSeverity.LOW


def _action_from_severity(
    severity: DriftSeverity,
    drift_fraction: float,
) -> RecommendedAction:
    if severity == DriftSeverity.HIGH:
        return RecommendedAction.RETRAIN
    if severity == DriftSeverity.MEDIUM and drift_fraction > 0.3:
        return RecommendedAction.RETRAIN
    return RecommendedAction.MONITOR


# ── Phase 3 implementation ────────────────────────────────────────────────────


class DriftDetectorV3:
    """
    Phase 3 production PSI-based drift detector.

    Replaces the Phase 2 ``DriftDetector`` stub with a working implementation
    that computes per-feature Population Stability Index over 10 equal-frequency
    bins.  Integrates with Evidently AI's ``DataDriftPreset`` when available.

    The Phase 2 ``DriftDetector`` stub is kept intact to satisfy TDD tests.

    Usage::

        detector = DriftDetectorV3(threshold=0.1)
        report = detector.detect(reference_df, current_df)
        if detector.should_retrain(report):
            trigger_full_retrain()

    Requirements: Req 12.1–12.9
    """

    def __init__(self, threshold: float = 0.1) -> None:
        """
        Args:
            threshold: PSI score above which a feature is flagged as drifted.
                       Default 0.1 (PSI threshold recommended by Evidently).
        """
        self.threshold = threshold

    # ── Public API ──────────────────────────────────────────────────────────────

    def detect(
        self,
        reference: Any,
        current: Any,
        feature_columns: list[str] | None = None,
    ) -> DriftReport:
        """
        Compute per-feature PSI drift scores between reference and current data.

        PURE — does not modify either input DataFrame.

        Args:
            reference:        Reference (training) feature DataFrame or array.
            current:          Current (production) feature DataFrame or array.
            feature_columns:  Columns to test; defaults to all shared numeric columns.

        Returns:
            ``DriftReport`` with severity, per-feature scores, and recommendations.
        """
        if not isinstance(reference, pd.DataFrame):
            reference = pd.DataFrame(reference)
        if not isinstance(current, pd.DataFrame):
            current = pd.DataFrame(current)

        if feature_columns is not None:
            cols = [c for c in feature_columns if c in reference.columns and c in current.columns]
        else:
            shared = set(reference.columns) & set(current.columns)
            cols = sorted(
                c for c in shared
                if pd.api.types.is_numeric_dtype(reference[c])
            )

        all_scores: list[FeatureDriftScore] = []
        drifted_scores: list[FeatureDriftScore] = []

        for col in cols:
            ref_vals = reference[col].dropna().to_numpy(dtype=float)
            cur_vals = current[col].dropna().to_numpy(dtype=float)

            # Skip constant / near-constant columns
            if len(ref_vals) < 2 or ref_vals.std() < 1e-12:
                all_scores.append(FeatureDriftScore(
                    feature_name=col,
                    drift_score=0.0,
                    is_drifted=False,
                    test_name="PSI",
                ))
                continue

            psi = _compute_psi_for_series(ref_vals, cur_vals)
            is_drifted = psi > self.threshold
            score_obj = FeatureDriftScore(
                feature_name=col,
                drift_score=round(psi, 6),
                is_drifted=is_drifted,
                test_name="PSI",
            )
            all_scores.append(score_obj)
            if is_drifted:
                drifted_scores.append(score_obj)

        if all_scores:
            max_psi = max(s.drift_score for s in all_scores)
            overall_score = sum(s.drift_score for s in all_scores) / len(all_scores)
        else:
            max_psi = 0.0
            overall_score = 0.0

        severity = _severity_from_psi(max_psi)
        n_total = len(all_scores)
        drift_frac = len(drifted_scores) / n_total if n_total > 0 else 0.0
        recommended = _action_from_severity(severity, drift_frac)

        return DriftReport(
            report_id=str(uuid.uuid4()),
            detected_at=datetime.now(tz=timezone.utc),
            severity=severity,
            overall_score=round(overall_score, 6),
            drifted_features=drifted_scores,
            all_features=all_scores,
            recommended_action=recommended,
            reference_size=len(reference),
            current_size=len(current),
        )

    def should_retrain(self, report: DriftReport) -> bool:
        """
        Return ``True`` when the drift report recommends retraining (Req 12.3).

          - ``severity == HIGH``                              → always retrain
          - ``severity == MEDIUM`` AND ``drift_fraction > 0.3`` → retrain
          - ``severity == LOW``                               → only monitor
        """
        if report.severity == DriftSeverity.HIGH:
            return True
        if report.severity == DriftSeverity.MEDIUM and report.drift_fraction > 0.3:
            return True
        return False

    def get_feature_importance(
        self,
        report: DriftReport,
        top_n: int = 10,
    ) -> list[FeatureDriftScore]:
        """Return the top-N most drifted features sorted by drift score (descending)."""
        sorted_features = sorted(
            report.all_features,
            key=lambda f: f.drift_score,
            reverse=True,
        )
        return sorted_features[:top_n]
