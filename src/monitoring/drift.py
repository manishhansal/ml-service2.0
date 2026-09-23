"""
src.monitoring.drift — Phase 2 stub interface for Evidently AI drift detection.

This module defines the ``DriftDetector`` interface that the TDD tests in
Phase 2 drive.  All methods raise ``NotImplementedError`` until Phase 3
wires in the real Evidently / NannyML implementation.

Interface contract
------------------
``DriftDetector.detect(reference, current)``
  - Accepts reference and current feature DataFrames.
  - Returns a ``DriftReport`` with per-feature drift scores and a severity.
  - Must NOT modify either input DataFrame (pure function, no side effects).

``DriftDetector.should_retrain(report)``
  - Returns ``True`` when ``report.severity`` is HIGH or when drift score
    exceeds the configured threshold.

``DriftReport``
  - Pydantic model with ``severity``, ``drifted_features``, ``overall_score``,
    and ``recommended_action``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import Field

from src.schemas.base import BaseSchema, DriftSeverity, RecommendedAction


# ── Phase 2 output contract ───────────────────────────────────────────────────


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


# ── Phase 2 stub ──────────────────────────────────────────────────────────────


class DriftDetector:
    """Evidently AI / NannyML drift detection interface — Phase 2 stub.

    Phase 2 TDD mandate: all methods raise ``NotImplementedError``.
    Tests assert the expected behaviour; Phase 3 provides the real
    Evidently / NannyML-backed implementation.

    Usage (Phase 3)::

        detector = DriftDetector(threshold=0.1)
        report = detector.detect(reference_df, current_df)
        if detector.should_retrain(report):
            trigger_retraining_pipeline()
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
        reference: Any,   # pandas DataFrame in Phase 3
        current: Any,     # pandas DataFrame in Phase 3
        feature_columns: list[str] | None = None,
    ) -> DriftReport:
        """Compute per-feature drift scores between reference and current data.

        Args:
            reference:        Reference (training) feature DataFrame.
            current:          Current (production) feature DataFrame.
            feature_columns:  Subset of columns to test; defaults to all shared columns.

        Returns:
            ``DriftReport`` with severity, per-feature scores, and recommendations.

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
        raise NotImplementedError(
            "DriftDetector.detect() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )

    def should_retrain(self, report: DriftReport) -> bool:
        """Return ``True`` when the drift report recommends retraining.

        Decision logic (Phase 3):
          - ``severity == HIGH``                          → always retrain
          - ``severity == MEDIUM`` AND ``drift_fraction > 0.3`` → retrain
          - ``severity == LOW``                           → only monitor

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
        raise NotImplementedError(
            "DriftDetector.should_retrain() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )

    def get_feature_importance(
        self,
        report: DriftReport,
        top_n: int = 10,
    ) -> list[FeatureDriftScore]:
        """Return the top-N most drifted features sorted by drift score.

        Raises:
            NotImplementedError: Until Phase 3 implementation.
        """
        raise NotImplementedError(
            "DriftDetector.get_feature_importance() is not yet implemented. "
            "Phase 2 TDD stub — implement in Phase 3."
        )
