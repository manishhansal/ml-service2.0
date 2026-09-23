"""
DriftMonitor — PSI-based feature drift detection with Evidently AI and NannyML integration.

Design:
- PSI computation over 10 equal-frequency bins from the reference distribution
- Evidently AI DataDriftPreset for comprehensive feature drift analysis
- NannyML CBPE for performance estimation on unlabeled production data
- DriftSeverity thresholds: MEDIUM (PSI > 0.2), HIGH (PSI > 0.25)

Requirements: Req 12.1 through Req 12.9
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.logging_config import get_logger
from src.schemas.base import DriftSeverity, RecommendedAction
from src.schemas.monitoring import DriftAlert

logger = get_logger(__name__)

# PSI thresholds
PSI_MEDIUM_THRESHOLD = 0.2
PSI_HIGH_THRESHOLD = 0.25

EVIDENTLY_AVAILABLE = False
try:
    import evidently  # noqa: F401

    EVIDENTLY_AVAILABLE = True
except ImportError:
    pass

NANNYML_AVAILABLE = False
try:
    import nannyml  # noqa: F401

    NANNYML_AVAILABLE = True
except ImportError:
    pass


class DriftMonitor:
    """
    PSI-based feature drift detector with Evidently AI and NannyML support.

    Usage::
        dm = DriftMonitor()
        psi = dm.compute_psi(reference_data, current_data)
        severity = dm.get_severity(psi)
    """

    def compute_psi(
        self,
        reference: list[float],
        current: list[float],
        n_bins: int = 10,
    ) -> float:
        """
        Compute Population Stability Index (PSI) between reference and current distributions.

        Algorithm (Req 12.8):
        1. Compute n_bins equal-frequency bin edges from the reference distribution.
        2. For each bin, compute the fraction of reference and current samples.
        3. PSI = Σ (actual_pct - expected_pct) × ln(actual_pct / expected_pct)

        Returns 0.0 for identical distributions (PSI(D, D) = 0).

        Args:
            reference: Reference distribution samples (training distribution).
            current:   Current distribution samples (production window).
            n_bins:    Number of equal-frequency bins (default 10 per Req 12.8).

        Returns:
            PSI as a non-negative float.
        """
        if not reference or not current:
            return 0.0

        ref_arr = np.array(reference, dtype=float)
        cur_arr = np.array(current, dtype=float)

        # Compute equal-frequency bin edges from reference (percentile-based).
        # Equal-frequency means each bin contains roughly the same number of
        # reference samples.
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(ref_arr, percentiles)

        # Make edges unique to handle constant/near-constant distributions.
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Constant distribution — no drift is computable.
            return 0.0

        n_ref = len(ref_arr)
        n_cur = len(cur_arr)
        # Small epsilon avoids log(0) when a bin is empty in one distribution.
        # Value is intentionally tiny so it does not distort the PSI signal.
        eps = 1e-10

        psi = 0.0
        n_actual_bins = len(bin_edges) - 1

        for i in range(n_actual_bins):
            lo = bin_edges[i]
            hi = bin_edges[i + 1]

            # Include the right edge only in the last bin so no sample is
            # double-counted and every sample falls in exactly one bin.
            if i == n_actual_bins - 1:
                ref_mask = (ref_arr >= lo) & (ref_arr <= hi)
                cur_mask = (cur_arr >= lo) & (cur_arr <= hi)
            else:
                ref_mask = (ref_arr >= lo) & (ref_arr < hi)
                cur_mask = (cur_arr >= lo) & (cur_arr < hi)

            expected_pct = max(float(ref_mask.sum()) / n_ref, eps)
            actual_pct = max(float(cur_mask.sum()) / n_cur, eps)

            psi += (actual_pct - expected_pct) * math.log(actual_pct / expected_pct)

        # PSI is mathematically non-negative; clamp for floating-point safety.
        return max(0.0, float(psi))

    def get_severity(self, psi: float) -> DriftSeverity:
        """
        Map a PSI value to a DriftSeverity level.

        Thresholds per Req 12.2 / 12.3:
        - PSI > 0.25         → HIGH
        - 0.2 < PSI <= 0.25  → MEDIUM
        - PSI <= 0.2         → LOW
        """
        if psi > PSI_HIGH_THRESHOLD:
            return DriftSeverity.HIGH
        elif psi > PSI_MEDIUM_THRESHOLD:
            return DriftSeverity.MEDIUM
        else:
            return DriftSeverity.LOW

    def get_recommended_action(
        self,
        psi: float,
        online_learning_triggered: bool = False,
        estimated_ic: float | None = None,
    ) -> RecommendedAction:
        """
        Determine the recommended action based on PSI and monitoring state (Req 12.9).

        Rules:
        - PSI <= 0.25                                                → MONITOR
        - PSI > 0.25 AND online learning not triggered               → RETRAIN
        - PSI > 0.25 AND online learning triggered AND IC < 0.015   → ROLLBACK
        - PSI > 0.25 AND online learning triggered AND IC >= 0.015  → RETRAIN
        """
        if psi <= PSI_HIGH_THRESHOLD:
            return RecommendedAction.MONITOR
        elif not online_learning_triggered:
            return RecommendedAction.RETRAIN
        elif estimated_ic is not None and estimated_ic < 0.015:
            return RecommendedAction.ROLLBACK
        else:
            return RecommendedAction.RETRAIN

    # ── Optional integrations ─────────────────────────────────────────────────

    def run_evidently_drift(
        self,
        reference_df: Any,
        current_df: Any,
    ) -> dict[str, Any]:
        """
        Run Evidently AI DataDriftPreset analysis.

        Returns a dict with per-feature drift detection results.
        Falls back gracefully if Evidently is not installed.
        """
        if not EVIDENTLY_AVAILABLE:
            logger.warning("evidently_not_available_using_psi_fallback")
            return {"available": False}

        try:
            from evidently.metric_preset import DataDriftPreset  # noqa: PLC0415
            from evidently.report import Report  # noqa: PLC0415

            report = Report(metrics=[DataDriftPreset()])
            report.run(reference_data=reference_df, current_data=current_df)
            return {"available": True, "result": report.as_dict()}

        except Exception as exc:
            logger.warning("evidently_drift_failed", error=str(exc))
            return {"available": False, "error": str(exc)}

    def run_nannyml_cbpe(
        self,
        reference_df: Any,
        analysis_df: Any,
        model_metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Run NannyML CBPE (Confidence-Based Performance Estimation).

        Returns estimated performance metrics without ground-truth labels.
        Falls back gracefully if NannyML is not installed.
        """
        if not NANNYML_AVAILABLE:
            logger.warning("nannyml_not_available")
            return {"available": False}

        try:
            import nannyml as nml  # noqa: PLC0415

            estimator = nml.CBPE(
                y_pred_proba=model_metadata.get("y_pred_proba_col", "y_pred_proba"),
                y_pred=model_metadata.get("y_pred_col", "y_pred"),
                y_true=model_metadata.get("y_true_col", "y_true"),
                metrics=["roc_auc"],
                chunk_size=model_metadata.get("chunk_size", 100),
            )
            estimator.fit(reference_df)
            results = estimator.estimate(analysis_df)
            return {"available": True, "estimated_auc": results.to_df().to_dict()}

        except Exception as exc:
            logger.warning("nannyml_cbpe_failed", error=str(exc))
            return {"available": False, "error": str(exc)}

    def create_drift_alert(
        self,
        model_name: str,
        feature_name: str,
        psi: float,
        reference_mean: float,
        reference_std: float,
        current_mean: float,
        current_std: float,
        online_learning_triggered: bool = False,
        estimated_ic: float | None = None,
    ) -> DriftAlert:
        """Create a structured DriftAlert from PSI and distribution statistics."""
        severity = self.get_severity(psi)
        action = self.get_recommended_action(psi, online_learning_triggered, estimated_ic)

        return DriftAlert(
            alert_id=str(uuid.uuid4()),
            model_name=model_name,
            feature_name=feature_name,
            severity=severity,
            psi_value=psi,
            reference_mean=reference_mean,
            reference_std=reference_std,
            current_mean=current_mean,
            current_std=current_std,
            recommended_action=action,
            triggered_at=datetime.now(tz=timezone.utc),
            online_learning_triggered=online_learning_triggered,
        )
