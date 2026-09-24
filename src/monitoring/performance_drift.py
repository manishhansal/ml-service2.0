"""
src.monitoring.performance_drift — rolling performance-drift tracker (Phase 52).

Tracks predicted probability vs realized outcome over a rolling window and
compares against the model's baseline (champion) metrics. Detects IC decay,
Brier degradation, and calibration drift so degradation can trigger a
challenger or downgrade.

Requirements: Phase 52, Phase 69, 17_MONITORING_SPEC.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from src.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class PerformanceDriftStatus:
    """Rolling performance-drift assessment."""

    n: int
    rolling_ic: float
    rolling_brier: float
    baseline_ic: float
    ic_degradation_pct: float
    degraded: bool
    action: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class PerformanceDriftTracker:
    """
    Maintains a rolling window of (predicted_prob, realized_outcome) pairs and
    detects performance degradation relative to a baseline IC.

    Usage::

        tracker = PerformanceDriftTracker(baseline_ic=0.05, window=100)
        for pred, outcome in stream:
            tracker.record(pred, outcome)
        status = tracker.assess()
    """

    def __init__(
        self,
        baseline_ic: float,
        window: int = 100,
        min_samples: int = 30,
        degradation_threshold: float = 0.20,
    ) -> None:
        self.baseline_ic = baseline_ic
        self.window = window
        self.min_samples = min_samples
        self.degradation_threshold = degradation_threshold
        self._preds: deque[float] = deque(maxlen=window)
        self._outcomes: deque[float] = deque(maxlen=window)

    def record(self, predicted_prob: float, realized_outcome: float) -> None:
        self._preds.append(float(predicted_prob))
        self._outcomes.append(float(realized_outcome))

    def assess(self) -> PerformanceDriftStatus:
        n = len(self._preds)
        if n < self.min_samples:
            return PerformanceDriftStatus(
                n=n, rolling_ic=0.0, rolling_brier=1.0,
                baseline_ic=self.baseline_ic, ic_degradation_pct=0.0,
                degraded=False, action="INSUFFICIENT_DATA",
            )

        preds = np.array(self._preds)
        outs = np.array(self._outcomes)

        if np.std(preds) < 1e-12 or np.std(outs) < 1e-12:
            rolling_ic = 0.0
        else:
            r, _ = spearmanr(preds, outs)
            rolling_ic = float(r) if np.isfinite(r) else 0.0

        rolling_brier = float(np.mean((preds - outs) ** 2))

        if self.baseline_ic > 0:
            degradation = (self.baseline_ic - rolling_ic) / self.baseline_ic
        else:
            degradation = 0.0

        degraded = degradation > self.degradation_threshold
        action = "TRIGGER_CHALLENGER" if degraded else "MONITOR"

        if degraded:
            logger.warning(
                "performance_degradation_detected",
                rolling_ic=round(rolling_ic, 4),
                baseline_ic=round(self.baseline_ic, 4),
                degradation_pct=round(degradation * 100, 2),
            )

        return PerformanceDriftStatus(
            n=n,
            rolling_ic=round(rolling_ic, 6),
            rolling_brier=round(rolling_brier, 6),
            baseline_ic=round(self.baseline_ic, 6),
            ic_degradation_pct=round(degradation * 100, 4),
            degraded=degraded,
            action=action,
        )
