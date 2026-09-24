"""
src.meta.calibration_eval — calibration quality metrics + gate (Phase I / 37).

Computes the calibration diagnostics required by the certification:
    - Brier score
    - log loss
    - ECE (Expected Calibration Error, 10 equal-width bins)
    - MCE (Maximum Calibration Error)
    - reliability curve (bin confidences vs bin accuracies)

Provides ``calibration_gate`` which enforces the promotion requirement that a
model with poor calibration (ECE above threshold) cannot become production.

Requirements: Phase I, Phase 36, Phase 37, 13_MODEL_REGISTRY_SPEC.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.logging_config import get_logger

logger = get_logger(__name__)

DEFAULT_MAX_ECE = 0.10   # production calibration gate (mandate Phase 36/37)
DEFAULT_MAX_BRIER = 0.25


@dataclass
class CalibrationMetrics:
    """Calibration diagnostics for a set of predicted probabilities."""

    brier: float
    log_loss: float
    ece: float
    mce: float
    n: int
    reliability: list[dict[str, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "brier": self.brier,
            "log_loss": self.log_loss,
            "ece": self.ece,
            "mce": self.mce,
            "n": self.n,
            "reliability": self.reliability,
        }


def evaluate_calibration(
    predictions: np.ndarray | list[float],
    labels: np.ndarray | list[float],
    n_bins: int = 10,
) -> CalibrationMetrics:
    """
    Compute calibration metrics for predicted probabilities against binary labels.

    Args:
        predictions: predicted probabilities in [0, 1].
        labels:      binary outcomes (0/1).
        n_bins:      number of equal-width reliability bins.
    """
    p = np.clip(np.asarray(predictions, dtype=float), 1e-7, 1 - 1e-7)
    y = np.asarray(labels, dtype=float)
    n = len(p)
    if n == 0:
        return CalibrationMetrics(brier=1.0, log_loss=1.0, ece=1.0, mce=1.0, n=0)

    brier = float(np.mean((p - y) ** 2))
    log_loss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    mce = 0.0
    reliability: list[dict[str, float]] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p <= hi) if i == n_bins - 1 else (p >= lo) & (p < hi)
        count = int(mask.sum())
        if count == 0:
            continue
        avg_conf = float(p[mask].mean())
        avg_acc = float(y[mask].mean())
        gap = abs(avg_conf - avg_acc)
        ece += (count / n) * gap
        mce = max(mce, gap)
        reliability.append({
            "bin_lo": round(lo, 3),
            "bin_hi": round(hi, 3),
            "count": count,
            "avg_confidence": round(avg_conf, 4),
            "avg_accuracy": round(avg_acc, 4),
        })

    return CalibrationMetrics(
        brier=round(brier, 6),
        log_loss=round(log_loss, 6),
        ece=round(float(ece), 6),
        mce=round(float(mce), 6),
        n=n,
        reliability=reliability,
    )


def calibration_gate(
    metrics: CalibrationMetrics,
    max_ece: float = DEFAULT_MAX_ECE,
    max_brier: float = DEFAULT_MAX_BRIER,
) -> tuple[bool, str]:
    """
    Enforce the production calibration gate.

    Returns (passed, reason). A model that fails this gate CANNOT be promoted
    to production — the gate must not be bypassed.
    """
    if metrics.n < 30:
        return False, "INSUFFICIENT_CALIBRATION_SAMPLES"
    if metrics.ece > max_ece:
        return False, f"ECE_TOO_HIGH ({metrics.ece:.4f} > {max_ece})"
    if metrics.brier > max_brier:
        return False, f"BRIER_TOO_HIGH ({metrics.brier:.4f} > {max_brier})"
    return True, "CALIBRATION_OK"
