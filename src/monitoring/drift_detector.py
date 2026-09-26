"""
src.monitoring.drift_detector — Statistical drift detection (PSI, KS, JS).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats


class DriftSeverity(str, Enum):
    NONE = "none"
    MINOR = "minor"
    MAJOR = "major"


@dataclass
class DriftResult:
    feature_name: str
    psi: float
    ks_statistic: float
    ks_p_value: float
    js_divergence: float
    ref_mean: float
    cur_mean: float
    ref_std: float
    cur_std: float
    ref_n: int
    cur_n: int
    severity: DriftSeverity
    has_drift: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "severity": self.severity.value,
            "has_drift": self.has_drift,
            "psi": self.psi,
            "ks_statistic": self.ks_statistic,
            "ks_p_value": self.ks_p_value,
            "js_divergence": self.js_divergence,
            "ref_mean": self.ref_mean,
            "cur_mean": self.cur_mean,
        }


_PSI_MINOR = 0.10
_PSI_MAJOR = 0.20
_N_BINS = 10
_EPS = 1e-9


def _compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = _N_BINS) -> float:
    """Population Stability Index."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    all_data = np.concatenate([ref, cur])
    bins = np.percentile(all_data, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    ref_counts = np.histogram(ref, bins=bins)[0].astype(float)
    cur_counts = np.histogram(cur, bins=bins)[0].astype(float)
    ref_pct = (ref_counts + _EPS) / (len(ref) + _EPS * len(ref_counts))
    cur_pct = (cur_counts + _EPS) / (len(cur) + _EPS * len(cur_counts))
    psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
    return max(0.0, psi)


def _compute_js_divergence(reference: np.ndarray, current: np.ndarray, n_bins: int = _N_BINS) -> float:
    """Jensen-Shannon divergence (0–1 scale)."""
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    all_data = np.concatenate([ref, cur])
    bins = np.percentile(all_data, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    if len(bins) < 2:
        return 0.0
    p = np.histogram(ref, bins=bins)[0].astype(float) + _EPS
    q = np.histogram(cur, bins=bins)[0].astype(float) + _EPS
    p /= p.sum()
    q /= q.sum()
    m = 0.5 * (p + q)
    js = 0.5 * np.sum(p * np.log(p / m)) + 0.5 * np.sum(q * np.log(q / m))
    return float(np.clip(js, 0.0, 1.0))


def detect_drift(feature_name: str, reference: np.ndarray, current: np.ndarray) -> DriftResult:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    psi = _compute_psi(ref, cur)
    ks_stat, ks_p = stats.ks_2samp(ref, cur)
    js = _compute_js_divergence(ref, cur)

    if psi >= _PSI_MAJOR:
        severity = DriftSeverity.MAJOR
    elif psi >= _PSI_MINOR:
        severity = DriftSeverity.MINOR
    else:
        severity = DriftSeverity.NONE

    return DriftResult(
        feature_name=feature_name,
        psi=psi,
        ks_statistic=float(ks_stat),
        ks_p_value=float(ks_p),
        js_divergence=js,
        ref_mean=float(np.mean(ref)),
        cur_mean=float(np.mean(cur)),
        ref_std=float(np.std(ref)),
        cur_std=float(np.std(cur)),
        ref_n=len(ref),
        cur_n=len(cur),
        severity=severity,
        has_drift=severity != DriftSeverity.NONE,
    )


class DriftDetector:
    """Stateful detector that holds reference distributions per feature."""

    def __init__(self) -> None:
        self._refs: dict[str, np.ndarray] = {}

    def set_reference(self, feature_name: str, data: np.ndarray) -> None:
        self._refs[feature_name] = np.asarray(data, dtype=float)

    def has_reference(self, feature_name: str) -> bool:
        return feature_name in self._refs

    def detect(self, feature_name: str, current: np.ndarray) -> DriftResult | None:
        if feature_name not in self._refs:
            return None
        return detect_drift(feature_name, self._refs[feature_name], current)

    def detect_all(self, current_map: dict[str, np.ndarray]) -> dict[str, DriftResult]:
        results = {}
        for name, data in current_map.items():
            r = self.detect(name, data)
            if r is not None:
                results[name] = r
        return results

    @staticmethod
    def prediction_distribution_summary(
        actions: list[str],
        confidences: list[float],
    ) -> dict:
        """
        Summarise prediction action distribution and confidence statistics.

        Returns {} when actions is empty.
        """
        if not actions:
            return {}

        from collections import Counter
        counts = Counter(actions)
        total = len(actions)
        ratios = {k: v / total for k, v in counts.items()}

        buy_count = counts.get("BUY", 0)
        sell_count = counts.get("SELL", 0)
        buy_sell_ratio = (
            buy_count / sell_count if sell_count > 0 else float("inf")
        )

        conf_arr = np.asarray(confidences, dtype=float)
        result: dict = {
            "counts": dict(counts),
            "ratios": ratios,
            "total": total,
            "buy_sell_ratio": buy_sell_ratio,
        }
        if len(conf_arr) > 0:
            result["confidence_mean"] = float(np.mean(conf_arr))
            result["confidence_std"] = float(np.std(conf_arr))
            result["confidence_p10"] = float(np.percentile(conf_arr, 10))
            result["confidence_p90"] = float(np.percentile(conf_arr, 90))
        return result
