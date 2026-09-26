"""
src.monitoring.feature_monitor — Per-feature drift monitoring.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.monitoring.drift_detector import DriftDetector, DriftResult, DriftSeverity


@dataclass
class FeatureDriftReport:
    model_name: str
    n_features: int
    n_drifted: int
    overall_severity: DriftSeverity
    per_feature: dict[str, dict]
    timestamp: str = ""


class FeatureMonitor:
    """Monitors feature distributions per model and triggers reports."""

    def __init__(
        self,
        alert_system=None,
        model_registry=None,
        window_size: int = 50,
        min_check_size: int = 10,
    ) -> None:
        self._alerts = alert_system
        self._registry = model_registry
        self._window_size = window_size
        self._min_check = min_check_size
        self._detector = DriftDetector()
        # buffer: model_name -> {feature -> list[float]}
        self._buffers: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        self._last_report: dict[str, FeatureDriftReport] = {}

    def set_reference(self, model_name: str, feature_name: str, data: np.ndarray) -> None:
        key = f"{model_name}::{feature_name}"
        self._detector.set_reference(key, data)

    def set_references_bulk(self, model_name: str, refs: dict[str, np.ndarray]) -> None:
        for fname, data in refs.items():
            self.set_reference(model_name, fname, data)

    def monitored_features(self, model_name: str) -> list[str]:
        prefix = f"{model_name}::"
        return [k[len(prefix):] for k in self._detector._refs if k.startswith(prefix)]

    def observe(self, model_name: str, feature_vector: dict[str, float]) -> None:
        buf = self._buffers[model_name]
        for fname, val in feature_vector.items():
            key = f"{model_name}::{fname}"
            if self._detector.has_reference(key):
                buf[fname].append(float(val))
        # Auto-trigger when window fills
        any_filled = any(len(v) >= self._window_size for v in buf.values())
        if any_filled:
            self._run_check(model_name)

    def observe_batch(self, model_name: str, vectors: list[dict[str, float]]) -> None:
        for v in vectors:
            self.observe(model_name, v)

    def force_check(self, model_name: str) -> FeatureDriftReport | None:
        buf = self._buffers[model_name]
        if not buf or all(len(v) < self._min_check for v in buf.values()):
            return None
        return self._run_check(model_name)

    def get_last_report(self, model_name: str) -> FeatureDriftReport | None:
        return self._last_report.get(model_name)

    def get_buffer_fill(self, model_name: str) -> dict[str, int]:
        return {k: len(v) for k, v in self._buffers[model_name].items()}

    def _run_check(self, model_name: str) -> FeatureDriftReport:
        from datetime import UTC, datetime
        buf = self._buffers[model_name]
        per_feature: dict[str, dict] = {}
        worst = DriftSeverity.NONE
        n_drifted = 0

        for fname, vals in buf.items():
            if len(vals) < self._min_check:
                continue
            key = f"{model_name}::{fname}"
            result = self._detector.detect(key, np.array(vals, dtype=float))
            if result is None:
                continue
            per_feature[fname] = result.to_dict()
            if result.severity == DriftSeverity.MAJOR:
                worst = DriftSeverity.MAJOR
                n_drifted += 1
            elif result.severity == DriftSeverity.MINOR and worst != DriftSeverity.MAJOR:
                worst = DriftSeverity.MINOR
                n_drifted += 1

        report = FeatureDriftReport(
            model_name=model_name,
            n_features=len(per_feature),
            n_drifted=n_drifted,
            overall_severity=worst,
            per_feature=per_feature,
            timestamp=datetime.now(tz=UTC).isoformat(),
        )
        self._last_report[model_name] = report

        # Propagate to registry
        if self._registry is not None and worst == DriftSeverity.MAJOR:
            self._registry.set_warning(model_name, reasons=[f"Major feature drift: {n_drifted} features"])

        # Clear buffers after check
        for fname in buf:
            buf[fname].clear()

        return report
