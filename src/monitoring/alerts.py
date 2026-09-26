"""
src.monitoring.alerts — Alert system for the AlphaForge ML monitoring stack.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    DISABLED = "disabled"


class AlertCategory(str, Enum):
    FEATURE_DRIFT = "feature_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PERFORMANCE = "performance"
    SYSTEM = "system"
    MODEL_REGISTRY = "model_registry"
    DATA_QUALITY = "data_quality"


@dataclass
class Alert:
    alert_id: str
    severity: AlertSeverity
    category: AlertCategory
    model_name: str
    title: str
    message: str
    timestamp: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "alertId": self.alert_id,
            "severity": self.severity.value,
            "category": self.category.value,
            "modelName": self.model_name,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class AlertSystem:
    """Stores and retrieves monitoring alerts."""

    def __init__(self, max_history: int = 10_000) -> None:
        self._history: list[Alert] = []
        self._max = max_history

    def _emit(
        self,
        severity: AlertSeverity,
        category: AlertCategory,
        model_name: str,
        title: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> Alert:
        alert = Alert(
            alert_id=str(uuid.uuid4()),
            severity=severity,
            category=category,
            model_name=model_name,
            title=title,
            message=message,
            timestamp=datetime.now(tz=UTC).isoformat(),
            metadata=metadata or {},
        )
        self._history.append(alert)
        if len(self._history) > self._max:
            self._history = self._history[-self._max:]
        return alert

    def info(self, category: AlertCategory, model_name: str, title: str,
             message: str, metadata: dict | None = None) -> Alert:
        return self._emit(AlertSeverity.INFO, category, model_name, title, message, metadata)

    def warn(self, category: AlertCategory, model_name: str, title: str,
             message: str, metadata: dict | None = None) -> Alert:
        return self._emit(AlertSeverity.WARNING, category, model_name, title, message, metadata)

    def critical(self, category: AlertCategory, model_name: str, title: str,
                 message: str, metadata: dict | None = None) -> Alert:
        return self._emit(AlertSeverity.CRITICAL, category, model_name, title, message, metadata)

    def disabled_alert(self, model_name: str, message: str) -> Alert:
        return self._emit(AlertSeverity.DISABLED, AlertCategory.MODEL_REGISTRY,
                          model_name, "Model Disabled", message)

    def get_recent(
        self,
        limit: int = 100,
        severity: AlertSeverity | None = None,
        model_name: str | None = None,
    ) -> list[Alert]:
        results = self._history
        if severity is not None:
            results = [a for a in results if a.severity == severity]
        if model_name is not None:
            results = [a for a in results if a.model_name == model_name]
        return results[-limit:]

    def get_summary(self) -> dict[str, Any]:
        counts: dict[str, int] = {s.value: 0 for s in AlertSeverity}
        for a in self._history:
            counts[a.severity.value] = counts.get(a.severity.value, 0) + 1
        return {
            "total": len(self._history),
            "counts": counts,
        }

    def clear(self) -> None:
        self._history.clear()
