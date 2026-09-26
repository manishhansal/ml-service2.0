"""
src.monitoring.model_registry — Model lifecycle registry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


class ModelState(str, Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    DISABLED = "disabled"


_STATE_WEIGHTS = {
    ModelState.HEALTHY: 1.0,
    ModelState.WARNING: 0.75,
    ModelState.DEGRADED: 0.30,
    ModelState.DISABLED: 0.0,
}


@dataclass
class RetrainingRecommendation:
    model_name: str
    current_state: ModelState
    reasons: list[str]
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    acknowledged: bool = False
    acknowledged_by: str | None = None
    notes: str = ""


@dataclass
class StateTransition:
    from_state: ModelState
    to_state: ModelState
    reasons: list[str]
    timestamp: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())


@dataclass
class ModelRecord:
    model_name: str
    model_version: str
    dataset_version: str = ""
    feature_version: str = ""
    training_period: str = ""
    deployment_date: str = ""
    validation_metrics: dict[str, float] = field(default_factory=dict)
    state: ModelState = ModelState.HEALTHY
    consecutive_warnings: int = 0
    retraining_recommendations: list[RetrainingRecommendation] = field(default_factory=list)
    state_history: list[StateTransition] = field(default_factory=list)

    @property
    def weight_multiplier(self) -> float:
        return _STATE_WEIGHTS[self.state]

    def to_dict(self) -> dict[str, Any]:
        return {
            "modelName": self.model_name,
            "modelVersion": self.model_version,
            "datasetVersion": self.dataset_version,
            "featureVersion": self.feature_version,
            "trainingPeriod": self.training_period,
            "deploymentDate": self.deployment_date,
            "validationMetrics": self.validation_metrics,
            "state": self.state.value,
            "weightMultiplier": self.weight_multiplier,
            "consecutiveWarnings": self.consecutive_warnings,
        }


class ModelRegistry:
    def __init__(self, alert_system=None, warn_after_n: int = 3) -> None:
        self._models: dict[str, ModelRecord] = {}
        self._alerts = alert_system
        self._warn_after_n = warn_after_n

    def register(self, record: ModelRecord) -> None:
        self._models[record.model_name] = record

    def is_registered(self, name: str) -> bool:
        return name in self._models

    def get(self, name: str) -> ModelRecord | None:
        return self._models.get(name)

    def get_weight(self, name: str) -> float:
        rec = self._models.get(name)
        return rec.weight_multiplier if rec else 1.0

    def set_warning(self, name: str, reasons: list[str]) -> ModelState:
        rec = self._models.get(name)
        if rec is None:
            return ModelState.HEALTHY
        if rec.state == ModelState.DISABLED:
            return ModelState.DISABLED
        rec.consecutive_warnings += 1
        rec.state_history.append(StateTransition(rec.state, ModelState.WARNING, reasons))
        if rec.consecutive_warnings >= self._warn_after_n:
            rec.state = ModelState.DEGRADED
            rec.retraining_recommendations.append(RetrainingRecommendation(
                model_name=name, current_state=ModelState.DEGRADED,
                reasons=reasons + [f"escalated after {rec.consecutive_warnings} warnings"],
            ))
        else:
            rec.state = ModelState.WARNING
        return rec.state

    def set_degraded(self, name: str, reasons: list[str]) -> ModelState:
        rec = self._models.get(name)
        if rec is None:
            return ModelState.HEALTHY
        rec.state_history.append(StateTransition(rec.state, ModelState.DEGRADED, reasons))
        rec.state = ModelState.DEGRADED
        rec.retraining_recommendations.append(RetrainingRecommendation(
            model_name=name, current_state=ModelState.DEGRADED, reasons=reasons,
        ))
        return ModelState.DEGRADED

    def set_disabled(self, name: str, reasons: list[str]) -> ModelState:
        rec = self._models.get(name)
        if rec is None:
            return ModelState.HEALTHY
        rec.state_history.append(StateTransition(rec.state, ModelState.DISABLED, reasons))
        rec.state = ModelState.DISABLED
        if self._alerts:
            from src.monitoring.alerts import AlertCategory
            self._alerts.disabled_alert(name, "; ".join(reasons))
        return ModelState.DISABLED

    def recover(self, name: str, notes: str = "") -> ModelState:
        rec = self._models.get(name)
        if rec is None:
            return ModelState.HEALTHY
        rec.state_history.append(StateTransition(rec.state, ModelState.HEALTHY, ["recovery"]))
        rec.state = ModelState.HEALTHY
        rec.consecutive_warnings = 0
        # Emit an INFO alert so tests can observe the recovery
        if self._alerts is not None:
            from src.monitoring.alerts import AlertCategory
            self._alerts.info(
                AlertCategory.MODEL_REGISTRY, name,
                "Model Recovered",
                f"Model '{name}' has been recovered to HEALTHY state. {notes}",
            )
        return ModelState.HEALTHY

    def get_all_weights(self) -> dict[str, float]:
        """Return {model_name: weight} for every registered model."""
        return {name: rec.weight_multiplier for name, rec in self._models.items()}

    def get_active_models(self) -> list[str]:
        return [n for n, r in self._models.items() if r.state != ModelState.DISABLED]

    def get_disabled_models(self) -> list[str]:
        return [n for n, r in self._models.items() if r.state == ModelState.DISABLED]

    def get_retraining_recommendations(
        self, unacknowledged_only: bool = False
    ) -> list[RetrainingRecommendation]:
        recs = []
        for rec in self._models.values():
            for r in rec.retraining_recommendations:
                if unacknowledged_only and r.acknowledged:
                    continue
                recs.append(r)
        return recs

    def acknowledge_recommendation(self, name: str, acknowledged_by: str = "") -> int:
        rec = self._models.get(name)
        if rec is None:
            return 0
        count = 0
        for r in rec.retraining_recommendations:
            if not r.acknowledged:
                r.acknowledged = True
                r.acknowledged_by = acknowledged_by
                count += 1
        return count

    def health_summary(self) -> dict[str, Any]:
        state_counts: dict[str, int] = {s.value: 0 for s in ModelState}
        for rec in self._models.values():
            state_counts[rec.state.value] += 1
        return {
            "totalModels": len(self._models),
            "stateCounts": state_counts,
            "models": {n: r.to_dict() for n, r in self._models.items()},
            "activeModels": self.get_active_models(),
            "disabledModels": self.get_disabled_models(),
        }


def build_default_registry(alert_system=None) -> ModelRegistry:
    """Build a registry pre-populated with all AlphaForge model names."""
    try:
        from src.meta.ensemble import MODEL_NAMES
    except Exception:
        MODEL_NAMES = [
            "regime", "ranker", "strategy", "risk",
            "price_forecaster", "iv_regime", "rl_execution",
        ]
    reg = ModelRegistry(alert_system=alert_system)
    for name in MODEL_NAMES:
        reg.register(ModelRecord(
            model_name=name,
            model_version=f"{name}-v1",
            dataset_version="ds-default",
            feature_version="fv4",
        ))
    return reg
