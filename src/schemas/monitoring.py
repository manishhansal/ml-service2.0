"""
Monitoring schemas for ml-service2.0.

DriftAlert — Structured alert emitted by the DriftMonitor on feature
             distribution shift or estimated performance degradation.

Severity thresholds (PSI-based):
  LOW    — PSI <= 0.2
  MEDIUM — 0.2 < PSI <= 0.25 → MONITOR
  HIGH   — PSI > 0.25         → RETRAIN or ROLLBACK (online learning may trigger)

Performance alerts (NannyML CBPE-based):
  estimated IC < 0.015 → PERFORMANCE_DEGRADATION_ALERT → blocks model from MetaDecisionEngine
"""
from __future__ import annotations

from datetime import datetime

from src.schemas.base import BaseSchema, DriftSeverity, RecommendedAction


class DriftAlert(BaseSchema):
    """Structured alert produced by DriftMonitor for a single model/feature pair.

    Emitted by ``GET /monitoring/alerts`` and stored in Redis under
    ``drift:latest`` (TTL=86400s).  ``online_learning_triggered`` is set to
    ``True`` when severity=HIGH and the OnlineLearner has been activated in
    response to this alert.
    """

    alert_id: str  # UUID generated at alert creation time
    model_name: str  # e.g. "market_regime", "stock_ranker"
    feature_name: str  # the drifted feature or "estimated_ic" for performance alerts
    severity: DriftSeverity
    psi_value: float  # computed via DriftMonitor.compute_psi()
    reference_mean: float  # mean of the training reference distribution
    reference_std: float  # std dev of the training reference distribution
    current_mean: float  # mean of the 7-day rolling production distribution
    current_std: float  # std dev of the 7-day rolling production distribution
    recommended_action: RecommendedAction  # MONITOR | RETRAIN | ROLLBACK
    triggered_at: datetime  # UTC timestamp when alert was raised
    online_learning_triggered: bool = False  # True when OnlineLearner was activated
