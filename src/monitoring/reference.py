"""
src.monitoring.reference — training reference distributions + drift gate (Phase P).

At training time we snapshot the reference distributions (feature values,
prediction scores) into an immutable JSON artifact. At inference/monitoring
time we compute rolling PSI of the production window against these references
and map the drift severity to an operational action:

    LOW      -> MONITOR
    MEDIUM   -> ALERT
    HIGH     -> TRAIN_CHALLENGER
    CRITICAL -> BLOCK (downgrade affected model to validated fallback)

Do not blindly retrain on every PSI uptick (Phase 51): only HIGH triggers a
challenger, and only CRITICAL blocks.

Requirements: Phase P, Phase 50, Phase 51, Phase 52, 17_MONITORING_SPEC.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.logging_config import get_logger
from src.monitoring.drift_monitor import DriftMonitor

logger = get_logger(__name__)

# PSI severity thresholds (extends DriftMonitor with a CRITICAL tier).
PSI_MEDIUM = 0.20
PSI_HIGH = 0.25
PSI_CRITICAL = 0.40


class DriftAction(str, Enum):
    MONITOR = "MONITOR"
    ALERT = "ALERT"
    TRAIN_CHALLENGER = "TRAIN_CHALLENGER"
    BLOCK = "BLOCK"


@dataclass
class ReferenceDistribution:
    """Snapshot of a model's training-time reference distributions."""

    model_name: str
    version: str
    feature_samples: dict[str, list[float]]      # feature -> reference values
    prediction_samples: list[float]              # reference prediction scores
    created_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "feature_samples": self.feature_samples,
            "prediction_samples": self.prediction_samples,
            "created_at": self.created_at,
        }


class ReferenceDistributionStore:
    """Persists and loads training reference distributions."""

    def __init__(self, root: Path) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, model_name: str, version: str) -> Path:
        d = self._root / model_name
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{version}_reference.json"

    def save(self, ref: ReferenceDistribution) -> Path:
        path = self._path(ref.model_name, ref.version)
        path.write_text(json.dumps(ref.to_dict()))
        logger.info("reference_distribution_saved", model_name=ref.model_name, version=ref.version)
        return path

    def load(self, model_name: str, version: str) -> ReferenceDistribution | None:
        path = self._path(model_name, version)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return ReferenceDistribution(**raw)

    @staticmethod
    def from_training(
        model_name: str,
        version: str,
        feature_frame: pd.DataFrame,
        predictions: np.ndarray,
        max_samples: int = 5000,
    ) -> ReferenceDistribution:
        """Build a reference snapshot from a training feature frame + predictions."""
        feats: dict[str, list[float]] = {}
        for col in feature_frame.columns:
            vals = feature_frame[col].dropna().to_numpy(dtype=float)
            if len(vals) > max_samples:
                idx = np.linspace(0, len(vals) - 1, max_samples).astype(int)
                vals = vals[idx]
            feats[col] = vals.tolist()
        preds = np.asarray(predictions, dtype=float)
        if len(preds) > max_samples:
            idx = np.linspace(0, len(preds) - 1, max_samples).astype(int)
            preds = preds[idx]
        return ReferenceDistribution(
            model_name=model_name, version=version,
            feature_samples=feats, prediction_samples=preds.tolist(),
        )


@dataclass
class DriftGateResult:
    """Result of evaluating drift against reference distributions."""

    max_psi: float
    severity: str
    action: str
    feature_psi: dict[str, float]
    prediction_psi: float
    drifted_features: list[str]

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class DriftGate:
    """
    Evaluates production data against training reference distributions and maps
    the worst drift to an operational action.
    """

    def __init__(self, monitor: DriftMonitor | None = None) -> None:
        self._monitor = monitor or DriftMonitor()

    def evaluate(
        self,
        reference: ReferenceDistribution,
        current_features: pd.DataFrame,
        current_predictions: np.ndarray | None = None,
    ) -> DriftGateResult:
        feature_psi: dict[str, float] = {}
        for col, ref_vals in reference.feature_samples.items():
            if col not in current_features.columns:
                continue
            cur_vals = current_features[col].dropna().to_numpy(dtype=float).tolist()
            if not cur_vals:
                continue
            feature_psi[col] = round(self._monitor.compute_psi(ref_vals, cur_vals), 6)

        prediction_psi = 0.0
        if current_predictions is not None and reference.prediction_samples:
            prediction_psi = round(
                self._monitor.compute_psi(
                    reference.prediction_samples,
                    np.asarray(current_predictions, dtype=float).tolist(),
                ),
                6,
            )

        all_psi = list(feature_psi.values()) + [prediction_psi]
        max_psi = max(all_psi) if all_psi else 0.0
        severity = self._severity(max_psi)
        action = self._action(severity)
        drifted = [f for f, p in feature_psi.items() if p > PSI_MEDIUM]

        logger.info(
            "drift_gate_evaluated",
            model_name=reference.model_name,
            max_psi=max_psi, severity=severity, action=action,
            n_drifted=len(drifted),
        )
        return DriftGateResult(
            max_psi=round(max_psi, 6),
            severity=severity,
            action=action,
            feature_psi=feature_psi,
            prediction_psi=prediction_psi,
            drifted_features=drifted,
        )

    @staticmethod
    def _severity(psi: float) -> str:
        if psi > PSI_CRITICAL:
            return "CRITICAL"
        if psi > PSI_HIGH:
            return "HIGH"
        if psi > PSI_MEDIUM:
            return "MEDIUM"
        return "LOW"

    @staticmethod
    def _action(severity: str) -> str:
        return {
            "LOW": DriftAction.MONITOR.value,
            "MEDIUM": DriftAction.ALERT.value,
            "HIGH": DriftAction.TRAIN_CHALLENGER.value,
            "CRITICAL": DriftAction.BLOCK.value,
        }[severity]
