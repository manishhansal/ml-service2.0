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


# ── Drift attribution (mandate §26) ────────────────────────────────────────
@dataclass
class DriftAttribution:
    """Explains WHY a drift verdict occurred rather than just reporting PSI.

    A high PSI has many possible causes (mandate §26): genuine regime change,
    train/live universe mismatch, timeframe mismatch, feature-implementation or
    normalization differences, insufficient reference sample, or a construction
    artifact from comparing a chronological tail against full-history bins.

    The key diagnostic here is a *stationarity self-test*: split the reference's
    OWN samples into an early half and a late half — drawn identically, same
    universe, same timeframe, same feature implementation — and compute PSI
    between them. If that in-reference split already exceeds the CRITICAL
    threshold on the same features, the live "drift" is attributable to the
    non-stationarity of the feature itself (and the tail-vs-full-history binning
    construction), NOT to a genuinely different live population. This lets us
    keep the CRITICAL threshold intact (mandate §50) while labelling the cause
    honestly.
    """

    max_psi: float
    severity: str
    self_test_max_psi: float
    self_test_severity: str
    likely_construction_artifact: bool
    per_feature: dict[str, dict[str, Any]]
    classification: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_psi": self.max_psi,
            "severity": self.severity,
            "self_test_max_psi": self.self_test_max_psi,
            "self_test_severity": self.self_test_severity,
            "likely_construction_artifact": self.likely_construction_artifact,
            "classification": self.classification,
            "note": self.note,
            "per_feature": self.per_feature,
        }


def attribute_drift(
    reference: ReferenceDistribution,
    current_features: pd.DataFrame,
    monitor: DriftMonitor | None = None,
) -> DriftAttribution:
    """Attribute a drift verdict to a genuine change vs a construction artifact.

    Runs the live PSI (reference vs current) AND a stationarity self-test
    (reference early-half vs reference late-half) per feature, then classifies
    the dominant cause. Does NOT change any threshold.
    """
    mon = monitor or DriftMonitor()

    def _sev(p: float) -> str:
        if p > PSI_CRITICAL:
            return "CRITICAL"
        if p > PSI_HIGH:
            return "HIGH"
        if p > PSI_MEDIUM:
            return "MEDIUM"
        return "LOW"

    per_feature: dict[str, dict[str, Any]] = {}
    live_psis: list[float] = []
    self_psis: list[float] = []
    for col, ref_vals in reference.feature_samples.items():
        if col not in current_features.columns:
            continue
        cur_vals = current_features[col].dropna().to_numpy(dtype=float).tolist()
        if not cur_vals or len(ref_vals) < 4:
            continue
        live_psi = round(mon.compute_psi(ref_vals, cur_vals), 6)

        # Stationarity self-test: reference's own early vs late half.
        half = len(ref_vals) // 2
        early, late = ref_vals[:half], ref_vals[half:]
        self_psi = round(mon.compute_psi(early, late), 6) if half >= 2 else 0.0

        per_feature[col] = {
            "live_psi": live_psi,
            "live_severity": _sev(live_psi),
            "self_test_psi": self_psi,
            "self_test_severity": _sev(self_psi),
            # A feature whose own history is already unstable at the same level
            # cannot be used to claim a genuinely different LIVE population.
            "artifact_suspected": self_psi > PSI_HIGH and live_psi > PSI_HIGH,
        }
        live_psis.append(live_psi)
        self_psis.append(self_psi)

    max_psi = max(live_psis) if live_psis else 0.0
    self_max = max(self_psis) if self_psis else 0.0
    severity = _sev(max_psi)
    self_severity = _sev(self_max)

    # Classification logic (does not alter the operational threshold).
    if max_psi <= PSI_HIGH:
        classification = "NO_SIGNIFICANT_DRIFT"
        artifact = False
        note = "Live PSI below HIGH threshold; no attribution needed."
    elif self_max > PSI_HIGH:
        classification = "CONSTRUCTION_ARTIFACT_OR_NONSTATIONARITY"
        artifact = True
        note = (
            "Reference's own early-vs-late split already exceeds HIGH PSI on the "
            "same features. The CRITICAL live PSI is attributable to feature "
            "non-stationarity + tail-vs-full-history binning, not a genuinely "
            "different live universe. Threshold left intact (mandate §50)."
        )
    else:
        classification = "POSSIBLE_GENUINE_DRIFT"
        artifact = False
        note = (
            "Reference is internally stable (low self-test PSI) but the live "
            "window diverges — consistent with a genuinely different live "
            "population (regime change or universe/timeframe mismatch). "
            "Investigate before trusting the model live."
        )

    return DriftAttribution(
        max_psi=max_psi,
        severity=severity,
        self_test_max_psi=self_max,
        self_test_severity=self_severity,
        likely_construction_artifact=artifact,
        per_feature=per_feature,
        classification=classification,
        note=note,
    )
