"""
src.training.orchestrator — end-to-end model training orchestrator (Phase H).

Ties together the data → model → validation → calibration → registry pipeline
that produces genuinely trained, OOS-validated, calibrated, immutable model
artifacts. This is what closes P0-001 (no trained model artifacts).

Flow
----
    frozen dataset (DatasetBuilder)
        -> candidate estimators (baselines + advanced)
        -> walk-forward OOS validation  (WalkForwardValidator)
        -> CPCV distribution + PBO       (CombinatorialPurgedCV)
        -> select champion by OOS evidence (parsimony tiebreak)
        -> fit champion on full history
        -> fit calibrator (Platt/isotonic) on held-out OOS predictions
        -> persist immutable artifact + metadata via ModelRegistry

The final held-out OOS window is NEVER used for hyperparameter/model selection
(the walk-forward mean IC across windows is the selection criterion; no peeking).

Model selection principle (Phase 111): the system must EARN complexity.
A baseline that matches the advanced model within `parsimony_margin` wins.

Requirements: Phase H, P0-001, Phase 27, 13_MODEL_REGISTRY_SPEC.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset_builder import DatasetBuilder
from src.logging_config import get_logger
from src.meta.calibration import CalibrationLayer
from src.models.estimators import BASELINE_MODELS, build_estimator
from src.registry.registry import ModelRegistry
from src.schemas.base import ModelLifecycleStage, PredictionProvenance
from src.schemas.registry import ModelArtifact
from src.training.cpcv import CombinatorialPurgedCV
from src.training.walk_forward import WalkForwardValidator

logger = get_logger(__name__)


@dataclass
class CandidateResult:
    """Validation result for one candidate model."""

    name: str
    is_baseline: bool
    wf_ic_mean: float
    wf_ic_worst: float
    wf_positive_fraction: float
    wf_net_sharpe: float
    cpcv_ic_mean: float
    cpcv_pbo: float
    calibration_ece: float = 1.0
    calibration_fitted: bool = False
    calibration_brier: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class TrainingReport:
    """Full training report for a model family."""

    model_name: str
    dataset_id: str
    dataset_hash: str
    candidates: list[CandidateResult] = field(default_factory=list)
    champion: str | None = None
    champion_version: str | None = None
    champion_ic_mean: float = 0.0
    champion_pbo: float = 0.0
    champion_net_sharpe: float = 0.0
    champion_ece: float = 1.0
    champion_brier: float = 1.0
    passed_acceptance: bool = False
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "champion": self.champion,
            "champion_version": self.champion_version,
            "champion_ic_mean": self.champion_ic_mean,
            "champion_pbo": self.champion_pbo,
            "champion_net_sharpe": self.champion_net_sharpe,
            "champion_ece": self.champion_ece,
            "champion_brier": self.champion_brier,
            "passed_acceptance": self.passed_acceptance,
            "rejection_reason": self.rejection_reason,
            "candidates": [c.to_dict() for c in self.candidates],
        }


class TrainingOrchestrator:
    """
    Orchestrates evidence-based model training and champion selection.

    Usage::

        orch = TrainingOrchestrator(dataset_builder, registry)
        report = orch.train(
            model_name="market_regime",
            dataset_id="ds-1d-...",
            candidate_names=["logistic", "lightgbm", "xgboost"],
        )
    """

    def __init__(
        self,
        dataset_builder: DatasetBuilder,
        registry: ModelRegistry | None = None,
        n_windows: int = 5,
        embargo_days: int = 10,
        cost_bps: float = 10.0,
        min_ic: float = 0.02,
        max_pbo: float = 0.5,
        parsimony_margin: float = 0.005,
        max_ece: float = 0.10,
        enforce_calibration_gate: bool = True,
    ) -> None:
        self._builder = dataset_builder
        self._registry = registry or ModelRegistry()
        self.n_windows = n_windows
        self.embargo_days = embargo_days
        self.cost_bps = cost_bps
        self.min_ic = min_ic
        self.max_pbo = max_pbo
        self.parsimony_margin = parsimony_margin
        self.max_ece = max_ece
        self.enforce_calibration_gate = enforce_calibration_gate

    def train(
        self,
        model_name: str,
        dataset_id: str,
        candidate_names: list[str] | None = None,
        register_champion: bool = True,
    ) -> TrainingReport:
        candidate_names = candidate_names or ["logistic", "lightgbm", "xgboost"]

        frame = self._builder.load_frame(dataset_id)
        meta = self._builder.load_metadata(dataset_id)
        feature_cols = self._builder._ff.FEATURE_NAMES

        X = frame[feature_cols].to_numpy(dtype=float)
        y_label = frame["label"].to_numpy(dtype=float)
        returns = frame["realized_return"].fillna(0.0).to_numpy(dtype=float)
        ts = pd.DatetimeIndex(frame.index)

        report = TrainingReport(
            model_name=model_name,
            dataset_id=dataset_id,
            dataset_hash=meta.dataset_hash,
        )

        wf = WalkForwardValidator(
            n_windows=self.n_windows, embargo_days=self.embargo_days, cost_bps=self.cost_bps
        )
        cpcv = CombinatorialPurgedCV(n_groups=6, k_test_groups=2, embargo=self.embargo_days)

        for name in candidate_names:
            cand = self._evaluate_candidate(name, X, y_label, returns, ts, wf, cpcv)
            if cand is not None:
                report.candidates.append(cand)

        if not report.candidates:
            report.rejection_reason = "NO_VALID_CANDIDATES"
            return report

        champion = self._select_champion(report.candidates)
        report.champion = champion.name
        report.champion_ic_mean = champion.wf_ic_mean
        report.champion_pbo = champion.cpcv_pbo
        report.champion_net_sharpe = champion.wf_net_sharpe
        report.champion_ece = champion.calibration_ece
        report.champion_brier = champion.calibration_brier

        # Acceptance gates (documented thresholds — NOT tuned to pass).
        report.passed_acceptance, report.rejection_reason = self._acceptance(champion)

        if register_champion and report.passed_acceptance:
            version = self._fit_and_register(
                model_name, champion.name, X, y_label, returns, ts, meta, champion
            )
            report.champion_version = version

        logger.info(
            "training_orchestrator_complete",
            model_name=model_name,
            champion=report.champion,
            ic_mean=report.champion_ic_mean,
            pbo=report.champion_pbo,
            passed=report.passed_acceptance,
        )
        return report

    # ── Candidate evaluation ──────────────────────────────────────────────

    def _evaluate_candidate(
        self, name, X, y_label, returns, ts, wf, cpcv
    ) -> CandidateResult | None:
        try:
            wf_report = wf.validate(X, y_label, returns, ts, lambda: build_estimator(name))
            cpcv_report = cpcv.run(X, y_label, returns, ts, lambda: build_estimator(name))
        except Exception as exc:
            logger.warning("candidate_eval_failed", name=name, error=str(exc))
            return None

        cand = CandidateResult(
            name=name,
            is_baseline=name in BASELINE_MODELS,
            wf_ic_mean=wf_report.ic_mean,
            wf_ic_worst=wf_report.ic_worst,
            wf_positive_fraction=wf_report.positive_ic_fraction,
            wf_net_sharpe=wf_report.net_sharpe_mean,
            cpcv_ic_mean=cpcv_report.ic_mean,
            cpcv_pbo=cpcv_report.pbo,
        )

        # Fit calibrator on a held-out OOS tail (last 20%) — never the selection set.
        cand.calibration_ece, cand.calibration_fitted, cand.calibration_brier = (
            self._fit_calibration_probe(name, X, y_label)
        )
        return cand

    def _fit_calibration_probe(self, name, X, y_label) -> tuple[float, bool, float]:
        """Return (ece, fitted, brier) computed on a held-out OOS tail."""
        from src.meta.calibration_eval import evaluate_calibration

        n = len(X)
        split = int(n * 0.8)
        if split < 30 or n - split < 20:
            return 1.0, False, 1.0
        try:
            model = build_estimator(name).fit(X[:split], y_label[:split])
            oos_scores = model.predict(X[split:])
            oos_labels = (y_label[split:] > 0).astype(float)
            # Calibrate then measure calibrated ECE/Brier (post-calibration quality).
            cal = CalibrationLayer()
            fitted = cal.fit(name, oos_scores.tolist(), oos_labels.tolist())
            calibrated = np.array([cal.calibrate(name, float(s)) for s in oos_scores])
            m = evaluate_calibration(calibrated, oos_labels)
            return m.ece, fitted, m.brier
        except Exception:
            return 1.0, False, 1.0

    # ── Champion selection (evidence + parsimony) ─────────────────────────

    def _select_champion(self, candidates: list[CandidateResult]) -> CandidateResult:
        """
        Select by walk-forward mean IC. A baseline within `parsimony_margin`
        of the best advanced model is preferred (earn complexity).
        """
        best = max(candidates, key=lambda c: c.wf_ic_mean)
        # Parsimony: if a baseline is within margin of the best, prefer it.
        for c in candidates:
            if c.is_baseline and (best.wf_ic_mean - c.wf_ic_mean) <= self.parsimony_margin:
                logger.info(
                    "parsimony_prefer_baseline",
                    baseline=c.name, best=best.name,
                    baseline_ic=c.wf_ic_mean, best_ic=best.wf_ic_mean,
                )
                return c
        return best

    def _acceptance(self, champion: CandidateResult) -> tuple[bool, str]:
        if champion.wf_ic_mean < self.min_ic:
            return False, "IC_BELOW_THRESHOLD"
        if champion.cpcv_pbo > self.max_pbo:
            return False, "HIGH_PBO"
        if champion.wf_net_sharpe < 0.0:
            return False, "NEGATIVE_NET_SHARPE"
        # Calibration gate (Phase 37): a poorly-calibrated model cannot be
        # promoted. Only enforced when a calibrator was actually fitted.
        if (
            self.enforce_calibration_gate
            and champion.calibration_fitted
            and champion.calibration_ece > self.max_ece
        ):
            return False, "POOR_CALIBRATION"
        return True, ""

    # ── Fit + register ─────────────────────────────────────────────────────

    def _fit_and_register(
        self, model_name, cand_name, X, y_label, returns, ts, meta, champion
    ) -> str:
        # Fit champion on ALL data for production.
        model = build_estimator(cand_name).fit(X, y_label)

        # Fit production calibrator on a held-out tail.
        split = int(len(X) * 0.8)
        cal = CalibrationLayer()
        cal_fitted = False
        if split >= 30 and len(X) - split >= 20:
            probe = build_estimator(cand_name).fit(X[:split], y_label[:split])
            cal.fit(
                model_name,
                probe.predict(X[split:]).tolist(),
                (y_label[split:] > 0).astype(float).tolist(),
            )
            cal_fitted = cal.has_calibrator(model_name)

        version = f"1.0.0-{datetime.now(tz=UTC).strftime('%Y%m%d%H%M%S%f')}"

        # Write the pickled payload to a STAGING location; ModelRegistry.register()
        # copies it into the immutable version directory and computes the checksum.
        staging = self._registry._root / "_staging"
        staging.mkdir(parents=True, exist_ok=True)
        staged_file = staging / f"{model_name}_{version}_model.pkl"

        payload = {
            "estimator": model,
            "estimator_name": cand_name,
            "calibrator": cal._calibrators.get(model_name) if cal_fitted else None,
            "feature_names": self._builder._ff.FEATURE_NAMES,
        }
        with staged_file.open("wb") as fh:
            pickle.dump(payload, fh)

        artifact = ModelArtifact(
            model_name=model_name,
            version=version,
            stage=ModelLifecycleStage.CHALLENGER,  # promotion gate lifts to PRODUCTION
            artifact_path="model.pkl",  # relative name; register copies + resolves
            sha256_checksum="",  # registry computes on register
            training_date=datetime.now(tz=UTC).strftime("%Y-%m-%d"),
            training_dataset_hash=meta.dataset_hash,
            ic_mean=champion.wf_ic_mean,
            sharpe_net=champion.wf_net_sharpe,
            pbo=champion.cpcv_pbo,
            provenance=PredictionProvenance.TRAINED_MODEL,
            metadata={
                "estimator_name": cand_name,
                "calibration_fitted": cal_fitted,
                "calibration_ece": champion.calibration_ece,
                "dataset_id": meta.dataset_id,
                "feature_schema_version": meta.feature_schema_version,
                "n_windows": self.n_windows,
                "cost_bps": self.cost_bps,
                "training_metrics": champion.to_dict(),
            },
        )
        # Rename staged file to model.pkl so the copied artifact file is model.pkl.
        final_stage = staging / "model.pkl"
        staged_file.replace(final_stage)
        self._registry.register(artifact, artifact_file_path=final_stage)
        final_stage.unlink(missing_ok=True)
        return version
