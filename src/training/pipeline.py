"""
TrainingPipeline — model training orchestration with purged K-fold CV.

Orchestrates:
1. Data ingestion and PIT validation
2. Purged K-fold CV via PurgedKFoldSplitter
3. Model acceptance gate (IC >= 0.02, Sharpe >= 0.0, PBO <= 0.5)
4. SDLC lifecycle enforcement
5. Audit log entries for every training run

Requirements: Req 3.1, Req 3.3, Req 3.4, Req 3.5, Req 3.9, Req 3.10
"""
from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.audit.logger import AuditLogger
from src.config import settings
from src.logging_config import get_logger
from src.schemas.base import ModelLifecycleStage
from src.schemas.registry import TrainingRunStatus
from src.training.purged_kfold import PurgedKFoldSplitter

logger = get_logger(__name__)

# 10 basis points per round-trip trade
TRANSACTION_COST_BPS: float = 10.0
TRANSACTION_COST_DECIMAL: float = TRANSACTION_COST_BPS / 10_000.0

# SDLC stage ordering — used to validate forward-only transitions
_LIFECYCLE_ORDER: list[ModelLifecycleStage] = [
    ModelLifecycleStage.HYPOTHESIS,
    ModelLifecycleStage.BACKTEST,
    ModelLifecycleStage.CHALLENGER,
    ModelLifecycleStage.SHADOW,
    ModelLifecycleStage.APPROVED,
    ModelLifecycleStage.PRODUCTION,
]


class TrainingAbortedError(Exception):
    """Raised when training is aborted due to a PIT violation or data integrity failure."""


class TrainingResult:
    """Mutable result object populated by ``TrainingPipeline.run_training``."""

    def __init__(self) -> None:
        self.run_id: str = str(uuid.uuid4())
        self.model_name: str = ""
        self.model_version: str = ""
        self.ic_per_fold: list[float] = []
        self.ic_mean: float = 0.0
        self.sharpe_net: float = 0.0
        self.max_drawdown: float = 0.0
        self.pbo: float = 0.0
        self.gate_results: dict[str, str] = {}
        self.outcome: str = ""
        self.rejection_reason: str = ""
        self.dataset_hash: str = ""
        self.hyperparameters: dict[str, Any] = {}
        self.lifecycle_stage: ModelLifecycleStage = ModelLifecycleStage.BACKTEST
        self.started_at: datetime = datetime.now(tz=timezone.utc)
        self.completed_at: datetime | None = None
        self.training_date_range: tuple[str, str] = ("", "")
        self.validation_date_range: tuple[str, str] = ("", "")


class TrainingPipeline:
    """
    Orchestrates model training with purged K-fold CV and acceptance gates.

    The pipeline implements the SDLC enforcement defined in the design:
    HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION.

    A model that passes all three acceptance gates (IC, Sharpe, PBO) advances
    from BACKTEST to CHALLENGER.  A model that fails any gate remains at
    BACKTEST with a documented rejection reason.

    Usage::

        pipeline = TrainingPipeline()
        result = pipeline.run_training(
            model_name="market_regime",
            X=feature_matrix,
            y=label_vector,
            timestamps=pd.DatetimeIndex([...]),
            model_factory=lambda params: XGBClassifier(**params),
            hyperparameters={"max_depth": 5, "learning_rate": 0.05},
        )
        if result.outcome == "PASSED":
            print(f"Challenger ready: IC={result.ic_mean:.4f}")
    """

    def __init__(
        self,
        audit_logger: AuditLogger | None = None,
        n_splits: int = 5,
        embargo_days: int | None = None,
        n_cpcv_paths: int | None = None,
    ) -> None:
        self._audit = audit_logger or AuditLogger()
        self.n_splits = n_splits
        self.embargo_days = embargo_days if embargo_days is not None else settings.embargo_period_days
        self.n_cpcv_paths = n_cpcv_paths if n_cpcv_paths is not None else settings.cpcv_n_paths
        self._run_statuses: dict[str, TrainingRunStatus] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def run_training(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
        model_factory: Callable[[dict[str, Any]], Any],
        search_space: dict[str, Any] | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ) -> TrainingResult:
        """
        Run a complete training cycle with purged K-fold CV and acceptance gates.

        Steps:
        1. Compute dataset hash.
        2. Run Optuna HPO (if ``search_space`` provided and no explicit
           ``hyperparameters``).  Minimum 50 trials enforced.
        3. Evaluate the chosen hyperparameters with ``PurgedKFoldSplitter``
           (``self.n_splits`` folds, ``self.embargo_days`` embargo).
        4. Compute per-fold Spearman IC and net Sharpe (after 10bp tx cost).
        5. Estimate PBO as the fraction of folds with negative Sharpe.
        6. Apply acceptance gates:
           - IC gate: mean IC < ``settings.model_acceptance_min_ic`` → REJECTED
             (reason: ``IC_BELOW_THRESHOLD``)
           - Sharpe gate: net Sharpe < 0.0 → REJECTED
             (reason: ``NEGATIVE_NET_SHARPE``)
           - PBO gate: PBO > ``settings.model_acceptance_max_pbo`` → REJECTED
             (reason: ``HIGH_PBO``)
        7. Advance lifecycle stage to CHALLENGER on success.
        8. Write immutable audit log entry.

        Args:
            model_name:       Model family identifier (e.g. ``"market_regime"``).
            X:                2-D feature matrix, shape ``(n_samples, n_features)``.
            y:                Target label vector, shape ``(n_samples,)``.
            timestamps:       PIT-correct timestamps aligned with ``X`` and ``y``.
            model_factory:    Callable that accepts a hyperparameter dict and returns
                              a fitted-able model (must expose ``.fit`` and ``.predict``).
            search_space:     Optuna search-space definition.  When provided *and*
                              ``hyperparameters`` is ``None``, HPO is run automatically.
            hyperparameters:  Explicit hyperparameter dict.  Takes precedence over
                              ``search_space`` when both are provided.

        Returns:
            ``TrainingResult`` populated with metrics, gate results, and outcome.

        Raises:
            TrainingAbortedError: if ``X`` and ``y`` are misaligned or timestamps
                length does not match the sample count.
        """
        if len(X) != len(y) or len(X) != len(timestamps):
            raise TrainingAbortedError(
                f"X ({len(X)}), y ({len(y)}), and timestamps ({len(timestamps)}) "
                "must have the same length."
            )

        result = TrainingResult()
        result.model_name = model_name
        result.model_version = (
            f"1.0.0-{datetime.now(tz=timezone.utc).strftime('%Y%m%d')}"
        )
        result.training_date_range = (
            str(timestamps.min().date()),
            str(timestamps.max().date()),
        )
        result.validation_date_range = result.training_date_range

        logger.info(
            "training_pipeline_started",
            run_id=result.run_id,
            model_name=model_name,
            n_samples=len(X),
            n_features=X.shape[1] if X.ndim > 1 else 1,
        )

        # 1. Dataset fingerprint
        result.dataset_hash = hashlib.sha256(
            X.tobytes() + y.tobytes()
        ).hexdigest()

        # 2. Hyperparameter resolution (HPO or explicit)
        best_params = self._resolve_hyperparameters(
            model_name=model_name,
            X=X,
            y=y,
            timestamps=timestamps,
            model_factory=model_factory,
            search_space=search_space,
            hyperparameters=hyperparameters,
        )
        result.hyperparameters = best_params

        # 3-5. K-fold evaluation
        fold_ics, fold_sharpes = self._evaluate_folds(
            X=X,
            y=y,
            timestamps=timestamps,
            model_factory=model_factory,
            params=best_params,
        )

        result.ic_per_fold = fold_ics
        result.ic_mean = float(np.mean(fold_ics)) if fold_ics else 0.0
        result.sharpe_net = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
        result.max_drawdown = self._compute_max_drawdown(fold_sharpes)
        result.pbo = self._compute_pbo(fold_sharpes)

        # 6. Acceptance gates
        self._apply_gates(result)

        # 7. SDLC stage
        if result.outcome == "PASSED":
            result.lifecycle_stage = ModelLifecycleStage.CHALLENGER

        result.completed_at = datetime.now(tz=timezone.utc)

        # 8. Audit log
        self._write_audit_entry(result)

        logger.info(
            "training_pipeline_completed",
            run_id=result.run_id,
            model_name=model_name,
            outcome=result.outcome,
            ic_mean=round(result.ic_mean, 4),
            sharpe_net=round(result.sharpe_net, 4),
            pbo=round(result.pbo, 4),
            rejection_reason=result.rejection_reason or None,
        )

        return result

    def get_status(self, run_id: str) -> TrainingRunStatus | None:
        """Return the status of a tracked training run, or ``None`` if not found."""
        return self._run_statuses.get(run_id)

    @staticmethod
    def validate_lifecycle_transition(
        from_stage: ModelLifecycleStage,
        to_stage: ModelLifecycleStage,
    ) -> bool:
        """
        Return ``True`` iff ``from_stage → to_stage`` is a valid SDLC transition.

        Valid transitions are strictly forward-only:
        HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION.

        A model may also remain in the same stage (no-op transition).
        """
        from_idx = _LIFECYCLE_ORDER.index(from_stage)
        to_idx = _LIFECYCLE_ORDER.index(to_stage)
        return to_idx >= from_idx

    # ── Private helpers ───────────────────────────────────────────────────────

    def _resolve_hyperparameters(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
        model_factory: Callable[[dict[str, Any]], Any],
        search_space: dict[str, Any] | None,
        hyperparameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Return final hyperparameters, running HPO if necessary."""
        if hyperparameters is not None:
            return dict(hyperparameters)

        if search_space is not None:
            return self._run_hpo(
                model_name=model_name,
                X=X,
                y=y,
                timestamps=timestamps,
                model_factory=model_factory,
                search_space=search_space,
            )

        return {}

    def _run_hpo(
        self,
        model_name: str,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
        model_factory: Callable[[dict[str, Any]], Any],
        search_space: dict[str, Any],
    ) -> dict[str, Any]:
        """Run Optuna HPO with minimum 50 trials, returning best params."""
        from src.training.hpo import HyperparameterOptimizer, suggest_params_from_space

        splitter = PurgedKFoldSplitter(
            n_splits=self.n_splits,
            embargo_days=self.embargo_days,
        )

        def objective(trial: Any) -> float:
            params = suggest_params_from_space(trial, search_space)
            fold_ics: list[float] = []
            for tr_idx, val_idx in splitter.split(X, y, timestamps):
                model = model_factory(params)
                model.fit(X[tr_idx], y[tr_idx])
                preds = model.predict(X[val_idx])
                ic, _ = spearmanr(preds, y[val_idx])
                fold_ics.append(float(ic) if not np.isnan(ic) else 0.0)
            return float(np.mean(fold_ics)) if fold_ics else 0.0

        optimizer = HyperparameterOptimizer(model_name)
        best_params, best_value = optimizer.optimize(
            objective, n_trials=settings.optuna_n_trials
        )

        logger.info(
            "hpo_complete",
            model_name=model_name,
            best_ic=round(best_value, 4),
            best_params=best_params,
        )
        return best_params

    def _evaluate_folds(
        self,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: pd.DatetimeIndex,
        model_factory: Callable[[dict[str, Any]], Any],
        params: dict[str, Any],
    ) -> tuple[list[float], list[float]]:
        """
        Evaluate ``params`` across all purged K-folds.

        Returns:
            (fold_ics, fold_sharpes) — per-fold Spearman IC and net Sharpe values.
        """
        splitter = PurgedKFoldSplitter(
            n_splits=self.n_splits,
            embargo_days=self.embargo_days,
        )

        fold_ics: list[float] = []
        fold_sharpes: list[float] = []

        for fold_idx, (tr_idx, val_idx) in enumerate(
            splitter.split(X, y, timestamps)
        ):
            model = model_factory(params)
            model.fit(X[tr_idx], y[tr_idx])
            preds = model.predict(X[val_idx])

            # Spearman IC
            ic_val, _ = spearmanr(preds, y[val_idx])
            ic = float(ic_val) if not np.isnan(ic_val) else 0.0
            fold_ics.append(ic)

            # Net Sharpe with 10bp round-trip transaction cost
            # Simplified: predicted signal × actual return, minus cost
            returns = preds * y[val_idx]
            net_returns = returns - TRANSACTION_COST_DECIMAL
            std = float(np.std(net_returns))
            sharpe = (
                float(np.mean(net_returns)) / (std + 1e-8) * np.sqrt(252)
                if std > 0
                else 0.0
            )
            fold_sharpes.append(sharpe)

            logger.debug(
                "training_fold_complete",
                fold=fold_idx,
                n_train=len(tr_idx),
                n_val=len(val_idx),
                ic=round(ic, 4),
                sharpe=round(sharpe, 4),
            )

        return fold_ics, fold_sharpes

    @staticmethod
    def _compute_max_drawdown(fold_sharpes: list[float]) -> float:
        """
        Approximate max drawdown from fold-level Sharpe ratios.

        Converts each Sharpe to a daily return proxy (Sharpe / sqrt(252))
        and computes the peak-to-trough decline over the cumulative series.
        """
        if not fold_sharpes:
            return 0.0

        # Approximate daily return from annualised Sharpe
        daily_returns = np.array([s / np.sqrt(252) for s in fold_sharpes])
        cum = np.cumprod(1.0 + daily_returns)
        running_max = np.maximum.accumulate(cum)
        drawdowns = (cum - running_max) / (running_max + 1e-8)
        return float(np.min(drawdowns))

    @staticmethod
    def _compute_pbo(fold_sharpes: list[float]) -> float:
        """
        Estimate Probability of Backtest Overfitting (PBO).

        Simplified estimate: fraction of folds with negative Sharpe.
        A PBO > 0.5 means the majority of evaluation windows had negative
        risk-adjusted returns after transaction costs.
        """
        if not fold_sharpes:
            return 0.0
        n_negative = sum(1 for s in fold_sharpes if s < 0)
        return float(n_negative / len(fold_sharpes))

    def _apply_gates(self, result: TrainingResult) -> None:
        """
        Evaluate the three model-acceptance gates in order and record results.

        Gate order (fail-fast):
        1. IC gate  — mean IC must be >= ``settings.model_acceptance_min_ic``
        2. Sharpe gate — net Sharpe must be >= 0.0
        3. PBO gate — PBO must be <= ``settings.model_acceptance_max_pbo``

        Modifies ``result`` in place: sets ``gate_results``, ``outcome``,
        and ``rejection_reason``.
        """
        # IC gate
        if result.ic_mean < settings.model_acceptance_min_ic:
            result.gate_results = {"IC": "FAIL", "SHARPE": "SKIP", "PBO": "SKIP"}
            result.outcome = "REJECTED"
            result.rejection_reason = "IC_BELOW_THRESHOLD"
            logger.warning(
                "training_gate_failed",
                gate="IC",
                ic_mean=round(result.ic_mean, 4),
                threshold=settings.model_acceptance_min_ic,
            )
            return

        # Sharpe gate
        if result.sharpe_net < 0.0:
            result.gate_results = {"IC": "PASS", "SHARPE": "FAIL", "PBO": "SKIP"}
            result.outcome = "REJECTED"
            result.rejection_reason = "NEGATIVE_NET_SHARPE"
            logger.warning(
                "training_gate_failed",
                gate="SHARPE",
                sharpe_net=round(result.sharpe_net, 4),
            )
            return

        # PBO gate
        if result.pbo > settings.model_acceptance_max_pbo:
            result.gate_results = {"IC": "PASS", "SHARPE": "PASS", "PBO": "FAIL"}
            result.outcome = "REJECTED"
            result.rejection_reason = "HIGH_PBO"
            logger.warning(
                "training_gate_failed",
                gate="PBO",
                pbo=round(result.pbo, 4),
                threshold=settings.model_acceptance_max_pbo,
            )
            return

        # All gates passed
        result.gate_results = {"IC": "PASS", "SHARPE": "PASS", "PBO": "PASS"}
        result.outcome = "PASSED"
        result.rejection_reason = ""

    def _write_audit_entry(self, result: TrainingResult) -> None:
        """Write an immutable training-run audit log entry.  Failures are logged but not re-raised."""
        try:
            self._audit.log_training_run(
                run_id=result.run_id,
                model_name=result.model_name,
                model_version=result.model_version,
                started_at=result.started_at.isoformat(),
                completed_at=(
                    result.completed_at.isoformat()
                    if result.completed_at
                    else datetime.now(tz=timezone.utc).isoformat()
                ),
                dataset_hash=result.dataset_hash,
                training_date_range=result.training_date_range,
                validation_date_range=result.validation_date_range,
                hyperparameters=result.hyperparameters,
                ic_per_fold=result.ic_per_fold,
                ic_mean=result.ic_mean,
                sharpe_net=result.sharpe_net,
                max_drawdown=result.max_drawdown,
                pbo=result.pbo,
                gate_results=result.gate_results,
                outcome=result.outcome,
            )
        except Exception as exc:
            logger.warning(
                "audit_log_write_failed",
                run_id=result.run_id,
                error=str(exc),
            )
