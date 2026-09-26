"""
src.validation.metrics — Financial ML validation metrics and result types.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


# ── Classification metrics ────────────────────────────────────────────────────

@dataclass
class ClassificationMetrics:
    accuracy: float
    f1: float
    precision: float
    recall: float
    n_samples: int
    n_classes: int
    roc_auc: float | None = None

    def to_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "n_samples": self.n_samples,
            "n_classes": self.n_classes,
            "roc_auc": self.roc_auc,
        }


@dataclass
class TradingMetrics:
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    n_bars: int
    total_return: float = 0.0

    def to_dict(self) -> dict:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "n_bars": self.n_bars,
            "total_return": self.total_return,
        }


@dataclass
class FoldResult:
    fold_index: int
    n_train: int
    n_test: int
    classification: ClassificationMetrics
    trading: TradingMetrics

    def to_dict(self) -> dict:
        return {
            "fold_index": self.fold_index,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "classification": self.classification.to_dict(),
            "trading": self.trading.to_dict(),
        }


# ── Stability analysis ────────────────────────────────────────────────────────

@dataclass
class StabilityResult:
    mean_sharpe: float
    std_sharpe: float
    min_sharpe: float
    max_sharpe: float
    mean_accuracy: float
    min_accuracy: float
    max_accuracy: float
    sharpe_decay: float
    stability_score: float
    is_dominated_by_single_period: bool
    max_fold_return_contribution: float


@dataclass
class AcceptanceThresholds:
    min_accuracy: float = 0.52
    min_sharpe: float = 0.0
    max_drawdown: float = 0.25
    min_win_rate: float = 0.0


@dataclass
class AcceptanceDecision:
    accepted: bool
    score: float
    reasons: list[str]
    thresholds_checked: dict[str, Any]


# ── ValidationResult ──────────────────────────────────────────────────────────

@dataclass
class ValidationResult:
    model_version: str
    dataset_version: str
    feature_version: str
    validation_method: str
    evaluated_at: str
    fold_results: list[FoldResult]
    aggregate_classification: ClassificationMetrics
    aggregate_trading: TradingMetrics
    stability: StabilityResult
    accepted: bool
    acceptance_reasons: list[str]

    def to_dict(self) -> dict:
        return {
            "modelVersion": self.model_version,
            "datasetVersion": self.dataset_version,
            "featureVersion": self.feature_version,
            "validationMethod": self.validation_method,
            "evaluatedAt": self.evaluated_at,
            "nFolds": len(self.fold_results),
            "accepted": self.accepted,
            "acceptanceReasons": self.acceptance_reasons,
            "aggregateClassification": self.aggregate_classification.to_dict(),
            "aggregateTrading": self.aggregate_trading.to_dict(),
            "foldResults": [f.to_dict() for f in self.fold_results],
            "stability": asdict(self.stability),
        }


# ── Evaluator ─────────────────────────────────────────────────────────────────

class FinancialMetricsEvaluator:

    def classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray | None = None,
    ) -> ClassificationMetrics:
        n = len(y_true)
        n_classes = len(np.unique(y_true))
        avg = "macro" if n_classes > 2 else "binary"
        labels = np.unique(y_true)

        acc = float(accuracy_score(y_true, y_pred))
        f1 = float(f1_score(y_true, y_pred, average=avg, zero_division=0, labels=labels))
        prec = float(precision_score(y_true, y_pred, average=avg, zero_division=0, labels=labels))
        rec = float(recall_score(y_true, y_pred, average=avg, zero_division=0, labels=labels))

        roc_auc = None
        if y_prob is not None and n_classes == 2:
            try:
                roc_auc = float(roc_auc_score(y_true, y_prob))
            except Exception:
                pass

        return ClassificationMetrics(
            accuracy=acc, f1=f1, precision=prec, recall=rec,
            n_samples=n, n_classes=n_classes, roc_auc=roc_auc,
        )

    def trading_metrics(self, returns: pd.Series) -> TradingMetrics:
        if len(returns) == 0:
            return TradingMetrics(
                sharpe_ratio=0.0, sortino_ratio=0.0, max_drawdown=0.0,
                win_rate=0.0, profit_factor=0.0, n_bars=0,
            )
        arr = returns.to_numpy(dtype=float)
        n = len(arr)
        mu = np.mean(arr)
        sigma = np.std(arr, ddof=1) if n > 1 else 0.0
        # Annualised Sharpe
        sharpe = float(mu / sigma * math.sqrt(252)) if sigma > 1e-12 else 0.0

        # Sortino: scaled semi-deviation
        # downside_dev = sqrt(mean(min(r,0)^2)) * sqrt(n_total/n_downside)
        # This correctly weights the downside by the frequency of losses,
        # giving sortino >= sharpe for positive-mean returns and a bounded
        # result relative to sharpe for mixed/negative returns.
        n_down = int(np.sum(arr < 0))
        down_base = math.sqrt(float(np.mean(np.minimum(arr, 0.0) ** 2)))
        if down_base < 1e-12 or n_down == 0:
            sortino = 0.0
        else:
            down_scaled = down_base * math.sqrt(n / n_down)
            sortino = float(mu / down_scaled * math.sqrt(252))

        equity = np.cumprod(1 + arr)
        running_max = np.maximum.accumulate(equity)
        dd = (equity - running_max) / running_max
        max_dd = float(abs(np.min(dd)))

        wins = arr[arr > 0]
        losses = arr[arr <= 0]
        win_rate = float(len(wins) / n)
        profit_factor = (
            float(np.sum(wins) / abs(np.sum(losses)))
            if len(losses) > 0 and abs(np.sum(losses)) > 1e-12
            else float(np.sum(wins)) if np.sum(wins) > 0 else 0.0
        )

        return TradingMetrics(
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            n_bars=n,
            total_return=float(np.sum(arr)),
        )

    def evaluate_fold(
        self,
        fold_index: int,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        returns: pd.Series,
        n_train: int,
        y_prob: np.ndarray | None = None,
    ) -> FoldResult:
        clf = self.classification_metrics(y_true, y_pred, y_prob)
        trd = self.trading_metrics(returns)
        return FoldResult(
            fold_index=fold_index,
            n_train=n_train,
            n_test=len(y_true),
            classification=clf,
            trading=trd,
        )


# ── Stability ─────────────────────────────────────────────────────────────────

def compute_stability(folds: list[FoldResult]) -> StabilityResult:
    if not folds:
        return StabilityResult(
            mean_sharpe=0.0, std_sharpe=0.0, min_sharpe=0.0, max_sharpe=0.0,
            mean_accuracy=0.0, min_accuracy=0.0, max_accuracy=0.0,
            sharpe_decay=0.0, stability_score=0.0,
            is_dominated_by_single_period=False, max_fold_return_contribution=0.0,
        )

    sharpes = np.array([f.trading.sharpe_ratio for f in folds])
    accs = np.array([f.classification.accuracy for f in folds])
    totals = np.array([f.trading.total_return for f in folds])

    # Sharpe decay: compare first half vs second half
    mid = len(folds) // 2
    first_half = sharpes[:mid]
    second_half = sharpes[mid:]
    decay = float(np.mean(first_half) - np.mean(second_half)) if len(first_half) and len(second_half) else 0.0

    # Stability score: 1 - cv of sharpes (capped at 0)
    cv = float(np.std(sharpes) / (abs(np.mean(sharpes)) + 1e-12))
    stability_score = max(0.0, 1.0 - cv)

    # Concentration: does one fold dominate?
    total_sum = abs(np.sum(totals))
    if total_sum > 1e-12:
        contributions = np.abs(totals) / total_sum
        max_contrib = float(np.max(contributions))
    else:
        max_contrib = 0.0

    return StabilityResult(
        mean_sharpe=float(np.mean(sharpes)),
        std_sharpe=float(np.std(sharpes)),
        min_sharpe=float(np.min(sharpes)),
        max_sharpe=float(np.max(sharpes)),
        mean_accuracy=float(np.mean(accs)),
        min_accuracy=float(np.min(accs)),
        max_accuracy=float(np.max(accs)),
        sharpe_decay=decay,
        stability_score=stability_score,
        is_dominated_by_single_period=max_contrib > 0.6,
        max_fold_return_contribution=max_contrib,
    )


def aggregate_fold_results(
    folds: list[FoldResult],
) -> tuple[ClassificationMetrics, TradingMetrics]:
    if not folds:
        return (
            ClassificationMetrics(0.0, 0.0, 0.0, 0.0, 0, 2),
            TradingMetrics(0.0, 0.0, 0.0, 0.0, 0.0, 0),
        )
    accs = [f.classification.accuracy for f in folds]
    f1s = [f.classification.f1 for f in folds]
    n_samples = sum(f.classification.n_samples for f in folds)
    n_classes = folds[0].classification.n_classes

    sharpes = [f.trading.sharpe_ratio for f in folds]
    dds = [f.trading.max_drawdown for f in folds]
    wrs = [f.trading.win_rate for f in folds]
    totals = [f.trading.total_return for f in folds]

    agg_clf = ClassificationMetrics(
        accuracy=float(np.mean(accs)),
        f1=float(np.mean(f1s)),
        precision=float(np.mean([f.classification.precision for f in folds])),
        recall=float(np.mean([f.classification.recall for f in folds])),
        n_samples=n_samples,
        n_classes=n_classes,
    )
    agg_trd = TradingMetrics(
        sharpe_ratio=float(np.mean(sharpes)),
        sortino_ratio=float(np.mean([f.trading.sortino_ratio for f in folds])),
        max_drawdown=float(np.mean(dds)),
        win_rate=float(np.mean(wrs)),
        profit_factor=float(np.mean([f.trading.profit_factor for f in folds])),
        n_bars=sum(f.trading.n_bars for f in folds),
        total_return=float(np.sum(totals)),
    )
    return agg_clf, agg_trd


# ── Acceptance gate ───────────────────────────────────────────────────────────

class ModelAcceptanceGate:
    def __init__(self, thresholds: AcceptanceThresholds | None = None) -> None:
        self.thresholds = thresholds or AcceptanceThresholds()

    def evaluate(
        self,
        folds: list[FoldResult],
        stability: StabilityResult,
        agg_clf: ClassificationMetrics,
        agg_trd: TradingMetrics,
    ) -> AcceptanceDecision:
        reasons: list[str] = []
        checked: dict[str, Any] = {}
        t = self.thresholds

        checked["accuracy"] = {"value": agg_clf.accuracy, "threshold": t.min_accuracy}
        if agg_clf.accuracy < t.min_accuracy:
            reasons.append(
                f"Accuracy {agg_clf.accuracy:.3f} < floor {t.min_accuracy:.3f}"
            )

        checked["sharpe"] = {"value": agg_trd.sharpe_ratio, "threshold": t.min_sharpe}
        if agg_trd.sharpe_ratio < t.min_sharpe:
            reasons.append(
                f"Sharpe {agg_trd.sharpe_ratio:.3f} < floor {t.min_sharpe:.3f}. "
                "Poor out-of-sample return."
            )

        checked["total_return"] = {"value": agg_trd.total_return, "threshold": 0.0}
        if agg_trd.total_return < 0:
            reasons.append(
                f"Total return {agg_trd.total_return:.4f} < 0. "
                "Negative out-of-sample return."
            )

        checked["max_drawdown"] = {"value": agg_trd.max_drawdown, "threshold": t.max_drawdown}
        if agg_trd.max_drawdown > t.max_drawdown:
            reasons.append(
                f"Max drawdown {agg_trd.max_drawdown:.3f} > limit {t.max_drawdown:.3f}"
            )

        accepted = len(reasons) == 0
        n_checks = len(checked)
        n_passed = sum(
            1 for k, v in checked.items()
            if not any(k.lower() in r.lower() for r in reasons)
        )
        score = n_passed / max(n_checks, 1)

        return AcceptanceDecision(
            accepted=accepted,
            score=float(score),
            reasons=reasons,
            thresholds_checked=checked,
        )


# ── Persistence ───────────────────────────────────────────────────────────────

def save_validation_result(result: ValidationResult, directory: Path) -> Path:
    """Save result to a unique JSON file (never overwrites)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S%f")
    safe_ver = result.model_version.replace("/", "_").replace(".", "-")
    filename = f"validation_{safe_ver}_{ts}.json"
    path = directory / filename
    path.write_text(json.dumps(result.to_dict(), indent=2))
    return path


def load_validation_history(
    directory: Path,
    model_version: str | None = None,
) -> list[dict]:
    directory = Path(directory)
    if not directory.exists():
        return []
    results = []
    for f in sorted(directory.glob("validation_*.json")):
        try:
            data = json.loads(f.read_text())
            if model_version is None or data.get("modelVersion") == model_version:
                results.append(data)
        except Exception:
            continue
    return results
