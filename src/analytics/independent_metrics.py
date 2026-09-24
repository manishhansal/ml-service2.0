"""
src.analytics.independent_metrics — independent metric recomputation (mandate §7, §37, §38, §73, §74).

The certification harness must NOT blindly trust metric objects emitted by the
models/backtest. This module recomputes the headline metrics *from raw
persisted predictions / trades* so the certification report can cross-check
model-reported values against an independent calculation. Any material
disagreement is a certification finding.

Everything here is deliberately simple, seed-free, and free of the training
code path so it forms a genuine second opinion:

    - information_coefficient (Spearman rank IC + Pearson IC)
    - brier_score
    - expected_calibration_error (+ bin diagnostics, empty bins reported)
    - net_consistency (net == gross - cost, per-trade and aggregate)
    - sharpe / win_rate from a net-return series

Adversarial calibration note (§37): a reported ECE of exactly 0.0 is treated as
*suspicious* unless it is corroborated here on a non-degenerate prediction set.
``ece_is_suspicious`` flags the degenerate cases (constant predictions, too few
samples, all predictions in one bin) that produce a spurious perfect ECE.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field


def _rankdata(values: Sequence[float]) -> list[float]:
    """Average-rank of values (ties share the mean rank). No SciPy dependency."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    n = len(values)
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0  # 1-based average rank
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    n = len(x)
    if n < 2:
        return 0.0
    mx = sum(x) / n
    my = sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return 0.0
    return cov / math.sqrt(vx * vy)


def pearson_ic(predictions: Sequence[float], outcomes: Sequence[float]) -> float:
    """Pearson correlation between prediction and realized outcome/return."""
    if len(predictions) != len(outcomes) or len(predictions) < 2:
        return 0.0
    return round(_pearson(list(predictions), list(outcomes)), 6)


def rank_ic(predictions: Sequence[float], outcomes: Sequence[float]) -> float:
    """Spearman rank IC — correlation of ranks (robust to outliers)."""
    if len(predictions) != len(outcomes) or len(predictions) < 2:
        return 0.0
    rp = _rankdata(list(predictions))
    ro = _rankdata(list(outcomes))
    return round(_pearson(rp, ro), 6)


def brier_score(probabilities: Sequence[float], outcomes: Sequence[float]) -> float:
    """Mean squared error between predicted probability and binary outcome."""
    n = len(probabilities)
    if n == 0 or n != len(outcomes):
        return float("nan")
    return round(sum((p - o) ** 2 for p, o in zip(probabilities, outcomes)) / n, 6)


@dataclass
class CalibrationResult:
    ece: float
    mce: float
    n_samples: int
    n_nonempty_bins: int
    bin_edges: list[float]
    bin_confidence: list[float] = field(default_factory=list)
    bin_accuracy: list[float] = field(default_factory=list)
    bin_count: list[int] = field(default_factory=list)
    ece_is_suspicious: bool = False
    suspicious_reason: str = ""

    def to_dict(self) -> dict:
        return {
            "ece": self.ece,
            "mce": self.mce,
            "n_samples": self.n_samples,
            "n_nonempty_bins": self.n_nonempty_bins,
            "ece_is_suspicious": self.ece_is_suspicious,
            "suspicious_reason": self.suspicious_reason,
            "bin_count": self.bin_count,
        }


def expected_calibration_error(
    probabilities: Sequence[float],
    outcomes: Sequence[float],
    n_bins: int = 10,
    min_samples: int = 30,
) -> CalibrationResult:
    """Independent ECE with adversarial degeneracy detection (mandate §37).

    Flags ``ece_is_suspicious`` when the inputs cannot support a meaningful ECE:
    too few samples, (near-)constant predictions, or all mass in a single bin —
    the exact conditions that manufacture a spurious ECE ≈ 0.
    """
    n = len(probabilities)
    edges = [i / n_bins for i in range(n_bins + 1)]
    if n == 0 or n != len(outcomes):
        return CalibrationResult(
            ece=float("nan"), mce=float("nan"), n_samples=n, n_nonempty_bins=0,
            bin_edges=edges, ece_is_suspicious=True, suspicious_reason="NO_SAMPLES",
        )

    counts = [0] * n_bins
    conf_sum = [0.0] * n_bins
    acc_sum = [0.0] * n_bins
    for p, o in zip(probabilities, outcomes):
        b = min(int(p * n_bins), n_bins - 1)
        b = max(0, b)
        counts[b] += 1
        conf_sum[b] += p
        acc_sum[b] += o

    bin_conf, bin_acc, ece, mce, nonempty = [], [], 0.0, 0.0, 0
    for b in range(n_bins):
        if counts[b] == 0:
            bin_conf.append(0.0)
            bin_acc.append(0.0)
            continue
        nonempty += 1
        c = conf_sum[b] / counts[b]
        a = acc_sum[b] / counts[b]
        bin_conf.append(round(c, 6))
        bin_acc.append(round(a, 6))
        gap = abs(c - a)
        ece += (counts[b] / n) * gap
        mce = max(mce, gap)

    p_min, p_max = min(probabilities), max(probabilities)
    suspicious, reason = False, ""
    if n < min_samples:
        suspicious, reason = True, f"TOO_FEW_SAMPLES({n}<{min_samples})"
    elif (p_max - p_min) < 1e-6:
        suspicious, reason = True, "CONSTANT_PREDICTIONS"
    elif nonempty <= 1:
        suspicious, reason = True, "SINGLE_POPULATED_BIN"

    return CalibrationResult(
        ece=round(ece, 6), mce=round(mce, 6), n_samples=n, n_nonempty_bins=nonempty,
        bin_edges=edges, bin_confidence=bin_conf, bin_accuracy=bin_acc,
        bin_count=counts, ece_is_suspicious=suspicious, suspicious_reason=reason,
    )


@dataclass
class NetConsistencyResult:
    n_trades: int
    n_inconsistent: int
    max_abs_error: float
    aggregate_gross: float
    aggregate_cost: float
    aggregate_net: float
    aggregate_consistent: bool

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def net_consistency(
    gross: Sequence[float],
    cost: Sequence[float],
    net: Sequence[float],
    tol: float = 1e-6,
) -> NetConsistencyResult:
    """Verify net == gross - cost per trade and in aggregate (mandate §74)."""
    n = len(gross)
    n_bad = 0
    max_err = 0.0
    for g, c, nt in zip(gross, cost, net):
        err = abs((g - c) - nt)
        max_err = max(max_err, err)
        if err > tol:
            n_bad += 1
    agg_g, agg_c, agg_n = sum(gross), sum(cost), sum(net)
    agg_ok = abs((agg_g - agg_c) - agg_n) <= max(tol, tol * n)
    return NetConsistencyResult(
        n_trades=n, n_inconsistent=n_bad, max_abs_error=round(max_err, 9),
        aggregate_gross=round(agg_g, 6), aggregate_cost=round(agg_c, 6),
        aggregate_net=round(agg_n, 6), aggregate_consistent=agg_ok,
    )


def sharpe(returns: Sequence[float], periods_per_year: int = 252) -> float:
    """Annualised Sharpe of a per-trade/per-period net-return series."""
    n = len(returns)
    if n < 2:
        return 0.0
    mean = sum(returns) / n
    var = sum((r - mean) ** 2 for r in returns) / (n - 1)
    if var <= 0:
        return 0.0
    return round((mean / math.sqrt(var)) * math.sqrt(periods_per_year), 4)


def win_rate(net_returns: Sequence[float]) -> float:
    if not net_returns:
        return 0.0
    return round(sum(1 for r in net_returns if r > 0) / len(net_returns), 4)


def compare(reported: float, independent: float, rel_tol: float = 0.05,
            abs_tol: float = 1e-3) -> dict:
    """Compare a reported metric with the independent recomputation."""
    if reported is None or independent is None:
        return {"reported": reported, "independent": independent, "agrees": False,
                "note": "missing value"}
    diff = abs(reported - independent)
    scale = max(abs(reported), abs(independent))
    agrees = diff <= max(abs_tol, rel_tol * scale)
    return {"reported": round(reported, 6), "independent": round(independent, 6),
            "abs_diff": round(diff, 6), "agrees": agrees}
