"""
Feature Quality Gate — Phase 3D.

Evaluates the quality of a computed feature matrix and produces
diagnostics reports.  Does NOT delete or filter features automatically;
generates diagnostics only.

Quality checks performed
------------------------
1.  Missingness     — % of NaN / None values per feature
2.  Infinities      — % of inf / -inf values
3.  Constants       — features with zero variance (or near-zero)
4.  Outliers        — % of values beyond 5 IQR from median
5.  Scale           — mean absolute value (cross-stock scale comparison)
6.  PIT safety      — cross-reference against registry pit_safety field
7.  Availability    — what fraction of feature rows have OK status
8.  Deprecated      — flag features that are still in use but deprecated
9.  Unregistered    — features not found in FEATURE_REGISTRY
10. Redundancy      — groups of features with pairwise |Pearson| > 0.95

All checks use only the data provided; no external data is fetched.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from .registry import FEATURE_REGISTRY
from .schemas import AvailabilityStatus, FeaturePromotion, PITSafety


# ── Result dataclasses ────────────────────────────────────────────────────────

@dataclass
class FeatureQualityDiagnostic:
    """Quality metrics for a single feature column."""
    feature_name:      str
    n_total:           int
    n_nan:             int
    n_inf:             int
    n_finite:          int
    missing_pct:       float    # NaN + inf as % of total
    mean:              Optional[float]
    std:               Optional[float]
    median:            Optional[float]
    iqr:               Optional[float]
    min_val:           Optional[float]
    max_val:           Optional[float]
    outlier_pct:       float    # % beyond 5*IQR from median
    zero_pct:          float    # % exactly zero
    is_constant:       bool     # std < 1e-10 or all identical
    pit_safety:        str      # from registry
    is_deprecated:     bool
    is_unregistered:   bool
    availability_ok_pct: Optional[float]  # % rows with OK status (if status_map provided)
    warnings:          list[str] = field(default_factory=list)


@dataclass
class RedundancyGroup:
    """A group of features with very high pairwise correlation."""
    features:    list[str]
    max_corr:    float
    correlation_type: str  # "pearson" or "spearman"


@dataclass
class FeatureQualityReport:
    """
    Complete quality report for a feature matrix.

    Produced by run_quality_gate().  Contains per-feature diagnostics,
    redundancy groups, and overall pass/fail verdict.

    Does NOT remove features — caller decides what to do with findings.
    """
    n_features:        int
    n_rows:            int
    n_missing_above_threshold: int   # features with missing_pct > threshold
    n_constant:        int
    n_deprecated_in_use: int
    n_unregistered:    int
    n_pit_unsafe:      int
    diagnostics:       dict[str, FeatureQualityDiagnostic]
    redundancy_groups: list[RedundancyGroup]
    verdict:           str   # "PASS" | "WARN" | "FAIL"
    missing_threshold: float
    constant_threshold: float
    notes:             list[str] = field(default_factory=list)


# ── Main quality gate ─────────────────────────────────────────────────────────

def run_quality_gate(
    feature_df: pd.DataFrame,
    status_df: Optional[pd.DataFrame] = None,
    missing_threshold: float = 0.30,
    constant_threshold: float = 1e-8,
    redundancy_corr_threshold: float = 0.95,
    compute_redundancy: bool = True,
) -> FeatureQualityReport:
    """
    Run the full feature quality gate on a feature matrix.

    Parameters
    ----------
    feature_df : DataFrame where rows are observations, columns are features.
                 Values should be float; NaN indicates unavailable/missing.
    status_df  : Optional DataFrame with same shape as feature_df, values are
                 AvailabilityStatus strings.  Used to compute availability_ok_pct.
    missing_threshold    : Features with missing_pct > this trigger FAIL.
    constant_threshold   : std < this is treated as constant.
    redundancy_corr_threshold : |Pearson| > this groups features as redundant.
    compute_redundancy   : If False, skip the O(N²) redundancy check.

    Returns
    -------
    FeatureQualityReport
    """
    cols = list(feature_df.columns)
    n_rows = len(feature_df)
    diagnostics: dict[str, FeatureQualityDiagnostic] = {}

    for col in cols:
        series = feature_df[col].astype(float)
        arr    = series.to_numpy(dtype=float)

        n_nan  = int(np.isnan(arr).sum())
        n_inf  = int(np.isinf(arr).sum())
        n_fin  = n_rows - n_nan - n_inf
        missing_pct = (n_nan + n_inf) / n_rows * 100 if n_rows > 0 else 0.0

        finite = arr[np.isfinite(arr)]
        if len(finite) > 0:
            mean   = float(np.mean(finite))
            std    = float(np.std(finite))
            median = float(np.median(finite))
            q1, q3 = float(np.percentile(finite, 25)), float(np.percentile(finite, 75))
            iqr    = q3 - q1
            min_v  = float(np.min(finite))
            max_v  = float(np.max(finite))
            outlier_fence = 5 * iqr
            n_out  = int(np.sum(np.abs(finite - median) > outlier_fence)) if iqr > 0 else 0
            outlier_pct = n_out / len(finite) * 100
            zero_pct = float((finite == 0.0).sum()) / len(finite) * 100
            is_constant = std < constant_threshold
        else:
            mean = std = median = iqr = min_v = max_v = None
            outlier_pct = zero_pct = 0.0
            is_constant = True

        # Registry checks
        spec = FEATURE_REGISTRY.get(col)
        pit_safety = spec.pit_safety.value if spec else PITSafety.UNVERIFIED.value
        is_deprecated = (
            spec.promotion == FeaturePromotion.DEPRECATED if spec else False
        )
        is_unregistered = spec is None

        # Availability OK %
        av_ok_pct = None
        if status_df is not None and col in status_df.columns:
            ok_mask = status_df[col] == AvailabilityStatus.OK.value
            av_ok_pct = float(ok_mask.sum()) / n_rows * 100 if n_rows > 0 else 0.0

        # Warnings
        warnings: list[str] = []
        if missing_pct > missing_threshold * 100:
            warnings.append(f"HIGH_MISSING: {missing_pct:.1f}%")
        if is_constant:
            warnings.append("CONSTANT: zero variance")
        if n_inf > 0:
            warnings.append(f"INF_VALUES: {n_inf} rows")
        if is_deprecated:
            warnings.append("DEPRECATED: use replacement feature")
        if is_unregistered:
            warnings.append("UNREGISTERED: not in FEATURE_REGISTRY")
        if pit_safety == PITSafety.UNSAFE.value:
            warnings.append("PIT_UNSAFE: uses future data")
        if pit_safety == PITSafety.UNVERIFIED.value:
            warnings.append("PIT_UNVERIFIED: timing not established")
        if zero_pct > 80 and is_deprecated:
            warnings.append(f"HIGH_ZERO: {zero_pct:.1f}% — likely silent default")

        diagnostics[col] = FeatureQualityDiagnostic(
            feature_name=col,
            n_total=n_rows,
            n_nan=n_nan,
            n_inf=n_inf,
            n_finite=n_fin,
            missing_pct=missing_pct,
            mean=mean,
            std=std,
            median=median,
            iqr=iqr,
            min_val=min_v,
            max_val=max_v,
            outlier_pct=outlier_pct,
            zero_pct=zero_pct,
            is_constant=is_constant,
            pit_safety=pit_safety,
            is_deprecated=is_deprecated,
            is_unregistered=is_unregistered,
            availability_ok_pct=av_ok_pct,
            warnings=warnings,
        )

    # Aggregate counts
    n_miss_above   = sum(1 for d in diagnostics.values() if d.missing_pct > missing_threshold * 100)
    n_constant     = sum(1 for d in diagnostics.values() if d.is_constant)
    n_deprecated   = sum(1 for d in diagnostics.values() if d.is_deprecated)
    n_unreg        = sum(1 for d in diagnostics.values() if d.is_unregistered)
    n_pit_unsafe   = sum(1 for d in diagnostics.values()
                         if d.pit_safety == PITSafety.UNSAFE.value)

    # Redundancy check
    redundancy_groups: list[RedundancyGroup] = []
    if compute_redundancy and len(cols) > 1:
        redundancy_groups = _find_redundancy_groups(
            feature_df, cols, redundancy_corr_threshold
        )

    # Verdict
    notes: list[str] = []
    if n_pit_unsafe > 0:
        verdict = "FAIL"
        notes.append(f"{n_pit_unsafe} features have PITSafety.UNSAFE — cannot train.")
    elif n_constant > len(cols) * 0.3:
        verdict = "FAIL"
        notes.append(f"{n_constant}/{len(cols)} features are constant — likely data issue.")
    elif n_miss_above > len(cols) * 0.5:
        verdict = "WARN"
        notes.append(f"{n_miss_above}/{len(cols)} features exceed missing threshold.")
    elif n_deprecated > 0 or n_unreg > 0:
        verdict = "WARN"
        notes.append(
            f"{n_deprecated} deprecated features in use; {n_unreg} unregistered."
        )
    else:
        verdict = "PASS"

    return FeatureQualityReport(
        n_features=len(cols),
        n_rows=n_rows,
        n_missing_above_threshold=n_miss_above,
        n_constant=n_constant,
        n_deprecated_in_use=n_deprecated,
        n_unregistered=n_unreg,
        n_pit_unsafe=n_pit_unsafe,
        diagnostics=diagnostics,
        redundancy_groups=redundancy_groups,
        verdict=verdict,
        missing_threshold=missing_threshold,
        constant_threshold=constant_threshold,
        notes=notes,
    )


def _find_redundancy_groups(
    df: pd.DataFrame,
    cols: list[str],
    threshold: float,
) -> list[RedundancyGroup]:
    """
    Find groups of features with pairwise |Pearson| >= threshold.

    Uses finite-value masking per pair (not listwise deletion).
    O(N²) over features — acceptable for N < 200.
    """
    groups: list[RedundancyGroup] = []
    assigned: set[str] = set()

    corr_cache: dict[tuple[str, str], float] = {}
    for i, c1 in enumerate(cols):
        for c2 in cols[i + 1:]:
            key = (c1, c2)
            arr1 = df[c1].to_numpy(dtype=float)
            arr2 = df[c2].to_numpy(dtype=float)
            mask = np.isfinite(arr1) & np.isfinite(arr2)
            if mask.sum() < 10:
                continue
            try:
                r = float(np.corrcoef(arr1[mask], arr2[mask])[0, 1])
            except Exception:
                r = 0.0
            if math.isfinite(r):
                corr_cache[key] = abs(r)

    # Union-find style grouping
    parent: dict[str, str] = {c: c for c in cols}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        parent[find(x)] = find(y)

    max_corrs: dict[tuple[str, str], float] = {}

    for (c1, c2), r in corr_cache.items():
        if r >= threshold:
            key = (find(c1), find(c2))
            max_corrs[key] = max(max_corrs.get(key, 0.0), r)
            union(c1, c2)

    # Build groups from union-find
    group_map: dict[str, list[str]] = {}
    for c in cols:
        root = find(c)
        if root not in group_map:
            group_map[root] = []
        group_map[root].append(c)

    for root, members in group_map.items():
        if len(members) > 1:
            # Max corr in this group
            max_r = 0.0
            for i, c1 in enumerate(members):
                for c2 in members[i + 1:]:
                    k = (c1, c2) if (c1, c2) in corr_cache else (c2, c1)
                    max_r = max(max_r, corr_cache.get(k, 0.0))
            groups.append(RedundancyGroup(
                features=sorted(members),
                max_corr=round(max_r, 4),
                correlation_type="pearson",
            ))

    return sorted(groups, key=lambda g: -g.max_corr)


def summarise_quality_report(report: FeatureQualityReport) -> str:
    """Return a concise human-readable summary."""
    lines = [
        f"Feature Quality Report: {report.verdict}",
        f"  Features: {report.n_features} | Rows: {report.n_rows}",
        f"  High-missing: {report.n_missing_above_threshold}",
        f"  Constant:     {report.n_constant}",
        f"  Deprecated:   {report.n_deprecated_in_use}",
        f"  Unregistered: {report.n_unregistered}",
        f"  PIT unsafe:   {report.n_pit_unsafe}",
        f"  Redundancy groups: {len(report.redundancy_groups)}",
    ]
    if report.notes:
        lines.append("  Notes:")
        for note in report.notes:
            lines.append(f"    - {note}")
    if report.redundancy_groups:
        lines.append("  Redundant groups (|r| >= threshold):")
        for grp in report.redundancy_groups[:5]:
            lines.append(f"    {grp.features} max_r={grp.max_corr:.3f}")
    return "\n".join(lines)
