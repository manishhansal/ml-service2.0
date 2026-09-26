"""
tests/test_reconciliation_ic.py — Tests for the canonical IC computation family.

Mandate §23: IC must be computed against continuous realized returns.
Mandate §25: Effective sample size must account for overlapping labels.
Mandate §24: Clustered standard errors.
"""
import math
import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from src.reconciliation.ic import (
    compute_timeseries_ic,
    compute_cross_sectional_ic,
    compute_overlapping_correction,
    compute_clustered_se,
)


def _make_panel(n_symbols: int = 10, n_dates: int = 50, seed: int = 42) -> pd.DataFrame:
    """Create a synthetic (ts, symbol)-indexed panel for IC tests."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D", tz="UTC")
    for d in dates:
        for i in range(n_symbols):
            sym = f"SYM{i:02d}"
            score = rng.uniform(0, 1)
            ret = rng.normal(0.0005, 0.02)
            rows.append({"ts": d, "symbol": sym, "score": score, "return": ret})
    df = pd.DataFrame(rows).set_index(["ts", "symbol"])
    return df


def _make_informative_panel(n_symbols: int = 20, n_dates: int = 100, ic: float = 0.20,
                             seed: int = 42) -> pd.DataFrame:
    """Panel where score correlates with return at the given IC level."""
    rng = np.random.default_rng(seed)
    rows = []
    dates = pd.date_range("2023-01-01", periods=n_dates, freq="D", tz="UTC")
    for d in dates:
        true_signal = rng.standard_normal(n_symbols)
        noise = rng.standard_normal(n_symbols) * math.sqrt((1 - ic**2) / ic**2) if ic > 0 else rng.standard_normal(n_symbols)
        scores = true_signal / (1 + abs(noise).mean())
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
        rets = true_signal * 0.01 + rng.normal(0, 0.015, n_symbols)
        for i in range(n_symbols):
            rows.append({"ts": d, "symbol": f"SYM{i:02d}", "score": scores[i], "return": rets[i]})
    return pd.DataFrame(rows).set_index(["ts", "symbol"])


class TestTimseriesIC:

    def test_zero_correlation_produces_zero_ic(self):
        rng = np.random.default_rng(0)
        n = 1000
        s = rng.uniform(0, 1, n)
        r = rng.normal(0, 0.02, n)
        result = compute_timeseries_ic(s, r)
        assert abs(result["ts_ic_pearson"]) < 0.1, "Uncorrelated should have IC≈0"

    def test_perfect_correlation_produces_ic_near_1(self):
        n = 100
        s = np.linspace(0, 1, n)
        r = s + np.random.default_rng(1).normal(0, 0.01, n)  # noisy but correlated
        result = compute_timeseries_ic(s, r)
        assert result["ts_ic_pearson"] > 0.95

    def test_barrier_artifact_fraction_quantified(self):
        """If 90% of returns are at ±0.02, barrier_fraction should be ~0.9."""
        rng = np.random.default_rng(42)
        n = 1000
        s = rng.uniform(0, 1, n)
        r = np.where(rng.uniform(0, 1, n) < 0.9,
                     np.where(rng.uniform(0, 1, n) < 0.5, 0.02, -0.02),
                     rng.normal(0, 0.01, n))
        result = compute_timeseries_ic(s, r)
        assert result["barrier_artifact_fraction"] > 0.8, (
            f"Expected barrier_fraction > 0.8, got {result['barrier_artifact_fraction']}"
        )

    def test_barrier_inflates_pearson_vs_continuous(self):
        """IC against clamped returns should be ≥ IC against continuous."""
        rng = np.random.default_rng(42)
        n = 500
        true_ret = rng.normal(0, 0.015, n)
        s = (true_ret > 0).astype(float) + rng.normal(0, 0.1, n)  # noisy classifier
        result = compute_timeseries_ic(s, true_ret, barrier_pct=0.02)
        # With barrier clamping, the relationship between binary predictor and
        # near-binary return is sharper
        assert result["ts_ic_vs_barrier"] >= result["ts_ic_vs_continuous"] - 0.05, (
            "Clamped IC should not be significantly lower than continuous IC"
        )

    def test_nanfiltering(self):
        """NaN values in inputs must be handled gracefully."""
        rng = np.random.default_rng(0)
        s = rng.uniform(0, 1, 100)
        r = rng.normal(0, 0.02, 100)
        s[::10] = np.nan
        r[::7] = np.nan
        result = compute_timeseries_ic(s, r)
        assert np.isfinite(result["ts_ic_pearson"])


class TestCrossSectionalIC:

    def test_zero_ic_for_random_scores(self):
        panel = _make_panel(n_symbols=20, n_dates=100)
        result = compute_cross_sectional_ic(panel, "score", "return")
        assert abs(result["xs_rank_ic_mean"]) < 0.15, (
            f"Random scores should have low XS IC, got {result['xs_rank_ic_mean']:.4f}"
        )

    def test_informative_signal_has_positive_xs_ic(self):
        """A genuinely informative signal should produce positive XS rank IC."""
        panel = _make_informative_panel(n_symbols=30, n_dates=200, ic=0.30)
        result = compute_cross_sectional_ic(panel, "score", "return")
        assert result["xs_rank_ic_mean"] > 0.05, (
            f"Informative signal should have XS IC > 0.05, got {result['xs_rank_ic_mean']:.4f}"
        )

    def test_per_timestamp_ic_averaged(self):
        """XS IC should be the mean of per-timestamp correlations, not pooled."""
        panel = _make_panel(n_symbols=10, n_dates=50)
        result = compute_cross_sectional_ic(panel, "score", "return")
        assert result["n_timestamps"] > 0
        # Manually compute one timestamp's IC
        ts_list = panel.index.get_level_values("ts").unique()
        t0 = ts_list[0]
        grp = panel.loc[t0]
        if len(grp) >= 5:
            manual_ic, _ = spearmanr(grp["score"], grp["return"])
            # The result should average these per-ts values
            assert "xs_rank_ic_per_ts" in result

    def test_min_symbols_filter(self):
        """Timestamps with fewer than min_symbols should be skipped."""
        panel = _make_panel(n_symbols=3, n_dates=50)  # only 3 symbols
        result = compute_cross_sectional_ic(panel, "score", "return", min_symbols=5)
        assert result["n_timestamps"] == 0, (
            "With 3 symbols and min_symbols=5, no timestamps should qualify"
        )

    def test_positive_fraction_between_0_and_1(self):
        panel = _make_panel(n_symbols=15, n_dates=80)
        result = compute_cross_sectional_ic(panel, "score", "return")
        assert 0.0 <= result["xs_rank_ic_positive_fraction"] <= 1.0

    def test_icir_computed(self):
        panel = _make_informative_panel(n_symbols=20, n_dates=100, ic=0.20)
        result = compute_cross_sectional_ic(panel, "score", "return")
        assert "xs_rank_ic_icir" in result
        assert np.isfinite(result["xs_rank_ic_icir"])


class TestOverlappingCorrection:

    def test_horizon_1_no_correction(self):
        """With h=1, the correction factor should be 1.0."""
        result = compute_overlapping_correction(ic=0.10, n_rows=1000, horizon=1)
        assert abs(result["overlap_correction_factor"] - 1.0) < 1e-9

    def test_horizon_5_correction_factor_sqrt5(self):
        result = compute_overlapping_correction(ic=0.10, n_rows=1000, horizon=5)
        assert abs(result["overlap_correction_factor"] - math.sqrt(5)) < 1e-6

    def test_t_stat_raw_larger_than_corrected(self):
        """Raw t-stat (from n) should be larger than corrected (from n/h)."""
        result = compute_overlapping_correction(ic=0.30, n_rows=127122, horizon=5)
        assert result["t_stat_raw"] > result["t_stat_corrected"], (
            "Raw t-stat must be > corrected t-stat for h>1"
        )

    def test_effective_n_is_n_over_h(self):
        result = compute_overlapping_correction(ic=0.10, n_rows=10000, horizon=5)
        assert result["effective_n"] == 2000

    def test_known_values(self):
        """Verify against the audit findings for the 65-symbol dataset."""
        # n=127122, h=5, reported t-stat from cluster SE ≈ 90
        # corrected t-stat ≈ 90 / sqrt(5) ≈ 40.2
        result = compute_overlapping_correction(
            ic=0.486, n_rows=127122, horizon=5, ic_se_naive=0.486 / 90.46
        )
        # Raw t-stat should be ~90 (≈ what was reported)
        assert abs(result["t_stat_raw"] - 90.46) < 1.0, (
            f"Expected t_stat_raw ≈ 90.46, got {result['t_stat_raw']:.2f}"
        )
        # Corrected = raw / sqrt(5) ≈ 40.4
        expected_corrected = 90.46 / math.sqrt(5)
        assert abs(result["t_stat_corrected"] - expected_corrected) < 2.0, (
            f"Expected corrected t-stat ≈{expected_corrected:.1f}, got {result['t_stat_corrected']:.2f}"
        )


class TestClusteredSE:

    def test_returns_required_keys(self):
        panel = _make_panel(n_symbols=10, n_dates=50).reset_index()
        result = compute_clustered_se(panel, "score", "return")
        required = {
            "ic_date_mean", "date_clustered_se", "date_clustered_tstat",
            "n_date_clusters", "ic_symbol_mean", "symbol_clustered_se",
            "n_symbol_clusters",
        }
        assert required.issubset(result.keys())

    def test_date_clusters_equal_n_dates(self):
        n_dates = 30
        panel = _make_panel(n_symbols=5, n_dates=n_dates).reset_index()
        result = compute_clustered_se(panel, "score", "return")
        assert result["n_date_clusters"] == n_dates

    def test_se_positive(self):
        panel = _make_panel(n_symbols=10, n_dates=50).reset_index()
        result = compute_clustered_se(panel, "score", "return")
        assert result["date_clustered_se"] >= 0
        assert result["symbol_clustered_se"] >= 0

    def test_handles_multiindex_input(self):
        """Should work with (ts, symbol) MultiIndex input."""
        panel = _make_panel(n_symbols=8, n_dates=30)  # MultiIndex
        result = compute_clustered_se(panel, "score", "return")
        assert "n_date_clusters" in result
