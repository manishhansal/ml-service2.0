"""
tests/test_reconciliation_targets.py — Tests for the pre-registered target family.

Mandate §8: "Pre-register the target family before final evaluation."
Mandate §8: "Do NOT select the best target after looking at final OOS results."
"""
import numpy as np
import pandas as pd
import pytest

from src.reconciliation.targets import (
    REGISTERED_TARGETS,
    TARGET_REGISTRATION_HASH,
    TARGET_REGISTRATION_CERTIFICATE,
    compute_next_open_raw,
    compute_residual_next_open,
    compute_cross_sectional_rank,
    _compute_registration_hash,
)


def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1 + rng.normal(0.0002, 0.015, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": price * (1 + rng.normal(0, 0.003, n)),
        "high": price * (1 + np.abs(rng.normal(0, 0.01, n))),
        "low": price * (1 - np.abs(rng.normal(0, 0.01, n))),
        "close": price,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=idx)


class TestTargetRegistration:

    def test_registration_hash_is_deterministic(self):
        """Same target definitions must always produce same hash."""
        h1 = _compute_registration_hash()
        h2 = _compute_registration_hash()
        assert h1 == h2

    def test_registration_hash_matches_constant(self):
        """The published constant must match the computed hash."""
        assert TARGET_REGISTRATION_HASH == _compute_registration_hash()

    def test_six_targets_registered(self):
        """Exactly 6 pre-registered targets."""
        assert len(REGISTERED_TARGETS) == 6

    def test_primary_targets_exist(self):
        """next_open_raw and cross_sectional_rank must be primary."""
        primaries = {k for k, v in REGISTERED_TARGETS.items() if v.is_primary}
        assert "next_open_raw" in primaries
        assert "cross_sectional_rank" in primaries

    def test_all_targets_use_open_prices(self):
        """All targets must use open prices (not close) for entry and exit."""
        for name, t in REGISTERED_TARGETS.items():
            assert "open" in t.entry.lower(), f"{name}: entry must use open price"
            assert "open" in t.exit_.lower(), f"{name}: exit must use open price"

    def test_all_targets_horizon_5(self):
        """All pre-registered targets use h=5 to match the trained model."""
        for name, t in REGISTERED_TARGETS.items():
            assert t.horizon_days == 5, (
                f"{name}: horizon should be 5 (matches model), got {t.horizon_days}"
            )

    def test_certificate_contains_required_fields(self):
        cert = TARGET_REGISTRATION_CERTIFICATE
        assert "registration_hash" in cert
        assert "primary_targets" in cert
        assert "registration_timestamp" in cert


class TestNextOpenRaw:

    def test_returns_multiindex(self):
        ohlcv = {"RELIANCE": _make_ohlcv(80)}
        result = compute_next_open_raw(ohlcv, horizon=5)
        assert "ts" in result.index.names
        assert "symbol" in result.index.names

    def test_no_lookahead_in_return(self):
        """fwd_return_next_open_raw must be NaN for the last h rows."""
        ohlcv = {"RELIANCE": _make_ohlcv(50)}
        result = compute_next_open_raw(ohlcv, horizon=5)
        sym_result = result.xs("RELIANCE", level="symbol")
        # Last 6 rows (h+1) should have NaN exit price
        tail = sym_result.tail(7)
        nan_count = tail["fwd_return_next_open_raw"].isna().sum()
        assert nan_count >= 5, f"Expected ≥5 NaN at tail, got {nan_count}"

    def test_return_is_open_to_open(self):
        """Verify return = (open[T+1+h] - open[T+1]) / open[T+1]."""
        df = _make_ohlcv(30)
        ohlcv = {"TEST": df}
        result = compute_next_open_raw(ohlcv, horizon=1)
        sym = result.xs("TEST", level="symbol")
        opens = df["open"].values
        # Row i: entry = opens[i+1], exit = opens[i+2], return = (opens[i+2]-opens[i+1])/opens[i+1]
        for i in range(min(5, len(sym) - 3)):
            ts = sym.index[i]
            expected = (opens[i + 2] - opens[i + 1]) / opens[i + 1]
            actual = sym.loc[ts, "fwd_return_next_open_raw"]
            if not np.isnan(actual):
                assert abs(actual - expected) < 1e-9, (
                    f"Row {i}: expected {expected:.6f}, got {actual:.6f}"
                )

    def test_is_economic_evidence_true(self):
        ohlcv = {"A": _make_ohlcv(60)}
        result = compute_next_open_raw(ohlcv, horizon=5)
        assert result["is_economic"].all()

    def test_no_close_price_used(self):
        """The return must NOT depend on close price for barrier-contamination check."""
        df1 = _make_ohlcv(50, seed=1)
        df2 = df1.copy()
        # Change only close prices
        df2["close"] = df2["close"] * 2.0
        ohlcv1 = {"A": df1}
        ohlcv2 = {"A": df2}
        r1 = compute_next_open_raw(ohlcv1, horizon=5)
        r2 = compute_next_open_raw(ohlcv2, horizon=5)
        # Returns should be identical (open prices unchanged)
        r1_vals = r1["fwd_return_next_open_raw"].dropna()
        r2_vals = r2["fwd_return_next_open_raw"].dropna()
        pd.testing.assert_series_equal(r1_vals, r2_vals)


class TestCrossSectionalRank:

    def test_rank_is_in_01(self):
        ohlcv = {f"SYM{i}": _make_ohlcv(60, seed=i) for i in range(5)}
        result = compute_cross_sectional_rank(ohlcv, horizon=5)
        rank_vals = result["fwd_return_xs_rank"].dropna()
        assert (rank_vals >= 0).all() and (rank_vals <= 1).all()

    def test_cross_sectional_mean_rank_near_05(self):
        """Within each timestamp the mean rank should be ~0.5."""
        ohlcv = {f"SYM{i}": _make_ohlcv(80, seed=i) for i in range(10)}
        result = compute_cross_sectional_rank(ohlcv, horizon=5)
        per_ts_mean = (
            result["fwd_return_xs_rank"]
            .dropna()
            .groupby(level="ts")
            .mean()
        )
        assert abs(per_ts_mean.mean() - 0.5) < 0.1


class TestResidualNextOpen:

    def test_residual_without_benchmark_is_raw(self):
        """If benchmark not in ohlcv, residual falls back to raw return."""
        ohlcv = {"RELIANCE": _make_ohlcv(60)}
        result = compute_residual_next_open(ohlcv, benchmark_sym="NIFTY", horizon=5)
        # benchmark_available should be False
        assert not result["benchmark_available"].any()
        # Residual values should equal raw values (names may differ)
        res_vals = result["fwd_return_residual"].dropna().values
        raw_vals = result["fwd_return_next_open_raw"].dropna().values
        np.testing.assert_allclose(res_vals, raw_vals, rtol=1e-6)

    def test_residual_with_benchmark_differs_from_raw(self):
        """When benchmark is available, residual ≠ raw."""
        ohlcv = {
            "RELIANCE": _make_ohlcv(60, seed=1),
            "NIFTY": _make_ohlcv(60, seed=99),
        }
        result = compute_residual_next_open(ohlcv, benchmark_sym="NIFTY", horizon=5)
        if result["benchmark_available"].any():
            raw = result["fwd_return_next_open_raw"].dropna()
            res = result["fwd_return_residual"].dropna()
            common = raw.index.intersection(res.index)
            # They should NOT be identical when benchmark exists
            if len(common) > 0:
                assert not (raw.loc[common] == res.loc[common]).all(), (
                    "Residual should differ from raw when benchmark is subtracted"
                )
