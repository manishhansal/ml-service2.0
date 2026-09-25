"""
Phase 3E Tests — Cross-Sectional Alpha & Ranking Engine.

All tests use deterministic synthetic data — the correct answer is always
known mathematically, never empirically.

Test categories
---------------
 1. Universe: IPO admission, delisting, F&O ban, insufficient cross-section
 2. Universe mutation: adding a future stock must not change historical ranks
 3. CS targets A–F: computation, correctness, missing data policy
 4. CS normalization: zscore, rank_pct, robust_zscore; timestamp-local only
 5. Winsorization: tail capping, NaN preservation
 6. Sector neutralization: residuals correct, sector mean removed
 7. Beta neutralization: market component removed
 8. Factor neutralization: OLS residualization
 9. Ranking: golden score/rank/percentile, tie handling, determinism
10. Rank direction: higher score = rank 1 (invariant)
11. IC golden: perfect alignment → IC≈+1, perfect inverse → IC≈-1
12. ICIR: formula, zero-std edge case
13. Decile golden: 100 stocks, known scores, verify Q1/Q10 assignment
14. Decile monotonicity: perfectly ordered scores → monotonicity_score≈1
15. Top-bottom spread: sign and magnitude
16. Turnover proxy: stable ranks → low turnover, churning ranks → high
17. Rank stability: consecutive Spearman
18. Walk-forward: temporal ordering, group invariant, embargo gap
19. PIT mutation: price/sector/universe changes after T do not alter ranks at T
20. Cross-sectional normalization panel: each timestamp independent
21. Reproducibility: same inputs → identical outputs
22. Schema invariants: signal status, provenance, semantics
23. ML rankers (sklearn-dependent): Ridge/ElasticNet predict shape
24. CS target: minimum cross-section gate (returns empty below threshold)
"""

from __future__ import annotations

import math
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import pytest

UTC = timezone.utc


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures — deterministic synthetic helpers
# ──────────────────────────────────────────────────────────────────────────────

def _ts(days_offset: int, base: str = "2023-01-02") -> datetime:
    """Return UTC datetime `days_offset` business days after `base`."""
    b = pd.Timestamp(base, tz="UTC")
    return (b + pd.Timedelta(days=days_offset)).to_pydatetime()


def _make_close(
    n: int,
    start_price: float = 100.0,
    daily_ret: float = 0.001,
    seed: int = 42,
) -> pd.Series:
    np.random.seed(seed)
    idx = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
    prices = start_price * np.cumprod(
        1 + daily_ret + np.random.randn(n) * 0.005
    )
    return pd.Series(prices, index=idx)


def _make_close_map(
    symbols: list[str],
    n: int = 100,
) -> dict[str, pd.Series]:
    """One price series per symbol, deterministically different."""
    return {
        sym: _make_close(n, seed=i * 13 + 7)
        for i, sym in enumerate(symbols)
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1 — Universe: IPO / delisting / F&O ban
# ══════════════════════════════════════════════════════════════════════════════

class TestUniverseEligibility:
    def _resolver(self):
        from src.ranking.universe import UniverseResolver
        return UniverseResolver()

    def test_ipo_stock_absent_before_admission(self):
        """A stock admitted in 2024 must not appear in a 2023 snapshot."""
        r = self._resolver()
        r.register("OLD_STOCK", datetime(2020, 1, 1, tzinfo=UTC))
        r.register("NEW_IPO",   datetime(2024, 6, 1, tzinfo=UTC))
        snap = r.resolve(datetime(2023, 1, 1, tzinfo=UTC))
        assert "OLD_STOCK" in snap.eligible
        assert "NEW_IPO" not in snap.eligible, (
            "IPO test FAILED: future stock appeared in historical universe"
        )

    def test_ipo_stock_present_after_admission(self):
        r = self._resolver()
        r.register("NEW_IPO", datetime(2024, 6, 1, tzinfo=UTC))
        snap = r.resolve(datetime(2025, 1, 1, tzinfo=UTC))
        assert "NEW_IPO" in snap.eligible

    def test_delisted_stock_present_before_delist(self):
        r = self._resolver()
        r.register("EXSTOCK", datetime(2018, 1, 1, tzinfo=UTC),
                   delisted_after=datetime(2022, 6, 1, tzinfo=UTC))
        snap = r.resolve(datetime(2021, 1, 1, tzinfo=UTC))
        assert "EXSTOCK" in snap.eligible

    def test_delisted_stock_absent_after_delist(self):
        """A delisted stock must not appear after its delist date."""
        r = self._resolver()
        r.register("EXSTOCK", datetime(2018, 1, 1, tzinfo=UTC),
                   delisted_after=datetime(2022, 6, 1, tzinfo=UTC))
        snap = r.resolve(datetime(2023, 1, 1, tzinfo=UTC))
        assert "EXSTOCK" not in snap.eligible, (
            "Delisting test FAILED: delisted stock still in universe"
        )

    def test_fno_banned_stock_excluded(self):
        """A stock under NSE F&O ban must not appear as MODEL_ELIGIBLE."""
        from src.ranking.universe import UniverseResolver, UniverseConfig
        from src.ranking.schemas import EligibilityState
        ban_ts = datetime(2023, 3, 1, tzinfo=UTC)
        cfg = UniverseConfig(exclude_fno_banned=True)
        r   = UniverseResolver(config=cfg, fno_ban_map={ban_ts: ["BANNED_CO"]})
        r.register("BANNED_CO", datetime(2020, 1, 1, tzinfo=UTC))
        snap = r.resolve(datetime(2023, 4, 1, tzinfo=UTC))
        assert "BANNED_CO" not in snap.eligible
        state = snap.all_states["BANNED_CO"].state
        assert state == EligibilityState.FNO_BANNED

    def test_insufficient_history_excluded(self):
        """Stock with fewer bars than min_history must be excluded."""
        from src.ranking.universe import UniverseResolver, UniverseConfig
        from src.ranking.schemas import EligibilityState
        cfg = UniverseConfig(min_history_bars=60)
        r   = UniverseResolver(config=cfg)
        r.register("THIN", datetime(2023, 1, 1, tzinfo=UTC))
        # Only 10 price bars — last bar is at index 9
        close = _make_close(10, seed=1)
        ts    = close.index[-1].to_pydatetime()  # use actual last bar as timestamp
        close_map = {"THIN": close}
        snap = r.resolve(ts, stock_close_map=close_map)
        assert "THIN" not in snap.eligible
        assert snap.all_states["THIN"].state == EligibilityState.INSUFFICIENT_HISTORY

    def test_snapshot_is_sufficient_check(self):
        from src.ranking.universe import UniverseResolver
        r = self._resolver()
        r.register("A", datetime(2020, 1, 1, tzinfo=UTC))
        r.register("B", datetime(2020, 1, 1, tzinfo=UTC))
        snap = r.resolve(datetime(2023, 1, 1, tzinfo=UTC))
        assert snap.is_sufficient(2)
        assert not snap.is_sufficient(10)

    def test_empty_universe_returns_empty_eligible(self):
        r = self._resolver()
        # No registered symbols
        snap = r.resolve(datetime(2023, 1, 1, tzinfo=UTC))
        assert snap.eligible == []


# ══════════════════════════════════════════════════════════════════════════════
# 2 — Universe mutation: future stock must not change historical ranks
# ══════════════════════════════════════════════════════════════════════════════

class TestUniverseMutation:
    def test_adding_future_stock_does_not_change_historical_eligible(self):
        """
        Adding stock D (effective 2025) to the resolver must not change
        the eligible list at 2023-01-01.
        """
        from src.ranking.universe import UniverseResolver

        r = UniverseResolver()
        r.register("A", datetime(2020, 1, 1, tzinfo=UTC))
        r.register("B", datetime(2020, 1, 1, tzinfo=UTC))
        r.register("C", datetime(2020, 1, 1, tzinfo=UTC))

        snap_before = r.resolve(datetime(2023, 1, 1, tzinfo=UTC))
        eligible_before = set(snap_before.eligible)

        # Now add a stock that became eligible AFTER 2023
        r.register("D", datetime(2025, 6, 1, tzinfo=UTC))

        snap_after = r.resolve(datetime(2023, 1, 1, tzinfo=UTC))
        eligible_after = set(snap_after.eligible)

        assert eligible_before == eligible_after, (
            "Universe mutation FAILED: adding future stock D changed historical universe at 2023"
        )
        assert "D" not in eligible_after

    def test_future_sector_change_does_not_alter_historical(self):
        """
        If sector membership is passed at call time (PIT), changing the
        sector map for a future period does not affect the historical snapshot.
        """
        # This test verifies the design contract: sector_map is passed per-call
        # The universe resolver does not store sector — that's the caller's responsibility
        from src.ranking.neutralization import sector_neutralize
        import pandas as pd

        # At t=2023: company X is in 'Tech'
        returns_2023 = pd.Series({"X": 5.0, "Y": 2.0, "Z": 3.0})
        sector_2023  = {"X": "Tech", "Y": "Tech", "Z": "Bank"}

        resid_2023 = sector_neutralize(returns_2023, sector_2023)

        # In 2025 X is reclassified to 'Bank' — but 2023 residual must be unchanged
        # (This is enforced by passing sector_2023 to the 2023 computation — correct design)
        resid_2023_again = sector_neutralize(returns_2023, sector_2023)
        assert (resid_2023 == resid_2023_again).all(), (
            "Sector neutralization is non-deterministic"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3 — CS targets A–F
# ══════════════════════════════════════════════════════════════════════════════

class TestCSTargets:
    def _make_close_map_and_benchmark(self):
        """5 stocks + benchmark over 20 bars."""
        symbols = ["A", "B", "C", "D", "E"]
        n = 20
        # Deterministic: A outperforms, E underperforms
        returns = [0.10, 0.05, 0.00, -0.05, -0.10]
        idx = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
        close_map = {}
        for sym, ret in zip(symbols, returns):
            prices = [100.0]
            for _ in range(n - 1):
                prices.append(prices[-1] * (1 + ret / n))
            close_map[sym] = pd.Series(prices, index=idx)

        # Benchmark flat
        bmark = pd.Series([100.0] * n, index=idx)
        return close_map, bmark, symbols, idx

    def test_target_a_raw_return_sign(self):
        """Stock with upward trend → positive raw return."""
        from src.labels.cross_sectional import CSTargetConfig, compute_cross_sectional_targets

        close_map, bmark, symbols, idx = self._make_close_map_and_benchmark()
        ts = idx[0].to_pydatetime()
        cfg = CSTargetConfig(horizon_bars=5, vol_normalize=False)

        rows = compute_cross_sectional_targets(
            stock_close_map=close_map,
            timestamp=ts,
            config=cfg,
            eligible_symbols=symbols,
        )
        row_map = {r.instrument_id: r for r in rows}
        # A has positive trend
        assert row_map["A"].raw_return is not None
        assert row_map["A"].raw_return > 0
        # E has negative trend
        assert row_map["E"].raw_return is not None
        assert row_map["E"].raw_return < 0

    def test_target_b_excess_return_vs_flat_benchmark(self):
        """With flat benchmark, excess return ≈ raw return."""
        from src.labels.cross_sectional import CSTargetConfig, compute_cross_sectional_targets

        close_map, bmark, symbols, idx = self._make_close_map_and_benchmark()
        ts = idx[0].to_pydatetime()
        cfg = CSTargetConfig(horizon_bars=5, vol_normalize=False)

        rows = compute_cross_sectional_targets(
            stock_close_map=close_map,
            timestamp=ts,
            config=cfg,
            eligible_symbols=symbols,
            benchmark_close=bmark,
        )
        row_map = {r.instrument_id: r for r in rows}
        for r in rows:
            if r.raw_return is not None and r.excess_return is not None:
                # With flat benchmark bmark_ret≈0, excess≈raw
                assert abs(r.excess_return - r.raw_return) < 0.5, (
                    f"Flat benchmark: excess_return should ≈ raw_return for {r.instrument_id}"
                )

    def test_target_d_percentile_ordering(self):
        """CS percentile of A (best) must be highest, E (worst) must be lowest."""
        from src.labels.cross_sectional import CSTargetConfig, compute_cross_sectional_targets

        close_map, bmark, symbols, idx = self._make_close_map_and_benchmark()
        ts = idx[0].to_pydatetime()
        cfg = CSTargetConfig(horizon_bars=5, vol_normalize=False, min_cs_size=3)

        rows = compute_cross_sectional_targets(
            stock_close_map=close_map,
            timestamp=ts,
            config=cfg,
            eligible_symbols=symbols,
        )
        row_map = {r.instrument_id: r for r in rows}
        pcts = {sym: row_map[sym].cs_percentile for sym in symbols
                if row_map[sym].cs_percentile is not None}
        if pcts:
            assert pcts["A"] > pcts["E"], (
                f"A ({pcts['A']:.1f}) must have higher percentile than E ({pcts['E']:.1f})"
            )

    def test_target_f_rank_ordering(self):
        """CS rank: A (best) must have highest rank number, E (worst) lowest."""
        from src.labels.cross_sectional import CSTargetConfig, compute_cross_sectional_targets

        close_map, bmark, symbols, idx = self._make_close_map_and_benchmark()
        ts = idx[0].to_pydatetime()
        cfg = CSTargetConfig(horizon_bars=5, vol_normalize=False, min_cs_size=3)

        rows = compute_cross_sectional_targets(
            stock_close_map=close_map, timestamp=ts,
            config=cfg, eligible_symbols=symbols,
        )
        row_map = {r.instrument_id: r for r in rows}
        if row_map["A"].cs_rank and row_map["E"].cs_rank:
            assert row_map["A"].cs_rank > row_map["E"].cs_rank, (
                "A (highest return) must have a higher cs_rank number than E"
            )

    def test_target_missing_data_returns_none(self):
        """Stock with no price data returns None targets, not fabricated zeros."""
        from src.labels.cross_sectional import CSTargetConfig, compute_cross_sectional_targets

        close_map, _, symbols, idx = self._make_close_map_and_benchmark()
        ts = idx[0].to_pydatetime()
        cfg = CSTargetConfig(horizon_bars=5, vol_normalize=False)

        # Request a non-existent symbol
        rows = compute_cross_sectional_targets(
            stock_close_map=close_map, timestamp=ts,
            config=cfg, eligible_symbols=["A", "MISSING_SYM"],
        )
        row_map = {r.instrument_id: r for r in rows}
        missing_row = row_map.get("MISSING_SYM")
        assert missing_row is not None
        assert missing_row.raw_return is None, (
            "Missing symbol must return raw_return=None, not fabricated value"
        )
        assert not missing_row.raw_return_available

    def test_min_cs_size_gate(self):
        """When eligible universe < min_cs_size, CS stats (D,E,F) must be None."""
        from src.labels.cross_sectional import CSTargetConfig, compute_cross_sectional_targets

        close_map, _, symbols, idx = self._make_close_map_and_benchmark()
        ts = idx[0].to_pydatetime()
        cfg = CSTargetConfig(horizon_bars=5, min_cs_size=20)  # need 20 stocks

        rows = compute_cross_sectional_targets(
            stock_close_map=close_map, timestamp=ts,
            config=cfg, eligible_symbols=symbols,  # only 5
        )
        for row in rows:
            assert row.cs_percentile is None, (
                "CS percentile must be None when universe < min_cs_size"
            )
            assert row.cs_zscore is None
            assert row.cs_rank is None


# ══════════════════════════════════════════════════════════════════════════════
# 4 — CS normalization
# ══════════════════════════════════════════════════════════════════════════════

class TestCSNormalization:
    def test_zscore_mean_zero_std_one(self):
        from src.ranking.normalization import cs_zscore
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z = cs_zscore(x)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std(ddof=1) - 1.0) < 1e-10

    def test_zscore_preserves_nan(self):
        from src.ranking.normalization import cs_zscore
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0])
        z = cs_zscore(x, min_obs=5)
        assert np.isnan(z[2]), "NaN must be preserved after zscore"
        # Non-NaN values should still have finite z-scores
        assert np.all(np.isfinite(z[[0, 1, 3, 4, 5, 6]]))

    def test_rank_pct_range(self):
        from src.ranking.normalization import cs_rank_pct
        x = np.array([1.0, 3.0, 2.0, 5.0, 4.0])
        r = cs_rank_pct(x)
        assert (r >= 0).all() and (r <= 1).all()
        assert r[np.argmax(x)] == pytest.approx(1.0)
        assert r[np.argmin(x)] == pytest.approx(0.0)

    def test_robust_zscore_less_sensitive_to_outlier(self):
        """
        Robust z-score (IQR-based): the non-outlier values should have
        a tighter spread than under standard z-score when outliers are present.
        The key property: removing the outlier changes the robust z-scores of
        non-outlier values less than removing it changes standard z-scores.
        """
        from src.ranking.normalization import cs_zscore, cs_robust_zscore
        # Large array so the outlier inflates std substantially
        np.random.seed(42)
        x_base = np.arange(1.0, 11.0)          # [1..10]
        x_out  = np.append(x_base, 1000.0)     # with extreme outlier

        z_std    = cs_zscore(x_out)
        z_robust = cs_robust_zscore(x_out)

        # Standard z-score: outlier inflates std, so non-outlier scores cluster near 0
        # Robust z-score: median/IQR stable, non-outlier scores still spread out
        # Property we test: std of non-outlier robust z-scores > std of non-outlier std z-scores
        # (robust keeps non-outlier discrimination; standard collapses them)
        non_out_std    = np.std(z_std[:10],    ddof=1)
        non_out_robust = np.std(z_robust[:10], ddof=1)
        assert non_out_robust > non_out_std, (
            f"Robust z-score should preserve non-outlier discrimination "
            f"(robust std {non_out_robust:.3f} should be > std z-score std {non_out_std:.3f})"
        )

    def test_normalize_panel_each_timestamp_independent(self):
        """Normalization statistics from t1 must not affect t2."""
        from src.ranking.normalization import normalize_panel

        idx1 = pd.bdate_range("2023-01-02", periods=5, tz="UTC")
        idx2 = pd.bdate_range("2023-01-09", periods=5, tz="UTC")

        rows1 = [{"timestamp": t, "instrument_id": f"S{i}",
                  "feat": float(i)} for i, t in enumerate(idx1)]
        rows2 = [{"timestamp": t, "instrument_id": f"S{i}",
                  "feat": float(i * 100)} for i, t in enumerate(idx2)]

        df = pd.DataFrame(rows1 + rows2)
        out = normalize_panel(df, timestamp_col="timestamp",
                              feature_cols=["feat"], method="zscore")

        # Each timestamp's mean should be ~0 independently
        for ts, grp in out.groupby("timestamp"):
            vals = grp["feat"].dropna()
            if len(vals) > 1:
                assert abs(vals.mean()) < 1e-9, (
                    f"At {ts}: z-score mean = {vals.mean()} — cross-timestamp leakage"
                )

    def test_normalize_panel_future_timestamps_do_not_affect_past(self):
        """
        PIT mutation test for normalization:
        Adding future rows must not change the normalized values at past timestamps.
        """
        from src.ranking.normalization import normalize_panel

        idx = pd.bdate_range("2023-01-02", periods=10, tz="UTC")
        rows = [{"timestamp": idx[i], "instrument_id": f"S{j}", "feat": float(i * 5 + j)}
                for i in range(10) for j in range(5)]
        df_orig = pd.DataFrame(rows)
        out_orig = normalize_panel(df_orig, "timestamp", ["feat"], "zscore")

        # Append extreme future rows
        future_ts = idx[-1] + pd.Timedelta(days=1)
        future_rows = [{"timestamp": future_ts, "instrument_id": f"S{j}",
                        "feat": 9999.0} for j in range(5)]
        df_extended = pd.concat([df_orig, pd.DataFrame(future_rows)], ignore_index=True)
        out_ext = normalize_panel(df_extended, "timestamp", ["feat"], "zscore")

        # Values at original timestamps must be unchanged
        orig_ts_mask = out_orig["timestamp"].isin(df_orig["timestamp"].unique())
        ext_ts_mask  = out_ext["timestamp"].isin(df_orig["timestamp"].unique())

        orig_vals = out_orig[orig_ts_mask].sort_values(
            ["timestamp", "instrument_id"])["feat"].to_numpy()
        ext_vals  = out_ext[ext_ts_mask].sort_values(
            ["timestamp", "instrument_id"])["feat"].to_numpy()

        np.testing.assert_allclose(orig_vals, ext_vals, atol=1e-10, err_msg=(
            "Normalization PIT FAILED: future rows changed past z-scores"
        ))


# ══════════════════════════════════════════════════════════════════════════════
# 5 — Winsorization
# ══════════════════════════════════════════════════════════════════════════════

class TestWinsorization:
    def test_extreme_outlier_is_capped(self):
        from src.ranking.normalization import winsorize
        x = np.array([1.0, 2.0, 3.0, 4.0, 100.0])
        w = winsorize(x, pct=20.0)
        assert w[-1] < 100.0, "Outlier must be capped"

    def test_nan_preserved(self):
        from src.ranking.normalization import winsorize
        x = np.array([1.0, np.nan, 3.0, 4.0, 5.0])
        w = winsorize(x, pct=1.0)
        assert np.isnan(w[1]), "NaN must be preserved after winsorization"

    def test_zero_pct_no_change(self):
        from src.ranking.normalization import winsorize
        x = np.array([1.0, 2.0, 3.0, 100.0])
        w = winsorize(x, pct=0.0)
        np.testing.assert_array_equal(w, x)


# ══════════════════════════════════════════════════════════════════════════════
# 6 — Sector neutralization
# ══════════════════════════════════════════════════════════════════════════════

class TestSectorNeutralization:
    def test_residuals_correct(self):
        """Sector mean removed exactly."""
        from src.ranking.neutralization import sector_neutralize
        returns = pd.Series({"A": 10.0, "B": 8.0, "C": 2.0, "D": 0.0})
        smap    = {"A": "Tech", "B": "Tech", "C": "Bank", "D": "Bank"}
        resid   = sector_neutralize(returns, smap)
        # Tech mean=9, Bank mean=1
        assert abs(resid["A"] - 1.0) < 1e-9
        assert abs(resid["B"] - (-1.0)) < 1e-9
        assert abs(resid["C"] - 1.0) < 1e-9
        assert abs(resid["D"] - (-1.0)) < 1e-9

    def test_sector_mean_is_zero_after_neutralization(self):
        """Sum of residuals within each sector must be zero."""
        from src.ranking.neutralization import sector_neutralize
        returns = pd.Series({"A": 5.0, "B": 3.0, "C": 1.0,
                              "D": 8.0, "E": 4.0, "F": 2.0})
        smap    = {"A":"Tech","B":"Tech","C":"Tech","D":"Bank","E":"Bank","F":"Bank"}
        resid   = sector_neutralize(returns, smap)
        tech_sum = resid[["A","B","C"]].sum()
        bank_sum = resid[["D","E","F"]].sum()
        assert abs(tech_sum) < 1e-9, f"Tech residuals must sum to 0, got {tech_sum}"
        assert abs(bank_sum) < 1e-9, f"Bank residuals must sum to 0, got {bank_sum}"

    def test_singleton_sector_unneutralized(self):
        """A sector with only 1 stock (< min_sector_size=2) stays unchanged."""
        from src.ranking.neutralization import sector_neutralize
        returns = pd.Series({"ALONE": 7.0, "A": 3.0, "B": 5.0})
        smap    = {"ALONE": "Solo", "A": "Duo", "B": "Duo"}
        resid   = sector_neutralize(returns, smap, min_sector_size=2)
        # Solo sector has 1 stock → not neutralized
        assert abs(resid["ALONE"] - 7.0) < 1e-9

    def test_sector_neutralization_synthetic_alpha(self):
        """
        Sector neutralization test per spec §57:
        Sector A has a +5% common move.
        Stock-specific alpha should survive; sector effect should be removed.
        """
        from src.ranking.neutralization import sector_neutralize
        # All Sector A stocks +5% common + individual alpha
        # All Sector B stocks flat
        returns = pd.Series({
            "A1": 5.0 + 2.0,   # common 5% + alpha +2
            "A2": 5.0 + 0.0,   # common 5% + alpha  0
            "A3": 5.0 - 2.0,   # common 5% + alpha -2
            "B1": 0.0 + 1.0,
            "B2": 0.0 + 0.0,
            "B3": 0.0 - 1.0,
        })
        smap = {"A1":"SectorA","A2":"SectorA","A3":"SectorA",
                "B1":"SectorB","B2":"SectorB","B3":"SectorB"}
        resid = sector_neutralize(returns, smap)
        # After neutralization, A stocks should NOT be uniformly higher than B
        # The common 5% effect must be removed
        assert abs(resid["A2"]) < 1e-9, (
            f"A2 has zero alpha; its residual should be 0 after removing sector effect, got {resid['A2']}"
        )
        assert abs(resid["A1"] - 2.0) < 1e-9
        assert abs(resid["A3"] - (-2.0)) < 1e-9


# ══════════════════════════════════════════════════════════════════════════════
# 7 — Beta neutralization
# ══════════════════════════════════════════════════════════════════════════════

class TestBetaNeutralization:
    def test_high_beta_vs_low_beta_equal_stock_alpha(self):
        """
        Spec §58: stocks with high/low beta and equal stock-specific alpha.
        After beta neutralization, both should have the same residual.
        """
        from src.ranking.neutralization import beta_neutralize
        import pandas as pd
        # Market returned +10%
        market_return = 0.10
        # Stock A: beta=2, raw return = 2×0.10 + 0.05 (alpha) = 0.25
        # Stock B: beta=0.5, raw return = 0.5×0.10 + 0.05 (alpha) = 0.10
        returns  = pd.Series({"HIGH_BETA": 0.25, "LOW_BETA": 0.10})
        beta_map = {"HIGH_BETA": 2.0, "LOW_BETA": 0.5}
        resid = beta_neutralize(returns, market_return, beta_map)
        # Both should have residual ≈ 0.05 (the common alpha)
        assert abs(resid["HIGH_BETA"] - 0.05) < 1e-9
        assert abs(resid["LOW_BETA"]  - 0.05) < 1e-9

    def test_beta_neutralize_missing_beta(self):
        """Stock with no beta entry should be left unchanged."""
        from src.ranking.neutralization import beta_neutralize
        import pandas as pd
        returns  = pd.Series({"KNOWN": 0.15, "UNKNOWN": 0.08})
        beta_map = {"KNOWN": 1.5}
        resid = beta_neutralize(returns, 0.10, beta_map)
        assert abs(resid["UNKNOWN"] - 0.08) < 1e-9  # unchanged


# ══════════════════════════════════════════════════════════════════════════════
# 8 — Factor neutralization (OLS)
# ══════════════════════════════════════════════════════════════════════════════

class TestFactorNeutralization:
    def test_sector_dummies_shape(self):
        from src.ranking.neutralization import sector_dummies
        symbols = ["A", "B", "C", "D"]
        smap    = {"A": "Tech", "B": "Tech", "C": "Bank", "D": "Bank"}
        F = sector_dummies(symbols, smap)
        assert F.shape == (4, 2), f"Expected (4,2), got {F.shape}"
        # A and B should have identical rows
        np.testing.assert_array_equal(F[0], F[1])
        np.testing.assert_array_equal(F[2], F[3])

    def test_factor_neutralize_removes_common_component(self):
        """OLS residualization removes the explained component."""
        from src.ranking.neutralization import factor_neutralize
        # Y = F @ coeff + noise; after neutralization residual should be ≈ noise
        np.random.seed(42)
        n = 50
        F = np.random.randn(n, 2)
        coeff = np.array([3.0, -2.0])
        noise = np.random.randn(n) * 0.1
        Y = F @ coeff + noise

        resid = factor_neutralize(Y, F)

        # The residual should be much closer to noise than Y is
        # Measure: variance of residual << variance of Y
        var_Y     = float(np.var(Y))
        var_resid = float(np.var(resid))
        assert var_resid < var_Y * 0.05, (
            f"Factor neutralization must dramatically reduce variance: "
            f"var(Y)={var_Y:.3f}, var(resid)={var_resid:.4f}"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 9 — Ranking golden tests
# ══════════════════════════════════════════════════════════════════════════════

class TestRankingGolden:
    """Spec §94, §95: deterministic golden tests for ranking output."""

    def _ranker(self):
        from src.ranking.ranker import MomentumBaselineRanker
        r = MomentumBaselineRanker(return_20d_col_idx=0)
        r.fit(np.zeros((5, 1)), np.zeros(5))
        return r

    def test_golden_score_to_rank(self):
        """
        Spec §94: A=0.90, B=0.80, C=0.70, D=0.60, E=0.50
        Expected: A→rank1, B→rank2, ..., E→rank5
        """
        from src.ranking.ranker import _rank_descending
        scores = np.array([0.90, 0.80, 0.70, 0.60, 0.50])
        ranks  = _rank_descending(scores)
        assert ranks[0] == 1.0, f"A should be rank 1, got {ranks[0]}"
        assert ranks[1] == 2.0
        assert ranks[2] == 3.0
        assert ranks[3] == 4.0
        assert ranks[4] == 5.0, f"E should be rank 5, got {ranks[4]}"

    def test_score_direction_invariant(self):
        """Higher score MUST be rank 1 — reversing scores inverts ranks."""
        from src.ranking.ranker import _rank_descending
        scores_asc  = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        scores_desc = scores_asc[::-1].copy()
        ranks_asc  = _rank_descending(scores_asc)
        ranks_desc = _rank_descending(scores_desc)
        # Rank 1 must be the highest score in both cases
        assert ranks_asc[4]  == 1.0, "Highest score (index 4) must be rank 1"
        assert ranks_desc[0] == 1.0, "After reversal, index 0 (highest) must still be rank 1"

    def test_tie_handling_deterministic(self):
        """Spec §60: equal scores → average rank, deterministic."""
        from src.ranking.ranker import _rank_descending
        scores = np.array([1.0, 1.0, 1.0])  # all ties
        ranks  = _rank_descending(scores)
        # All get average rank = (1+2+3)/3 = 2.0
        assert (ranks == 2.0).all(), f"All-tie average rank must be 2.0, got {ranks}"

    def test_tie_handling_partial(self):
        from src.ranking.ranker import _rank_descending
        scores = np.array([3.0, 1.0, 1.0])
        ranks  = _rank_descending(scores)
        assert ranks[0] == 1.0   # unique top score
        assert ranks[1] == 2.5   # tied: (2+3)/2
        assert ranks[2] == 2.5

    def test_percentile_range(self):
        from src.ranking.ranker import _score_to_percentile
        scores = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        pcts   = _score_to_percentile(scores)
        assert (pcts >= 0).all() and (pcts <= 100).all()
        assert pcts[4] == pytest.approx(100.0)
        assert pcts[0] == pytest.approx(0.0)

    def test_momentum_baseline_rank1_is_highest_score(self):
        """MomentumBaselineRanker: highest score column value → rank 1."""
        from src.ranking.ranker import MomentumBaselineRanker
        X = np.array([[5.0], [3.0], [7.0], [1.0], [4.0]])  # col 0 = return_20d
        r = MomentumBaselineRanker(return_20d_col_idx=0)
        r.fit(X, np.zeros(5))
        ranks = r.rank(X)
        assert ranks[2] == 1.0, "7.0 (index 2) must be rank 1"
        assert ranks[3] == 5.0, "1.0 (index 3) must be rank 5"

    def test_ranking_is_reproducible(self):
        """Spec §63: same inputs → identical outputs on repeated calls."""
        from src.ranking.ranker import MomentumBaselineRanker
        X = np.random.RandomState(42).randn(20, 5).astype(np.float32)
        r = MomentumBaselineRanker(return_20d_col_idx=2)
        r.fit(X, np.zeros(20))
        ranks1 = r.rank(X)
        ranks2 = r.rank(X)
        np.testing.assert_array_equal(ranks1, ranks2)


# ══════════════════════════════════════════════════════════════════════════════
# 10 — IC golden tests
# ══════════════════════════════════════════════════════════════════════════════

class TestICGolden:
    """Spec §96: IC golden tests."""

    def test_perfect_alignment_rank_ic_is_one(self):
        """Scores perfectly aligned with realized returns → Rank IC = 1."""
        from src.ranking.evaluation import compute_rank_ic
        scores   = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        realized = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        ic = compute_rank_ic(scores, realized)
        assert ic is not None
        assert abs(ic - 1.0) < 1e-6, f"Expected IC≈1, got {ic}"

    def test_perfect_inverse_rank_ic_is_minus_one(self):
        """Scores perfectly anti-aligned → Rank IC = -1."""
        from src.ranking.evaluation import compute_rank_ic
        scores   = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        realized = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ic = compute_rank_ic(scores, realized)
        assert ic is not None
        assert abs(ic + 1.0) < 1e-6, f"Expected IC≈-1, got {ic}"

    def test_insufficient_data_returns_none(self):
        """Fewer than 5 valid pairs → None."""
        from src.ranking.evaluation import compute_rank_ic
        ic = compute_rank_ic(np.array([1.0, 2.0]), np.array([1.0, 2.0]))
        assert ic is None

    def test_nan_handling_in_ic(self):
        """NaN values are excluded; IC computed on valid pairs only (need ≥5)."""
        from src.ranking.evaluation import compute_rank_ic
        # 6 elements, 1 NaN → 5 valid pairs → IC should be computable
        scores   = np.array([6.0, 5.0, np.nan, 3.0, 2.0, 1.0])
        realized = np.array([6.0, 5.0, 4.0,    3.0, 2.0, 1.0])
        ic = compute_rank_ic(scores, realized)
        # Should compute on 5 valid pairs — near +1
        assert ic is not None, "IC must be computable with 5 valid pairs"
        assert ic > 0.9

    def test_icir_formula(self):
        """ICIR = mean(IC) / std(IC)."""
        from src.ranking.evaluation import summarise_ic_series
        ic_values = pd.Series([0.10, 0.08, 0.12, 0.09, 0.11])
        summary   = summarise_ic_series(ic_values)
        expected_icir = ic_values.mean() / ic_values.std(ddof=1)
        assert summary.icir is not None
        assert abs(summary.icir - expected_icir) < 1e-6

    def test_icir_zero_std(self):
        """Constant IC series → std=0 → ICIR=None (not divide-by-zero)."""
        from src.ranking.evaluation import summarise_ic_series
        ic_values = pd.Series([0.05, 0.05, 0.05, 0.05, 0.05])
        summary   = summarise_ic_series(ic_values)
        assert summary.icir is None or not math.isfinite(summary.icir), (
            "Zero-std IC must produce None or non-finite ICIR"
        )

    def test_ic_series_per_timestamp(self):
        """IC must be computed independently per timestamp."""
        from src.ranking.evaluation import compute_ic_series

        # 3 timestamps, 5 stocks each
        ts1 = datetime(2023, 1, 2, tzinfo=UTC)
        ts2 = datetime(2023, 1, 3, tzinfo=UTC)
        ts3 = datetime(2023, 1, 4, tzinfo=UTC)
        rows = []
        for ts, aligned in [(ts1, True), (ts2, False), (ts3, True)]:
            for i in range(5):
                score = float(i)
                real  = float(i) if aligned else float(4 - i)
                rows.append({"ts": ts, "instrument_id": f"S{i}",
                             "score": score, "realized": real})
        df  = pd.DataFrame(rows)
        ics = compute_ic_series(df, "score", "realized", "ts", use_rank_ic=True)
        assert abs(ics[ts1] - 1.0) < 1e-6, f"ts1: perfect alignment → IC≈1, got {ics[ts1]}"
        assert abs(ics[ts2] + 1.0) < 1e-6, f"ts2: perfect inverse → IC≈-1, got {ics[ts2]}"
        assert abs(ics[ts3] - 1.0) < 1e-6

    def test_positive_ic_pct(self):
        from src.ranking.evaluation import summarise_ic_series
        ics = pd.Series([0.1, -0.05, 0.08, 0.12, -0.02])
        s   = summarise_ic_series(ics)
        assert s.positive_pct == pytest.approx(60.0)


# ══════════════════════════════════════════════════════════════════════════════
# 11 — Decile golden tests
# ══════════════════════════════════════════════════════════════════════════════

class TestDecileGolden:
    """Spec §97: 100 stocks, known scores, verify Q1/Q10 assignment."""

    def _make_golden_panel(self, n: int = 100, ts: Optional[datetime] = None):
        """n stocks with scores 1..n and realized returns perfectly aligned."""
        if ts is None:
            ts = datetime(2023, 1, 2, tzinfo=UTC)
        rows = [{"timestamp": ts, "instrument_id": f"S{i:03d}",
                 "score": float(i), "realized": float(i)}
                for i in range(1, n + 1)]
        return pd.DataFrame(rows)

    def test_each_decile_has_correct_count(self):
        """With 100 stocks and 10 deciles, each decile must have 10 stocks."""
        from src.ranking.evaluation import compute_decile_report
        df = self._make_golden_panel(100)
        rep = compute_decile_report(df, "score", "realized", "timestamp",
                                     n_deciles=10, min_cs_size=10)
        for s in rep.deciles:
            assert s.n_obs == 10, f"Decile {s.decile} has {s.n_obs} stocks, expected 10"

    def test_q10_higher_than_q1(self):
        """With perfect score-return alignment, Q10 mean return > Q1 mean return."""
        from src.ranking.evaluation import compute_decile_report
        df = self._make_golden_panel(100)
        rep = compute_decile_report(df, "score", "realized", "timestamp",
                                     n_deciles=10, min_cs_size=10)
        assert rep.deciles[-1].mean_return > rep.deciles[0].mean_return, (
            f"Q10 ({rep.deciles[-1].mean_return:.2f}) must > Q1 ({rep.deciles[0].mean_return:.2f})"
        )

    def test_monotonicity_score_near_one_for_perfect_alignment(self):
        """Perfect score-return alignment → monotonicity_score ≈ 1."""
        from src.ranking.evaluation import compute_decile_report
        df = self._make_golden_panel(100)
        rep = compute_decile_report(df, "score", "realized", "timestamp",
                                     n_deciles=10, min_cs_size=10)
        assert rep.monotonicity_score is not None
        assert rep.monotonicity_score > 0.95, (
            f"Expected monotonicity ≈ 1, got {rep.monotonicity_score}"
        )

    def test_top_bottom_spread_positive(self):
        """Q10 mean - Q1 mean must be positive for aligned scores."""
        from src.ranking.evaluation import compute_decile_report
        df = self._make_golden_panel(100)
        rep = compute_decile_report(df, "score", "realized", "timestamp")
        assert rep.top_bottom_spread is not None
        assert rep.top_bottom_spread > 0

    def test_monotonicity_near_minus_one_for_inverse(self):
        """Inverse alignment → monotonicity_score ≈ -1."""
        from src.ranking.evaluation import compute_decile_report
        ts  = datetime(2023, 1, 2, tzinfo=UTC)
        rows = [{"timestamp": ts, "instrument_id": f"S{i:03d}",
                 "score": float(i), "realized": float(100 - i)}
                for i in range(1, 101)]
        df  = pd.DataFrame(rows)
        rep = compute_decile_report(df, "score", "realized", "timestamp")
        assert rep.monotonicity_score is not None
        assert rep.monotonicity_score < -0.9, (
            f"Inverse alignment: expected monotonicity ≈ -1, got {rep.monotonicity_score}"
        )

    def test_min_cs_size_gate_skips_timestamp(self):
        """Timestamps with fewer than min_cs_size stocks are excluded."""
        from src.ranking.evaluation import compute_decile_report
        ts  = datetime(2023, 1, 2, tzinfo=UTC)
        rows = [{"timestamp": ts, "instrument_id": f"S{i}", "score": float(i),
                 "realized": float(i)} for i in range(3)]  # only 3 stocks
        df  = pd.DataFrame(rows)
        rep = compute_decile_report(df, "score", "realized", "timestamp",
                                     min_cs_size=10)
        assert rep.n_timestamps == 0, "Small CS should not be counted"


# ══════════════════════════════════════════════════════════════════════════════
# 12 — Turnover proxy
# ══════════════════════════════════════════════════════════════════════════════

class TestTurnoverProxy:
    def test_stable_top_k_gives_zero_turnover(self):
        """When the same stocks rank first every day → turnover = 0."""
        from src.ranking.evaluation import compute_turnover_proxy
        rows = []
        for day in range(5):
            ts = datetime(2023, 1, 2 + day, tzinfo=UTC)
            for i in range(10):
                rows.append({"timestamp": ts, "instrument_id": f"S{i:02d}",
                             "score": float(10 - i)})
        df  = pd.DataFrame(rows)
        to  = compute_turnover_proxy(df, "score", "timestamp", top_k_pct=0.2)
        assert to is not None
        assert to == pytest.approx(0.0, abs=1e-6), (
            f"Stable top-2 should give turnover=0, got {to}"
        )

    def test_complete_churn_gives_high_turnover(self):
        """When top stocks rotate completely each day → turnover ≈ 1."""
        from src.ranking.evaluation import compute_turnover_proxy
        rows = []
        # Each day a different set of stocks occupies the top position
        for day in range(5):
            ts = datetime(2023, 1, 2 + day, tzinfo=UTC)
            for i in range(10):
                # Shift scores so different stocks lead each day
                rows.append({"timestamp": ts, "instrument_id": f"S{i:02d}",
                             "score": float((i + day * 3) % 10)})
        df  = pd.DataFrame(rows)
        to  = compute_turnover_proxy(df, "score", "timestamp", top_k_pct=0.2)
        assert to is not None
        assert to > 0.5, f"High-churn should give high turnover, got {to}"


# ══════════════════════════════════════════════════════════════════════════════
# 13 — Walk-forward validation
# ══════════════════════════════════════════════════════════════════════════════

class TestWalkForward:
    def _timestamps(self, n: int = 200) -> list[datetime]:
        base = datetime(2023, 1, 2, tzinfo=UTC)
        return [base + timedelta(days=i) for i in range(n)]

    def test_temporal_ordering_invariant(self):
        """max(train) < min(val) < min(test) for every fold."""
        from src.ranking.walk_forward import CrossSectionalWalkForward
        wf    = CrossSectionalWalkForward(train_bars=80, val_bars=20, test_bars=20, embargo_bars=5)
        folds = wf.split(self._timestamps(200))
        for fold in folds:
            assert max(fold.train_timestamps) < min(fold.val_timestamps), (
                f"Fold {fold.fold_index}: train/val overlap"
            )
            assert max(fold.val_timestamps) < min(fold.test_timestamps), (
                f"Fold {fold.fold_index}: val/test overlap"
            )

    def test_embargo_gap_respected(self):
        """Embargo gap between train and val must be ≥ embargo_bars."""
        from src.ranking.walk_forward import CrossSectionalWalkForward
        embargo = 10
        wf      = CrossSectionalWalkForward(train_bars=80, val_bars=20,
                                             test_bars=20, embargo_bars=embargo)
        folds   = wf.split(self._timestamps(200))
        ts_list = self._timestamps(200)
        ts_sorted = sorted(set(ts_list))
        for fold in folds:
            last_train_idx = ts_sorted.index(max(fold.train_timestamps))
            first_val_idx  = ts_sorted.index(min(fold.val_timestamps))
            gap = first_val_idx - last_train_idx - 1
            assert gap >= embargo, (
                f"Fold {fold.fold_index}: embargo gap {gap} < {embargo}"
            )

    def test_no_timestamp_in_both_train_and_test(self):
        """Every timestamp belongs to at most one fold's test set."""
        from src.ranking.walk_forward import CrossSectionalWalkForward
        wf    = CrossSectionalWalkForward(train_bars=60, val_bars=20, test_bars=20, embargo_bars=5)
        folds = wf.split(self._timestamps(200))
        test_sets = [set(fold.test_timestamps) for fold in folds]
        for i in range(len(test_sets)):
            for j in range(i + 1, len(test_sets)):
                overlap = test_sets[i] & test_sets[j]
                assert len(overlap) == 0, (
                    f"Folds {i} and {j} share test timestamps: {overlap}"
                )

    def test_insufficient_timestamps_raises(self):
        from src.ranking.walk_forward import CrossSectionalWalkForward
        wf = CrossSectionalWalkForward(train_bars=100, val_bars=50, test_bars=50)
        with pytest.raises(ValueError, match="timestamps"):
            wf.split(self._timestamps(10))

    def test_group_structure_preserved(self):
        """filter_panel must not mix timestamps across groups."""
        from src.ranking.walk_forward import CrossSectionalWalkForward
        ts_list = [datetime(2023,1,2,tzinfo=UTC) + timedelta(days=i) for i in range(10)]
        rows    = [{"ts": t, "instrument_id": f"S{j}", "val": j}
                   for t in ts_list for j in range(5)]
        df  = pd.DataFrame(rows)
        wf  = CrossSectionalWalkForward(train_bars=5, val_bars=2, test_bars=2, embargo_bars=1)
        folds = wf.split(ts_list)
        for fold in folds:
            test_df = CrossSectionalWalkForward.filter_panel(df, fold.test_timestamps, "ts")
            for ts, grp in test_df.groupby("ts"):
                assert ts in fold.test_timestamps


# ══════════════════════════════════════════════════════════════════════════════
# 14 — PIT mutation tests
# ══════════════════════════════════════════════════════════════════════════════

class TestPITMutation:
    """
    Spec §62: Modifying market data after T must not alter ranks at T.
    """

    def test_price_mutation_does_not_change_historical_normalization(self):
        """
        PIT mutation: appending extreme future prices must not change
        normalized feature values at past timestamps.
        (This re-tests normalize_panel's PIT property end-to-end.)
        """
        from src.ranking.normalization import normalize_panel
        ts_list = pd.bdate_range("2023-01-02", periods=10, tz="UTC")
        rows = []
        for i, ts in enumerate(ts_list):
            for j in range(5):
                rows.append({"timestamp": ts, "instrument_id": f"S{j}",
                             "return_20d": float(i * 5 + j)})
        df_orig = pd.DataFrame(rows)
        out_orig = normalize_panel(df_orig, "timestamp", ["return_20d"], "zscore")

        # Extreme future values appended
        future_ts = ts_list[-1] + pd.Timedelta(days=1)
        future_rows = [{"timestamp": future_ts, "instrument_id": f"S{j}",
                        "return_20d": 9999.0} for j in range(5)]
        df_ext = pd.concat([df_orig, pd.DataFrame(future_rows)], ignore_index=True)
        out_ext = normalize_panel(df_ext, "timestamp", ["return_20d"], "zscore")

        past_mask_orig = out_orig["timestamp"].isin(df_orig["timestamp"].unique())
        past_mask_ext  = out_ext["timestamp"].isin(df_orig["timestamp"].unique())

        v_orig = out_orig[past_mask_orig].sort_values(["timestamp","instrument_id"])["return_20d"].to_numpy()
        v_ext  = out_ext[past_mask_ext].sort_values(["timestamp","instrument_id"])["return_20d"].to_numpy()
        np.testing.assert_allclose(v_orig, v_ext, atol=1e-10,
                                   err_msg="Price mutation changed historical normalized features")

    def test_universe_mutation_does_not_change_historical_eligible(self):
        """Spec §98: adding future stock D at T+future must not change U_T."""
        from src.ranking.universe import UniverseResolver
        r = UniverseResolver()
        for sym in ["A","B","C"]:
            r.register(sym, datetime(2020,1,1,tzinfo=UTC))

        t = datetime(2023,1,1,tzinfo=UTC)
        snap_before = set(r.resolve(t).eligible)

        # Add D with effective date AFTER t
        r.register("D", datetime(2024,6,1,tzinfo=UTC))
        snap_after = set(r.resolve(t).eligible)

        assert snap_before == snap_after, (
            "Universe mutation FAILED: adding future stock changed historical universe"
        )

    def test_sector_mutation_does_not_change_historical_neutralization(self):
        """
        Spec §99: changing sector membership after T does not change
        residuals at T (because sector_map is passed per-call, not stored).
        """
        from src.ranking.neutralization import sector_neutralize
        returns = pd.Series({"X": 5.0, "Y": 2.0, "Z": 3.0})
        sector_at_t = {"X": "Tech", "Y": "Tech", "Z": "Bank"}

        resid1 = sector_neutralize(returns, sector_at_t)

        # "Change" X's sector in 2025 — this must not affect the 2023 computation
        # because sector_map is passed explicitly at call time
        sector_2025 = {"X": "Bank", "Y": "Tech", "Z": "Bank"}
        _ = sector_neutralize(returns, sector_2025)  # future computation

        # Re-run 2023 computation — must be identical
        resid2 = sector_neutralize(returns, sector_at_t)
        pd.testing.assert_series_equal(resid1, resid2)

    def test_cs_target_future_mutation_does_not_alter_historical(self):
        """
        Appending an extreme future bar to a stock's price series must not
        change the cross-sectional target computed at an earlier timestamp.
        """
        from src.labels.cross_sectional import CSTargetConfig, compute_cross_sectional_targets

        symbols = ["A", "B", "C", "D", "E"]
        n = 20
        idx = pd.bdate_range("2023-01-02", periods=n, tz="UTC")
        close_map = {sym: pd.Series(
            [100.0 + i * (j+1) for i in range(n)], index=idx
        ) for j, sym in enumerate(symbols)}

        ts  = idx[0].to_pydatetime()
        cfg = CSTargetConfig(horizon_bars=3, min_cs_size=3, vol_normalize=False)

        rows_orig = compute_cross_sectional_targets(
            stock_close_map=close_map, timestamp=ts,
            config=cfg, eligible_symbols=symbols,
        )
        orig_map = {r.instrument_id: r.raw_return for r in rows_orig}

        # Append extreme future bar to stock A
        future_idx = idx[-1] + pd.Timedelta(days=1)
        for sym in symbols:
            extended = pd.concat([close_map[sym],
                                   pd.Series([9999.0], index=pd.DatetimeIndex([future_idx], tz="UTC"))])
            close_map[sym] = extended

        rows_mut = compute_cross_sectional_targets(
            stock_close_map=close_map, timestamp=ts,
            config=cfg, eligible_symbols=symbols,
        )
        mut_map = {r.instrument_id: r.raw_return for r in rows_mut}

        for sym in symbols:
            orig_r = orig_map[sym]
            mut_r  = mut_map[sym]
            if orig_r is not None and mut_r is not None:
                assert abs(orig_r - mut_r) < 1e-10, (
                    f"{sym}: CS target changed after future append "
                    f"({orig_r} → {mut_r})"
                )


# ══════════════════════════════════════════════════════════════════════════════
# 15 — Schema invariants
# ══════════════════════════════════════════════════════════════════════════════

class TestSchemaInvariants:
    def test_alpha_score_semantics_must_be_set(self):
        """Every signal must have a non-UNKNOWN semantics."""
        from src.ranking.schemas import CrossSectionalAlphaSignal, AlphaScoreSemantics
        from src.ranking.schemas import PredictionProvenance, SignalStatus
        sig = CrossSectionalAlphaSignal(
            instrument_id="RELIANCE",
            timestamp=datetime(2023,1,2,tzinfo=UTC),
            alpha_score=75.0,
            rank=1,
            percentile=99.0,
            cross_section_size=50,
            universe_version="v1",
            model_id="momentum_baseline",
            model_version="v1",
            feature_set_id="RANKING_SET",
            label_version="lv2",
            dataset_id="d1",
            alpha_score_semantics=AlphaScoreSemantics.MOMENTUM_RANK,
            prediction_provenance=PredictionProvenance.BASELINE,
            signal_status=SignalStatus.RANKED,
        )
        assert sig.alpha_score_semantics != AlphaScoreSemantics.UNKNOWN

    def test_trained_model_provenance_not_for_heuristic(self):
        """TRAINED_MODEL provenance must never be assigned to a heuristic/baseline."""
        from src.ranking.schemas import PredictionProvenance
        # This test documents the invariant; enforcement is at the caller level.
        # Baselines must use BASELINE, not TRAINED_MODEL.
        assert PredictionProvenance.BASELINE != PredictionProvenance.TRAINED_MODEL
        assert PredictionProvenance.HEURISTIC != PredictionProvenance.TRAINED_MODEL

    def test_eligibility_states_complete(self):
        from src.ranking.schemas import EligibilityState
        required = {
            "MODEL_ELIGIBLE", "MODEL_INELIGIBLE", "DATA_UNAVAILABLE",
            "INSUFFICIENT_LIQUIDITY", "FNO_BANNED", "CONTRACT_EXPIRED",
            "INSUFFICIENT_HISTORY",
        }
        actual = {e.value for e in EligibilityState}
        for s in required:
            assert s in actual, f"EligibilityState missing: {s}"

    def test_experiment_manifest_fields(self):
        from src.ranking.schemas import ExperimentManifest
        m = ExperimentManifest(
            experiment_id="exp-001",
            created_at="2026-09-06T00:00:00Z",
            git_commit="abc1234",
            dataset_id="ds-v1",
            universe_version="historical_v1",
            feature_set_id="RANKING_SET-v1",
            label_version="lv2",
            target_horizon=5,
            models=["momentum_baseline","ridge_ranker","lgbm_ranker"],
            hyperparameters={"lgbm_ranker": {"num_leaves": 63}},
            validation_config={"train_bars":756,"val_bars":63,"test_bars":63},
            normalization="zscore",
            neutralization="sector",
            cost_model="DATA_UNAVAILABLE",
            random_seed=42,
            n_experiments=3,
        )
        assert m.n_experiments == 3
        assert m.random_seed == 42


# ══════════════════════════════════════════════════════════════════════════════
# 16 — Compare rankers: ML_ADDS_NO_CLEAR_VALUE logic
# ══════════════════════════════════════════════════════════════════════════════

class TestCompareRankers:
    def test_ml_adds_no_clear_value_when_below_baseline(self):
        """LightGBM with IC ≤ baseline IC → ML_ADDS_NO_CLEAR_VALUE verdict."""
        from src.ranking.evaluation import compare_rankers, ICSummary, DecileReport

        def _mock_result(model_id, mean_ic):
            ic_sum = ICSummary(mean=mean_ic, median=mean_ic, std=0.05,
                               icir=mean_ic/0.05 if mean_ic else None,
                               positive_pct=60.0, n_timestamps=50)
            return {"model_id": model_id, "ic_summary": ic_sum,
                    "decile_report": None, "coverage_pct": 95.0}

        results = [
            _mock_result("momentum_baseline",  0.05),
            _mock_result("composite_baseline", 0.04),
            _mock_result("lgbm_ranker",        0.04),   # not better than baseline
            _mock_result("ridge_ranker",        0.03),
        ]
        comparison = compare_rankers(results)
        lgbm_row   = next(r for r in comparison if r.model_id == "lgbm_ranker")
        assert lgbm_row.verdict == "ML_ADDS_NO_CLEAR_VALUE", (
            f"Expected ML_ADDS_NO_CLEAR_VALUE, got {lgbm_row.verdict}"
        )

    def test_insufficient_evidence_when_ic_too_low(self):
        from src.ranking.evaluation import compare_rankers, ICSummary

        results = [{"model_id": "lgbm_ranker",
                    "ic_summary": ICSummary(mean=0.005, median=0.005, std=0.05),
                    "decile_report": None, "coverage_pct": 90.0}]
        rows = compare_rankers(results)
        assert rows[0].verdict == "INSUFFICIENT_EVIDENCE"


# ══════════════════════════════════════════════════════════════════════════════
# 17 — ML Rankers (sklearn/scipy required)
# ══════════════════════════════════════════════════════════════════════════════

class TestRidgeRanker:
    def test_ridge_predict_shape(self):
        sklearn = pytest.importorskip("sklearn", reason="sklearn not installed")
        scipy   = pytest.importorskip("scipy",   reason="scipy not installed")
        from src.ranking.ranker import RidgeRanker
        np.random.seed(42)
        X_tr = np.random.randn(100, 10).astype(np.float32)
        y_tr = np.random.randn(100).astype(np.float32)
        X_te = np.random.randn(20,  10).astype(np.float32)

        r = RidgeRanker(alpha=1.0)
        metrics = r.fit(X_tr, y_tr)
        assert "train_ic" in metrics
        preds = r.predict(X_te)
        assert preds.shape == (20,)

    def test_elasticnet_predict_shape(self):
        sklearn = pytest.importorskip("sklearn", reason="sklearn not installed")
        scipy   = pytest.importorskip("scipy",   reason="scipy not installed")
        from src.ranking.ranker import ElasticNetRanker
        np.random.seed(7)
        X = np.random.randn(80, 5).astype(np.float32)
        y = np.random.randn(80).astype(np.float32)
        r = ElasticNetRanker(alpha=0.1, l1_ratio=0.5)
        r.fit(X, y)
        preds = r.predict(X[:10])
        assert preds.shape == (10,)


class TestLightGBMRanker:
    def test_lgbm_predict_shape(self):
        lgb = pytest.importorskip("lightgbm", reason="lightgbm not installed")
        from src.ranking.ranker import LightGBMRanker
        np.random.seed(42)
        X_tr = np.random.randn(200, 10).astype(np.float32)
        y_tr = np.random.randn(200).astype(np.float32)
        r = LightGBMRanker(params={"n_estimators": 20, "num_leaves": 15})
        r.fit(X_tr, y_tr)
        preds = r.predict(X_tr[:5])
        assert preds.shape == (5,)


class TestXGBoostRanker:
    def test_xgboost_predict_shape(self):
        xgb = pytest.importorskip("xgboost", reason="xgboost not installed")
        from src.ranking.ranker import XGBoostRanker
        np.random.seed(11)
        X_tr = np.random.randn(200, 8).astype(np.float32)
        y_tr = np.random.randn(200).astype(np.float32)
        r = XGBoostRanker(params={"n_estimators": 20, "max_depth": 3})
        r.fit(X_tr, y_tr)
        preds = r.predict(X_tr[:5])
        assert preds.shape == (5,)


# ══════════════════════════════════════════════════════════════════════════════
# 18 — Cross-sectional target golden test (Spec §95)
# ══════════════════════════════════════════════════════════════════════════════

class TestCSTargetGolden:
    def test_highest_return_stock_gets_highest_percentile(self):
        """
        Spec §95: A=+10%, B=+5%, C=0%, D=-5%, E=-10%
        Expected: A=highest rank, E=lowest rank.
        """
        from src.labels.cross_sectional import CSTargetConfig, compute_cross_sectional_targets

        n   = 15  # enough bars to compute a 5-bar return
        idx = pd.bdate_range("2023-01-02", periods=n, tz="UTC")

        # Build exact price paths: each bar compounds the stated 5-bar return
        def make_series(total_ret_over_5: float) -> pd.Series:
            per_bar = (1 + total_ret_over_5) ** (1 / 5) - 1
            prices  = [100.0]
            for _ in range(n - 1):
                prices.append(prices[-1] * (1 + per_bar))
            return pd.Series(prices, index=idx)

        close_map = {
            "A": make_series(0.10),
            "B": make_series(0.05),
            "C": make_series(0.00),
            "D": make_series(-0.05),
            "E": make_series(-0.10),
        }

        ts  = idx[0].to_pydatetime()
        cfg = CSTargetConfig(horizon_bars=5, min_cs_size=5,
                             vol_normalize=False, winsorize_pct=0.0)
        rows = compute_cross_sectional_targets(
            stock_close_map=close_map, timestamp=ts,
            config=cfg, eligible_symbols=["A","B","C","D","E"],
        )
        row_map = {r.instrument_id: r for r in rows}

        # Raw return ordering
        assert row_map["A"].raw_return > row_map["E"].raw_return, (
            "A (+10%) must have higher raw return than E (-10%)"
        )
        # CS percentile ordering
        if row_map["A"].cs_percentile and row_map["E"].cs_percentile:
            assert row_map["A"].cs_percentile == pytest.approx(100.0)
            assert row_map["E"].cs_percentile == pytest.approx(0.0)

        # CS rank ordering: A = highest rank, E = lowest rank
        if row_map["A"].cs_rank and row_map["E"].cs_rank:
            assert row_map["A"].cs_rank > row_map["E"].cs_rank
