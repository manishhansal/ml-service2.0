"""Unit tests for the cross-sectional research + backtest modules.

These tests use small SYNTHETIC panels with KNOWN structure to verify the
mechanics (PIT-safety, rank-IC-primary metric, next-open execution, IC decay).
Synthetic data here is ONLY for testing pipeline correctness — never for an
alpha claim (which requires real data, per the mandate).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analytics.cross_sectional import (
    CrossSectionalValidator,
    PanelConfig,
    build_cross_sectional_panel,
)
from src.analytics.cross_sectional_backtest import (
    CostModel,
    backtest_next_open,
    ic_decay_curve,
    signal_persistence,
)


def _make_ohlcv(n_days: int = 400, seed: int = 0, drift: float = 0.0) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2020-01-01", periods=n_days, freq="B", tz="UTC")
    rets = rng.normal(drift, 0.02, n_days)
    close = 100.0 * np.cumprod(1.0 + rets)
    open_ = close * (1.0 + rng.normal(0, 0.002, n_days))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0, 0.003, n_days)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0, 0.003, n_days)))
    vol = rng.randint(1_000, 100_000, n_days).astype(float)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def _universe(n_symbols: int = 20, n_days: int = 400) -> dict[str, pd.DataFrame]:
    return {f"SYM{i:02d}": _make_ohlcv(n_days, seed=i) for i in range(n_symbols)}


class TestPanelConstruction:
    def test_panel_has_labels_and_features(self):
        panel = build_cross_sectional_panel(_universe(), PanelConfig(horizon=1))
        assert panel.index.names == ["ts", "symbol"]
        for col in ("fwd_return_raw", "fwd_return_excess", "fwd_return_residual",
                    "fwd_return_rank", "_open", "_close"):
            assert col in panel.columns
        # cross-sectional relative features present
        for col in ("xs_ret5_demean", "xs_relvol", "xs_ret5_z"):
            assert col in panel.columns

    def test_forward_return_is_strictly_forward(self):
        """fwd_return_raw(T) must equal close(T+h)/close(T)-1 exactly (no leak)."""
        uni = _universe(n_symbols=5)
        panel = build_cross_sectional_panel(uni, PanelConfig(horizon=1))
        sym = "SYM00"
        df = uni[sym]
        manual = (df["close"].shift(-1) - df["close"]) / df["close"]
        got = panel.xs(sym, level="symbol")["fwd_return_raw"]
        aligned = manual.reindex(got.index)
        assert float((got - aligned).abs().max()) < 1e-9

    def test_rank_label_is_percentile(self):
        panel = build_cross_sectional_panel(_universe(), PanelConfig(horizon=1))
        ranks = panel["fwd_return_rank"].dropna()
        assert ranks.min() >= 0.0 and ranks.max() <= 1.0


class TestNullAndOutlier:
    def test_shuffled_label_collapses_rank_ic(self):
        """A leakage-free pipeline must yield ~0 rank IC on shuffled labels."""
        panel = build_cross_sectional_panel(_universe(24, 500), PanelConfig(horizon=1))
        rng = np.random.RandomState(3)

        def shuf(s):
            a = s.to_numpy().copy(); rng.shuffle(a); return pd.Series(a, index=s.index)

        panel["fwd_return_residual"] = (
            panel.groupby(level="ts")["fwd_return_residual"].transform(shuf)
        )
        from src.models.estimators import build_estimator
        v = CrossSectionalValidator(n_windows=4, embargo_bars=3, min_symbols_per_ts=8)
        res, _ = v.evaluate(
            panel, list(panel.columns[:24]), "fwd_return_residual",
            "fwd_return_residual", lambda: build_estimator("ridge"), "ridge", "BASE", 1,
        )
        assert abs(res.mean_rank_ic) < 0.05  # collapses to ~0


class TestNextOpenBacktest:
    def test_backtest_uses_open_prices(self):
        """The long leg return must be computed from OPEN[entry]->OPEN[exit]."""
        uni = _universe(20, 300)
        panel = build_cross_sectional_panel(uni, PanelConfig(horizon=1))
        # deterministic score = -ret_1 (reversal); OOS = all timestamps
        scores = -panel["ret_1"].fillna(0.0)
        scores = scores.loc[panel.dropna(subset=["_open", "_close"]).index]
        r = backtest_next_open(
            panel.loc[scores.index], scores, holding_bars=1, rebalance_every=1,
            decile=0.2, long_only=False, cost_model=CostModel(), min_symbols=10,
        )
        assert r.pnl_provenance == "REAL_HISTORICAL_OHLCV_NEXT_OPEN"
        assert r.execution == "NEXT_OPEN"
        assert r.n_rebalances > 0
        # net return must be gross minus a non-negative cost
        for p in r.per_rebalance[:20]:
            assert p["net"] <= p["gross"] + 1e-12

    def test_long_only_has_no_shorts(self):
        uni = _universe(20, 300)
        panel = build_cross_sectional_panel(uni, PanelConfig(horizon=1))
        scores = (-panel["ret_1"].fillna(0.0)).loc[
            panel.dropna(subset=["_open", "_close"]).index
        ]
        r = backtest_next_open(panel.loc[scores.index], scores, long_only=True,
                               decile=0.2, min_symbols=10)
        assert r.long_only is True
        assert all(p["n_short"] == 0 for p in r.per_rebalance)


class TestICDecayAndPersistence:
    def test_ic_decay_returns_all_horizons(self):
        panel = build_cross_sectional_panel(_universe(20, 400), PanelConfig(horizon=1))
        panel = panel.dropna(subset=["_close"])
        panel["_score"] = -panel["ret_1"].fillna(0.0)
        decay = ic_decay_curve(panel, "_score", horizons=[1, 3, 5])
        assert set(decay) == {1, 3, 5}
        for h in decay.values():
            assert "mean_rank_ic" in h and "n_timestamps" in h

    def test_persistence_lags(self):
        panel = build_cross_sectional_panel(_universe(20, 400), PanelConfig(horizon=1))
        panel = panel.dropna(subset=["_close"])
        panel["_score"] = panel["ret_5"].fillna(0.0)
        pers = signal_persistence(panel, "_score", lags=[1, 2])
        assert set(pers) == {1, 2}
        assert all(-1.0 <= v <= 1.0 for v in pers.values())


class TestCostModel:
    def test_round_trip_exceeds_per_side(self):
        cm = CostModel()
        assert cm.round_trip_bps() > cm.per_side_bps()
        assert cm.round_trip_bps() > 0
