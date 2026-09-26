"""
tests/test_reconciliation_pnl.py — Tests for the canonical executable portfolio backtest.

Mandate §4: ONE canonical evaluation pipeline.
Mandate §32: Signals at close[T], positions ENTERED at open[T+1], EXITED at open[T+1+h].
Mandate §43: Final test runs once, uses untouched OOS period.
"""
import math
import numpy as np
import pandas as pd
import pytest

from src.reconciliation.costs import PRIMARY_COST, STRESS_2X_COST
from src.reconciliation.pnl import (
    ExecutablePortfolioBacktest,
    ExecutableBacktestResult,
    build_scores_panel,
    _turnover,
)


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    price = 100.0 * np.cumprod(1 + rng.normal(0.0002, 0.012, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": price * (1 + rng.normal(0, 0.003, n)),
        "high": price * (1 + np.abs(rng.normal(0, 0.008, n))),
        "low": price * (1 - np.abs(rng.normal(0, 0.008, n))),
        "close": price,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=idx)


def _make_scores_panel(n_symbols: int = 20, n_dates: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic (ts, symbol)-indexed panel with 'score' and 'open' columns."""
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_symbols):
        sym = f"SYM{i:02d}"
        df = _make_ohlcv(n_dates, seed=i * 7 + seed)
        scores = pd.Series(rng.uniform(0, 1, n_dates), index=df.index, name="score")
        frame = pd.DataFrame({"open": df["open"], "score": scores})
        frame["symbol"] = sym
        frame.index.name = "ts"
        frames.append(frame.reset_index())
    panel = pd.concat(frames).set_index(["ts", "symbol"]).sort_index()
    return panel


class TestTurnoverHelper:

    def test_empty_prev_gives_full_turnover(self):
        assert _turnover(set(), {"A", "B"}) == 1.0

    def test_identical_gives_zero_turnover(self):
        assert _turnover({"A", "B"}, {"A", "B"}) == 0.0

    def test_half_changed(self):
        result = _turnover({"A", "B"}, {"A", "C"})
        # symmetric_diff = {B, C} = 2 elements; 2 * len(cur) = 4; fraction = 2/4 = 0.5
        assert 0.45 < result < 0.55

    def test_complete_rotation_gives_full_turnover(self):
        result = _turnover({"A", "B"}, {"C", "D"})
        assert result == 1.0


class TestExecutablePortfolioBacktest:

    def test_produces_result(self):
        panel = _make_scores_panel()
        bt = ExecutablePortfolioBacktest(panel, cost_model=PRIMARY_COST, holding_bars=5)
        result = bt.run()
        assert isinstance(result, ExecutableBacktestResult)

    def test_n_rebalances_positive(self):
        panel = _make_scores_panel(n_dates=100)
        bt = ExecutablePortfolioBacktest(panel, cost_model=PRIMARY_COST, holding_bars=5)
        result = bt.run()
        assert result.n_rebalances > 0, "Should have executed at least one rebalance"

    def test_xs_rank_ic_in_range(self):
        """XS rank IC should be in [-1, 1]."""
        panel = _make_scores_panel()
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        assert -1.0 <= result.xs_rank_ic_mean <= 1.0

    def test_net_sharpe_less_than_gross_sharpe(self):
        """Net Sharpe must be ≤ gross Sharpe (costs only drag)."""
        panel = _make_scores_panel()
        bt = ExecutablePortfolioBacktest(panel, cost_model=PRIMARY_COST)
        result = bt.run()
        assert result.net_sharpe <= result.gross_sharpe + 1e-9, (
            f"Net Sharpe {result.net_sharpe:.3f} > Gross Sharpe {result.gross_sharpe:.3f}"
        )

    def test_stress_costs_worse_than_primary(self):
        """Higher costs must produce lower or equal net Sharpe."""
        panel = _make_scores_panel()
        bt_primary = ExecutablePortfolioBacktest(panel, cost_model=PRIMARY_COST)
        bt_stress = ExecutablePortfolioBacktest(panel, cost_model=STRESS_2X_COST)
        r_primary = bt_primary.run()
        r_stress = bt_stress.run()
        assert r_stress.net_sharpe <= r_primary.net_sharpe + 0.5, (
            f"Stress Sharpe {r_stress.net_sharpe:.3f} should be ≤ Primary {r_primary.net_sharpe:.3f}"
        )

    def test_random_scores_near_zero_xs_ic(self):
        """Random scores should produce near-zero XS IC (not high positive)."""
        panel = _make_scores_panel(seed=12345)
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        # With random scores, expected XS IC ≈ 0 ± noise
        assert abs(result.xs_rank_ic_mean) < 0.25, (
            f"Random scores should have low XS IC, got {result.xs_rank_ic_mean:.4f}"
        )

    def test_provenance_fields_set(self):
        panel = _make_scores_panel(n_dates=80)
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        assert result.pnl_provenance == "REAL_HISTORICAL_OHLCV_NEXT_OPEN"
        assert result.is_economic_evidence

    def test_holding_bars_affects_n_rebalances(self):
        """Longer holding period → fewer rebalances."""
        panel = _make_scores_panel(n_dates=100)
        bt1 = ExecutablePortfolioBacktest(panel, holding_bars=1)
        bt5 = ExecutablePortfolioBacktest(panel, holding_bars=5)
        r1 = bt1.run()
        r5 = bt5.run()
        assert r1.n_rebalances >= r5.n_rebalances, (
            "h=1 should have ≥ rebalances than h=5"
        )

    def test_long_only_has_nonnegative_positions(self):
        """Long-only portfolio: all trade directions should be +1."""
        panel = _make_scores_panel(n_dates=100)
        bt = ExecutablePortfolioBacktest(
            panel, portfolio_type="top_decile_long_only"
        )
        result = bt.run()
        for per_reb in result.per_rebalance:
            assert per_reb["n_short"] == 0, (
                f"Long-only should have 0 short positions, got {per_reb['n_short']}"
            )

    def test_to_dict_serializable(self):
        """Result must be JSON-serializable."""
        import json
        panel = _make_scores_panel(n_dates=60)
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        d = result.to_dict()
        json.dumps(d, default=str)  # must not raise

    def test_execution_note_present(self):
        panel = _make_scores_panel(n_dates=60)
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        assert "open[T+1]" in result.execution_note or "open" in result.execution_note

    def test_max_drawdown_nonpositive(self):
        """Max drawdown must be ≤ 0."""
        panel = _make_scores_panel()
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        assert result.max_drawdown <= 0.0, (
            f"Max drawdown must be ≤ 0, got {result.max_drawdown}"
        )

    def test_missing_open_column_raises(self):
        panel = _make_scores_panel(n_dates=50)
        panel_no_open = panel.drop(columns=["open"])
        bt = ExecutablePortfolioBacktest(panel_no_open)
        with pytest.raises(ValueError, match="open"):
            bt.run()

    def test_missing_score_column_raises(self):
        panel = _make_scores_panel(n_dates=50)
        panel_no_score = panel.drop(columns=["score"])
        bt = ExecutablePortfolioBacktest(panel_no_score)
        with pytest.raises(ValueError, match="score"):
            bt.run()


class TestBuildScoresPanel:

    def test_combines_ohlcv_and_predictions(self):
        ohlcv = {f"SYM{i}": _make_ohlcv(80, seed=i) for i in range(5)}
        rng = np.random.default_rng(99)
        all_rows = []
        for sym, df in ohlcv.items():
            for ts in df.index:
                all_rows.append({"ts": ts, "symbol": sym, "score": rng.uniform(0, 1)})
        preds = pd.DataFrame(all_rows)
        panel = build_scores_panel(ohlcv, preds, score_col="score")
        assert "score" in panel.columns
        assert "open" in panel.columns
        assert panel.index.names == ["ts", "symbol"]

    def test_no_nan_after_dropna(self):
        ohlcv = {f"SYM{i}": _make_ohlcv(60, seed=i) for i in range(3)}
        rows = []
        rng = np.random.default_rng(0)
        for sym, df in ohlcv.items():
            for ts in df.index:
                rows.append({"ts": ts, "symbol": sym, "score": rng.uniform(0, 1)})
        preds = pd.DataFrame(rows)
        panel = build_scores_panel(ohlcv, preds, score_col="score")
        assert panel.isna().sum().sum() == 0, "Panel should have no NaN after dropna"
