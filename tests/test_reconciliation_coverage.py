"""
tests/test_reconciliation_coverage.py — Coverage tests for reconciliation modules.

Covers the uncovered paths in:
- src/reconciliation/baselines.py  (BaselineFamily, compare_model_to_baselines)
- src/reconciliation/targets.py    (compute_residual_next_open, beta_neutral, xs_rank, risk_adjusted)
- src/reconciliation/pnl.py        (multiple portfolio types, edge cases)
- src/reconciliation/matrix.py     (reconciler economic evaluation branches)
- src/validation/purged_kfold.py   (_iter_test_masks, CPCV build_backtest_paths)
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

# ─── helpers ──────────────────────────────────────────────────────────────────

def _make_ohlcv(n: int = 100, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = 100.0 * np.cumprod(1 + rng.normal(0.0002, 0.012, n))
    idx = pd.date_range("2022-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({
        "open": p * (1 + rng.normal(0, 0.003, n)),
        "high": p * (1 + np.abs(rng.normal(0, 0.008, n))),
        "low":  p * (1 - np.abs(rng.normal(0, 0.008, n))),
        "close": p,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=idx)


def _make_scores_panel(n_symbols: int = 15, n_dates: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for i in range(n_symbols):
        sym = f"SYM{i:02d}"
        df = _make_ohlcv(n_dates, seed=i * 3 + seed)
        scores = pd.Series(rng.uniform(0, 1, n_dates), index=df.index, name="score")
        frame = pd.DataFrame({"open": df["open"], "score": scores})
        frame["symbol"] = sym
        frame.index.name = "ts"
        frames.append(frame.reset_index())
    panel = pd.concat(frames).set_index(["ts", "symbol"]).sort_index()
    return panel


# ─── BaselineFamily ────────────────────────────────────────────────────────────

class TestBaselineFamily:

    def _make_panel(self, n_symbols: int = 10, n_dates: int = 80, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        rows = []
        dates = pd.date_range("2023-01-01", periods=n_dates, freq="D", tz="UTC")
        for d in dates:
            for i in range(n_symbols):
                sym = f"SYM{i:02d}"
                rows.append({
                    "ts": d, "symbol": sym,
                    "score": rng.uniform(0, 1),
                    "true_ret": rng.normal(0.0005, 0.015),
                    "ret_1": rng.normal(0, 0.01),
                    "ret_5": rng.normal(0, 0.02),
                    "vol_20": abs(rng.normal(0.015, 0.003)),
                    "rel_volume_20": abs(rng.normal(1.0, 0.3)),
                    "label": int(rng.integers(0, 2)),
                })
        return pd.DataFrame(rows).set_index(["ts", "symbol"])

    def test_run_all_returns_list(self):
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        feature_cols = ["ret_1", "ret_5", "vol_20", "rel_volume_20"]
        bf = BaselineFamily(panel, feature_cols=feature_cols, return_col="true_ret")
        results = bf.run_all()
        assert isinstance(results, list)
        assert len(results) > 0

    def test_zero_prediction_baseline(self):
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["ret_1"], return_col="true_ret")
        results = bf.run_all()
        zero = next((r for r in results if r.name == "zero_prediction"), None)
        assert zero is not None
        assert zero.xs_rank_ic_mean == 0.0
        assert not zero.beats_zero

    def test_momentum_baseline_runs(self):
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["ret_1", "ret_5"], return_col="true_ret")
        results = bf.run_all()
        mom = next((r for r in results if r.name == "momentum_5d"), None)
        assert mom is not None
        assert -1.0 <= mom.xs_rank_ic_mean <= 1.0

    def test_reversal_baseline_runs(self):
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["ret_1"], return_col="true_ret")
        results = bf.run_all()
        rev = next((r for r in results if r.name == "reversal_1d"), None)
        assert rev is not None

    def test_volatility_baseline_runs(self):
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["vol_20"], return_col="true_ret")
        results = bf.run_all()
        vol = next((r for r in results if r.name == "volatility_rank"), None)
        assert vol is not None

    def test_volume_baseline_runs(self):
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["rel_volume_20"], return_col="true_ret")
        results = bf.run_all()
        vol = next((r for r in results if r.name == "volume_rank"), None)
        assert vol is not None

    def test_baseline_result_to_dict(self):
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["ret_1"], return_col="true_ret")
        results = bf.run_all()
        for r in results:
            d = r.to_dict()
            assert "name" in d
            assert "xs_rank_ic_mean" in d

    def test_missing_ret_5_gracefully_handled(self):
        """Momentum baseline gracefully skips when col not in panel."""
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        panel_no_ret5 = panel.drop(columns=["ret_5"])
        bf = BaselineFamily(panel_no_ret5, feature_cols=["ret_1"], return_col="true_ret")
        results = bf.run_all()
        mom = next((r for r in results if r.name == "momentum_5d"), None)
        assert mom is not None
        assert mom.n_observations == 0  # gracefully returns empty result

    def test_historical_mean_baseline(self):
        from src.reconciliation.baselines import BaselineFamily
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["ret_1"], return_col="true_ret")
        results = bf.run_all()
        hm = next((r for r in results if r.name == "historical_mean"), None)
        assert hm is not None
        assert not hm.beats_zero

    def test_compare_model_to_baselines(self):
        from src.reconciliation.baselines import BaselineFamily, compare_model_to_baselines
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["ret_1", "ret_5"], return_col="true_ret")
        baseline_results = bf.run_all()
        model_result = {"xs_rank_ic_mean": 0.15}
        comparison = compare_model_to_baselines(model_result, baseline_results)
        assert "model_value" in comparison
        assert "verdict" in comparison
        assert "n_baselines" in comparison
        assert comparison["n_baselines"] > 0

    def test_compare_model_beats_some(self):
        from src.reconciliation.baselines import BaselineFamily, compare_model_to_baselines
        panel = self._make_panel()
        bf = BaselineFamily(panel, feature_cols=["ret_1"], return_col="true_ret")
        baseline_results = bf.run_all()
        # With model IC=0, it beats zero_prediction (IC=0) but may not beat others
        model_result = {"xs_rank_ic_mean": 0.0}
        comparison = compare_model_to_baselines(model_result, baseline_results)
        assert comparison["verdict"] in (
            "MODEL_BEATS_ALL_BASELINES",
            "MODEL_BEATS_SOME_BASELINES",
            "MODEL_DOES_NOT_BEAT_BASELINES",
        )


# ─── compute_residual_next_open with benchmark ───────────────────────────────

class TestComputeResidualWithBenchmark:

    def test_residual_with_nifty_benchmark(self):
        from src.reconciliation.targets import compute_residual_next_open
        ohlcv = {
            "RELIANCE": _make_ohlcv(80, seed=1),
            "NIFTY": _make_ohlcv(80, seed=99),
        }
        result = compute_residual_next_open(ohlcv, benchmark_sym="NIFTY", horizon=5)
        assert "fwd_return_residual" in result.columns
        assert result["benchmark_available"].any()
        raw = result["fwd_return_next_open_raw"].dropna()
        res = result["fwd_return_residual"].dropna()
        assert len(raw) > 0 and len(res) > 0

    def test_beta_neutral_with_benchmark(self):
        from src.reconciliation.targets import compute_beta_neutral
        ohlcv = {
            "RELIANCE": _make_ohlcv(120, seed=2),
            "NIFTY": _make_ohlcv(120, seed=10),
        }
        result = compute_beta_neutral(ohlcv, benchmark_sym="NIFTY", horizon=5, beta_window=30)
        assert "fwd_return_beta_neutral" in result.columns

    def test_beta_neutral_without_benchmark(self):
        from src.reconciliation.targets import compute_beta_neutral
        ohlcv = {"RELIANCE": _make_ohlcv(80, seed=3)}
        result = compute_beta_neutral(ohlcv, benchmark_sym="NIFTY", horizon=5)
        # Falls back gracefully
        assert "fwd_return_next_open_raw" in result.columns

    def test_compute_risk_adjusted(self):
        from src.reconciliation.targets import compute_risk_adjusted
        ohlcv = {
            "RELIANCE": _make_ohlcv(80, seed=4),
            "TCS": _make_ohlcv(80, seed=5),
        }
        result = compute_risk_adjusted(ohlcv, horizon=5, vol_window=20)
        assert "fwd_return_risk_adjusted" in result.columns
        # Should be mostly finite
        valid = result["fwd_return_risk_adjusted"].dropna()
        assert len(valid) > 0

    def test_compute_xs_rank_with_multiple_symbols(self):
        from src.reconciliation.targets import compute_cross_sectional_rank
        ohlcv = {f"SYM{i}": _make_ohlcv(80, seed=i) for i in range(6)}
        result = compute_cross_sectional_rank(ohlcv, horizon=5)
        assert "fwd_return_xs_rank" in result.columns
        rank_vals = result["fwd_return_xs_rank"].dropna()
        assert len(rank_vals) > 0
        assert (rank_vals >= 0).all() and (rank_vals <= 1).all()

    def test_registration_hash_matches(self):
        from src.reconciliation.targets import (
            TARGET_REGISTRATION_HASH, _compute_registration_hash
        )
        assert TARGET_REGISTRATION_HASH == _compute_registration_hash()


# ─── ExecutablePortfolioBacktest — additional portfolio types ─────────────────

class TestPortfolioTypes:

    def test_rank_weight_long_short(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel(n_dates=80)
        bt = ExecutablePortfolioBacktest(
            panel, holding_bars=5,
            portfolio_type="rank_weight_long_short",
        )
        result = bt.run()
        assert result.n_rebalances >= 0

    def test_top_quintile_long_only(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel(n_dates=80)
        bt = ExecutablePortfolioBacktest(
            panel, holding_bars=5,
            portfolio_type="top_quintile_long_only",
        )
        result = bt.run()
        assert result.n_rebalances >= 0
        for pr in result.per_rebalance:
            assert pr["n_short"] == 0

    def test_top_bottom_quintile_long_short(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel(n_dates=80)
        bt = ExecutablePortfolioBacktest(
            panel, holding_bars=5,
            portfolio_type="top_bottom_quintile_long_short",
        )
        result = bt.run()
        assert result.n_rebalances >= 0

    def test_equal_weight_long(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel(n_dates=80)
        bt = ExecutablePortfolioBacktest(
            panel, holding_bars=5,
            portfolio_type="equal_weight_long",
        )
        result = bt.run()
        assert result.n_rebalances >= 0
        for pr in result.per_rebalance:
            assert pr["n_short"] == 0

    def test_equal_weight_long_short(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel(n_dates=80)
        bt = ExecutablePortfolioBacktest(
            panel, holding_bars=5,
            portfolio_type="equal_weight_long_short",
        )
        result = bt.run()
        assert result.n_rebalances >= 0

    def test_insufficient_timestamps_raises(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel(n_dates=3)
        bt = ExecutablePortfolioBacktest(panel, holding_bars=5)
        with pytest.raises(ValueError):
            bt.run()

    def test_result_has_xs_ic_in_range(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel()
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        if result.n_rebalances > 0:
            assert -1.0 <= result.xs_rank_ic_mean <= 1.0

    def test_sortino_returned(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel()
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        assert math.isfinite(result.sortino)

    def test_calmar_returned(self):
        from src.reconciliation.pnl import ExecutablePortfolioBacktest
        panel = _make_scores_panel()
        bt = ExecutablePortfolioBacktest(panel)
        result = bt.run()
        assert math.isfinite(result.calmar)


# ─── ReconciliationMatrix economic evaluation branches ───────────────────────

class TestMatrixEconomicEvaluation:

    def _make_minimal_reconciler(self):
        """Build a reconciler with empty OHLCV (exercises the no-OHLCV path)."""
        import pickle, hashlib, tempfile
        from pathlib import Path
        from src.reconciliation.matrix import PredictionToPnLReconciler
        from src.reconciliation.costs import PRIMARY_COST
        from sklearn.linear_model import LogisticRegression

        # Synthetic dataset
        rng = np.random.default_rng(42)
        n_sym, n_dates = 5, 40
        dates = pd.date_range("2022-01-01", periods=n_dates, freq="D", tz="UTC")
        rows = []
        for d in dates:
            for s in range(n_sym):
                ret = rng.choice([-0.02, 0.02, rng.normal(0, 0.01)], p=[0.44, 0.44, 0.12])
                rows.append({
                    "symbol": f"SYM{s}",
                    "label": int(ret > 0),
                    "realized_return": float(ret),
                    "ret_1": rng.normal(0, 0.01),
                    "ret_5": rng.normal(0, 0.02),
                    "ret_10": rng.normal(0, 0.025),
                    "ret_20": rng.normal(0, 0.03),
                    "log_ret_1": rng.normal(0, 0.01),
                    "vol_5": abs(rng.normal(0.01, 0.003)),
                    "vol_10": abs(rng.normal(0.012, 0.003)),
                    "vol_20": abs(rng.normal(0.015, 0.003)),
                    "atr_14_pct": abs(rng.normal(0.02, 0.005)),
                    "rel_volume_20": abs(rng.normal(1.0, 0.3)),
                    "volume_zscore_20": rng.normal(0, 1),
                    "vwap_distance_pct": rng.normal(0, 0.005),
                    "rsi_14": rng.uniform(20, 80),
                    "macd_hist": rng.normal(0, 0.001),
                    "stoch_k_14": rng.uniform(0, 1),
                    "ema_5_20": rng.normal(0, 0.005),
                    "ema_10_50": rng.normal(0, 0.008),
                    "adx_14": rng.uniform(15, 45),
                    "hl_range_pct": abs(rng.normal(0.015, 0.005)),
                    "close_position": rng.uniform(0, 1),
                    "gap_pct": rng.normal(0, 0.003),
                    "bb_zscore_20": rng.normal(0, 1),
                    "skew_20": rng.normal(0, 0.5),
                    "kurt_20": rng.normal(3, 1),
                })
        df = pd.DataFrame(rows, index=np.tile(dates, n_sym).flatten()[:len(rows)])
        df.index.name = "ts"
        df = df.sort_index()

        # Minimal sklearn model
        X = df[["ret_1","ret_5","ret_10","ret_20","log_ret_1",
                "vol_5","vol_10","vol_20","atr_14_pct","rel_volume_20",
                "volume_zscore_20","vwap_distance_pct","rsi_14","macd_hist",
                "stoch_k_14","ema_5_20","ema_10_50","adx_14","hl_range_pct",
                "close_position","gap_pct","bb_zscore_20","skew_20","kurt_20"]].to_numpy(float)
        y = df["label"].to_numpy(int)
        clf = LogisticRegression(max_iter=100, random_state=0)
        clf.fit(X, y)
        model_dict = {"estimator": clf, "calibrator": None, "feature_names": []}

        # Write to temp files
        tmp = tempfile.mkdtemp()
        ds_path = Path(tmp) / "data.parquet"
        df.to_parquet(ds_path)
        model_bytes = pickle.dumps(model_dict)
        sha = hashlib.sha256(model_bytes).hexdigest()
        model_path = Path(tmp) / "model.pkl"
        model_path.write_bytes(model_bytes)

        r = PredictionToPnLReconciler.__new__(PredictionToPnLReconciler)
        r.dataset_path = ds_path
        r.model_path = model_path
        r.ohlcv = {}  # no OHLCV — tests the no-OHLCV branch
        r.cost = PRIMARY_COST
        r.h = 5
        r.code_sha = "test"
        r.MODEL_SHA256 = sha
        r.BARRIER_PCT = 0.02
        r.HORIZON = 5
        r.ML_REPORTED_IC = 0.486
        r.ML_REPORTED_SHARPE_10BPS = 5.47
        r.ML_REPORTED_PBO = 0.0
        r.ML_REPORTED_N_TRADES = 200
        return r, df

    def test_economic_evaluation_no_ohlcv(self):
        """Economic evaluation without OHLCV falls back gracefully."""
        r, df = self._make_minimal_reconciler()
        model_bytes = r.model_path.read_bytes()
        import pickle
        model_dict = pickle.loads(model_bytes)
        pred_df = r._generate_predictions(df, model_dict)
        econ, details = r._economic_evaluation(pred_df, df)
        # With no OHLCV, XS IC computed from predictions only
        assert econ.xs_rank_ic_h5 == 0.0 or math.isfinite(econ.xs_rank_ic_h5)
        assert math.isfinite(econ.ts_ic_vs_continuous)

    def test_full_run_without_ohlcv(self):
        """Full reconciler.run() without OHLCV should not raise."""
        r, _ = self._make_minimal_reconciler()
        matrix = r.run()
        assert matrix.lifecycle_state in ("RESEARCH_READY", "PAPER_ELIGIBLE_PENDING_ROBUSTNESS")
        assert len(matrix.divergences) == 5

    def test_xs_ic_from_ohlcv_branch(self):
        """Test _compute_xs_ic_from_ohlcv with actual OHLCV data."""
        r, df = self._make_minimal_reconciler()
        # Add some OHLCV
        ohlcv = {f"SYM{i}": _make_ohlcv(80, seed=i) for i in range(5)}
        r.ohlcv = ohlcv
        import pickle
        model_dict = pickle.loads(r.model_path.read_bytes())
        pred_df = r._generate_predictions(df, model_dict)
        xs_result = r._compute_xs_ic_from_ohlcv(pred_df)
        assert "xs_rank_ic_mean" in xs_result

    def test_attach_true_continuous_return(self):
        """_attach_true_continuous_return populates true_next_open_return."""
        r, df = self._make_minimal_reconciler()
        ohlcv = {f"SYM{i}": _make_ohlcv(80, seed=i) for i in range(5)}
        r.ohlcv = ohlcv
        import pickle
        model_dict = pickle.loads(r.model_path.read_bytes())
        pred_df = r._generate_predictions(df, model_dict)
        # true_next_open_return should be populated for symbols in OHLCV
        result = r._attach_true_continuous_return(pred_df)
        assert "true_next_open_return" in result.columns

    def test_certification_gates_structure(self):
        """Gates dict must have all 16 required gates."""
        from src.reconciliation.matrix import (
            PredictionToPnLReconciler, MLEvaluationAudit, EconomicEvaluationAudit
        )
        r, _ = self._make_minimal_reconciler()
        from src.reconciliation.matrix import MLEvaluationAudit, EconomicEvaluationAudit
        ml = MLEvaluationAudit(
            ic_pearson=0.4, ic_rank=0.3, ic_vs_barrier_clamped=0.4,
            ic_inflation_from_barrier=1.0, net_sharpe_reported=3.0,
            n_trades_in_sharpe=100, sharpe_computation_note="",
            overlapping_label_autocorr=0.1, t_stat_raw=20.0,
            t_stat_corrected=9.0, pbo=0.0, methodology="",
        )
        econ = EconomicEvaluationAudit(
            xs_rank_ic_h5=0.05, xs_rank_ic_std=0.05,
            xs_rank_ic_positive_fraction=0.6, ts_ic_vs_continuous=0.2,
            net_sharpe_ls=0.5, net_sharpe_lo=0.3, gross_sharpe_ls=1.0,
            cost_drag_annual=0.05, max_drawdown=-0.1, mean_daily_turnover=0.2,
            n_rebalances=10, n_trades_actual=50, cost_scenario="conservative",
            methodology="",
        )
        gates, blockers, verdict, state = r._lifecycle_determination(ml, econ)
        assert len(gates) == 16
        for gate_name in gates:
            assert gate_name.startswith("GATE_")


# ─── PurgedKFold — _iter_test_masks and sklearn compatibility ─────────────────

class TestPurgedKFoldCoverage:

    def test_iter_test_masks(self):
        from src.validation.purged_kfold import PurgedKFold
        n = 100
        X = np.zeros((n, 3))
        splitter = PurgedKFold(n_splits=5)
        masks = list(splitter._iter_test_masks(X))
        assert len(masks) == 5
        for mask in masks:
            assert mask.sum() > 0

    def test_get_n_splits(self):
        from src.validation.purged_kfold import PurgedKFold
        splitter = PurgedKFold(n_splits=4)
        assert splitter.get_n_splits() == 4

    def test_split_with_t1_purging(self):
        """Verify purging removes training obs whose t1 overlaps test."""
        from src.validation.purged_kfold import PurgedKFold, build_t1_series
        n = 100
        X = np.zeros((n, 2))
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        t1 = build_t1_series(idx, horizon_bars=10)
        splitter = PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.02)
        folds = list(splitter.split(X, groups=idx))
        assert len(folds) == 5
        for train_idx, test_idx in folds:
            # training indices should only be before test start
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert max(train_idx) < min(test_idx), (
                    "Training obs must be chronologically before test obs"
                )

    def test_cpcv_build_backtest_paths(self):
        from src.validation.purged_kfold import CombinatorialPurgedCV, CPCVConfig
        n = 120
        X = np.zeros((n, 2))
        cfg = CPCVConfig(n_splits=6, n_test_splits=2)
        engine = CombinatorialPurgedCV(cfg)
        folds = engine.split(X)
        paths = engine.build_backtest_paths(folds)
        assert len(paths) > 0
        for path in paths:
            assert len(path) >= 1

    def test_cpcv_with_t1_purging(self):
        """CPCV with t1 series sets n_purged > 0."""
        from src.validation.purged_kfold import CombinatorialPurgedCV, CPCVConfig, build_t1_series
        n = 120
        X = np.zeros((n, 2))
        idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
        t1 = build_t1_series(idx, horizon_bars=10)
        cfg = CPCVConfig(n_splits=6, n_test_splits=2)
        engine = CombinatorialPurgedCV(cfg, t1=t1)
        folds = engine.split(X, groups=idx)
        total_purged = sum(f.n_purged for f in folds)
        assert total_purged > 0

    def test_cpcv_config_invalid_raises(self):
        from src.validation.purged_kfold import CPCVConfig
        with pytest.raises(ValueError):
            CPCVConfig(n_splits=5, n_test_splits=5)
        with pytest.raises(ValueError):
            CPCVConfig(n_splits=1, n_test_splits=1)

    def test_build_t1_series_basic(self):
        from src.validation.purged_kfold import build_t1_series
        idx = pd.date_range("2023-01-01", periods=30, freq="D", tz="UTC")
        t1 = build_t1_series(idx, horizon_bars=5)
        assert len(t1) == 30
        assert t1.iloc[0] == idx[5]
        assert t1.iloc[-1] == idx[-1]  # capped at last index


# ─── Full run with OHLCV (exercises portfolio backtest path) ─────────────────

class TestMatrixWithOHLCV:
    """Exercises the portfolio backtest branch of PredictionToPnLReconciler."""

    def _make_reconciler_with_ohlcv(self):
        """Build reconciler WITH OHLCV data so the portfolio backtest runs."""
        import pickle, hashlib, tempfile
        from pathlib import Path
        from src.reconciliation.matrix import PredictionToPnLReconciler
        from src.reconciliation.costs import PRIMARY_COST
        from sklearn.linear_model import LogisticRegression

        rng = np.random.default_rng(7)
        n_sym, n_dates = 12, 80
        syms = [f"SYM{i:02d}" for i in range(n_sym)]
        dates = pd.date_range("2022-01-01", periods=n_dates, freq="D", tz="UTC")

        rows = []
        for d in dates:
            for sym in syms:
                ret = rng.choice([-0.02, 0.02, rng.normal(0, 0.01)], p=[0.44, 0.44, 0.12])
                rows.append({
                    "symbol": sym, "label": int(ret > 0),
                    "realized_return": float(ret),
                    "ret_1": rng.normal(0, 0.01), "ret_5": rng.normal(0, 0.02),
                    "ret_10": rng.normal(0, 0.025), "ret_20": rng.normal(0, 0.03),
                    "log_ret_1": rng.normal(0, 0.01),
                    "vol_5": abs(rng.normal(0.01, 0.003)),
                    "vol_10": abs(rng.normal(0.012, 0.003)),
                    "vol_20": abs(rng.normal(0.015, 0.003)),
                    "atr_14_pct": abs(rng.normal(0.02, 0.005)),
                    "rel_volume_20": abs(rng.normal(1.0, 0.3)),
                    "volume_zscore_20": rng.normal(0, 1),
                    "vwap_distance_pct": rng.normal(0, 0.005),
                    "rsi_14": rng.uniform(20, 80), "macd_hist": rng.normal(0, 0.001),
                    "stoch_k_14": rng.uniform(0, 1), "ema_5_20": rng.normal(0, 0.005),
                    "ema_10_50": rng.normal(0, 0.008), "adx_14": rng.uniform(15, 45),
                    "hl_range_pct": abs(rng.normal(0.015, 0.005)),
                    "close_position": rng.uniform(0, 1), "gap_pct": rng.normal(0, 0.003),
                    "bb_zscore_20": rng.normal(0, 1), "skew_20": rng.normal(0, 0.5),
                    "kurt_20": rng.normal(3, 1),
                })
        df = pd.DataFrame(rows, index=np.tile(dates, n_sym).flatten()[:len(rows)])
        df.index.name = "ts"
        df = df.sort_index()

        X = df[["ret_1","ret_5","ret_10","ret_20","log_ret_1",
                "vol_5","vol_10","vol_20","atr_14_pct","rel_volume_20",
                "volume_zscore_20","vwap_distance_pct","rsi_14","macd_hist",
                "stoch_k_14","ema_5_20","ema_10_50","adx_14","hl_range_pct",
                "close_position","gap_pct","bb_zscore_20","skew_20","kurt_20"]].to_numpy(float)
        y = df["label"].to_numpy(int)
        clf = LogisticRegression(max_iter=200, random_state=0)
        clf.fit(X, y)
        model_dict = {"estimator": clf, "calibrator": None, "feature_names": []}

        tmp = tempfile.mkdtemp()
        ds_path = Path(tmp) / "data.parquet"
        df.to_parquet(ds_path)
        model_bytes = pickle.dumps(model_dict)
        sha = hashlib.sha256(model_bytes).hexdigest()
        model_path = Path(tmp) / "model.pkl"
        model_path.write_bytes(model_bytes)

        # Build aligned OHLCV with matching dates and symbols
        ohlcv = {}
        for sym in syms:
            ohlcv[sym] = _make_ohlcv(n_dates, seed=int(sym[3:]) + 10)

        r = PredictionToPnLReconciler.__new__(PredictionToPnLReconciler)
        r.dataset_path = ds_path
        r.model_path = model_path
        r.ohlcv = ohlcv
        r.cost = PRIMARY_COST
        r.h = 5
        r.code_sha = "test_ohlcv"
        r.MODEL_SHA256 = sha
        r.BARRIER_PCT = 0.02
        r.HORIZON = 5
        r.ML_REPORTED_IC = 0.486
        r.ML_REPORTED_SHARPE_10BPS = 5.47
        r.ML_REPORTED_PBO = 0.0
        r.ML_REPORTED_N_TRADES = 200
        return r, df

    def test_full_run_with_ohlcv(self):
        """Full reconciler.run() with OHLCV exercises the portfolio backtest path."""
        r, _ = self._make_reconciler_with_ohlcv()
        matrix = r.run()
        assert matrix.lifecycle_state in ("RESEARCH_READY", "PAPER_ELIGIBLE_PENDING_ROBUSTNESS")
        # Economic evaluation should have run
        econ = matrix.economic_evaluation
        assert math.isfinite(econ.xs_rank_ic_h5)

    def test_scores_panel_built_from_predictions(self):
        """_build_scores_panel_from_predictions should succeed with OHLCV."""
        r, df = self._make_reconciler_with_ohlcv()
        import pickle
        model_dict = pickle.loads(r.model_path.read_bytes())
        pred_df = r._generate_predictions(df, model_dict)
        panel = r._build_scores_panel_from_predictions(pred_df)
        assert "score" in panel.columns
        assert "open" in panel.columns
        assert len(panel) > 0

    def test_economic_evaluation_with_ohlcv(self):
        """Economic evaluation with OHLCV runs portfolio backtest."""
        r, df = self._make_reconciler_with_ohlcv()
        import pickle
        model_dict = pickle.loads(r.model_path.read_bytes())
        pred_df = r._generate_predictions(df, model_dict)
        econ, details = r._economic_evaluation(pred_df, df)
        # With OHLCV provided, should have portfolio results
        assert math.isfinite(econ.ts_ic_vs_continuous)
        # n_rebalances may be 0 if panel is too short, but no crash
        assert econ.n_rebalances >= 0

    def test_ts_ic_vs_continuous_branch(self):
        """When true_next_open_return is populated, ts_ic branch runs."""
        r, df = self._make_reconciler_with_ohlcv()
        import pickle
        model_dict = pickle.loads(r.model_path.read_bytes())
        pred_df = r._generate_predictions(df, model_dict)
        # Attach true returns first
        pred_df = r._attach_true_continuous_return(pred_df)
        # Now force has_true_ret = True by inflating count
        if "true_next_open_return" in pred_df.columns:
            pred_df["true_next_open_return"] = pred_df["true_next_open_return"].fillna(
                pred_df["realized_return"]
            )
        econ, _ = r._economic_evaluation(pred_df, df)
        assert math.isfinite(econ.ts_ic_vs_continuous)
