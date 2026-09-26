"""
tests/test_reconciliation_matrix.py — Tests for the prediction-to-P&L reconciliation matrix.

Mandate §6: EXACT RECONCILIATION MATRIX.
Mandate §5 PHASE 2: take exact persisted predictions, DO NOT retrain.
"""
import json
import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.reconciliation.matrix import (
    PredictionToPnLReconciler,
    ReconciliationMatrix,
    DivergenceRecord,
)
from src.reconciliation.costs import PRIMARY_COST


def _make_synthetic_dataset(n_symbols: int = 5, n_dates: int = 60) -> pd.DataFrame:
    """Minimal synthetic dataset with the columns the reconciler expects."""
    rng = np.random.default_rng(42)
    rows = []
    dates = pd.date_range("2022-01-01", periods=n_dates, freq="D", tz="UTC")
    syms = [f"SYM{i}" for i in range(n_symbols)]
    for d in dates:
        for sym in syms:
            ret = rng.choice([-0.02, 0.02, rng.normal(0, 0.01)],
                             p=[0.45, 0.44, 0.11])
            rows.append({
                "symbol": sym,
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
                "open": rng.uniform(100, 200),
                "close": rng.uniform(100, 200),
                "high": rng.uniform(100, 200),
                "low": rng.uniform(100, 200),
                "volume": rng.uniform(1e6, 1e7),
            })
    df = pd.DataFrame(rows, index=np.tile(dates, n_symbols).flatten()[:len(rows)])
    df.index.name = "ts"
    # sort by (ts, symbol)
    df = df.sort_index()
    return df


def _make_fake_model() -> dict:
    """Minimal model dict compatible with the reconciler."""
    from sklearn.linear_model import LogisticRegression

    n_features = 24
    rng = np.random.default_rng(42)
    X = rng.normal(0, 1, (200, n_features))
    y = (X[:, 0] > 0).astype(int)

    clf = LogisticRegression(max_iter=200, random_state=42)
    clf.fit(X, y)

    return {
        "estimator": clf,
        "calibrator": None,
        "feature_names": [f"f{i}" for i in range(n_features)],
    }


class TestPredictionToPnLReconciler:

    def _make_reconciler(self, tmp_path: Path) -> PredictionToPnLReconciler:
        """Set up a reconciler with temporary synthetic artifacts."""
        # Write dataset
        df = _make_synthetic_dataset()
        ds_path = tmp_path / "data.parquet"
        df.to_parquet(ds_path)

        # Write model
        model_dict = _make_fake_model()
        model_bytes = pickle.dumps(model_dict)
        import hashlib
        sha = hashlib.sha256(model_bytes).hexdigest()
        model_path = tmp_path / "model.pkl"
        model_path.write_bytes(model_bytes)

        # Patch the expected SHA to match what we just wrote
        reconciler = PredictionToPnLReconciler.__new__(PredictionToPnLReconciler)
        reconciler.dataset_path = ds_path
        reconciler.model_path = model_path
        reconciler.ohlcv = {}
        reconciler.cost = PRIMARY_COST
        reconciler.h = 5
        reconciler.code_sha = "test_sha"
        # Override SHA check by patching the constant
        reconciler.MODEL_SHA256 = sha
        reconciler.BARRIER_PCT = 0.02
        reconciler.HORIZON = 5
        reconciler.ML_REPORTED_IC = 0.486
        reconciler.ML_REPORTED_SHARPE_10BPS = 5.47
        reconciler.ML_REPORTED_PBO = 0.0
        reconciler.ML_REPORTED_N_TRADES = 300
        return reconciler

    def test_audit_dataset_returns_datasetaudit(self, tmp_path):
        r = self._make_reconciler(tmp_path)
        df = r._load_dataset()
        audit = r._audit_dataset(df)
        assert audit.n_rows > 0
        assert audit.n_symbols > 0
        assert 0 <= audit.barrier_fraction <= 1
        assert len(audit.notes) > 0

    def test_barrier_fraction_quantified(self, tmp_path):
        """The dataset with 89% barrier hits should report that."""
        r = self._make_reconciler(tmp_path)
        df = r._load_dataset()
        audit = r._audit_dataset(df)
        # Our synthetic data is ~89% at ±0.02
        assert audit.barrier_fraction > 0.5, (
            f"Synthetic barrier data should show >50% at barrier, got {audit.barrier_fraction}"
        )

    def test_ml_audit_replicates_walkforward(self, tmp_path):
        r = self._make_reconciler(tmp_path)
        df = r._load_dataset()
        model_dict = _make_fake_model()
        pred_df = r._generate_predictions(df, model_dict)
        ml_audit = r._ml_evaluation_audit(pred_df, df)
        # IC should be numeric
        assert np.isfinite(ml_audit.ic_pearson)
        # t-stat raw > corrected for h=5
        assert ml_audit.t_stat_raw > ml_audit.t_stat_corrected

    def test_divergences_have_five_entries(self, tmp_path):
        """There are exactly 5 pre-identified divergence sources."""
        r = self._make_reconciler(tmp_path)
        from src.reconciliation.matrix import DatasetAudit, MLEvaluationAudit, EconomicEvaluationAudit
        # Mock minimal audit objects
        ds = DatasetAudit(
            n_rows=127122, n_symbols=65, n_dates=1525, symbols=[],
            date_start="2021-01-01", date_end="2026-09-23",
            barrier_fraction=0.888, overlap_correction_factor=2.236,
            effective_n=25424, label_autocorr_lag1=0.1128,
            outcome_dist={}, survivorship="CURRENT_UNIVERSE_ONLY", notes=[],
        )
        ml = MLEvaluationAudit(
            ic_pearson=0.486, ic_rank=0.32, ic_vs_barrier_clamped=0.486,
            ic_inflation_from_barrier=1.0, net_sharpe_reported=5.47,
            n_trades_in_sharpe=127122, sharpe_computation_note="test",
            overlapping_label_autocorr=0.1128, t_stat_raw=90.0,
            t_stat_corrected=40.2, pbo=0.0, methodology="test",
        )
        econ = EconomicEvaluationAudit(
            xs_rank_ic_h5=0.05, xs_rank_ic_std=0.08,
            xs_rank_ic_positive_fraction=0.6,
            ts_ic_vs_continuous=0.29, net_sharpe_ls=-0.5,
            net_sharpe_lo=0.0, gross_sharpe_ls=-0.3,
            cost_drag_annual=0.1, max_drawdown=-0.3,
            mean_daily_turnover=0.3, n_rebalances=100,
            n_trades_actual=500, cost_scenario="conservative",
            methodology="test",
        )
        divs = r._compute_divergences(ds, ml, econ)
        assert len(divs) == 5, f"Expected 5 divergences, got {len(divs)}"

    def test_divergences_have_increasing_steps(self, tmp_path):
        r = self._make_reconciler(tmp_path)
        from src.reconciliation.matrix import DatasetAudit, MLEvaluationAudit, EconomicEvaluationAudit
        ds = DatasetAudit(n_rows=100, n_symbols=5, n_dates=20, symbols=[],
            date_start="2021-01-01", date_end="2022-01-01",
            barrier_fraction=0.9, overlap_correction_factor=2.2,
            effective_n=20, label_autocorr_lag1=0.1, outcome_dist={},
            survivorship="CURRENT_UNIVERSE_ONLY", notes=[])
        ml = MLEvaluationAudit(ic_pearson=0.4, ic_rank=0.3, ic_vs_barrier_clamped=0.4,
            ic_inflation_from_barrier=1.0, net_sharpe_reported=3.0,
            n_trades_in_sharpe=100, sharpe_computation_note="",
            overlapping_label_autocorr=0.1, t_stat_raw=20.0,
            t_stat_corrected=9.0, pbo=0.0, methodology="")
        econ = EconomicEvaluationAudit(xs_rank_ic_h5=0.03, xs_rank_ic_std=0.05,
            xs_rank_ic_positive_fraction=0.6, ts_ic_vs_continuous=0.2,
            net_sharpe_ls=0.5, net_sharpe_lo=0.3, gross_sharpe_ls=1.0,
            cost_drag_annual=0.05, max_drawdown=-0.1, mean_daily_turnover=0.2,
            n_rebalances=10, n_trades_actual=50, cost_scenario="conservative",
            methodology="")
        divs = r._compute_divergences(ds, ml, econ)
        steps = [d.step for d in divs]
        assert steps == sorted(steps), "Divergences must have increasing step numbers"

    def test_lifecycle_state_no_edge_for_negative_sharpe(self, tmp_path):
        r = self._make_reconciler(tmp_path)
        from src.reconciliation.matrix import MLEvaluationAudit, EconomicEvaluationAudit
        ml = MLEvaluationAudit(ic_pearson=0.4, ic_rank=0.3, ic_vs_barrier_clamped=0.4,
            ic_inflation_from_barrier=1.0, net_sharpe_reported=3.0,
            n_trades_in_sharpe=100, sharpe_computation_note="",
            overlapping_label_autocorr=0.1, t_stat_raw=20.0,
            t_stat_corrected=9.0, pbo=0.0, methodology="")
        econ = EconomicEvaluationAudit(xs_rank_ic_h5=0.05, xs_rank_ic_std=0.05,
            xs_rank_ic_positive_fraction=0.6, ts_ic_vs_continuous=0.2,
            net_sharpe_ls=-1.5, net_sharpe_lo=-0.5, gross_sharpe_ls=-0.8,
            cost_drag_annual=0.1, max_drawdown=-0.5, mean_daily_turnover=0.3,
            n_rebalances=50, n_trades_actual=200, cost_scenario="conservative",
            methodology="")
        _, blockers, verdict, state = r._lifecycle_determination(ml, econ)
        assert state == "RESEARCH_READY"
        assert "NO_VERIFIED_EDGE" in verdict
        # Must document the blocker
        has_net_sharpe_blocker = any("GATE_7" in b or "Net Sharpe" in b for b in blockers)
        assert has_net_sharpe_blocker

    def test_lifecycle_positive_economics_paper_eligible(self, tmp_path):
        r = self._make_reconciler(tmp_path)
        from src.reconciliation.matrix import MLEvaluationAudit, EconomicEvaluationAudit
        ml = MLEvaluationAudit(ic_pearson=0.4, ic_rank=0.3, ic_vs_barrier_clamped=0.4,
            ic_inflation_from_barrier=1.0, net_sharpe_reported=3.0,
            n_trades_in_sharpe=100, sharpe_computation_note="",
            overlapping_label_autocorr=0.1, t_stat_raw=20.0,
            t_stat_corrected=9.0, pbo=0.0, methodology="")
        econ = EconomicEvaluationAudit(xs_rank_ic_h5=0.08, xs_rank_ic_std=0.05,
            xs_rank_ic_positive_fraction=0.7, ts_ic_vs_continuous=0.15,
            net_sharpe_ls=1.2, net_sharpe_lo=0.8, gross_sharpe_ls=2.0,
            cost_drag_annual=0.08, max_drawdown=-0.1, mean_daily_turnover=0.2,
            n_rebalances=100, n_trades_actual=400, cost_scenario="conservative",
            methodology="")
        _, blockers, verdict, state = r._lifecycle_determination(ml, econ)
        # Should be paper eligible (robustness not done yet)
        assert "PAPER_ELIGIBLE" in state or "RESEARCH_READY" in state
        # Forward paper must still be listed as a blocker
        has_forward_paper_blocker = any("GATE_16" in b or "forward" in b.lower() for b in blockers)
        assert has_forward_paper_blocker

    def test_generate_predictions_has_required_columns(self, tmp_path):
        r = self._make_reconciler(tmp_path)
        df = r._load_dataset()
        model_dict = _make_fake_model()
        pred_df = r._generate_predictions(df, model_dict)
        required = {"prediction", "signal", "raw_score", "symbol", "realized_return"}
        assert required.issubset(set(pred_df.columns)), (
            f"Missing columns: {required - set(pred_df.columns)}"
        )

    def test_prediction_scores_in_01(self, tmp_path):
        """Calibrated predictions should be in [0, 1]."""
        r = self._make_reconciler(tmp_path)
        df = r._load_dataset()
        model_dict = _make_fake_model()
        pred_df = r._generate_predictions(df, model_dict)
        valid = pred_df["prediction"].dropna()
        assert (valid >= 0).all() and (valid <= 1).all(), (
            "Predictions must be in [0, 1]"
        )

    def test_to_dict_json_serializable(self, tmp_path):
        """ReconciliationMatrix.to_dict() must produce a JSON-serializable object."""
        import json
        r = self._make_reconciler(tmp_path)
        df = r._load_dataset()
        model_dict = _make_fake_model()
        pred_df = r._generate_predictions(df, model_dict)
        ds_audit = r._audit_dataset(df)
        ml_audit = r._ml_evaluation_audit(pred_df, df)
        econ_audit, _ = r._economic_evaluation(pred_df, df)
        divs = r._compute_divergences(ds_audit, ml_audit, econ_audit)
        _, blockers, verdict, state = r._lifecycle_determination(ml_audit, econ_audit)

        from src.reconciliation.matrix import ReconciliationMatrix
        m = ReconciliationMatrix(
            generated_at="2026-09-26T00:00:00+00:00",
            code_sha="test",
            dataset_id="test_ds",
            dataset_hash="abc",
            model_version="1.0.0-test",
            model_sha256="def",
            dataset=ds_audit,
            ml_evaluation=ml_audit,
            economic_evaluation=econ_audit,
            divergences=divs,
            lifecycle_state=state,
            certification_gate_results={},
            remaining_blockers=blockers,
            honest_verdict=verdict,
        )
        json.dumps(m.to_dict(), default=str)  # must not raise
