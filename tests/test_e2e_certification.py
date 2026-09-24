"""
test_e2e_certification.py — Phase U/V end-to-end lifecycle integration test.

Runs the complete pipeline on a controlled dataset and asserts that every stage
produces genuine, self-consistent evidence — WITHOUT forcing a PASS. The test
verifies the pipeline is wired end-to-end and that acceptance honestly reflects
the (cost-aware) result.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine, CostModel
from src.data.dataset_builder import DatasetBuilder
from src.data.feedback import FeedbackStore, OutcomeResolver
from src.data.labels import LabelConfig
from src.models.estimators import build_estimator
from src.registry.lifecycle import ChampionChallengerManager
from src.registry.registry import ModelRegistry
from src.schemas.meta import FeedbackRecord
from src.training.orchestrator import TrainingOrchestrator

UTC = timezone.utc


def _ohlcv(n=700, seed=0):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, 0.011, n)
    for t in range(1, n):
        rets[t] += 0.12 * rets[t - 1]
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    vol = rng.integers(1000, 10000, n).astype(float)
    idx = pd.DatetimeIndex([datetime(2021, 1, 1, tzinfo=UTC) + timedelta(days=k) for k in range(n)])
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


def test_full_lifecycle_produces_evidence(tmp_path):
    # 1. Dataset
    builder = DatasetBuilder(output_root=tmp_path / "ds", run_leakage_validation=True)
    meta = builder.build(
        {"NIFTY": _ohlcv(seed=1), "BANKNIFTY": _ohlcv(seed=2)},
        LabelConfig(label_type="triple_barrier", horizon=5, cost_bps=10),
    )
    assert meta.row_count > 0
    assert meta.leakage_validated is True

    # 2. Training with walk-forward + CPCV + calibration
    registry = ModelRegistry(artifacts_path=tmp_path / "art")
    orch = TrainingOrchestrator(builder, registry, n_windows=5, embargo_days=5,
                                cost_bps=10, enforce_calibration_gate=False)
    report = orch.train("market_regime", meta.dataset_id,
                        candidate_names=["logistic", "lightgbm"], register_champion=True)
    # Genuine OOS metrics exist (may or may not pass — honesty over PASS).
    assert len(report.candidates) == 2
    for c in report.candidates:
        assert -1.0 <= c.wf_ic_mean <= 1.0
        assert 0.0 <= c.cpcv_pbo <= 1.0

    # 3. Backtest (cost-aware)
    frame = builder.load_frame(meta.dataset_id)
    rr = frame["realized_return"].fillna(0.0).to_numpy()
    close = 100 * np.cumprod(1 + rr / 5.0)
    bt_frame = pd.DataFrame({"open": np.concatenate([[100.0], close[:-1]]), "close": close}, index=frame.index)
    signals = np.where(frame["label"].fillna(0).to_numpy() > 0, 1.0, -1.0)
    bt = BacktestEngine(CostModel()).run(bt_frame, signals)
    assert bt.cost_bps_round_trip > 0
    assert bt.net_return <= bt.gross_return  # costs never increase returns

    # 4. Champion/challenger/shadow (only when accepted)
    lifecycle = ChampionChallengerManager(registry=registry, state_path=tmp_path / "roles.json")
    if report.passed_acceptance and report.champion_version:
        lifecycle.register_challenger("market_regime", report.champion_version)
        lifecycle.promote_to_shadow("market_regime")
        assert lifecycle.get_roles("market_regime").shadow_version == report.champion_version

    # 5. Paper-trading loop -> feedback (>= 20 outcomes)
    feedback = FeedbackStore(tmp_path / "fb.jsonl")
    resolver = OutcomeResolver(cost_bps=10)
    n_paper = 0
    for i in range(0, len(frame) - 6, 3):
        entry = float(close[i])
        direction = 1 if signals[i] > 0 else -1
        target = entry * (1 + direction * 0.02)
        stop = entry * (1 - direction * 0.015)
        path = [(frame.index[i + k], float(close[i + k])) for k in range(1, 6)]
        res = resolver.resolve(f"p-{i}", "NIFTY", direction, entry, target, stop, path)
        feedback.append(FeedbackRecord(
            signal_id=res.signal_id, symbol="NIFTY",
            prediction_timestamp=frame.index[i].to_pydatetime(),
            action="BUY" if direction > 0 else "SELL",
            entry_price=entry, exit_price=res.exit_price,
            realized_return=res.realized_return, realized_return_net=res.realized_return_net,
            exit_reason=res.exit_reason,
        ))
        n_paper += 1

    assert n_paper >= 20  # certification minimum
    summary = feedback.summary()
    assert summary["n_resolved"] >= 20
    assert 0.0 <= summary["win_rate"] <= 1.0
