"""
test_training_orchestrator.py — Phase H end-to-end training tests (P0-001).

Builds a real dataset with a genuine signal, trains candidate models, selects
a champion by OOS evidence, calibrates, and registers an immutable artifact.
Also verifies the honest negative path: a no-signal dataset is REJECTED.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from src.data.dataset_builder import DatasetBuilder
from src.data.labels import LabelConfig
from src.registry.registry import ModelRegistry
from src.training.orchestrator import TrainingOrchestrator

UTC = timezone.utc


def _signal_ohlcv(n=700, seed=0, momentum=True):
    """
    Build OHLCV where forward returns are (weakly) predictable from trailing
    momentum — a realistic, PIT-safe, learnable structure.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    if momentum:
        # Inject mild momentum autocorrelation so trailing return predicts forward.
        for i in range(1, n):
            rets[i] += 0.15 * rets[i - 1]
    close = 100 * np.cumprod(1 + rets)
    high = close * (1 + np.abs(rng.normal(0, 0.003, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.003, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1000, 5000, n).astype(float)
    idx = pd.DatetimeIndex([datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)])
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )


def _build_dataset(tmp_path, ohlcv_by_symbol, horizon=3):
    builder = DatasetBuilder(output_root=tmp_path / "datasets", run_leakage_validation=True)
    meta = builder.build(
        ohlcv_by_symbol,
        LabelConfig(label_type="fixed_horizon", horizon=horizon, cost_bps=10),
        timeframe="1d",
    )
    return builder, meta


def test_orchestrator_trains_and_registers_champion(tmp_path):
    builder, meta = _build_dataset(
        tmp_path, {"AAA": _signal_ohlcv(seed=1), "BBB": _signal_ohlcv(seed=2)}
    )
    registry = ModelRegistry(artifacts_path=tmp_path / "artifacts")
    orch = TrainingOrchestrator(
        builder, registry, n_windows=5, embargo_days=5, cost_bps=10,
        min_ic=-1.0,  # relax acceptance so registration path is exercised deterministically
        max_pbo=1.0, enforce_calibration_gate=False,
    )
    report = orch.train(
        "market_regime", meta.dataset_id,
        candidate_names=["logistic", "lightgbm"],
    )
    assert report.champion in ("logistic", "lightgbm")
    assert len(report.candidates) == 2
    assert report.passed_acceptance is True
    assert report.champion_version is not None
    # Artifact registered on disk.
    artifact = registry.get_champion("market_regime") or None
    # champion.json only written for PRODUCTION stage; challenger registered under version dir.
    version_dir = (tmp_path / "artifacts" / "market_regime" / report.champion_version)
    assert version_dir.exists()
    assert (version_dir / "model.pkl").exists()
    assert (version_dir / "model.pkl.sha256").exists()
    assert (version_dir / "metadata.json").exists()


def test_orchestrator_reports_oos_metrics(tmp_path):
    builder, meta = _build_dataset(tmp_path, {"AAA": _signal_ohlcv(seed=3, n=800)})
    orch = TrainingOrchestrator(
        builder, ModelRegistry(artifacts_path=tmp_path / "art"),
        n_windows=5, embargo_days=5, min_ic=-1.0, max_pbo=1.0,
    )
    report = orch.train("market_regime", meta.dataset_id,
                        candidate_names=["logistic", "lightgbm", "xgboost"],
                        register_champion=False)
    # Every candidate has genuine OOS metrics.
    for c in report.candidates:
        assert -1.0 <= c.wf_ic_mean <= 1.0
        assert 0.0 <= c.cpcv_pbo <= 1.0
    d = report.to_dict()
    assert "candidates" in d


def test_no_signal_dataset_is_rejected(tmp_path):
    """Honest negative result: pure noise must FAIL the IC acceptance gate."""
    rng = np.random.default_rng(9)
    n = 700
    rets = rng.normal(0, 0.01, n)  # no autocorrelation → no predictable signal
    close = 100 * np.cumprod(1 + rets)
    # Non-degenerate OHLC (varied ranges/volume) so features are defined, but
    # forward returns remain independent of features → no learnable edge.
    high = close * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = close * (1 - np.abs(rng.normal(0, 0.004, n)))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    volume = rng.integers(1000, 8000, n).astype(float)
    idx = pd.DatetimeIndex([datetime(2020, 1, 1, tzinfo=UTC) + timedelta(days=i) for i in range(n)])
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=idx)
    builder, meta = _build_dataset(tmp_path, {"NOISE": df})
    orch = TrainingOrchestrator(
        builder, ModelRegistry(artifacts_path=tmp_path / "art2"),
        n_windows=5, embargo_days=5, min_ic=0.02, max_pbo=0.5,
    )
    report = orch.train("market_regime", meta.dataset_id, candidate_names=["logistic"])
    # With no genuine edge, acceptance should fail — we do NOT force a pass.
    assert report.passed_acceptance is False
    assert report.rejection_reason in ("IC_BELOW_THRESHOLD", "HIGH_PBO", "NEGATIVE_NET_SHARPE")


def test_parsimony_prefers_baseline(tmp_path):
    builder, meta = _build_dataset(tmp_path, {"AAA": _signal_ohlcv(seed=5, n=800)})
    orch = TrainingOrchestrator(
        builder, ModelRegistry(artifacts_path=tmp_path / "art3"),
        n_windows=5, embargo_days=5, min_ic=-1.0, max_pbo=1.0,
        parsimony_margin=1.0,  # huge margin → always prefer baseline
    )
    report = orch.train("market_regime", meta.dataset_id,
                        candidate_names=["logistic", "lightgbm"],
                        register_champion=False)
    assert report.champion == "logistic"  # baseline preferred under wide margin
