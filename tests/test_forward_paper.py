"""Tests for the genuine forward-paper runner (mandate §55-§61)."""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

from src.analytics.forward_paper import (
    MIN_TRADES_FOR_VALIDATION,
    ForwardPaperRunner,
    ForwardPaperStore,
)


def _runner(tmp_path, horizon=1, interval="1d"):
    store = ForwardPaperStore(tmp_path / "signals.jsonl")
    return ForwardPaperRunner(store=store, interval=interval, horizon_bars=horizon), store


def _record(runner, symbol="NIFTY", direction=1, edge=0.01, cost=0.002,
            signal_ts=None):
    signal_ts = signal_ts or datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    return runner.record_signal(
        symbol=symbol, signal_ts=signal_ts, direction=direction,
        prediction=0.7, probability=0.7, expected_edge=edge, expected_cost=cost,
        entry_price=100.0, model_version="1.0.0", experiment_id="xs-1d-rank-x",
        dataset_hash="abc", feature_schema="fs-2.0.0",
        feature_vector={"ret_1": 0.01, "vol_20": 0.02},
        feature_as_of=signal_ts, data_as_of=signal_ts, bar_seconds=86400,
    )


class TestImmutabilityAndVersioning:
    def test_signal_persisted_with_versioning(self, tmp_path):
        runner, store = _runner(tmp_path)
        sig = _record(runner)
        loaded = store.all_signals()
        assert len(loaded) == 1
        s = loaded[0]
        assert s.git_sha and s.model_version == "1.0.0"
        assert s.experiment_id == "xs-1d-rank-x"
        assert s.dataset_hash == "abc" and s.feature_schema == "fs-2.0.0"
        assert s.feature_hash != "none"
        assert s.decision == "TRADE"  # edge 0.01 > cost 0.002

    def test_store_is_append_only(self, tmp_path):
        runner, store = _runner(tmp_path)
        _record(runner, symbol="NIFTY")
        _record(runner, symbol="BANKNIFTY")
        assert len(store.all_signals()) == 2
        # a third append does not mutate the first two
        _record(runner, symbol="RELIANCE")
        sigs = store.all_signals()
        assert len(sigs) == 3
        assert sigs[0].symbol == "NIFTY"  # unchanged

    def test_no_trade_when_edge_below_cost(self, tmp_path):
        runner, _ = _runner(tmp_path)
        sig = _record(runner, edge=0.001, cost=0.005)
        assert sig.decision == "NO_TRADE"


class TestGenuineForward:
    def test_resolve_due_gates_on_wallclock(self, tmp_path):
        """A signal cannot be resolved before its horizon has elapsed (§55, §96)."""
        runner, _ = _runner(tmp_path, horizon=1)
        signal_ts = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        _record(runner, signal_ts=signal_ts)  # resolve_after = signal_ts + 1 day

        # "now" just after the signal: NOT due yet.
        not_due = runner.resolve_due(now=signal_ts + timedelta(hours=1))
        assert not_due == []

        # "now" after the horizon elapsed: due.
        due = runner.resolve_due(now=signal_ts + timedelta(days=1, minutes=1))
        assert len(due) == 1

    def test_no_trade_signals_never_resolve(self, tmp_path):
        runner, _ = _runner(tmp_path)
        signal_ts = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
        _record(runner, edge=0.0, cost=0.01, signal_ts=signal_ts)  # NO_TRADE
        due = runner.resolve_due(now=signal_ts + timedelta(days=5))
        assert due == []


class TestStatusSampleGates:
    def test_status_not_run_when_empty(self, tmp_path):
        runner, _ = _runner(tmp_path)
        assert runner.status(n_resolved=0)["state"] == "NOT_RUN"

    def test_status_insufficient_sample(self, tmp_path):
        runner, _ = _runner(tmp_path)
        _record(runner)
        st = runner.status(n_resolved=5)
        assert st["state"] == "INSUFFICIENT_SAMPLE"
        assert st["min_trades_for_validation"] == MIN_TRADES_FOR_VALIDATION

    def test_status_running_at_min_sample(self, tmp_path):
        runner, _ = _runner(tmp_path)
        _record(runner)
        st = runner.status(n_resolved=MIN_TRADES_FOR_VALIDATION)
        assert st["state"] == "RUNNING"
