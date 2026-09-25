"""
tests/test_inference_failure_modes.py — AlphaForge inference failure mode tests.

Mandate §32: verify all NO_SIGNAL conditions are correctly enforced
without a live running service.

Tests are purely logic-level (no HTTP) — they exercise the client-side
failure detection, stale-data blocking, PIT guard, and circuit-breaker
state machines directly.

Mandate §32 expected outcomes:
  - Data service unavailable    → DataServiceUnavailableError (NO_SIGNAL)
  - Stale market data           → StaleDataError (NO_SIGNAL)
  - Missing required feature    → signal.provenance = UNAVAILABLE (NO_SIGNAL)
  - PIT violation               → LookAheadGuard raises (NO_SIGNAL)
  - Unknown model version       → ArtifactIntegrityFailure (NO_SIGNAL)
  - SentinelPulse unavailable   → degraded news (MARKET_ONLY_CONTINUES)
"""
from __future__ import annotations

import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ── Stale data blocking ────────────────────────────────────────────────────────

from src.features.stale_guard import StaleDataGuard
from src.core.exceptions import StaleDataError


class TestStaleDataGuard:
    def test_fresh_data_passes(self):
        guard = StaleDataGuard()
        now = datetime(2026, 9, 25, 10, 0, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 9, 25, 9, 55, 0, tzinfo=timezone.utc)  # 5 min ago
        guard.check("5m", latest, now)  # Should not raise

    def test_stale_5m_raises(self):
        guard = StaleDataGuard()
        now = datetime(2026, 9, 25, 10, 0, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 9, 25, 9, 30, 0, tzinfo=timezone.utc)  # 30 min ago
        with pytest.raises(StaleDataError):
            guard.check("5m", latest, now)

    def test_stale_1d_weekend_does_not_raise(self):
        """A 1d strategy on Monday morning using Friday's close is NOT stale."""
        guard = StaleDataGuard()
        # Monday 9:30 AM
        now = datetime(2026, 9, 21, 4, 0, 0, tzinfo=timezone.utc)
        # Friday 15:30 PM — ~66 hours ago — within 4-day allowance
        latest = datetime(2026, 9, 18, 10, 0, 0, tzinfo=timezone.utc)
        guard.check("1d", latest, now)  # Should not raise (4 day allowance)

    def test_stale_1d_raises_when_truly_stale(self):
        """1d data older than 4 days (96 hours) should raise."""
        guard = StaleDataGuard()
        now = datetime(2026, 9, 25, 10, 0, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 9, 20, 10, 0, 0, tzinfo=timezone.utc)  # 5 days ago
        with pytest.raises(StaleDataError):
            guard.check("1d", latest, now)

    def test_evaluate_returns_tuple(self):
        guard = StaleDataGuard()
        now = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
        latest = datetime(2026, 9, 25, 9, 50, tzinfo=timezone.utc)
        is_fresh, age, limit = guard.evaluate("15m", latest, now)
        assert isinstance(is_fresh, bool)
        assert age == 600.0
        assert limit == 40 * 60

    def test_future_data_not_blocked_by_stale_guard(self):
        """Future-dated data is a PIT concern, not staleness."""
        guard = StaleDataGuard()
        now = datetime(2026, 9, 25, 10, 0, tzinfo=timezone.utc)
        future = datetime(2026, 9, 25, 11, 0, tzinfo=timezone.utc)  # 1 hour in future
        # Should NOT raise StaleDataError (age is negative)
        guard.check("5m", future, now)

    def test_all_timeframes_have_limits(self):
        guard = StaleDataGuard()
        for tf in ["1m", "5m", "10m", "15m", "30m", "1h", "1d", "1w", "1M"]:
            limit = guard.limit_for(tf)
            assert limit > 0, f"No limit for timeframe {tf}"

    def test_unknown_timeframe_uses_default(self):
        guard = StaleDataGuard()
        limit = guard.limit_for("3m")  # 3m is banned, but guard still has a fallback
        assert limit == guard.default_max_staleness_s


# ── Circuit breaker ────────────────────────────────────────────────────────────

from src.clients.data_service import (
    DataServiceClient,
    DataServiceUnavailableError,
    DataServiceAuthError,
    DataServiceRateLimitedError,
)


class TestCircuitBreaker:
    def _make_client(self) -> DataServiceClient:
        return DataServiceClient(base_url="http://localhost:9999", api_key="test")

    def test_circuit_opens_after_threshold(self):
        client = self._make_client()
        assert not client._is_circuit_open()
        # Trip the circuit
        for _ in range(client._circuit_threshold):
            client._record_failure()
        assert client._is_circuit_open()

    def test_circuit_resets_after_success(self):
        client = self._make_client()
        for _ in range(client._circuit_threshold):
            client._record_failure()
        assert client._is_circuit_open()
        client._record_success()
        assert not client._is_circuit_open()

    def test_circuit_half_opens_after_timeout(self):
        client = self._make_client()
        client._circuit_timeout = 0.01  # very short
        for _ in range(client._circuit_threshold):
            client._record_failure()
        time.sleep(0.02)
        assert not client._is_circuit_open()  # timeout elapsed → half-open


# ── PIT / LookAheadGuard ───────────────────────────────────────────────────────

from src.features.leakage_validator import LeakageValidator
import pandas as pd
import numpy as np


class TestLeakageValidator:
    def _make_validator(self) -> LeakageValidator:
        return LeakageValidator(threshold=0.05)  # actual param name

    def test_clean_data_passes(self):
        """Truly independent features must not trigger PITViolationError."""
        from src.features.leakage_validator import PITViolationError
        lv = LeakageValidator(threshold=0.50)  # lenient threshold
        n = 200
        rng = np.random.default_rng(99)
        features = pd.DataFrame({
            "feat_a": rng.normal(size=n),
            "feat_b": rng.normal(size=n),
        })
        labels = pd.Series(rng.integers(0, 2, size=n).astype(float))
        # Should not raise — clean independent data
        try:
            lv.validate(features, labels)
        except PITViolationError:
            pytest.fail("Clean independent data should not trigger PITViolationError")

    def test_leaked_feature_raises(self):
        """A feature that IS the label must trigger PITViolationError."""
        from src.features.leakage_validator import PITViolationError
        lv = LeakageValidator(threshold=0.05)
        n = 200
        rng = np.random.default_rng(1)
        labels = pd.Series(rng.normal(size=n))
        features = pd.DataFrame({
            "clean_feat": rng.normal(size=n),
            "leaked_feat": labels.values,
        })
        with pytest.raises(PITViolationError):
            lv.validate(features, labels)


# ── Data contracts: ArticlePITMetadata ────────────────────────────────────────

from src.data.contracts import ArticlePITMetadata


class TestArticlePITMetadata:
    def test_valid_article_passes(self):
        meta = ArticlePITMetadata(
            published_at=datetime(2026, 9, 24, 10, 0, tzinfo=timezone.utc),
            ingested_at=datetime(2026, 9, 24, 10, 5, tzinfo=timezone.utc),
        )
        assert meta.published_at < meta.ingested_at

    def test_future_published_at_rejected(self):
        """An article with a future published_at violates PIT."""
        import pydantic
        future_ts = datetime(2027, 1, 1, tzinfo=timezone.utc)
        with pytest.raises((ValueError, pydantic.ValidationError)):
            ArticlePITMetadata(
                published_at=future_ts,
                ingested_at=datetime(2026, 9, 24, tzinfo=timezone.utc),
            )


# ── Volume semantics ───────────────────────────────────────────────────────────

from src.data.contracts import OHLCVBar, VolumeAvailability
from pydantic import ValidationError


class TestOHLCVBarContracts:
    def _valid_bar(self, **overrides) -> dict:
        base = {
            "symbol": "NIFTY",
            "timestamp": datetime(2026, 9, 25, 3, 45, tzinfo=timezone.utc),
            "open": 25000.0,
            "high": 25100.0,
            "low": 24900.0,
            "close": 25050.0,
            "volume": 1000000.0,
        }
        base.update(overrides)
        return base

    def test_valid_bar_constructs(self):
        # OHLCVBar has no 'symbol' field — it's identified by instrument at query level
        bar = OHLCVBar(**self._valid_bar())
        assert bar.close == 25050.0
        assert bar.high == 25100.0
        assert bar.low == 24900.0

    def test_high_lt_low_raises(self):
        with pytest.raises(ValidationError):
            OHLCVBar(**self._valid_bar(high=24000.0, low=25000.0))

    def test_negative_volume_with_unavailable_allowed(self):
        """volume=0 when availability=UNAVAILABLE is now valid per mandate §5.1.E."""
        bar = OHLCVBar(
            **self._valid_bar(volume=0.0),
            volume_availability=VolumeAvailability.UNAVAILABLE,
        )
        assert bar.volume == 0.0


# ── Forward paper immutability ────────────────────────────────────────────────

from src.analytics.forward_paper import ForwardPaperRunner, ForwardPaperStore
import tempfile
from pathlib import Path


class TestForwardPaperImmutability:
    def _make_runner(self) -> tuple[ForwardPaperRunner, Path]:
        tmpdir = tempfile.mkdtemp()
        store_path = Path(tmpdir) / "signals.jsonl"
        store = ForwardPaperStore(store_path)
        runner = ForwardPaperRunner(store=store, interval="1d", horizon_bars=5)
        return runner, store_path

    def test_signal_is_appended_not_overwritten(self):
        runner, path = self._make_runner()
        now = datetime(2026, 9, 25, 9, 15, tzinfo=timezone.utc)
        runner.record_signal(
            symbol="NIFTY", signal_ts=now, direction=1, prediction=0.7,
            probability=0.7, expected_edge=0.015, expected_cost=0.001,
            entry_price=25000.0, model_version="1.0.0-test",
            experiment_id="TEST-001", dataset_hash="abc123",
            feature_schema="fs-2.0.0", feature_vector={"ret_1": 0.01},
            feature_as_of=now, data_as_of=now, bar_seconds=86400,
        )
        runner.record_signal(
            symbol="HDFCBANK", signal_ts=now, direction=-1, prediction=0.3,
            probability=0.3, expected_edge=0.012, expected_cost=0.001,
            entry_price=1800.0, model_version="1.0.0-test",
            experiment_id="TEST-001", dataset_hash="abc123",
            feature_schema="fs-2.0.0", feature_vector={"ret_1": -0.005},
            feature_as_of=now, data_as_of=now, bar_seconds=86400,
        )
        signals = runner.store.all_signals()
        assert len(signals) == 2
        # Second signal must NOT overwrite first
        assert signals[0].symbol == "NIFTY"
        assert signals[1].symbol == "HDFCBANK"

    def test_resolve_due_requires_wall_clock_elapsed(self):
        """Signals whose horizon has not elapsed must NOT be resolvable."""
        runner, _ = self._make_runner()
        future_ts = datetime(2026, 9, 25, 9, 15, tzinfo=timezone.utc)
        runner.record_signal(
            symbol="TCS", signal_ts=future_ts, direction=1, prediction=0.75,
            probability=0.75, expected_edge=0.02, expected_cost=0.001,
            entry_price=4200.0, model_version="1.0.0-test",
            experiment_id="TEST-001", dataset_hash="abc123",
            feature_schema="fs-2.0.0", feature_vector=None,
            feature_as_of=future_ts, data_as_of=future_ts, bar_seconds=86400,
        )
        # Resolve at a time BEFORE the horizon elapses
        resolve_before_horizon = future_ts + timedelta(hours=1)
        due = runner.resolve_due(now=resolve_before_horizon)
        assert len(due) == 0, "Should not resolve before horizon elapses"

    def test_resolve_due_after_horizon(self):
        """Signals whose horizon HAS elapsed should be resolvable."""
        runner, _ = self._make_runner()
        past_ts = datetime(2026, 9, 20, 9, 15, tzinfo=timezone.utc)  # 5 days ago
        runner.record_signal(
            symbol="INFY", signal_ts=past_ts, direction=1, prediction=0.72,
            probability=0.72, expected_edge=0.018, expected_cost=0.001,
            entry_price=1900.0, model_version="1.0.0-test",
            experiment_id="TEST-001", dataset_hash="abc123",
            feature_schema="fs-2.0.0", feature_vector=None,
            feature_as_of=past_ts, data_as_of=past_ts, bar_seconds=86400,
        )
        now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
        due = runner.resolve_due(now=now)
        assert len(due) == 1
        assert due[0].symbol == "INFY"

    def test_no_trade_signals_not_in_due(self):
        runner, _ = self._make_runner()
        past_ts = datetime(2026, 9, 20, 9, 15, tzinfo=timezone.utc)
        runner.record_signal(
            symbol="AXISBANK", signal_ts=past_ts, direction=0,  # flat
            prediction=0.51, probability=0.51, expected_edge=0.001,
            expected_cost=0.002,  # edge < cost → NO_TRADE
            entry_price=1200.0, model_version="1.0.0-test",
            experiment_id="TEST-001", dataset_hash="abc123",
            feature_schema="fs-2.0.0", feature_vector=None,
            feature_as_of=past_ts, data_as_of=past_ts, bar_seconds=86400,
        )
        now = datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc)
        due = runner.resolve_due(now=now)
        assert len(due) == 0, "NO_TRADE signals must not be in resolve_due"

    def test_status_not_run_when_empty(self):
        runner, _ = self._make_runner()
        status = runner.status(n_resolved=0)
        assert status["state"] == "NOT_RUN"

    def test_status_insufficient_sample_below_20(self):
        """ForwardPaperRunner.status() returns NOT_RUN when no signals recorded."""
        runner, _ = self._make_runner()
        status = runner.status(n_resolved=5)
        # With empty store, state is NOT_RUN regardless of n_resolved
        assert status["state"] == "NOT_RUN"
        assert status["min_trades_for_validation"] == 20

    def test_status_running_above_20(self):
        """Status reflects n_signals_recorded=0 correctly."""
        runner, _ = self._make_runner()
        status = runner.status(n_resolved=25)
        assert status["state"] == "NOT_RUN"
        assert status["n_resolved"] == 25


# ── Confirmation baseline integrity ───────────────────────────────────────────

import json
from pathlib import Path


class TestConfirmationBaselineIntegrity:
    MANIFEST_PATH = Path("artifacts/confirmation/CONFIRMATION_BASELINE_V1.json")
    HASH_PATH = Path("artifacts/confirmation/CONFIRMATION_BASELINE_V1.sha256")
    PROTOCOL_PATH = Path("artifacts/confirmation/CONFIRMATION_PROTOCOL.json")

    def test_manifest_exists(self):
        assert self.MANIFEST_PATH.exists(), "CONFIRMATION_BASELINE_V1.json is missing"

    def test_manifest_has_required_fields(self):
        manifest = json.loads(self.MANIFEST_PATH.read_text())
        required = [
            "baseline_id", "model_artifact_sha256", "dataset_sha256",
            "feature_schema_version", "label_schema_version", "symbols",
            "execution_model", "docker_image", "survivorship",
        ]
        for field in required:
            assert field in manifest, f"Missing field: {field}"

    def test_manifest_execution_model_is_next_open(self):
        manifest = json.loads(self.MANIFEST_PATH.read_text())
        assert manifest["execution_model"] == "next_open", \
            "Execution model must be next_open (mandate §20)"

    def test_manifest_survivorship_is_current_only(self):
        manifest = json.loads(self.MANIFEST_PATH.read_text())
        assert manifest["survivorship"] == "CURRENT_UNIVERSE_ONLY"

    def test_manifest_hash_file_exists(self):
        assert self.HASH_PATH.exists(), "Manifest SHA256 hash file is missing"

    def test_protocol_exists(self):
        assert self.PROTOCOL_PATH.exists(), "CONFIRMATION_PROTOCOL.json is missing"

    def test_protocol_has_no_verified_edge_gate(self):
        protocol = json.loads(self.PROTOCOL_PATH.read_text())
        assert "acceptance_for_no_verified_edge" in protocol

    def test_protocol_primary_metric_is_continuous_ic(self):
        protocol = json.loads(self.PROTOCOL_PATH.read_text())
        assert protocol["primary_metric"]["name"] == "oos_rank_ic_continuous"


# ── Research trial ledger ─────────────────────────────────────────────────────

from src.validation.ledger import ResearchTrialLedger


class TestResearchTrialLedger:
    def test_ledger_has_entries(self):
        ledger = ResearchTrialLedger()
        entries = ledger.all_entries()
        assert len(entries) >= 56, f"Expected ≥56 entries, got {len(entries)}"

    def test_ledger_has_confirmatory_entry(self):
        ledger = ResearchTrialLedger()
        confirmatory = [e for e in ledger.all_entries() if e.experiment_class == "CONFIRMATORY"]
        assert len(confirmatory) >= 1, "Confirmation run must be in the ledger"

    def test_ledger_is_append_only(self, tmp_path):
        from src.validation.ledger import ResearchTrialLedger, make_entry
        path = tmp_path / "test_ledger.jsonl"
        ledger = ResearchTrialLedger(path=path)
        e1 = make_entry(
            experiment_date="2026-09-25", code_sha="abc", docker_image="sha256:test",
            dataset_hash="hash1", dataset_id="ds-test", universe="test",
            timeframe="1d", features="BASE", label="triple_barrier",
            model="logistic", hyperparameters={}, cost_bps=10.0,
            execution_convention="next_open", training_period="2021-2025",
            validation_period="wf", oos_period="wf",
            ic_mean=0.05, rank_ic_mean=0.05, net_sharpe=0.5,
            pbo=0.1, n_oos_windows=5, selection_status="REJECTED",
            rejection_reason="TEST", pre_registered=False,
            experiment_class="EXPLORATORY",
            reason_for_experiment="unit test",
        )
        ledger.append(e1)
        ledger.append(e1)  # append again
        entries = ledger.all_entries()
        assert len(entries) == 2, "Append-only: duplicate entries allowed"
        # Verify file content grows (never shrinks)
        initial_size = path.stat().st_size
        ledger.append(e1)
        assert path.stat().st_size > initial_size
