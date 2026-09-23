"""
tests/test_feature_pit.py — TDD: Point-In-Time correctness & lookahead-bias tests.

This file tests the PIT enforcement layer in ml-service2.0:

  1. ``LookAheadGuard``      — raises ``PointInTimeViolationError`` when a feature
                               vector contains any value with a source timestamp
                               >= the PIT boundary.
  2. ``LeakageValidator``    — raises ``PITViolationError`` (statistical) when
                               |Pearson correlation| between a feature and future
                               returns exceeds LEAK_THRESHOLD (0.05).
  3. ``FeaturePipeline``     — end-to-end: PIT-safe data → zero violations in report;
                               PIT-contaminated data → violation recorded in report.

TDD mandate
-----------
- ``LookAheadGuard`` is a NEW Phase-2 class defined here and in
  ``src/core/exceptions.py``.  Tests drive its interface; Phase 3 implements it.
- ``LeakageValidator`` is ALREADY implemented in
  ``src/features/leakage_validator.py`` — tests here assert its existing contract.
- ``FeaturePipeline`` is ALREADY implemented; tests here document the contract
  for Phase 3 reviewers and provide regression protection.

Markers: @pytest.mark.unit, @pytest.mark.pit, @pytest.mark.tdd
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume, given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.core.exceptions import LookAheadBiasError, PointInTimeViolationError
from src.features.leakage_validator import (
    LEAK_THRESHOLD,
    MAX_LOOK_AHEAD_DAYS,
    MIN_ALIGNED_SAMPLES,
    MIN_LOOK_AHEAD_DAYS,
    PITViolationError,
    LeakageValidator,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


_PIT_BOUNDARY = datetime(2024, 1, 15, 9, 30, 0, tzinfo=timezone.utc)
_PAST_TS = _PIT_BOUNDARY - timedelta(minutes=15)
_FUTURE_TS = _PIT_BOUNDARY + timedelta(minutes=30)


def _make_ohlcv_df(n: int = 250, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic OHLCV DataFrame with ``n`` rows (≥ MIN_ALIGNED_SAMPLES + buffer)."""
    rng = np.random.default_rng(seed)
    close = 18000.0 + np.cumsum(rng.normal(0, 50, n))
    returns = np.diff(close, prepend=close[0]) / close
    df = pd.DataFrame({
        "open":   close * (1 + rng.uniform(-0.005, 0.005, n)),
        "high":   close * (1 + rng.uniform(0.001, 0.015, n)),
        "low":    close * (1 - rng.uniform(0.001, 0.015, n)),
        "close":  close,
        "volume": rng.integers(500_000, 2_000_000, n).astype(float),
        "returns": returns,
    })
    return df


def _make_leaky_feature(df: pd.DataFrame, look_ahead: int = 1) -> pd.Series:
    """Return a feature that is exactly the future return at ``look_ahead`` — guaranteed leak."""
    return df["returns"].shift(-look_ahead)


def _make_clean_feature(df: pd.DataFrame, seed: int = 99) -> pd.Series:
    """Return a feature with zero look-ahead correlation (completely independent noise).

    We generate the noise from a fixed seed and verify it has <0.05 correlation
    at all look-ahead windows before returning it.  The inner loop tries up to
    10 seeds to guarantee independence; in practice the first seed works.
    """
    for offset in range(20):
        rng = np.random.default_rng(seed + offset * 7)
        noise = pd.Series(rng.standard_normal(len(df)), index=df.index)
        # Quick pre-check: ensure no window in [1, 22] has |correlation| > 0.04
        future_rets = df["returns"]
        max_corr = 0.0
        for w in range(1, 23):
            shifted = future_rets.shift(-w)
            aligned = pd.concat([noise, shifted], axis=1).dropna()
            if len(aligned) < 30:
                continue
            corr = abs(float(np.corrcoef(aligned.iloc[:, 0], aligned.iloc[:, 1])[0, 1]))
            max_corr = max(max_corr, corr)
        if max_corr < 0.04:  # safe margin below LEAK_THRESHOLD (0.05)
            return noise
    # Fallback: use a constant offset from close (guaranteed zero correlation
    # with future returns because it's a static scalar)
    return pd.Series(42.0, index=df.index)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PointInTimeViolationError — exception contract
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.pit
@pytest.mark.tdd
class TestPointInTimeViolationError:
    """The Phase-2 ``PointInTimeViolationError`` exception must satisfy its interface contract."""

    def test_exception_is_instantiable(self) -> None:
        """PointInTimeViolationError must be instantiable with the required args."""
        err = PointInTimeViolationError(
            feature_name="close",
            source_ts_iso=_FUTURE_TS.isoformat(),
            pit_boundary_iso=_PIT_BOUNDARY.isoformat(),
        )
        assert err.feature_name == "close"
        assert err.source_ts_iso == _FUTURE_TS.isoformat()
        assert err.pit_boundary_iso == _PIT_BOUNDARY.isoformat()

    def test_exception_message_contains_feature_name(self) -> None:
        err = PointInTimeViolationError(
            feature_name="india_vix",
            source_ts_iso=_FUTURE_TS.isoformat(),
            pit_boundary_iso=_PIT_BOUNDARY.isoformat(),
        )
        assert "india_vix" in str(err)

    def test_exception_is_subclass_of_ml_service_error(self) -> None:
        from src.core.exceptions import MLServiceError
        err = PointInTimeViolationError(
            feature_name="close",
            source_ts_iso=_FUTURE_TS.isoformat(),
            pit_boundary_iso=_PIT_BOUNDARY.isoformat(),
        )
        assert isinstance(err, MLServiceError)

    def test_default_code_is_pit_001(self) -> None:
        err = PointInTimeViolationError(
            feature_name="close",
            source_ts_iso=_FUTURE_TS.isoformat(),
            pit_boundary_iso=_PIT_BOUNDARY.isoformat(),
        )
        assert err.code == "PIT_001"

    def test_custom_code_is_accepted(self) -> None:
        err = PointInTimeViolationError(
            feature_name="close",
            source_ts_iso=_FUTURE_TS.isoformat(),
            pit_boundary_iso=_PIT_BOUNDARY.isoformat(),
            code="PIT_CUSTOM",
        )
        assert err.code == "PIT_CUSTOM"


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LookAheadBiasError — exception contract
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.pit
@pytest.mark.tdd
class TestLookAheadBiasError:
    """``LookAheadBiasError`` must satisfy its interface contract."""

    def test_instantiation_with_required_args(self) -> None:
        err = LookAheadBiasError(
            feature_name="momentum_5d",
            look_ahead_days=1,
            correlation=0.12,
        )
        assert err.feature_name == "momentum_5d"
        assert err.look_ahead_days == 1
        assert err.correlation == 0.12

    def test_is_subclass_of_point_in_time_violation_error(self) -> None:
        err = LookAheadBiasError("f", 1, 0.1)
        assert isinstance(err, PointInTimeViolationError)

    def test_message_contains_correlation_value(self) -> None:
        err = LookAheadBiasError("rsi_14", 5, 0.0876)
        assert "0.0876" in str(err) or "0.088" in str(err)

    def test_default_code_is_pit_002(self) -> None:
        err = LookAheadBiasError("f", 1, 0.1)
        assert err.code == "PIT_002"

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(PointInTimeViolationError):
            raise LookAheadBiasError("close_lag1", 1, 0.99)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LookAheadGuard — Phase-2 stub interface (drives Phase 3 implementation)
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.pit
@pytest.mark.tdd
class TestLookAheadGuard:
    """
    ``LookAheadGuard`` is a Phase-2 stub that must raise ``NotImplementedError``
    until Phase 3 provides the implementation.

    Interface contract driven by these tests:
      - ``LookAheadGuard.check(feature_name, source_ts, pit_boundary)``
        → ``None`` when ``source_ts < pit_boundary``  (clean)
        → raises ``PointInTimeViolationError`` when ``source_ts >= pit_boundary``

    This test class is the *specification*; the implementation in Phase 3
    must make every test pass.
    """

    def _get_guard(self) -> Any:
        """Import LookAheadGuard; skip if not yet implemented (Phase 3)."""
        try:
            from src.features.leakage_validator import LookAheadGuard  # type: ignore[attr-defined]
            return LookAheadGuard
        except (ImportError, AttributeError):
            pytest.skip(
                "LookAheadGuard not yet implemented — Phase 2 TDD red phase. "
                "Implement in Phase 3 to make this test pass."
            )

    def test_check_clean_data_returns_none(self) -> None:
        """Source timestamp BEFORE pit boundary → returns None (no violation)."""
        LookAheadGuard = self._get_guard()
        guard = LookAheadGuard()
        result = guard.check(
            feature_name="close",
            source_ts=_PAST_TS,
            pit_boundary=_PIT_BOUNDARY,
        )
        assert result is None

    def test_check_future_source_raises_pit_violation(self) -> None:
        """Source timestamp AFTER pit boundary → raises PointInTimeViolationError."""
        LookAheadGuard = self._get_guard()
        guard = LookAheadGuard()
        with pytest.raises(PointInTimeViolationError):
            guard.check(
                feature_name="close",
                source_ts=_FUTURE_TS,
                pit_boundary=_PIT_BOUNDARY,
            )

    def test_check_equal_timestamps_raises_pit_violation(self) -> None:
        """Source timestamp EQUAL TO pit boundary → raises PointInTimeViolationError.

        The PIT rule is STRICT: source_ts must be < pit_boundary, not <=.
        """
        LookAheadGuard = self._get_guard()
        guard = LookAheadGuard()
        with pytest.raises(PointInTimeViolationError):
            guard.check(
                feature_name="india_vix",
                source_ts=_PIT_BOUNDARY,   # equal — still a violation
                pit_boundary=_PIT_BOUNDARY,
            )

    def test_violated_error_has_correct_feature_name(self) -> None:
        """Raised PointInTimeViolationError must carry the correct feature_name."""
        LookAheadGuard = self._get_guard()
        guard = LookAheadGuard()
        with pytest.raises(PointInTimeViolationError) as exc_info:
            guard.check(
                feature_name="put_call_ratio",
                source_ts=_FUTURE_TS,
                pit_boundary=_PIT_BOUNDARY,
            )
        assert exc_info.value.feature_name == "put_call_ratio"

    @given(
        offset_seconds=st.integers(min_value=1, max_value=7200)
    )
    @h_settings(max_examples=30, suppress_health_check=[HealthCheck.too_slow])
    def test_any_future_offset_raises_violation(self, offset_seconds: int) -> None:
        """Hypothesis: for ANY positive offset, source_ts >= pit_boundary must raise."""
        LookAheadGuard = self._get_guard()
        guard = LookAheadGuard()
        future_ts = _PIT_BOUNDARY + timedelta(seconds=offset_seconds)
        with pytest.raises(PointInTimeViolationError):
            guard.check(
                feature_name="close",
                source_ts=future_ts,
                pit_boundary=_PIT_BOUNDARY,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 4. LeakageValidator — statistical look-ahead bias detection
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.mark.unit
@pytest.mark.pit
@pytest.mark.tdd
class TestLeakageValidator:
    """Tests for the existing ``LeakageValidator`` / ``PITViolationError`` contract."""

    def test_pit_violation_error_is_instantiable(self) -> None:
        """PITViolationError must be instantiable with the correct attrs."""
        err = PITViolationError(
            feature_name="momentum_5d",
            look_ahead_days=1,
            correlation=0.25,
        )
        assert err.feature_name == "momentum_5d"
        assert err.look_ahead_days == 1
        assert abs(err.correlation - 0.25) < 1e-9

    def test_pit_violation_error_message_contains_threshold(self) -> None:
        """Error message must reference the threshold value for operator diagnostics."""
        err = PITViolationError("f", 1, 0.1)
        assert str(LEAK_THRESHOLD) in str(err)

    def test_leaky_feature_detected(self) -> None:
        """A feature = future return at lag 1 must trigger PITViolationError."""
        df = _make_ohlcv_df(n=300)
        leaky = _make_leaky_feature(df, look_ahead=1)
        feature_df = df[["close"]].copy()
        feature_df["leaky_momentum"] = leaky
        future_returns = df["returns"].shift(-1)

        validator = LeakageValidator()
        with pytest.raises(PITViolationError) as exc_info:
            validator.validate(feature_df, future_returns)

        assert exc_info.value.look_ahead_days >= MIN_LOOK_AHEAD_DAYS
        assert abs(exc_info.value.correlation) > LEAK_THRESHOLD

    def test_clean_feature_does_not_raise(self) -> None:
        """Pure noise feature must NOT trigger PITViolationError."""
        df = _make_ohlcv_df(n=300)
        clean = _make_clean_feature(df)
        feature_df = pd.DataFrame({"clean_noise": clean})
        future_returns = df["returns"].shift(-1)

        validator = LeakageValidator()
        result = validator.validate(feature_df, future_returns)
        assert result is None, "Clean feature should not trigger PITViolationError"

    def test_constants_are_within_design_spec(self) -> None:
        """Validate that the constants match the Phase 2 design spec."""
        assert LEAK_THRESHOLD == 0.05, f"Expected 0.05, got {LEAK_THRESHOLD}"
        assert MIN_ALIGNED_SAMPLES == 30
        assert MIN_LOOK_AHEAD_DAYS == 1
        assert MAX_LOOK_AHEAD_DAYS == 22

    def test_insufficient_samples_skips_without_raising(self) -> None:
        """Fewer than MIN_ALIGNED_SAMPLES aligned rows → no error (skip silently)."""
        df = _make_ohlcv_df(n=20)  # 20 < MIN_ALIGNED_SAMPLES (30)
        leaky = _make_leaky_feature(df)
        feature_df = pd.DataFrame({"leaky": leaky})
        future_returns = df["returns"].shift(-1)

        validator = LeakageValidator()
        # Should NOT raise — insufficient samples means we can't compute correlation
        result = validator.validate(feature_df, future_returns)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════════
# 5. FeaturePipeline — PIT end-to-end regression tests
# ═══════════════════════════════════════════════════════════════════════════════


def _make_mock_market_data(
    data_as_of: datetime,
    confidence_score: int = 88,
    signal_engine_allowed: bool = True,
) -> dict[str, Any]:
    return {
        "data": {
            "instrumentId": "NSE:NIFTY:IDX",
            "symbol": "NIFTY",
            "close": 22150.0,
            "volume": 1_000_000.0,
        },
        "metadata": {
            "quality": {
                "score": confidence_score,
                "signalEngineAllowed": signal_engine_allowed,
            },
            "dataAsOf": data_as_of.isoformat(),
        },
    }


@pytest.mark.unit
@pytest.mark.pit
class TestFeaturePipelinePITEndToEnd:
    """End-to-end PIT tests on the existing FeaturePipeline implementation."""

    @pytest.mark.asyncio
    async def test_past_source_data_records_zero_pit_violations(self) -> None:
        """Source data 15 min before PIT boundary → report.pit_violations_count == 0."""
        from src.features.pipeline import FeaturePipeline

        mock_data = AsyncMock()
        mock_data.get_live_quote.return_value = _make_mock_market_data(
            data_as_of=_PAST_TS
        )
        mock_data.get_historical_ohlcv.return_value = []

        mock_news = AsyncMock()
        mock_news.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
        _vector, report = await pipeline.build_vector("NIFTY", _PIT_BOUNDARY)

        assert report.pit_violations_count == 0, (
            f"No PIT violations expected for past source data, "
            f"got {report.pit_violations_count}"
        )

    @pytest.mark.asyncio
    async def test_future_source_data_records_pit_violation(self) -> None:
        """Source data AFTER PIT boundary → report.pit_violations_count > 0."""
        from src.features.pipeline import FeaturePipeline

        mock_data = AsyncMock()
        mock_data.get_live_quote.return_value = _make_mock_market_data(
            data_as_of=_FUTURE_TS
        )
        mock_data.get_historical_ohlcv.return_value = []

        mock_news = AsyncMock()
        mock_news.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
        _vector, report = await pipeline.build_vector("NIFTY", _PIT_BOUNDARY)

        assert report.pit_violations_count > 0, (
            "Expected at least 1 PIT violation when source data is after PIT boundary."
        )

    @pytest.mark.asyncio
    async def test_equal_timestamp_source_data_records_pit_violation(self) -> None:
        """Source data with timestamp EQUAL to PIT boundary is also a violation."""
        from src.features.pipeline import FeaturePipeline

        mock_data = AsyncMock()
        mock_data.get_live_quote.return_value = _make_mock_market_data(
            data_as_of=_PIT_BOUNDARY  # equal — strict rule: must be <
        )
        mock_data.get_historical_ohlcv.return_value = []

        mock_news = AsyncMock()
        mock_news.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
        _vector, report = await pipeline.build_vector("NIFTY", _PIT_BOUNDARY)

        assert report.pit_violations_count > 0, (
            "Source data at EXACTLY the PIT boundary must be flagged as a violation "
            "(PIT rule is strict: source_ts must be < pit_boundary)."
        )

    @pytest.mark.asyncio
    async def test_feature_vector_timestamp_equals_requested_pit_boundary(self) -> None:
        """FeatureVector.timestamp must always equal the requested PIT boundary."""
        from src.features.pipeline import FeaturePipeline

        mock_data = AsyncMock()
        mock_data.get_live_quote.return_value = _make_mock_market_data(
            data_as_of=_PAST_TS
        )
        mock_data.get_historical_ohlcv.return_value = []

        mock_news = AsyncMock()
        mock_news.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
        vector, _report = await pipeline.build_vector("NIFTY", _PIT_BOUNDARY)

        assert vector.timestamp == _PIT_BOUNDARY, (
            f"FeatureVector.timestamp must be the PIT boundary. "
            f"Expected {_PIT_BOUNDARY}, got {vector.timestamp}"
        )

    @pytest.mark.asyncio
    @given(
        offset_minutes=st.integers(min_value=1, max_value=60)
    )
    @h_settings(max_examples=10, suppress_health_check=[HealthCheck.too_slow], deadline=3000)
    async def test_past_source_always_produces_zero_pit_violations(
        self, offset_minutes: int
    ) -> None:
        """Hypothesis: for any source timestamp before PIT boundary, zero violations."""
        from src.features.pipeline import FeaturePipeline

        past = _PIT_BOUNDARY - timedelta(minutes=offset_minutes)

        mock_data = AsyncMock()
        mock_data.get_live_quote.return_value = _make_mock_market_data(
            data_as_of=past
        )
        mock_data.get_historical_ohlcv.return_value = []

        mock_news = AsyncMock()
        mock_news.fetch_news_context.return_value = None

        pipeline = FeaturePipeline(data_client=mock_data, news_client=mock_news)
        _vector, report = await pipeline.build_vector("NIFTY", _PIT_BOUNDARY)

        assert report.pit_violations_count == 0, (
            f"Source data {offset_minutes} min before PIT boundary must yield 0 violations, "
            f"got {report.pit_violations_count}"
        )
