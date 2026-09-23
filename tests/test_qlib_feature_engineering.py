"""
test_qlib_feature_engineering.py

TDD test file for QlibFeatureEngine (Alpha158/Alpha360 factor computation).

Written BEFORE QlibFeatureEngine implementation (red phase — tests skip until
src/features/qlib_engine.py is implemented).

Properties tested:
  Property 5a: compute_alpha158(df, symbol) == compute_alpha158(df, symbol) (idempotence)
  Property 5b: all 158 factor values are finite (no NaN or Inf for valid inputs)
  Property 5c: factor values are within documented bounds (reasonable range check)
  Property 5d: compute_alpha360() returns exactly 360 keys when enabled

Requirements: Req 2.3, Req 2.4, Req 18.5

Validates: Requirements 2.3, 2.4, 18.5
"""
from __future__ import annotations

import math
import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, assume
from hypothesis import given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")


# ── OHLCV DataFrame helpers ───────────────────────────────────────────────────


def make_valid_ohlcv_df(n_bars: int = 300, seed: int = 42) -> pd.DataFrame:
    """Create a realistic OHLCV DataFrame for testing.

    Prices start around 22 000 (NIFTY-like) and follow a random walk.
    All OHLC consistency constraints are enforced:
    high >= max(open, close), low <= min(open, close).
    Volume is drawn from a realistic range.
    """
    rng = np.random.default_rng(seed)

    close = 22_000.0 + np.cumsum(rng.normal(0, 50, n_bars))
    close = np.maximum(close, 1_000.0)  # floor: never go negative

    high = close * (1 + np.abs(rng.normal(0, 0.01, n_bars)))
    low = close * (1 - np.abs(rng.normal(0, 0.01, n_bars)))
    open_ = close * (1 + rng.normal(0, 0.005, n_bars))
    volume = rng.integers(100_000, 5_000_000, n_bars).astype(float)

    # Enforce OHLC consistency
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # Build a business-day DatetimeIndex starting from 2023-01-02
    base_date = date(2023, 1, 2)
    dates = pd.bdate_range(start=str(base_date), periods=n_bars)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=dates,
    )


@st.composite
def valid_ohlcv_dataframes(draw: Any) -> pd.DataFrame:
    """Hypothesis strategy: generate random but valid OHLCV DataFrames.

    n_bars is drawn from [252, 500] — the minimum (252) guarantees the
    200-bar history required by QlibFeatureEngine.compute_alpha158.
    seed is drawn randomly so the strategy produces genuinely distinct frames.
    """
    n_bars = draw(st.integers(min_value=252, max_value=500))
    seed = draw(st.integers(min_value=0, max_value=10_000))
    return make_valid_ohlcv_df(n_bars, seed=seed)


# ── Guard: skip all tests gracefully if qlib_engine is not yet implemented ────


def _import_engine() -> Any:
    """Import QlibFeatureEngine or skip the calling test."""
    try:
        from src.features.qlib_engine import QlibFeatureEngine  # type: ignore[import-not-found]

        return QlibFeatureEngine
    except (ImportError, ModuleNotFoundError):
        pytest.skip("QlibFeatureEngine not yet implemented — TDD red phase")


# ── Test classes ──────────────────────────────────────────────────────────────


class TestQlibFeatureEngineImport:
    """Verify that QlibFeatureEngine is importable and has the required interface."""

    def test_qlib_engine_importable(self) -> None:
        """QlibFeatureEngine must be importable from src.features.qlib_engine."""
        EngineClass = _import_engine()
        engine = EngineClass()
        assert engine is not None

    def test_alpha158_method_exists(self) -> None:
        """QlibFeatureEngine must expose a compute_alpha158 method."""
        EngineClass = _import_engine()
        assert hasattr(EngineClass, "compute_alpha158"), (
            "QlibFeatureEngine must have a compute_alpha158 method"
        )

    def test_alpha360_method_exists(self) -> None:
        """QlibFeatureEngine must expose a compute_alpha360 method."""
        EngineClass = _import_engine()
        assert hasattr(EngineClass, "compute_alpha360"), (
            "QlibFeatureEngine must have a compute_alpha360 method"
        )


class TestAlpha158OutputStructure:
    """Verify the structural contract of compute_alpha158 output."""

    def test_alpha158_returns_dict(self) -> None:
        """compute_alpha158 must return a plain dict."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        result = engine.compute_alpha158(df, "NIFTY")
        assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"

    def test_alpha158_returns_at_least_158_factors(self) -> None:
        """compute_alpha158 must return at least 158 factor keys."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        result = engine.compute_alpha158(df, "NIFTY")
        assert len(result) >= 158, (
            f"compute_alpha158 must return ≥ 158 factors, got {len(result)}"
        )

    def test_alpha158_factor_keys_are_strings(self) -> None:
        """All Alpha158 factor dictionary keys must be strings."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        result = engine.compute_alpha158(df, "NIFTY")
        non_string_keys = [k for k in result if not isinstance(k, str)]
        assert not non_string_keys, (
            f"Alpha158 keys must be strings; found non-string keys: {non_string_keys[:5]}"
        )

    def test_alpha158_factor_values_are_numeric(self) -> None:
        """All Alpha158 factor values must be numeric (int or float)."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        result = engine.compute_alpha158(df, "NIFTY")
        bad = {k: type(v).__name__ for k, v in result.items() if not isinstance(v, (int, float))}
        assert not bad, (
            f"Alpha158 must produce numeric values; got non-numeric for: {list(bad.items())[:5]}"
        )


class TestAlpha158Idempotence:
    """
    Property 5a: compute_alpha158(df, symbol) == compute_alpha158(df, symbol)

    The same OHLCV input always produces an identical factor dict.
    No random state, no side effects, no mutated internal buffers between calls.

    Validates: Requirements 2.3, 18.5
    """

    def test_alpha158_idempotent_fixed_input(self) -> None:
        """Two consecutive calls with the same DataFrame produce identical output."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        result1 = engine.compute_alpha158(df, "NIFTY")
        result2 = engine.compute_alpha158(df, "NIFTY")

        assert result1 == result2, (
            "compute_alpha158 is not idempotent: two calls with identical input "
            "produced different results"
        )

    def test_alpha158_idempotent_different_symbol_different_result(self) -> None:
        """The symbol parameter is used: different symbols may produce different factors.

        This is not a strict requirement — same OHLCV may yield the same numbers —
        but we verify the engine accepts arbitrary symbol names without error.
        """
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        # Should not raise for any alphanumeric symbol string
        result_nifty = engine.compute_alpha158(df, "NIFTY")
        result_reliance = engine.compute_alpha158(df, "RELIANCE")

        # Both calls must return dicts of the required length
        assert len(result_nifty) >= 158
        assert len(result_reliance) >= 158

    @given(ohlcv_df=valid_ohlcv_dataframes())
    @h_settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5_000,  # generous: Qlib init may take >500ms on first call
    )
    def test_alpha158_idempotence_hypothesis(self, ohlcv_df: pd.DataFrame) -> None:
        """
        Property 5a (Hypothesis): For all valid OHLCV DataFrames with ≥ 252 bars,
        compute_alpha158(df, symbol) == compute_alpha158(df, symbol).

        Validates: Requirements 2.3, 18.5
        """
        EngineClass = _import_engine()
        engine = EngineClass()

        result1 = engine.compute_alpha158(ohlcv_df, "TEST")
        result2 = engine.compute_alpha158(ohlcv_df, "TEST")

        assert result1 == result2, (
            f"compute_alpha158 is not idempotent for a DataFrame with {len(ohlcv_df)} bars"
        )


class TestAlpha158FiniteOutputs:
    """
    Property 5b: All 158 factor values must be finite (no NaN or Inf).

    For complete OHLCV data (no missing values, no zeros), every output factor
    must satisfy math.isfinite(v).

    Validates: Requirements 2.3, 2.4, 18.5
    """

    def test_alpha158_no_nan_or_inf_fixed(self) -> None:
        """Standard 300-bar OHLCV DataFrame produces no non-finite Alpha158 values."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        result = engine.compute_alpha158(df, "NIFTY")

        non_finite = {k: v for k, v in result.items() if not math.isfinite(v)}
        assert not non_finite, (
            f"Alpha158 produced non-finite values for factors: "
            f"{list(non_finite.items())[:5]}"
        )

    def test_alpha158_no_nan_with_500_bars(self) -> None:
        """Longer OHLCV history (500 bars) also produces no non-finite Alpha158 values."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(500)
        engine = EngineClass()

        result = engine.compute_alpha158(df, "BANKNIFTY")

        non_finite = {k: v for k, v in result.items() if not math.isfinite(v)}
        assert not non_finite, (
            f"Alpha158 produced non-finite values (500 bars): "
            f"{list(non_finite.items())[:5]}"
        )

    @given(ohlcv_df=valid_ohlcv_dataframes())
    @h_settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5_000,
    )
    def test_alpha158_finite_outputs_hypothesis(self, ohlcv_df: pd.DataFrame) -> None:
        """
        Property 5b (Hypothesis): For any valid OHLCV DataFrame with ≥ 252 bars,
        all Alpha158 factor values are finite.

        Validates: Requirements 2.4, 18.5
        """
        EngineClass = _import_engine()
        engine = EngineClass()

        result = engine.compute_alpha158(ohlcv_df, "TEST")

        non_finite = {k: v for k, v in result.items() if not math.isfinite(v)}
        assert not non_finite, (
            f"Alpha158 produced non-finite values for a DataFrame with "
            f"{len(ohlcv_df)} bars: {list(non_finite.items())[:5]}"
        )


class TestAlpha158BoundedRange:
    """
    Property 5c: Factor values must be within reasonable bounds.

    Raw Alpha158 factors are normalised cross-sectional ranks or z-scores;
    none should exceed ±1 000 000 in absolute value for realistic price data.
    This guards against division-by-zero, overflow, or unit errors.

    Validates: Requirements 2.4, 18.5
    """

    def test_alpha158_values_within_sanity_bounds(self) -> None:
        """All Alpha158 factor values must satisfy |v| <= 1e6 for NIFTY-like data."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        result = engine.compute_alpha158(df, "NIFTY")

        out_of_bounds = {k: v for k, v in result.items() if abs(v) > 1e6}
        assert not out_of_bounds, (
            f"Alpha158 produced unreasonably large values "
            f"(|v| > 1e6): {list(out_of_bounds.items())[:5]}"
        )

    @given(ohlcv_df=valid_ohlcv_dataframes())
    @h_settings(
        max_examples=10,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=5_000,
    )
    def test_alpha158_bounded_range_hypothesis(self, ohlcv_df: pd.DataFrame) -> None:
        """
        Property 5c (Hypothesis): For any valid OHLCV DataFrame,
        all Alpha158 factors satisfy |v| <= 1e6.

        Validates: Requirements 2.4
        """
        EngineClass = _import_engine()
        engine = EngineClass()

        result = engine.compute_alpha158(ohlcv_df, "TEST")

        out_of_bounds = {k: v for k, v in result.items() if abs(v) > 1e6}
        assert not out_of_bounds, (
            f"Alpha158 factors out of ±1e6 range: {list(out_of_bounds.items())[:5]}"
        )


class TestAlpha360Output:
    """
    Property 5d: compute_alpha360 must return exactly 360 keys when enabled.

    Also validates idempotence and finiteness for Alpha360.

    Validates: Requirements 2.3, 2.4
    """

    def test_alpha360_returns_exactly_360_factors(self) -> None:
        """compute_alpha360 must return a dict with exactly 360 keys."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        if not hasattr(engine, "compute_alpha360"):
            pytest.skip("compute_alpha360 not yet implemented")

        result = engine.compute_alpha360(df, "NIFTY")
        assert isinstance(result, dict), f"Expected dict, got {type(result).__name__}"
        assert len(result) == 360, (
            f"compute_alpha360 must return exactly 360 factors, got {len(result)}"
        )

    def test_alpha360_is_idempotent(self) -> None:
        """compute_alpha360 must produce identical output on two consecutive calls."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        if not hasattr(engine, "compute_alpha360"):
            pytest.skip("compute_alpha360 not yet implemented")

        result1 = engine.compute_alpha360(df, "NIFTY")
        result2 = engine.compute_alpha360(df, "NIFTY")

        assert result1 == result2, "compute_alpha360 is not idempotent"

    def test_alpha360_all_values_finite(self) -> None:
        """All Alpha360 factor values must be finite (no NaN or Inf)."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        if not hasattr(engine, "compute_alpha360"):
            pytest.skip("compute_alpha360 not yet implemented")

        result = engine.compute_alpha360(df, "NIFTY")

        non_finite = {k: v for k, v in result.items() if not math.isfinite(v)}
        assert not non_finite, (
            f"Alpha360 produced non-finite values: {list(non_finite.items())[:5]}"
        )

    def test_alpha360_keys_are_strings(self) -> None:
        """All Alpha360 factor keys must be strings."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(300)
        engine = EngineClass()

        if not hasattr(engine, "compute_alpha360"):
            pytest.skip("compute_alpha360 not yet implemented")

        result = engine.compute_alpha360(df, "NIFTY")
        non_string = [k for k in result if not isinstance(k, str)]
        assert not non_string, (
            f"Alpha360 has non-string keys: {non_string[:5]}"
        )

    @given(ohlcv_df=valid_ohlcv_dataframes())
    @h_settings(
        max_examples=5,
        suppress_health_check=[HealthCheck.too_slow],
        deadline=8_000,
    )
    def test_alpha360_idempotence_hypothesis(self, ohlcv_df: pd.DataFrame) -> None:
        """
        Property 5d (Hypothesis): For any valid OHLCV DataFrame,
        compute_alpha360 produces exactly 360 keys and is idempotent.

        Validates: Requirements 2.3, 2.4
        """
        EngineClass = _import_engine()
        engine = EngineClass()

        if not hasattr(engine, "compute_alpha360"):
            pytest.skip("compute_alpha360 not yet implemented")

        result1 = engine.compute_alpha360(ohlcv_df, "TEST")
        result2 = engine.compute_alpha360(ohlcv_df, "TEST")

        assert len(result1) == 360, (
            f"compute_alpha360 returned {len(result1)} keys (expected 360)"
        )
        assert result1 == result2, "compute_alpha360 is not idempotent"


class TestAlpha158MinimumHistoryGuard:
    """
    QlibFeatureEngine requires ≥ 200 bars of OHLCV history.
    Submitting fewer bars should either raise a clear exception
    or return an empty dict (not silently produce NaN-filled garbage).
    """

    def test_insufficient_history_raises_or_returns_empty(self) -> None:
        """With fewer than 200 bars, engine should raise ValueError or return {}."""
        EngineClass = _import_engine()
        df = make_valid_ohlcv_df(50)  # deliberately insufficient
        engine = EngineClass()

        try:
            result = engine.compute_alpha158(df, "NIFTY")
            # If it doesn't raise, the result must be empty or explicitly signal failure
            assert result == {} or len(result) == 0, (
                "With fewer than 200 bars, compute_alpha158 must return an empty dict "
                "or raise ValueError; got non-empty result"
            )
        except (ValueError, RuntimeError):
            pass  # acceptable: explicit error on insufficient history
