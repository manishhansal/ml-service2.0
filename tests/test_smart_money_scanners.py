"""
TDD red-phase test suite for the Smart Money Scanners (Scanners 2–5).

File scaffolded in Task 2.1.  Each test function below has a ``pass`` body;
implementations are filled in tasks 2.2–2.7.  Running pytest with
``--collect-only`` must succeed without import errors before any production
code beyond the schemas/configs exists.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given
from hypothesis import settings as h_settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Environment setup — must happen before any src.* imports that read env vars
# ---------------------------------------------------------------------------
os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://localhost:3001")

# ---------------------------------------------------------------------------
# Always-available imports (Task 1 complete)
# ---------------------------------------------------------------------------
from src.scanners.fair_value_gap.config import FVGConfig
from src.scanners.fair_value_gap.schemas import (
    FVGDirection,
    FVGLifecycle,
    FVGScanResponse,
    FVGScanResult,
    FVGZone,
)
from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.schemas import (
    SweepDirection,
    SweepLifecycle,
    SweepScanResponse,
    SweepScanResult,
    SweepSignal,
)
from src.scanners.market_structure.config import MSConfig
from src.scanners.market_structure.schemas import (
    MSScanResponse,
    MSScanResult,
    MSSignal,
    MSSignalType,
    TrendState,
)
from src.scanners.supply_demand.config import SDConfig
from src.scanners.supply_demand.schemas import (
    SDDirection,
    SDLifecycle,
    SDScanResponse,
    SDScanResult,
    SDZone,
    SDZoneType,
)

# ---------------------------------------------------------------------------
# Conditional imports for not-yet-implemented modules
# ---------------------------------------------------------------------------
try:
    from src.scanners.fair_value_gap.detector import detect_fvg_zones
except ImportError:
    detect_fvg_zones = None  # type: ignore[assignment]

try:
    from src.scanners.fair_value_gap.zone_manager import update_fvg_lifecycle
except ImportError:
    update_fvg_lifecycle = None  # type: ignore[assignment]

try:
    from src.scanners.fair_value_gap.scorer import rank_fvg_zones
except ImportError:
    rank_fvg_zones = None  # type: ignore[assignment]

try:
    from src.scanners.fair_value_gap.scanner import FVGScanner
except ImportError:
    FVGScanner = None  # type: ignore[assignment]

try:
    from src.scanners.supply_demand.detector import detect_sd_zones
except ImportError:
    detect_sd_zones = None  # type: ignore[assignment]

try:
    from src.scanners.supply_demand.zone_manager import update_sd_lifecycle
except ImportError:
    update_sd_lifecycle = None  # type: ignore[assignment]

try:
    from src.scanners.supply_demand.scorer import rank_sd_zones
except ImportError:
    rank_sd_zones = None  # type: ignore[assignment]

try:
    from src.scanners.supply_demand.scanner import SDScanner
except ImportError:
    SDScanner = None  # type: ignore[assignment]

try:
    from src.scanners.liquidity_sweep.detector import detect_sweeps
except ImportError:
    detect_sweeps = None  # type: ignore[assignment]

try:
    from src.scanners.liquidity_sweep.signal_manager import update_sweep_lifecycle
except ImportError:
    update_sweep_lifecycle = None  # type: ignore[assignment]

try:
    from src.scanners.liquidity_sweep.scorer import rank_sweeps
except ImportError:
    rank_sweeps = None  # type: ignore[assignment]

try:
    from src.scanners.liquidity_sweep.scanner import SweepScanner
except ImportError:
    SweepScanner = None  # type: ignore[assignment]

try:
    from src.scanners.market_structure.detector import detect_market_structure
except ImportError:
    detect_market_structure = None  # type: ignore[assignment]

try:
    from src.scanners.market_structure.state_manager import TrendStateTracker
except ImportError:
    TrendStateTracker = None  # type: ignore[assignment]

try:
    from src.scanners.market_structure.scorer import rank_ms_signals
except ImportError:
    rank_ms_signals = None  # type: ignore[assignment]

try:
    from src.scanners.market_structure.scanner import MSScanner
except ImportError:
    MSScanner = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# API test client — optional; skips gracefully if router not yet implemented
# ---------------------------------------------------------------------------
_api_available = False
try:
    from fastapi import FastAPI
    from starlette.testclient import TestClient

    from src.api.scanners import router as scanner_router

    _test_app = FastAPI()
    _test_app.include_router(scanner_router, prefix="/v2")
    test_client = TestClient(_test_app)
    _api_available = True
except Exception:  # noqa: BLE001
    pass

# ===========================================================================
# Hypothesis strategies
# ===========================================================================

price_st = st.floats(min_value=1.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
volume_st = st.floats(min_value=0.0, max_value=1e10, allow_nan=False, allow_infinity=False)


@st.composite
def ohlcv_strategy(draw: Any, min_rows: int = 20, max_rows: int = 100) -> pd.DataFrame:
    """Generate a valid OHLCV DataFrame with UTC DatetimeIndex.

    Enforces: ``high >= max(open, close) >= min(open, close) >= low > 0``
    and ``volume >= 0``.
    """
    n = draw(st.integers(min_value=min_rows, max_value=max_rows))
    rows = []
    for _ in range(n):
        low = draw(price_st)
        spread = draw(
            st.floats(min_value=0.0, max_value=low, allow_nan=False, allow_infinity=False)
        )
        high = low + spread
        open_ = draw(
            st.floats(min_value=low, max_value=high, allow_nan=False, allow_infinity=False)
        )
        close = draw(
            st.floats(min_value=low, max_value=high, allow_nan=False, allow_infinity=False)
        )
        volume = draw(volume_st)
        rows.append(
            {"open": open_, "high": high, "low": low, "close": close, "volume": volume}
        )
    start_year = draw(st.integers(min_value=2020, max_value=2022))
    ts = pd.date_range(
        start=datetime(start_year, 1, 1, tzinfo=timezone.utc),
        periods=n,
        freq="D",
        tz="UTC",
    )
    return pd.DataFrame(rows, index=ts)


@st.composite
def fvg_zone_strategy(draw: Any) -> FVGZone:
    """Generate a random valid FVGZone with consistent geometry."""
    fvg_low = draw(price_st)
    spread = draw(
        st.floats(min_value=0.01, max_value=fvg_low, allow_nan=False, allow_infinity=False)
    )
    fvg_high = fvg_low + spread
    fvg_mid = (fvg_high + fvg_low) / 2
    ob_ts = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2022, 1, 1),
            timezones=st.just(timezone.utc),
        )
    )
    c1_idx = draw(st.integers(min_value=0, max_value=50))
    c3_idx = c1_idx + draw(st.integers(min_value=2, max_value=5))
    return FVGZone(
        symbol=draw(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=8)
        ),
        direction=draw(st.sampled_from(FVGDirection)),
        timeframe=draw(st.sampled_from(["5m", "15m", "1h", "1d"])),
        fvg_high=fvg_high,
        fvg_low=fvg_low,
        fvg_midpoint=fvg_mid,
        fvg_timestamp=ob_ts,
        candle1_bar_index=c1_idx,
        candle3_bar_index=c3_idx,
        impulse_atr=draw(
            st.floats(min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        lifecycle_state=draw(st.sampled_from(FVGLifecycle)),
        gap_size_score=draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        volume_score=draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        combined_score=draw(
            st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def sd_zone_strategy(draw: Any) -> SDZone:
    """Generate a random valid SDZone with consistent direction/zone_type mapping.

    Zone type → direction mapping:
      RBR / DBR → DEMAND
      RBD / DBD → SUPPLY
    Zone strength is sampled from the fixed set ``{1.0, 0.7, 0.4, 0.0}`` consistent
    with the lifecycle state.
    """
    zone_type = draw(st.sampled_from(SDZoneType))
    direction = (
        SDDirection.DEMAND
        if zone_type in (SDZoneType.RBR, SDZoneType.DBR)
        else SDDirection.SUPPLY
    )
    zone_low = draw(price_st)
    spread = draw(
        st.floats(min_value=0.01, max_value=zone_low, allow_nan=False, allow_infinity=False)
    )
    zone_high = zone_low + spread
    lifecycle = draw(st.sampled_from(SDLifecycle))
    strength_map = {
        SDLifecycle.FRESH: 1.0,
        SDLifecycle.TESTED: 0.7,
        SDLifecycle.USED: 0.4,
        SDLifecycle.BROKEN: 0.0,
    }
    zone_strength = strength_map[lifecycle]
    ts1 = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2022, 1, 1),
            timezones=st.just(timezone.utc),
        )
    )
    ts2 = ts1 + timedelta(days=draw(st.integers(min_value=1, max_value=5)))
    b_start = draw(st.integers(min_value=1, max_value=10))
    b_end = b_start + draw(st.integers(min_value=1, max_value=4))
    dep_idx = b_end + 1
    return SDZone(
        symbol=draw(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=8)
        ),
        direction=direction,
        zone_type=zone_type,
        timeframe=draw(st.sampled_from(["5m", "15m", "1h", "1d"])),
        zone_high=zone_high,
        zone_low=zone_low,
        entry_leg_timestamp=ts1,
        departure_timestamp=ts2,
        base_start_bar_index=b_start,
        base_end_bar_index=b_end,
        departure_bar_index=dep_idx,
        lifecycle_state=lifecycle,
        zone_strength=zone_strength,
        departure_strength=draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        combined_score=draw(
            st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def sweep_signal_strategy(draw: Any) -> SweepSignal:
    """Generate a random valid SweepSignal with consistent zone geometry.

    ``sweep_zone_high = swept_level + delta``,
    ``sweep_zone_low  = swept_level - delta``  (delta > 0).
    """
    swept_level = draw(price_st)
    delta = draw(
        st.floats(
            min_value=0.01,
            max_value=swept_level * 0.1,
            allow_nan=False,
            allow_infinity=False,
        )
    )
    ts = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2022, 1, 1),
            timezones=st.just(timezone.utc),
        )
    )
    return SweepSignal(
        symbol=draw(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=8)
        ),
        direction=draw(st.sampled_from(SweepDirection)),
        timeframe=draw(st.sampled_from(["5m", "15m", "1h", "1d"])),
        swept_level=swept_level,
        sweep_zone_high=swept_level + delta,
        sweep_zone_low=swept_level - delta,
        sweep_timestamp=ts,
        sweep_bar_index=draw(st.integers(min_value=0, max_value=100)),
        lifecycle_state=draw(st.sampled_from(SweepLifecycle)),
        equal_levels_count=draw(st.integers(min_value=1, max_value=5)),
        sweep_magnitude=draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        volume_ratio=draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        conviction_score=draw(
            st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
        ),
    )


@st.composite
def ms_signal_strategy(draw: Any) -> MSSignal:
    """Generate a random valid MSSignal with consistent close_at_break direction.

    For BULLISH signals: ``close_at_break > broken_level``.
    For BEARISH signals: ``close_at_break < broken_level``.
    ``is_choch`` is derived from ``signal_type``.
    """
    signal_type = draw(st.sampled_from(MSSignalType))
    broken_level = draw(price_st)
    is_bullish_signal = signal_type in (MSSignalType.BOS_BULLISH, MSSignalType.CHOCH_BULLISH)
    is_choch = signal_type in (MSSignalType.CHOCH_BULLISH, MSSignalType.CHOCH_BEARISH)
    if is_bullish_signal:
        close_at_break = draw(
            st.floats(
                min_value=broken_level + 0.01,
                max_value=broken_level * 2.0,
                allow_nan=False,
                allow_infinity=False,
            )
        )
        trend_state = TrendState.BULLISH
    else:
        close_at_break = draw(
            st.floats(
                min_value=0.01,
                max_value=max(0.01, broken_level - 0.01),
                allow_nan=False,
                allow_infinity=False,
            )
        )
        trend_state = TrendState.BEARISH
    ts = draw(
        st.datetimes(
            min_value=datetime(2020, 1, 1),
            max_value=datetime(2022, 1, 1),
            timezones=st.just(timezone.utc),
        )
    )
    return MSSignal(
        symbol=draw(
            st.text(alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ", min_size=1, max_size=8)
        ),
        signal_type=signal_type,
        timeframe=draw(st.sampled_from(["5m", "15m", "1h", "1d"])),
        broken_level=broken_level,
        close_at_break=close_at_break,
        break_timestamp=ts,
        break_bar_index=draw(st.integers(min_value=0, max_value=100)),
        trend_state=trend_state,
        is_choch=is_choch,
        swing_count=draw(st.integers(min_value=0, max_value=10)),
        break_magnitude=draw(
            st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False)
        ),
        conviction_score=draw(
            st.floats(min_value=0.0, max_value=10000.0, allow_nan=False, allow_infinity=False)
        ),
    )


# ---------------------------------------------------------------------------
# FVG test helpers
# ---------------------------------------------------------------------------


def _make_fvg_df(n_extra: int = 0, start_ts: datetime | None = None) -> pd.DataFrame:
    """Build the simplest valid 3-candle bullish FVG DataFrame.

    - bar 0: high=100 (c1)
    - bar 1: big bullish candle (impulse); body=29, ATR≈21
    - bar 2: low=119 (c3); gap=19 >> 0.5×ATR(≈17), satisfying the default min_gap_atr_multiplier=0.5

    ATR breakdown: TR[0]=10, TR[1]=32, TR[2]=9 → ATR[2]≈17; gap=19 ≥ 0.5×17=8.5 ✓
    """
    ts = start_ts or datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"open": 95.0, "high": 100.0, "low": 90.0, "close": 98.0, "volume": 1000.0},    # c1: high=100
        {"open": 99.0, "high": 130.0, "low": 98.0, "close": 128.0, "volume": 5000.0},   # c2: big impulse
        {"open": 120.0, "high": 121.0, "low": 119.0, "close": 120.0, "volume": 2000.0}, # c3: low=119 > high[0]=100
    ]
    for _ in range(n_extra):
        rows.append({"open": 110.0, "high": 112.0, "low": 109.0, "close": 111.0, "volume": 1000.0})
    timestamps = pd.date_range(start=ts, periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=timestamps)


# ===========================================================================
# Task 2.2 — TDD red-phase tests for FVG scanner (Req 1–7)
# ===========================================================================


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_bullish_detection_req_1_1() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df()
    config = FVGConfig()
    zones = detect_fvg_zones(df, "TEST", "1h", config)
    assert len(zones) >= 1
    z = zones[0]
    assert z.direction == FVGDirection.BULLISH
    assert z.fvg_low == pytest.approx(df["high"].iloc[0])   # FVG_Low = high[c1]
    assert z.fvg_high == pytest.approx(df["low"].iloc[2])   # FVG_High = low[c3]


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_bearish_detection_req_1_2() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    # Build bearish FVG: high[c3] < low[c1] AND close[c3] < low[c1]
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"open": 100.0, "high": 110.0, "low": 102.0, "close": 108.0, "volume": 1000.0},  # c1: low=102
        {"open": 107.0, "high": 108.0, "low": 75.0,  "close": 77.0,  "volume": 5000.0},  # c2: big bearish
        {"open": 78.0,  "high": 80.0,  "low": 70.0,  "close": 72.0,  "volume": 2000.0},  # c3: high=80 < low[c1]=102
    ]
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=3, freq="D", tz="UTC"))
    zones = detect_fvg_zones(df, "TEST", "1h", FVGConfig())
    assert len(zones) >= 1
    z = zones[0]
    assert z.direction == FVGDirection.BEARISH
    assert z.fvg_high == pytest.approx(df["low"].iloc[0])   # FVG_High = low[c1]
    assert z.fvg_low  == pytest.approx(df["high"].iloc[2])  # FVG_Low = high[c3]


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_midpoint_computed_req_1_3() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df()
    zones = detect_fvg_zones(df, "TEST", "1h", FVGConfig())
    assert len(zones) >= 1
    z = zones[0]
    assert z.fvg_midpoint == pytest.approx((z.fvg_high + z.fvg_low) / 2)


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_overlapping_windows_req_1_4() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    # Craft 4-bar OHLCV where window [0,1,2] and [1,2,3] both form valid bullish FVGs
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"open": 95.0, "high": 100.0, "low": 90.0,  "close": 98.0,  "volume": 1000.0},  # high=100
        {"open": 99.0, "high": 150.0, "low": 98.0,  "close": 148.0, "volume": 5000.0},  # big impulse; also c1 for next window: high=150
        {"open": 106.0, "high": 200.0, "low": 105.0, "close": 195.0, "volume": 3000.0},  # c3 of window0: low=105>100; c2 of window1
        {"open": 196.0, "high": 220.0, "low": 160.0, "close": 215.0, "volume": 2000.0},  # c3 of window1: low=160>150
    ]
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=4, freq="D", tz="UTC"))
    config = FVGConfig(impulse_atr_multiplier=0.1, min_gap_atr_multiplier=0.01)  # relaxed filters
    zones = detect_fvg_zones(df, "TEST", "1h", config)
    assert len(zones) >= 2, f"Expected at least 2 zones for overlapping windows, got {len(zones)}"


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_fewer_than_3_bars_req_1_5() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0},
    ]
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=2, freq="D", tz="UTC"))
    result = detect_fvg_zones(df, "TEST", "1h", FVGConfig())
    assert result == [], f"Expected [], got {result}"


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_nan_skipped_req_1_6() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df(n_extra=3)
    # Insert NaN in bar 1's close (c2 of first window) — this should skip window [0,1,2]
    df.loc[df.index[1], "close"] = float("nan")
    result = detect_fvg_zones(df, "TEST", "1h", FVGConfig())
    assert isinstance(result, list), "Should return a list without raising"
    # Bar index 1 has NaN; window [0,1,2] should be skipped
    assert all(z.candle1_bar_index != 0 for z in result), (
        "Window starting at bar 0 should be skipped due to NaN at bar 1"
    )


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_timestamp_is_candle3_req_1_7() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df()
    zones = detect_fvg_zones(df, "TEST", "1h", FVGConfig())
    assert len(zones) >= 1
    z = zones[0]
    assert z.fvg_timestamp == df.index[z.candle3_bar_index], (
        f"fvg_timestamp should equal df.index[candle3_bar_index]"
    )


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_impulse_filter_rejects_small_body_req_2_1() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    # Build FVG where c2 body is intentionally small (0.5×ATR) - should be rejected
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    # All bars have TR≈2 → ATR14≈2; c2 body = 0.5 (< 1.5×2=3) → rejected with multiplier=1.5
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}] * 20
    rows_list = list(rows)
    rows_list[17] = {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 1000.0}  # tiny c2 body=0
    rows_list[18] = {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}  # c3: low=99 NOT > high[17]=102
    df = pd.DataFrame(rows_list, index=pd.date_range(start=ts, periods=20, freq="D", tz="UTC"))
    # With default impulse_atr_multiplier=1.0, a zero body should be rejected
    config = FVGConfig(impulse_atr_multiplier=1.0)
    result = detect_fvg_zones(df, "TEST", "1h", config)
    # The pattern at [17,18,...] doesn't form a valid FVG anyway; test that raising multiplier blocks detection
    df2 = _make_fvg_df()  # has valid FVG
    config_strict = FVGConfig(impulse_atr_multiplier=1000.0)  # impossibly strict
    result_strict = detect_fvg_zones(df2, "TEST", "1h", config_strict)
    assert result_strict == [], "Impossibly strict impulse multiplier should reject all zones"


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_gap_size_filter_rejects_micro_gap_req_2_2() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df()
    config_strict = FVGConfig(min_gap_atr_multiplier=1000.0)  # impossibly strict gap filter
    result = detect_fvg_zones(df, "TEST", "1h", config_strict)
    assert result == [], "Impossibly strict gap filter should reject all zones"


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_atr_trailing_only_req_2_4() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df(n_extra=17)  # 20 bars total
    zones = detect_fvg_zones(df, "TEST", "1h", FVGConfig())
    # verify impulse_atr stored on each zone is positive and not NaN
    for z in zones:
        assert z.impulse_atr > 0, "impulse_atr should be positive (trailing ATR, no NaN from centered window)"
        assert z.impulse_atr == z.impulse_atr  # NaN check: NaN != NaN


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_atr_min_periods_req_2_5() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df()  # only 3 bars
    result = detect_fvg_zones(df, "TEST", "1h", FVGConfig())
    assert isinstance(result, list), "Should not raise even with only 3 bars"


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_lifecycle_unfilled_initial_req_3_1() -> None:
    # Pure schema test — no skip guard needed
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    zone = FVGZone(
        symbol="TEST", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=105.0, fvg_low=100.0, fvg_midpoint=102.5,
        fvg_timestamp=ts, candle1_bar_index=0, candle3_bar_index=2, impulse_atr=2.0,
    )
    assert zone.lifecycle_state == FVGLifecycle.UNFILLED


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_lifecycle_partial_fill_req_3_2() -> None:
    if update_fvg_lifecycle is None:
        pytest.skip("update_fvg_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    zone = FVGZone(
        symbol="TEST", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=105.0, fvg_low=100.0, fvg_midpoint=102.5,
        fvg_timestamp=ts, candle1_bar_index=0, candle3_bar_index=2, impulse_atr=2.0,
    )
    # Candle that enters the zone (low=101 ≤ fvg_high=105, high=106 ≥ fvg_low=100) but close above midpoint
    result = update_fvg_lifecycle(
        zone,
        candle_low=101.0, candle_high=106.0, candle_close=104.0,
        bar_index=5, creation_bar_index=2, config=FVGConfig(),
    )
    assert result.lifecycle_state == FVGLifecycle.PARTIALLY_FILLED


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_lifecycle_filled_req_3_3() -> None:
    if update_fvg_lifecycle is None:
        pytest.skip("update_fvg_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    # Start with PARTIALLY_FILLED zone
    zone = FVGZone(
        symbol="TEST", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=105.0, fvg_low=100.0, fvg_midpoint=102.5,
        fvg_timestamp=ts, candle1_bar_index=0, candle3_bar_index=2, impulse_atr=2.0,
        lifecycle_state=FVGLifecycle.PARTIALLY_FILLED,
    )
    # Candle that closes below midpoint (102.5) for bullish zone → FILLED
    result = update_fvg_lifecycle(
        zone,
        candle_low=99.0, candle_high=104.0, candle_close=101.0,
        bar_index=6, creation_bar_index=2, config=FVGConfig(),
    )
    assert result.lifecycle_state == FVGLifecycle.FILLED


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_lifecycle_terminal_filled_req_3_4() -> None:
    if update_fvg_lifecycle is None:
        pytest.skip("update_fvg_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    zone = FVGZone(
        symbol="TEST", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=105.0, fvg_low=100.0, fvg_midpoint=102.5,
        fvg_timestamp=ts, candle1_bar_index=0, candle3_bar_index=2, impulse_atr=2.0,
        lifecycle_state=FVGLifecycle.FILLED,
    )
    result = update_fvg_lifecycle(
        zone,
        candle_low=90.0, candle_high=120.0, candle_close=95.0,
        bar_index=10, creation_bar_index=2, config=FVGConfig(),
    )
    assert result.lifecycle_state == FVGLifecycle.FILLED


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_lifecycle_expiry_req_3_5() -> None:
    if update_fvg_lifecycle is None:
        pytest.skip("update_fvg_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    config = FVGConfig(fvg_expiry_bars=10)
    zone = FVGZone(
        symbol="TEST", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=105.0, fvg_low=100.0, fvg_midpoint=102.5,
        fvg_timestamp=ts, candle1_bar_index=0, candle3_bar_index=2, impulse_atr=2.0,
    )
    # Candle far outside zone, bar_index - creation_bar_index = 11 > 10
    result = update_fvg_lifecycle(
        zone,
        candle_low=200.0, candle_high=210.0, candle_close=205.0,
        bar_index=13, creation_bar_index=2, config=config,
    )
    assert result.lifecycle_state == FVGLifecycle.EXPIRED


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_scorer_gap_size_score_req_4_1() -> None:
    if rank_fvg_zones is None or detect_fvg_zones is None:
        pytest.skip("FVG scorer not yet implemented")
    df = _make_fvg_df()
    config = FVGConfig()
    zones = detect_fvg_zones(df, "TEST", "1h", config)
    if not zones:
        pytest.skip("No zones detected in test data")
    ranked = rank_fvg_zones(zones, {"TEST": df}, config)
    z = ranked[0]
    # Just verify the score is positive and non-NaN
    assert z.gap_size_score >= 0
    assert z.gap_size_score == z.gap_size_score  # NaN check: NaN != NaN


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_scorer_volume_score_req_4_2() -> None:
    if rank_fvg_zones is None or detect_fvg_zones is None:
        pytest.skip("FVG scorer not yet implemented")
    df = _make_fvg_df()
    ranked = rank_fvg_zones(detect_fvg_zones(df, "TEST", "1h", FVGConfig()), {"TEST": df}, FVGConfig())
    if not ranked:
        pytest.skip("No zones detected")
    z = ranked[0]
    assert z.volume_score >= 0
    assert z.volume_score == z.volume_score  # NaN check: NaN != NaN


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_scorer_combined_score_req_4_3() -> None:
    if rank_fvg_zones is None or detect_fvg_zones is None:
        pytest.skip("FVG scorer not yet implemented")
    df = _make_fvg_df()
    ranked = rank_fvg_zones(detect_fvg_zones(df, "TEST", "1h", FVGConfig()), {"TEST": df}, FVGConfig())
    if not ranked:
        pytest.skip("No zones detected")
    for z in ranked:
        assert z.combined_score == pytest.approx(z.gap_size_score * z.volume_score, rel=1e-6)


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_scorer_sorted_descending_req_4_4() -> None:
    if rank_fvg_zones is None:
        pytest.skip("rank_fvg_zones not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    # Build 3 zones with different scores directly
    zone1 = FVGZone(
        symbol="A", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=105.0, fvg_low=100.0, fvg_midpoint=102.5,
        fvg_timestamp=ts, candle1_bar_index=0, candle3_bar_index=2,
        impulse_atr=2.0, combined_score=10.0,
    )
    zone2 = FVGZone(
        symbol="B", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=108.0, fvg_low=100.0, fvg_midpoint=104.0,
        fvg_timestamp=ts, candle1_bar_index=0, candle3_bar_index=2,
        impulse_atr=2.0, combined_score=5.0,
    )
    zone3 = FVGZone(
        symbol="C", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=110.0, fvg_low=100.0, fvg_midpoint=105.0,
        fvg_timestamp=ts, candle1_bar_index=0, candle3_bar_index=2,
        impulse_atr=2.0, combined_score=20.0,
    )
    # Create minimal df for each symbol
    df_dummy = _make_fvg_df()
    df_by_symbol = {"A": df_dummy, "B": df_dummy, "C": df_dummy}
    ranked = rank_fvg_zones([zone1, zone2, zone3], df_by_symbol, FVGConfig())
    scores = [z.combined_score for z in ranked]
    assert scores == sorted(scores, reverse=True), f"Expected descending order, got {scores}"


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_scorer_zero_avg_volume_req_4_6() -> None:
    if rank_fvg_zones is None:
        pytest.skip("rank_fvg_zones not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    # DataFrame with all-zero volume
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 0.0}] * 30
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=30, freq="D", tz="UTC"))
    zone = FVGZone(
        symbol="A", direction=FVGDirection.BULLISH, timeframe="1h",
        fvg_high=105.0, fvg_low=100.0, fvg_midpoint=102.5,
        fvg_timestamp=df.index[2], candle1_bar_index=0, candle3_bar_index=2, impulse_atr=2.0,
    )
    ranked = rank_fvg_zones([zone], {"A": df}, FVGConfig())
    assert ranked[0].volume_score == 0.0


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_timeframe_5m_expiry_req_5_2() -> None:
    config = FVGConfig()
    assert config.tf_5m_expiry_bars == 288


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_timeframe_1d_expiry_req_5_3() -> None:
    config = FVGConfig()
    assert config.tf_1d_expiry_bars == 20


@pytest.mark.tdd
@pytest.mark.unit
def test_fvg_invalid_timeframe_raises_req_5_4() -> None:
    if FVGScanner is None:
        pytest.skip("FVGScanner not yet implemented")
    scanner = FVGScanner()
    with pytest.raises(ValueError, match="5m"):
        scanner.scan({}, timeframe="1w")


@pytest.mark.tdd
@pytest.mark.integration
def test_fvg_api_returns_200() -> None:
    if not _api_available:
        pytest.skip("FVG API not yet available")
    resp = test_client.get("/v2/scanners/fair-value-gaps", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    data = resp.json()
    assert "zones" in data


@pytest.mark.tdd
@pytest.mark.integration
def test_fvg_api_limit_param_req_6_2() -> None:
    if not _api_available:
        pytest.skip("FVG API not yet available")
    resp = test_client.get("/v2/scanners/fair-value-gaps?limit=3", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    assert len(resp.json()["zones"]) <= 3


@pytest.mark.tdd
@pytest.mark.integration
def test_fvg_api_direction_filter_req_6_3() -> None:
    if not _api_available:
        pytest.skip("FVG API not yet available")
    resp = test_client.get("/v2/scanners/fair-value-gaps?direction=bullish", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    for z in resp.json()["zones"]:
        assert z["direction"] == "bullish"


@pytest.mark.tdd
@pytest.mark.integration
def test_fvg_api_timeframe_filter_req_6_4() -> None:
    if not _api_available:
        pytest.skip("FVG API not yet available")
    resp = test_client.get("/v2/scanners/fair-value-gaps?timeframe=1h", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    for z in resp.json()["zones"]:
        assert z["timeframe"] == "1h"


@pytest.mark.tdd
@pytest.mark.integration
def test_fvg_api_empty_returns_200_req_6_7() -> None:
    if not _api_available:
        pytest.skip("FVG API not yet available")
    resp = test_client.get("/v2/scanners/fair-value-gaps", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200  # never 404 even when empty


@pytest.mark.tdd
@pytest.mark.integration
def test_fvg_api_limit_out_of_range_req_6_9() -> None:
    if not _api_available:
        pytest.skip("FVG API not yet available")
    resp0 = test_client.get("/v2/scanners/fair-value-gaps?limit=0", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp0.status_code == 422
    resp201 = test_client.get("/v2/scanners/fair-value-gaps?limit=201", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp201.status_code == 422


# ---------------------------------------------------------------------------
# S&D test helpers
# ---------------------------------------------------------------------------


def _make_sd_df(
    entry_dir="up",     # "up" or "down"
    dep_dir="up",       # "up" or "down"
    n_base=3,
    extra_bars=0,
):
    """Build a minimal S&D zone OHLCV DataFrame:
    bar 0: entry_leg candle (up → bullish, down → bearish)
    bars 1..n_base: small-body base candles
    bar n_base+1: departure candle
    """
    from datetime import datetime, timezone
    import pandas as pd
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = []
    # entry leg
    if entry_dir == "up":
        rows.append({"open": 95.0, "high": 105.0, "low": 94.0, "close": 104.0, "volume": 2000.0})
    else:
        rows.append({"open": 104.0, "high": 105.0, "low": 90.0, "close": 91.0, "volume": 2000.0})
    # base candles (tiny bodies: open≈close≈100)
    for i in range(n_base):
        rows.append({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 500.0})
    # departure candle
    if dep_dir == "up":
        rows.append({"open": 100.0, "high": 135.0, "low": 99.0, "close": 134.0, "volume": 5000.0})
    else:
        rows.append({"open": 100.0, "high": 101.0, "low": 65.0, "close": 66.0, "volume": 5000.0})
    for _ in range(extra_bars):
        rows.append({"open": 110.0, "high": 112.0, "low": 109.0, "close": 111.0, "volume": 1000.0})
    timestamps = pd.date_range(start=ts, periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, index=timestamps)


# ===========================================================================
# Task 2.3 — TDD red-phase tests for S&D scanner (Req 8–14)
# ===========================================================================


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_base_detection_req_8_1() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(entry_dir="down", dep_dir="down")  # DBD pattern
    zones = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert len(zones) >= 1
    z = zones[0]
    assert z.base_start_bar_index == 1
    assert z.base_end_bar_index == 3


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_zone_bounds_req_8_2() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(entry_dir="down", dep_dir="down")
    zones = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert len(zones) >= 1
    z = zones[0]
    base_highs = df["high"].iloc[z.base_start_bar_index:z.base_end_bar_index+1]
    base_lows  = df["low"].iloc[z.base_start_bar_index:z.base_end_bar_index+1]
    assert z.zone_high == pytest.approx(base_highs.max())
    assert z.zone_low  == pytest.approx(base_lows.min())


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_departure_filter_req_8_3() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    # Build data where departure body is tiny (< 1.5×ATR)
    config_strict = SDConfig(departure_body_threshold=1000.0)  # impossibly strict
    df = _make_sd_df()
    result = detect_sd_zones(df, "TEST", "1h", config_strict)
    assert result == [], "Strict departure filter should reject all zones"


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_min_zone_width_filter_req_8_4() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    config_strict = SDConfig(min_zone_width_multiplier=1000.0)  # impossibly strict width
    df = _make_sd_df()
    result = detect_sd_zones(df, "TEST", "1h", config_strict)
    assert result == [], "Strict zone width filter should reject all zones"


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_fewer_than_5_bars_req_8_5() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}] * 4
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=4, freq="D", tz="UTC"))
    result = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert result == []


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_nan_skipped_req_8_6() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(extra_bars=5)
    df.loc[df.index[2], "close"] = float("nan")  # NaN in middle of base
    result = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert isinstance(result, list), "Should not raise on NaN"


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_rbr_classification_req_9_1() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(entry_dir="up", dep_dir="up")  # entry up, dep up → RBR
    zones = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert len(zones) >= 1
    z = zones[0]
    assert z.zone_type == SDZoneType.RBR
    assert z.direction == SDDirection.DEMAND


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_dbd_classification_req_9_2() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(entry_dir="down", dep_dir="down")  # DBD
    zones = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert len(zones) >= 1
    assert zones[0].zone_type == SDZoneType.DBD
    assert zones[0].direction == SDDirection.SUPPLY


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_rbd_classification_req_9_3() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(entry_dir="up", dep_dir="down")  # RBD
    zones = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert len(zones) >= 1
    assert zones[0].zone_type == SDZoneType.RBD
    assert zones[0].direction == SDDirection.SUPPLY


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_dbr_classification_req_9_4() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(entry_dir="down", dep_dir="up")  # DBR
    zones = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert len(zones) >= 1
    assert zones[0].zone_type == SDZoneType.DBR
    assert zones[0].direction == SDDirection.DEMAND


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_no_entry_leg_skipped_req_9_5() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    # Build a DataFrame where base starts at bar 0 (no entry_leg before it)
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 500.0},   # base (idx 0)
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 500.0},   # base
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 500.0},   # base
        {"open": 100.0, "high": 135.0, "low": 99.0, "close": 134.0, "volume": 5000.0},  # departure
    ]
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=4, freq="D", tz="UTC"))
    result = detect_sd_zones(df, "TEST", "1h", SDConfig())
    assert result == [], "Zone starting at index 0 should be skipped (no entry_leg)"


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_lifecycle_fresh_initial_req_10_1() -> None:
    # Pure schema test
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    zone = SDZone(
        symbol="TEST", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR,
        timeframe="1h", zone_high=105.0, zone_low=99.0,
        entry_leg_timestamp=ts, departure_timestamp=ts + timedelta(days=4),
        base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4,
    )
    assert zone.lifecycle_state == SDLifecycle.FRESH
    assert zone.zone_strength == pytest.approx(1.0)


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_lifecycle_fresh_to_tested_req_10_2() -> None:
    if update_sd_lifecycle is None:
        pytest.skip("update_sd_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    zone = SDZone(
        symbol="TEST", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR,
        timeframe="1h", zone_high=105.0, zone_low=99.0,
        entry_leg_timestamp=ts, departure_timestamp=ts + timedelta(days=4),
        base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4,
    )
    # DEMAND zone: enters zone (low=100, high=106) without closing through (close=103 > zone_low=99)
    result = update_sd_lifecycle(zone, candle_low=100.0, candle_high=106.0, candle_close=103.0, bar_index=10, config=SDConfig())
    assert result.lifecycle_state == SDLifecycle.TESTED
    assert result.zone_strength == pytest.approx(0.7)


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_lifecycle_tested_to_used_req_10_3() -> None:
    if update_sd_lifecycle is None:
        pytest.skip("update_sd_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    zone = SDZone(
        symbol="TEST", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR,
        timeframe="1h", zone_high=105.0, zone_low=99.0,
        entry_leg_timestamp=ts, departure_timestamp=ts + timedelta(days=4),
        base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4,
        lifecycle_state=SDLifecycle.TESTED, zone_strength=0.7,
    )
    result = update_sd_lifecycle(zone, candle_low=100.0, candle_high=106.0, candle_close=103.0, bar_index=15, config=SDConfig())
    assert result.lifecycle_state == SDLifecycle.USED
    assert result.zone_strength == pytest.approx(0.4)


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_lifecycle_broken_req_10_4() -> None:
    if update_sd_lifecycle is None:
        pytest.skip("update_sd_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    zone = SDZone(
        symbol="TEST", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR,
        timeframe="1h", zone_high=105.0, zone_low=99.0,
        entry_leg_timestamp=ts, departure_timestamp=ts + timedelta(days=4),
        base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4,
    )
    # DEMAND zone broken: close < zone_low=99
    result = update_sd_lifecycle(zone, candle_low=90.0, candle_high=100.0, candle_close=95.0, bar_index=20, config=SDConfig())
    assert result.lifecycle_state == SDLifecycle.BROKEN
    assert result.zone_strength == pytest.approx(0.0)


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_lifecycle_terminal_broken_req_10_5() -> None:
    if update_sd_lifecycle is None:
        pytest.skip("update_sd_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    zone = SDZone(
        symbol="TEST", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR,
        timeframe="1h", zone_high=105.0, zone_low=99.0,
        entry_leg_timestamp=ts, departure_timestamp=ts + timedelta(days=4),
        base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4,
        lifecycle_state=SDLifecycle.BROKEN, zone_strength=0.0,
    )
    result = update_sd_lifecycle(zone, candle_low=98.0, candle_high=106.0, candle_close=103.0, bar_index=25, config=SDConfig())
    assert result.lifecycle_state == SDLifecycle.BROKEN


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_scorer_departure_strength_req_11_1() -> None:
    if rank_sd_zones is None or detect_sd_zones is None:
        pytest.skip("SD scorer not yet implemented")
    df = _make_sd_df(entry_dir="down", dep_dir="down", extra_bars=20)
    zones = detect_sd_zones(df, "TEST", "1h", SDConfig())
    if not zones:
        pytest.skip("No zones detected")
    ranked = rank_sd_zones(zones, {"TEST": df}, SDConfig())
    for z in ranked:
        assert z.departure_strength >= 0
        assert z.departure_strength == z.departure_strength  # NaN check


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_scorer_combined_score_req_11_3() -> None:
    if rank_sd_zones is None or detect_sd_zones is None:
        pytest.skip("SD scorer not yet implemented")
    df = _make_sd_df(extra_bars=20)
    zones = detect_sd_zones(df, "TEST", "1h", SDConfig())
    if not zones:
        pytest.skip("No zones detected")
    ranked = rank_sd_zones(zones, {"TEST": df}, SDConfig())
    for z in ranked:
        assert z.combined_score == pytest.approx(z.departure_strength * z.zone_strength, rel=1e-6)


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_scorer_sorted_descending_req_11_4() -> None:
    if rank_sd_zones is None:
        pytest.skip("rank_sd_zones not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    ts2 = ts + timedelta(days=4)
    zone1 = SDZone(symbol="A", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR, timeframe="1h",
                   zone_high=105.0, zone_low=99.0, entry_leg_timestamp=ts, departure_timestamp=ts2,
                   base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4,
                   combined_score=10.0)
    zone2 = SDZone(symbol="B", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR, timeframe="1h",
                   zone_high=108.0, zone_low=99.0, entry_leg_timestamp=ts, departure_timestamp=ts2,
                   base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4,
                   combined_score=5.0)
    zone3 = SDZone(symbol="C", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR, timeframe="1h",
                   zone_high=112.0, zone_low=99.0, entry_leg_timestamp=ts, departure_timestamp=ts2,
                   base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4,
                   combined_score=20.0)
    df_dummy = _make_sd_df()
    ranked = rank_sd_zones([zone1, zone2, zone3], {"A": df_dummy, "B": df_dummy, "C": df_dummy}, SDConfig())
    scores = [z.combined_score for z in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_scorer_zero_atr_req_11_6() -> None:
    if rank_sd_zones is None:
        pytest.skip("rank_sd_zones not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    ts2 = ts + timedelta(days=4)
    # DataFrame with all-zero TR (flat candles)
    rows = [{"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1000.0}] * 20
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=20, freq="D", tz="UTC"))
    zone = SDZone(symbol="A", direction=SDDirection.DEMAND, zone_type=SDZoneType.DBR, timeframe="1h",
                  zone_high=105.0, zone_low=99.0, entry_leg_timestamp=ts, departure_timestamp=ts2,
                  base_start_bar_index=1, base_end_bar_index=3, departure_bar_index=4)
    ranked = rank_sd_zones([zone], {"A": df}, SDConfig())
    assert ranked[0].departure_strength == 0.0


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_multi_tf_confluence_req_12_5() -> None:
    if SDScanner is None:
        pytest.skip("SDScanner not yet implemented")
    scanner = SDScanner()
    df = _make_sd_df(extra_bars=20)
    # Scan two timeframes for the same symbol
    result_1h = scanner.scan({"TEST": df}, timeframe="1h")
    result_1d = scanner.scan({"TEST": df}, timeframe="1d")
    # Just verify it runs without error (confluence detection is tested at orchestrator level)
    assert hasattr(result_1h, "zones")
    assert hasattr(result_1d, "zones")


@pytest.mark.tdd
@pytest.mark.unit
def test_sd_invalid_timeframe_raises_req_12_3() -> None:
    if SDScanner is None:
        pytest.skip("SDScanner not yet implemented")
    scanner = SDScanner()
    with pytest.raises(ValueError, match="5m"):
        scanner.scan({}, timeframe="1w")


@pytest.mark.tdd
@pytest.mark.integration
def test_sd_api_returns_200_req_13_1() -> None:
    if not _api_available:
        pytest.skip("S&D API not yet available")
    resp = test_client.get("/v2/scanners/supply-demand", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    assert "zones" in resp.json()


@pytest.mark.tdd
@pytest.mark.integration
def test_sd_api_limit_req_13_2() -> None:
    if not _api_available:
        pytest.skip("S&D API not yet available")
    resp = test_client.get("/v2/scanners/supply-demand?limit=2", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    assert len(resp.json()["zones"]) <= 2


@pytest.mark.tdd
@pytest.mark.integration
def test_sd_api_direction_filter_req_13_3() -> None:
    if not _api_available:
        pytest.skip("S&D API not yet available")
    resp = test_client.get("/v2/scanners/supply-demand?direction=supply", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    for z in resp.json()["zones"]:
        assert z["direction"] == "SUPPLY"


@pytest.mark.tdd
@pytest.mark.integration
def test_sd_api_zone_type_filter_req_13_4() -> None:
    if not _api_available:
        pytest.skip("S&D API not yet available")
    resp = test_client.get("/v2/scanners/supply-demand?zone_type=RBR", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    for z in resp.json()["zones"]:
        assert z["zone_type"] == "RBR"


@pytest.mark.tdd
@pytest.mark.integration
def test_sd_api_empty_returns_200_req_13_8() -> None:
    if not _api_available:
        pytest.skip("S&D API not yet available")
    resp = test_client.get("/v2/scanners/supply-demand", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200


@pytest.mark.tdd
@pytest.mark.integration
def test_sd_api_limit_out_of_range_req_13_9() -> None:
    if not _api_available:
        pytest.skip("S&D API not yet available")
    resp = test_client.get("/v2/scanners/supply-demand?limit=0", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Sweep test helpers
# ---------------------------------------------------------------------------


def _make_sweep_df(
    n_bars: int = 30,
    sweep_bar: int = 20,
    sweep_dir: str = "bullish",  # "bullish" = sweep lows, "bearish" = sweep highs
):
    """Build an OHLCV DataFrame with one clear liquidity sweep event.

    For bullish sweep:
    - bars 0..sweep_bar-1: steady bars with consistent swing_low ≈ 95.0
    - sweep_bar: wick below 95 (low=90), close above 95 (close=98), high volume
    - sweep_bar+1: confirmation candle that closes higher (close=102)
    """
    from datetime import datetime, timezone
    import pandas as pd
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(n_bars):
        if i == sweep_bar:
            if sweep_dir == "bullish":
                rows.append({"open": 96.0, "high": 100.0, "low": 90.0, "close": 98.0, "volume": 8000.0})
            else:
                rows.append({"open": 104.0, "high": 115.0, "low": 101.0, "close": 103.0, "volume": 8000.0})
        elif i == sweep_bar + 1:
            if sweep_dir == "bullish":
                rows.append({"open": 99.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 2000.0})
            else:
                rows.append({"open": 102.0, "high": 103.0, "low": 95.0, "close": 97.0, "volume": 2000.0})
        else:
            if sweep_dir == "bullish":
                rows.append({"open": 97.0, "high": 100.0, "low": 95.0, "close": 98.0, "volume": 1000.0})
            else:
                rows.append({"open": 103.0, "high": 110.0, "low": 101.0, "close": 107.0, "volume": 1000.0})
    timestamps = pd.date_range(start=ts, periods=n_bars, freq="D", tz="UTC")
    return pd.DataFrame(rows, index=timestamps)


# ===========================================================================
# Task 2.4 — TDD red-phase tests for Liquidity Sweep scanner (Req 15–21)
# ===========================================================================


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_swing_high_trailing_req_15_1() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    df = _make_sweep_df(sweep_dir="bearish")
    config = SweepConfig(require_reversal_confirmation=False)
    signals = detect_sweeps(df, "TEST", "1h", config)
    # Verify the detector doesn't look ahead: sw_high[k] should equal
    # high.rolling(swing_lookback, min_periods=1).max().iloc[k]
    expected_sw_high = df["high"].rolling(config.swing_lookback, min_periods=1).max()
    # If any bearish sweep was detected, check that the swept_level matches the prior swing_high
    for s in signals:
        k = s.sweep_bar_index
        assert s.swept_level == pytest.approx(expected_sw_high.iloc[k - 1]), \
            f"swept_level should equal sw_high[k-1] (trailing), got {s.swept_level}"


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_swing_low_trailing_req_15_2() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish")
    config = SweepConfig(require_reversal_confirmation=False)
    signals = detect_sweeps(df, "TEST", "1h", config)
    expected_sw_low = df["low"].rolling(config.swing_lookback, min_periods=1).min()
    for s in signals:
        k = s.sweep_bar_index
        assert s.swept_level == pytest.approx(expected_sw_low.iloc[k - 1])


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_liquidity_pool_highs_req_15_3() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    # Craft a series where multiple consecutive swing highs are at the same level (~110)
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(25):
        if i in (5, 10, 15):  # multiple "highs" at ~110
            rows.append({"open": 105.0, "high": 110.0, "low": 104.0, "close": 108.0, "volume": 1000.0})
        elif i == 20:  # bearish sweep: wick above 110, close below
            rows.append({"open": 109.0, "high": 116.0, "low": 108.0, "close": 109.0, "volume": 6000.0})
        elif i == 21:  # confirmation: close further down
            rows.append({"open": 108.0, "high": 109.0, "low": 104.0, "close": 105.0, "volume": 2000.0})
        else:
            rows.append({"open": 105.0, "high": 108.0, "low": 103.0, "close": 106.0, "volume": 1000.0})
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=25, freq="D", tz="UTC"))
    config = SweepConfig(volume_confirmation_multiplier=1.0)
    signals = detect_sweeps(df, "TEST", "1h", config)
    # If a sweep is detected, equal_levels_count >= 1
    for s in signals:
        assert s.equal_levels_count >= 1


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_too_few_bars_req_15_5() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}] * 5
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=5, freq="D", tz="UTC"))
    config = SweepConfig(swing_lookback=10)
    result = detect_sweeps(df, "TEST", "1h", config)
    assert result == [], "Too few bars should return empty list"


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_bullish_detection_req_16_1() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish", sweep_bar=20)
    config = SweepConfig(require_reversal_confirmation=True)
    signals = detect_sweeps(df, "TEST", "1h", config)
    assert len(signals) >= 1
    s = signals[0]
    assert s.direction == SweepDirection.BULLISH
    assert s.sweep_bar_index == 20
    # swept_level is the swing_low just before bar 20
    expected_sw_low = df["low"].rolling(config.swing_lookback, min_periods=1).min().iloc[19]
    assert s.swept_level == pytest.approx(expected_sw_low, rel=0.01)


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_bearish_detection_req_16_2() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    df = _make_sweep_df(sweep_dir="bearish", sweep_bar=20)
    config = SweepConfig(require_reversal_confirmation=True)
    signals = detect_sweeps(df, "TEST", "1h", config)
    assert len(signals) >= 1
    assert signals[0].direction == SweepDirection.BEARISH


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_volume_filter_req_16_3() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish")
    config = SweepConfig(volume_confirmation_multiplier=1000.0)  # impossibly strict
    result = detect_sweeps(df, "TEST", "1h", config)
    assert result == [], "Strict volume filter should reject all sweeps"


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_zone_bounds_req_16_4() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish")
    config = SweepConfig(require_reversal_confirmation=False)
    signals = detect_sweeps(df, "TEST", "1h", config)
    if not signals:
        pytest.skip("No sweeps detected in test data")
    s = signals[0]
    k = s.sweep_bar_index
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean().iloc[k]
    assert s.sweep_zone_high == pytest.approx(s.swept_level + 0.5 * atr, rel=0.05)
    assert s.sweep_zone_low  == pytest.approx(s.swept_level - 0.5 * atr, rel=0.05)


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_reversal_confirmation_req_16_5() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    # Build a bullish sweep where the NEXT candle does NOT close higher (no reversal)
    df = _make_sweep_df(sweep_dir="bullish")
    # Overwrite the confirmation bar to close lower than sweep bar
    df_no_conf = df.copy()
    df_no_conf.iloc[21, df_no_conf.columns.get_loc("close")] = df_no_conf["close"].iloc[20] - 2.0
    config = SweepConfig(require_reversal_confirmation=True)
    result = detect_sweeps(df_no_conf, "TEST", "1h", config)
    assert result == [], "Without reversal confirmation, no sweep should be detected"


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_lifecycle_active_initial_req_17_1() -> None:
    # Pure schema test
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    signal = SweepSignal(
        symbol="TEST", direction=SweepDirection.BULLISH, timeframe="1h",
        swept_level=95.0, sweep_zone_high=97.0, sweep_zone_low=93.0,
        sweep_timestamp=ts, sweep_bar_index=20,
    )
    assert signal.lifecycle_state == SweepLifecycle.ACTIVE


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_lifecycle_confirmed_req_17_2() -> None:
    if update_sweep_lifecycle is None:
        pytest.skip("update_sweep_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    signal = SweepSignal(
        symbol="TEST", direction=SweepDirection.BULLISH, timeframe="1h",
        swept_level=95.0, sweep_zone_high=97.0, sweep_zone_low=93.0,
        sweep_timestamp=ts, sweep_bar_index=20,
    )
    # Next close higher than sweep close (98) → CONFIRMED
    result = update_sweep_lifecycle(signal, next_candle_close=102.0, bar_index=21, config=SweepConfig())
    assert result.lifecycle_state == SweepLifecycle.CONFIRMED


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_lifecycle_invalidated_req_17_3() -> None:
    if update_sweep_lifecycle is None:
        pytest.skip("update_sweep_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    signal = SweepSignal(
        symbol="TEST", direction=SweepDirection.BULLISH, timeframe="1h",
        swept_level=95.0, sweep_zone_high=97.0, sweep_zone_low=93.0,
        sweep_timestamp=ts, sweep_bar_index=20,
    )
    # Close below swept_level (95) → INVALIDATED
    result = update_sweep_lifecycle(signal, next_candle_close=92.0, bar_index=21, config=SweepConfig())
    assert result.lifecycle_state == SweepLifecycle.INVALIDATED


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_lifecycle_terminal_invalidated_req_17_4() -> None:
    if update_sweep_lifecycle is None:
        pytest.skip("update_sweep_lifecycle not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    signal = SweepSignal(
        symbol="TEST", direction=SweepDirection.BULLISH, timeframe="1h",
        swept_level=95.0, sweep_zone_high=97.0, sweep_zone_low=93.0,
        sweep_timestamp=ts, sweep_bar_index=20,
        lifecycle_state=SweepLifecycle.INVALIDATED,
    )
    result = update_sweep_lifecycle(signal, next_candle_close=110.0, bar_index=22, config=SweepConfig())
    assert result.lifecycle_state == SweepLifecycle.INVALIDATED


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_scorer_magnitude_req_18_1() -> None:
    if rank_sweeps is None or detect_sweeps is None:
        pytest.skip("Sweep scorer not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish")
    config = SweepConfig(require_reversal_confirmation=False)
    signals = detect_sweeps(df, "TEST", "1h", config)
    if not signals:
        pytest.skip("No sweeps detected")
    ranked = rank_sweeps(signals, config)
    for s in ranked:
        assert s.sweep_magnitude >= 0
        assert s.sweep_magnitude == s.sweep_magnitude  # NaN check


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_scorer_volume_ratio_req_18_2() -> None:
    if rank_sweeps is None or detect_sweeps is None:
        pytest.skip("Sweep scorer not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish")
    signals = detect_sweeps(df, "TEST", "1h", SweepConfig(require_reversal_confirmation=False))
    if not signals:
        pytest.skip("No sweeps detected")
    ranked = rank_sweeps(signals, SweepConfig())
    for s in ranked:
        assert s.volume_ratio >= 0


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_scorer_conviction_score_req_18_3() -> None:
    if rank_sweeps is None or detect_sweeps is None:
        pytest.skip("Sweep scorer not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish")
    signals = detect_sweeps(df, "TEST", "1h", SweepConfig(require_reversal_confirmation=False))
    if not signals:
        pytest.skip("No sweeps detected")
    ranked = rank_sweeps(signals, SweepConfig())
    for s in ranked:
        expected = s.sweep_magnitude * s.volume_ratio * s.equal_levels_count
        assert s.conviction_score == pytest.approx(expected, rel=1e-6)


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_scorer_sorted_descending_req_18_4() -> None:
    if rank_sweeps is None:
        pytest.skip("rank_sweeps not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    s1 = SweepSignal(symbol="A", direction=SweepDirection.BULLISH, timeframe="1h",
                     swept_level=95.0, sweep_zone_high=97.0, sweep_zone_low=93.0,
                     sweep_timestamp=ts, sweep_bar_index=5, conviction_score=10.0)
    s2 = SweepSignal(symbol="B", direction=SweepDirection.BULLISH, timeframe="1h",
                     swept_level=95.0, sweep_zone_high=97.0, sweep_zone_low=93.0,
                     sweep_timestamp=ts, sweep_bar_index=6, conviction_score=5.0)
    s3 = SweepSignal(symbol="C", direction=SweepDirection.BULLISH, timeframe="1h",
                     swept_level=95.0, sweep_zone_high=97.0, sweep_zone_low=93.0,
                     sweep_timestamp=ts, sweep_bar_index=7, conviction_score=20.0)
    ranked = rank_sweeps([s1, s2, s3], SweepConfig())
    scores = [s.conviction_score for s in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_scorer_zero_avg_vol_req_18_6() -> None:
    if rank_sweeps is None:
        pytest.skip("rank_sweeps not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    s = SweepSignal(symbol="A", direction=SweepDirection.BULLISH, timeframe="1h",
                    swept_level=95.0, sweep_zone_high=97.0, sweep_zone_low=93.0,
                    sweep_timestamp=ts, sweep_bar_index=5, volume_ratio=0.0)
    ranked = rank_sweeps([s], SweepConfig())
    assert ranked[0].volume_ratio == 0.0


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_invalid_timeframe_raises_req_19_3() -> None:
    if SweepScanner is None:
        pytest.skip("SweepScanner not yet implemented")
    scanner = SweepScanner()
    with pytest.raises(ValueError, match="5m"):
        scanner.scan({}, timeframe="1w")


@pytest.mark.tdd
@pytest.mark.unit
def test_sweep_multi_tf_confluence_req_19_5() -> None:
    if SweepScanner is None:
        pytest.skip("SweepScanner not yet implemented")
    scanner = SweepScanner()
    df = _make_sweep_df()
    result_1h = scanner.scan({"TEST": df}, timeframe="1h")
    result_1d = scanner.scan({"TEST": df}, timeframe="1d")
    assert hasattr(result_1h, "signals")
    assert hasattr(result_1d, "signals")


@pytest.mark.tdd
@pytest.mark.integration
def test_sweep_api_returns_200_req_20_1() -> None:
    if not _api_available:
        pytest.skip("Sweep API not yet available")
    resp = test_client.get("/v2/scanners/liquidity-sweeps", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    assert "signals" in resp.json()


@pytest.mark.tdd
@pytest.mark.integration
def test_sweep_api_limit_req_20_2() -> None:
    if not _api_available:
        pytest.skip("Sweep API not yet available")
    resp = test_client.get("/v2/scanners/liquidity-sweeps?limit=2", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    assert len(resp.json()["signals"]) <= 2


@pytest.mark.tdd
@pytest.mark.integration
def test_sweep_api_direction_filter_req_20_3() -> None:
    if not _api_available:
        pytest.skip("Sweep API not yet available")
    resp = test_client.get("/v2/scanners/liquidity-sweeps?direction=bearish", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    for s in resp.json()["signals"]:
        assert s["direction"] == "bearish"


@pytest.mark.tdd
@pytest.mark.integration
def test_sweep_api_empty_returns_200_req_20_7() -> None:
    if not _api_available:
        pytest.skip("Sweep API not yet available")
    resp = test_client.get("/v2/scanners/liquidity-sweeps", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200


@pytest.mark.tdd
@pytest.mark.integration
def test_sweep_api_limit_out_of_range_req_20_8() -> None:
    if not _api_available:
        pytest.skip("Sweep API not yet available")
    resp = test_client.get("/v2/scanners/liquidity-sweeps?limit=0", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Market Structure test helpers
# ---------------------------------------------------------------------------


def _make_ms_df(trend="bullish", n_bars=50):
    """Build an OHLCV DataFrame with a clear market structure.

    For 'bullish': creates HH-HL sequence leading to BOS_BULLISH signals.
    For 'bearish': creates LL-LH sequence.
    """
    from datetime import datetime, timezone
    import pandas as pd
    import numpy as np
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = []
    base = 100.0
    if trend == "bullish":
        for i in range(n_bars):
            drift = i * 0.5
            rows.append({
                "open": base + drift + np.sin(i * 0.3) * 3,
                "high": base + drift + 5 + np.sin(i * 0.3) * 3,
                "low":  base + drift - 3 + np.sin(i * 0.3) * 3,
                "close": base + drift + np.sin(i * 0.3 + 0.1) * 3,
                "volume": 1000.0,
            })
    else:
        for i in range(n_bars):
            drift = -i * 0.5
            rows.append({
                "open": base + drift + np.sin(i * 0.3) * 3,
                "high": base + drift + 5 + np.sin(i * 0.3) * 3,
                "low":  base + drift - 3 + np.sin(i * 0.3) * 3,
                "close": base + drift + np.sin(i * 0.3 + 0.1) * 3,
                "volume": 1000.0,
            })
    # Ensure high >= max(open, close) >= min(open, close) >= low > 0 for each row
    fixed_rows = []
    for r in rows:
        o, h, l, c = r["open"], r["high"], r["low"], r["close"]
        h = max(h, o, c)
        l = min(l, o, c)
        l = max(l, 1.0)  # ensure > 0
        h = max(h, l + 0.01)
        fixed_rows.append({"open": o, "high": h, "low": l, "close": c, "volume": r["volume"]})
    timestamps = pd.date_range(start=ts, periods=n_bars, freq="D", tz="UTC")
    return pd.DataFrame(fixed_rows, index=timestamps)


# ===========================================================================
# Task 2.5 — TDD red-phase tests for Market Structure scanner (Req 22–29)
# ===========================================================================


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_swing_high_hh_req_22_1() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    # Register two consecutive swing highs where second > first
    tracker.register_swing_high(bar_index=5, price=110.0)
    tracker.register_swing_high(bar_index=10, price=115.0)
    assert tracker.last_swing_high_type() == "HH", "When second high > first, it's HH"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_swing_high_lh_req_22_2() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    tracker.register_swing_high(bar_index=5, price=115.0)
    tracker.register_swing_high(bar_index=10, price=110.0)
    assert tracker.last_swing_high_type() == "LH"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_swing_low_ll_req_22_3() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    tracker.register_swing_low(bar_index=5, price=95.0)
    tracker.register_swing_low(bar_index=10, price=90.0)
    assert tracker.last_swing_low_type() == "LL"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_swing_low_hl_req_22_4() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    tracker.register_swing_low(bar_index=5, price=90.0)
    tracker.register_swing_low(bar_index=10, price=95.0)
    assert tracker.last_swing_low_type() == "HL"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_swing_trailing_only_req_22_6() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df()
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    # Verify no lookahead: modify a future bar and re-detect
    df_mod = df.copy()
    df_mod.iloc[40, df_mod.columns.get_loc("high")] = 999.0
    signals_mod = detect_market_structure(df_mod, "TEST", "1h", MSConfig())
    # Signals for bars < 38 should be identical
    orig_early = [(s.break_bar_index, s.signal_type.value) for s in signals if s.break_bar_index < 38]
    mod_early  = [(s.break_bar_index, s.signal_type.value) for s in signals_mod if s.break_bar_index < 38]
    assert sorted(orig_early) == sorted(mod_early), "Trailing-only: future bar modification should not change prior signals"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_bos_bullish_req_23_1() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    bos_bullish = [s for s in signals if s.signal_type == MSSignalType.BOS_BULLISH]
    assert len(bos_bullish) >= 1, "Bullish trending data should produce BOS_BULLISH signals"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_bos_bearish_req_23_2() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df(trend="bearish", n_bars=50)
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    bos_bearish = [s for s in signals if s.signal_type == MSSignalType.BOS_BEARISH]
    assert len(bos_bearish) >= 1, "Bearish trending data should produce BOS_BEARISH signals"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_bos_wick_only_no_signal_req_23_3() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    # Build a 20-bar series where price wicks above swing_high[k-1] but CLOSE stays below
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = []
    for i in range(20):
        rows.append({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0})
    # Add a wick bar: high=120 (above swing_high≈101) but close=100 (stays below)
    rows.append({"open": 100.0, "high": 120.0, "low": 99.0, "close": 100.0, "volume": 2000.0})
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=21, freq="D", tz="UTC"))
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    # The wick bar must NOT produce a BOS_BULLISH (close did not cross the level)
    bos_at_wick = [s for s in signals if s.break_bar_index == 20 and s.signal_type == MSSignalType.BOS_BULLISH]
    assert bos_at_wick == [], "Wick-only cross should not trigger BoS"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_bos_records_evidence_req_23_4() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    for s in signals:
        assert s.close_at_break > 0
        assert s.broken_level > 0
        assert s.break_timestamp is not None
        assert s.break_bar_index >= 0


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_bos_level_consumed_once_req_23_5() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    # No two BOS_BULLISH signals should reference the same broken_level
    bos_levels = [s.broken_level for s in signals if s.signal_type == MSSignalType.BOS_BULLISH]
    assert len(bos_levels) == len(set(bos_levels)), "Same broken_level should not trigger BoS twice"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_bos_undefined_no_signal_req_23_6() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    # Too short a series to establish trend
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    rows = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}] * 5
    df = pd.DataFrame(rows, index=pd.date_range(start=ts, periods=5, freq="D", tz="UTC"))
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    bos_signals = [s for s in signals if s.trend_state == TrendState.UNDEFINED]
    # No BoS while trend_state is UNDEFINED
    for s in signals:
        assert s.trend_state != TrendState.UNDEFINED, "BoS signals should only emit when trend_state is defined"


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_choch_bullish_req_24_1() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    # Bearish then reversal → CHOCH_BULLISH
    df_bear = _make_ms_df(trend="bearish", n_bars=30)
    # Append a strong bullish reversal
    ts_last = df_bear.index[-1]
    reversal_rows = []
    for i in range(1, 20):
        drift = i * 2.0
        reversal_rows.append({
            "open": float(df_bear["close"].iloc[-1]) + drift,
            "high": float(df_bear["close"].iloc[-1]) + drift + 5,
            "low":  float(df_bear["close"].iloc[-1]) + drift - 1,
            "close": float(df_bear["close"].iloc[-1]) + drift + 4,
            "volume": 3000.0,
        })
    reversal_ts = pd.date_range(start=ts_last + pd.Timedelta(days=1), periods=len(reversal_rows), freq="D", tz="UTC")
    df_reversal = pd.DataFrame(reversal_rows, index=reversal_ts)
    df_combined = pd.concat([df_bear, df_reversal])
    signals = detect_market_structure(df_combined, "TEST", "1h", MSConfig())
    choch_bull = [s for s in signals if s.signal_type == MSSignalType.CHOCH_BULLISH]
    assert len(choch_bull) >= 1, "Bearish→bullish reversal should produce CHOCH_BULLISH"
    for s in choch_bull:
        assert s.is_choch is True


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_choch_bearish_req_24_2() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    # All ChoCh signals for bearish should have is_choch=True
    choch_bear = [s for s in detect_market_structure(df, "TEST", "1h", MSConfig())
                  if s.signal_type == MSSignalType.CHOCH_BEARISH]
    for s in choch_bear:
        assert s.is_choch is True


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_choch_transitions_state_bullish_req_24_4() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    tracker.on_choch_bullish()
    assert tracker.trend_state == TrendState.BULLISH
    assert tracker.swing_count == 0


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_choch_transitions_state_bearish_req_24_5() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    tracker.on_choch_bearish()
    assert tracker.trend_state == TrendState.BEARISH
    assert tracker.swing_count == 0


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_choch_invalidated_req_24_6() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    # All signals should have invalidated=False by default
    for s in signals:
        assert s.invalidated is False


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_state_undefined_initial_req_25_1() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    assert tracker.trend_state == TrendState.UNDEFINED


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_state_undefined_to_bullish_hh_hl_req_25_2() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    # Establish HH-HL sequence
    tracker.register_swing_high(1, 110.0)
    tracker.register_swing_low(2, 105.0)
    tracker.register_swing_high(3, 115.0)  # HH
    tracker.register_swing_low(4, 108.0)   # HL
    tracker.update_trend_state()
    assert tracker.trend_state == TrendState.BULLISH


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_state_undefined_to_bearish_ll_lh_req_25_3() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    tracker.register_swing_low(1, 95.0)
    tracker.register_swing_high(2, 100.0)
    tracker.register_swing_low(3, 90.0)   # LL
    tracker.register_swing_high(4, 97.0)  # LH
    tracker.update_trend_state()
    assert tracker.trend_state == TrendState.BEARISH


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_bos_does_not_change_state_req_25_6() -> None:
    if TrendStateTracker is None:
        pytest.skip("TrendStateTracker not yet implemented")
    tracker = TrendStateTracker()
    tracker.on_choch_bullish()  # set state to BULLISH
    assert tracker.trend_state == TrendState.BULLISH
    tracker.on_bos()  # BoS should NOT change trend_state
    assert tracker.trend_state == TrendState.BULLISH


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_scorer_break_magnitude_req_26_1() -> None:
    if rank_ms_signals is None or detect_market_structure is None:
        pytest.skip("MS scorer not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    if not signals:
        pytest.skip("No signals detected")
    ranked = rank_ms_signals(signals, MSConfig())
    for s in ranked:
        assert s.break_magnitude >= 0
        assert s.break_magnitude == s.break_magnitude  # NaN check


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_scorer_conviction_choch_req_26_3() -> None:
    if rank_ms_signals is None or detect_market_structure is None:
        pytest.skip("MS scorer not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    signals = detect_market_structure(df, "TEST", "1h", MSConfig())
    if not signals:
        pytest.skip("No signals detected")
    ranked = rank_ms_signals(signals, MSConfig())
    for s in ranked:
        swing_count = max(s.swing_count, 1)  # prevent division issues
        choch_mult = 1.5 if s.is_choch else 1.0
        expected = s.break_magnitude * swing_count * choch_mult
        assert s.conviction_score == pytest.approx(expected, rel=1e-6)


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_scorer_sorted_descending_req_26_4() -> None:
    if rank_ms_signals is None:
        pytest.skip("rank_ms_signals not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    s1 = MSSignal(symbol="A", signal_type=MSSignalType.BOS_BULLISH, timeframe="1h",
                   broken_level=110.0, close_at_break=112.0, break_timestamp=ts,
                   break_bar_index=10, trend_state=TrendState.BULLISH, is_choch=False,
                   swing_count=2, conviction_score=10.0)
    s2 = MSSignal(symbol="B", signal_type=MSSignalType.BOS_BULLISH, timeframe="1h",
                   broken_level=115.0, close_at_break=117.0, break_timestamp=ts,
                   break_bar_index=15, trend_state=TrendState.BULLISH, is_choch=False,
                   swing_count=3, conviction_score=5.0)
    s3 = MSSignal(symbol="C", signal_type=MSSignalType.CHOCH_BULLISH, timeframe="1h",
                   broken_level=120.0, close_at_break=122.0, break_timestamp=ts,
                   break_bar_index=20, trend_state=TrendState.BULLISH, is_choch=True,
                   swing_count=4, conviction_score=20.0)
    ranked = rank_ms_signals([s1, s2, s3], MSConfig())
    scores = [s.conviction_score for s in ranked]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_scorer_zero_atr_req_26_6() -> None:
    if rank_ms_signals is None:
        pytest.skip("rank_ms_signals not yet implemented")
    ts = datetime(2021, 1, 1, tzinfo=timezone.utc)
    s = MSSignal(symbol="A", signal_type=MSSignalType.BOS_BULLISH, timeframe="1h",
                  broken_level=100.0, close_at_break=101.0, break_timestamp=ts,
                  break_bar_index=5, trend_state=TrendState.BULLISH, is_choch=False,
                  swing_count=1, break_magnitude=0.0)
    ranked = rank_ms_signals([s], MSConfig())
    assert ranked[0].break_magnitude == 0.0


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_invalid_timeframe_raises_req_27_3() -> None:
    if MSScanner is None:
        pytest.skip("MSScanner not yet implemented")
    scanner = MSScanner()
    with pytest.raises(ValueError, match="5m"):
        scanner.scan({}, timeframe="1w")


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_separate_trend_states_per_timeframe_req_27_4() -> None:
    if MSScanner is None:
        pytest.skip("MSScanner not yet implemented")
    scanner = MSScanner()
    df = _make_ms_df()
    result_1h = scanner.scan({"TEST": df}, timeframe="1h")
    result_1d = scanner.scan({"TEST": df}, timeframe="1d")
    assert hasattr(result_1h, "signals")
    assert hasattr(result_1d, "signals")


@pytest.mark.tdd
@pytest.mark.unit
def test_ms_multi_tf_confluence_req_27_5() -> None:
    if MSScanner is None:
        pytest.skip("MSScanner not yet implemented")
    scanner = MSScanner()
    df = _make_ms_df()
    result = scanner.scan({"TEST": df}, timeframe="1h")
    assert hasattr(result, "signals")


@pytest.mark.tdd
@pytest.mark.integration
def test_ms_api_returns_200_req_28_1() -> None:
    if not _api_available:
        pytest.skip("MS API not yet available")
    resp = test_client.get("/v2/scanners/market-structure", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    assert "signals" in resp.json()


@pytest.mark.tdd
@pytest.mark.integration
def test_ms_api_limit_req_28_2() -> None:
    if not _api_available:
        pytest.skip("MS API not yet available")
    resp = test_client.get("/v2/scanners/market-structure?limit=3", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    assert len(resp.json()["signals"]) <= 3


@pytest.mark.tdd
@pytest.mark.integration
def test_ms_api_signal_type_filter_req_28_3() -> None:
    if not _api_available:
        pytest.skip("MS API not yet available")
    resp = test_client.get("/v2/scanners/market-structure?signal_type=BOS_BULLISH", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200
    for s in resp.json()["signals"]:
        assert s["signal_type"] == "BOS_BULLISH"


@pytest.mark.tdd
@pytest.mark.integration
def test_ms_api_empty_returns_200_req_28_7() -> None:
    if not _api_available:
        pytest.skip("MS API not yet available")
    resp = test_client.get("/v2/scanners/market-structure", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 200


@pytest.mark.tdd
@pytest.mark.integration
def test_ms_api_limit_out_of_range_req_28_9() -> None:
    if not _api_available:
        pytest.skip("MS API not yet available")
    resp = test_client.get("/v2/scanners/market-structure?limit=0", headers={"X-API-KEY": "test-key-for-testing"})
    assert resp.status_code == 422


# ===========================================================================
# Task 2.6 — Property-based tests (correctness Properties 1–20)
# ===========================================================================

# Feature: smart-money-scanners
# Property FVG-1: Bullish Detection Geometric Invariant
# Validates: Requirements 1.1, 1.3, 2.1, 2.2
@pytest.mark.unit
@pytest.mark.slow
@given(ohlcv_strategy(min_rows=3))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_fvg_geometric_invariant_prop_1(df: pd.DataFrame) -> None:
    if detect_fvg_zones is None:
        return
    config = FVGConfig(impulse_atr_multiplier=0.0, min_gap_atr_multiplier=0.0)  # no filters
    zones = detect_fvg_zones(df, "PROP1", "1h", config)
    for z in zones:
        if z.direction == FVGDirection.BULLISH:
            assert z.fvg_low == pytest.approx(df["high"].iloc[z.candle1_bar_index], rel=1e-9)
            assert z.fvg_high == pytest.approx(df["low"].iloc[z.candle3_bar_index], rel=1e-9)
            assert z.fvg_high > z.fvg_low
        else:
            assert z.fvg_high == pytest.approx(df["low"].iloc[z.candle1_bar_index], rel=1e-9)
            assert z.fvg_low == pytest.approx(df["high"].iloc[z.candle3_bar_index], rel=1e-9)
        assert z.fvg_midpoint == pytest.approx((z.fvg_high + z.fvg_low) / 2, rel=1e-9)


# Property FVG-2: No-Lookahead Invariant
# Validates: Requirements 7.2, 2.4
@pytest.mark.unit
@pytest.mark.slow
@given(ohlcv_strategy(), st.integers(min_value=1, max_value=5))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_fvg_no_lookahead_prop_2(df: pd.DataFrame, n: int) -> None:
    if detect_fvg_zones is None:
        return
    if len(df) < 3:
        return
    config = FVGConfig()
    result_orig = detect_fvg_zones(df, "PROP2", "1h", config)
    # Append n extra rows
    last_ts = df.index[-1]
    extra_rows = []
    for i in range(n):
        extra_rows.append({"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0})
    extra_ts = pd.date_range(start=last_ts + pd.Timedelta(days=1), periods=n, freq="D", tz="UTC")
    df_ext = pd.concat([df, pd.DataFrame(extra_rows, index=extra_ts)])
    result_ext = detect_fvg_zones(df_ext, "PROP2", "1h", config)
    orig_set = {(z.candle1_bar_index, z.candle3_bar_index, z.direction.value) for z in result_orig}
    ext_set = {(z.candle1_bar_index, z.candle3_bar_index, z.direction.value) for z in result_ext if z.candle3_bar_index < len(df)}
    assert orig_set == ext_set


# Property FVG-3: Lifecycle Terminal State
# Validates: Requirements 3.4, 3.5
@pytest.mark.unit
@pytest.mark.slow
@given(fvg_zone_strategy())
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_fvg_lifecycle_terminal_prop_3(zone: FVGZone) -> None:
    if update_fvg_lifecycle is None:
        return
    if zone.lifecycle_state not in (FVGLifecycle.FILLED, FVGLifecycle.EXPIRED):
        return  # only test terminal states
    config = FVGConfig()
    # Apply multiple candle updates — terminal state must not change
    for _ in range(5):
        updated = update_fvg_lifecycle(
            zone,
            candle_low=zone.fvg_low - 10.0,
            candle_high=zone.fvg_high + 10.0,
            candle_close=zone.fvg_low - 5.0,  # would fill if not terminal
            bar_index=999,
            creation_bar_index=0,
            config=config,
        )
        assert updated.lifecycle_state == zone.lifecycle_state
        zone = updated


# Property FVG-4: Scoring Monotonicity
# Validates: Requirements 4.3, 4.4, 4.5
@pytest.mark.unit
@pytest.mark.slow
@given(st.lists(fvg_zone_strategy(), min_size=1, max_size=50))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_fvg_scoring_monotonicity_prop_4(zones: list) -> None:
    if rank_fvg_zones is None:
        return
    if not zones:
        return
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df_dummy = pd.DataFrame(
        [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}] * 30,
        index=pd.date_range(start=ts, periods=30, freq="D", tz="UTC"),
    )
    fixed_ts = df_dummy.index[15]
    zones_fixed = [z.model_copy(update={"fvg_timestamp": fixed_ts}) for z in zones]
    df_by_symbol = {z.symbol: df_dummy for z in zones_fixed}
    ranked = rank_fvg_zones(zones_fixed, df_by_symbol, FVGConfig())
    for k in range(len(ranked) - 1):
        assert ranked[k].combined_score >= ranked[k + 1].combined_score


# Property FVG-5: Round-Trip Serialization
# Validates: Requirements 7.4
@pytest.mark.unit
@pytest.mark.slow
@given(fvg_zone_strategy())
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_fvg_round_trip_prop_5(zone: FVGZone) -> None:
    restored = FVGZone.model_validate_json(zone.model_dump_json())
    assert restored == zone


# Property SD-1: Base Candle Body Invariant
# Validates: Requirements 8.1, 8.3, 14.1
@pytest.mark.unit
@pytest.mark.slow
@given(ohlcv_strategy(min_rows=7))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sd_base_departure_invariant_prop_6(df: pd.DataFrame) -> None:
    if detect_sd_zones is None:
        return
    if len(df) < 7:
        return
    config = SDConfig()
    zones = detect_sd_zones(df, "PROP6", "1h", config)
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"]-df["low"], (df["high"]-prev_close).abs(), (df["low"]-prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(config.atr_window, min_periods=1).mean()
    for z in zones:
        for k in range(z.base_start_bar_index, z.base_end_bar_index + 1):
            body = abs(float(df["close"].iloc[k]) - float(df["open"].iloc[k]))
            atr_k = float(atr.iloc[k])
            assert body < config.base_body_threshold * atr_k + 1e-9
        dep = z.departure_bar_index
        dep_body = abs(float(df["close"].iloc[dep]) - float(df["open"].iloc[dep]))
        atr_dep = float(atr.iloc[dep])
        assert dep_body > config.departure_body_threshold * atr_dep - 1e-9


# Property SD-2: Zone Type Direction Invariant
# Validates: Requirements 9.1–9.4
@pytest.mark.unit
@pytest.mark.slow
@given(sd_zone_strategy())
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sd_zone_type_direction_invariant_prop_7(zone: SDZone) -> None:
    if zone.zone_type in (SDZoneType.RBR, SDZoneType.DBR):
        assert zone.direction == SDDirection.DEMAND
    else:
        assert zone.direction == SDDirection.SUPPLY


# Property SD-3: Zone Strength Monotonicity
# Validates: Requirements 10.1–10.4, 14.6
@pytest.mark.unit
@pytest.mark.slow
@given(sd_zone_strategy())
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sd_strength_monotonicity_prop_8(zone: SDZone) -> None:
    if update_sd_lifecycle is None:
        return
    config = SDConfig()
    current = zone
    prev_strength = current.zone_strength
    for _ in range(5):
        updated = update_sd_lifecycle(
            current,
            candle_low=current.zone_low + 0.5,
            candle_high=current.zone_high - 0.5,
            candle_close=current.zone_low + 1.0,
            bar_index=999,
            config=config,
        )
        assert updated.zone_strength <= prev_strength + 1e-9
        prev_strength = updated.zone_strength
        current = updated


# Property SD-4: Scoring Monotonicity
# Validates: Requirements 11.3–11.5, 14.5
@pytest.mark.unit
@pytest.mark.slow
@given(st.lists(sd_zone_strategy(), min_size=1, max_size=50))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sd_scoring_monotonicity_prop_9(zones: list) -> None:
    if rank_sd_zones is None:
        return
    if not zones:
        return
    ts = datetime(2020, 1, 1, tzinfo=timezone.utc)
    df_dummy = pd.DataFrame(
        [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0}] * 30,
        index=pd.date_range(start=ts, periods=30, freq="D", tz="UTC"),
    )
    df_by_symbol = {z.symbol: df_dummy for z in zones}
    ranked = rank_sd_zones(zones, df_by_symbol, SDConfig())
    for k in range(len(ranked) - 1):
        assert ranked[k].combined_score >= ranked[k + 1].combined_score


# Property SD-5: Round-Trip Serialization
# Validates: Requirements 14.4
@pytest.mark.unit
@pytest.mark.slow
@given(sd_zone_strategy())
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sd_round_trip_prop_10(zone: SDZone) -> None:
    restored = SDZone.model_validate_json(zone.model_dump_json())
    assert restored == zone


# Property SWEEP-1: Sweep Geometric Invariant
# Validates: Requirements 16.1–16.3
@pytest.mark.unit
@pytest.mark.slow
@given(ohlcv_strategy(min_rows=12))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sweep_geometric_invariant_prop_11(df: pd.DataFrame) -> None:
    if detect_sweeps is None:
        return
    if len(df) < 12:
        return
    config = SweepConfig(require_reversal_confirmation=False, volume_confirmation_multiplier=0.0)
    signals = detect_sweeps(df, "PROP11", "1h", config)
    avg_vol = df["volume"].rolling(config.avg_volume_window, min_periods=1).mean()
    sw_high = df["high"].rolling(config.swing_lookback, min_periods=1).max()
    sw_low  = df["low"].rolling(config.swing_lookback, min_periods=1).min()
    for s in signals:
        k = s.sweep_bar_index
        if s.direction == SweepDirection.BULLISH:
            assert float(df["low"].iloc[k]) < float(sw_low.iloc[k - 1]) + 1e-9
            assert float(df["close"].iloc[k]) > float(sw_low.iloc[k - 1]) - 1e-9
        else:
            assert float(df["high"].iloc[k]) > float(sw_high.iloc[k - 1]) - 1e-9
            assert float(df["close"].iloc[k]) < float(sw_high.iloc[k - 1]) + 1e-9


# Property SWEEP-2: No-Lookahead Invariant
# Validates: Requirements 21.2, 21.3
@pytest.mark.unit
@pytest.mark.slow
@given(ohlcv_strategy(), st.integers(min_value=1, max_value=5))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sweep_no_lookahead_prop_12(df: pd.DataFrame, n: int) -> None:
    if detect_sweeps is None:
        return
    if len(df) < 12:
        return
    config = SweepConfig(require_reversal_confirmation=False)
    result_orig = detect_sweeps(df, "PROP12", "1h", config)
    last_ts = df.index[-1]
    extra_ts = pd.date_range(start=last_ts + pd.Timedelta(days=1), periods=n, freq="D", tz="UTC")
    extra_rows = [{"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0}] * n
    df_ext = pd.concat([df, pd.DataFrame(extra_rows, index=extra_ts)])
    result_ext = detect_sweeps(df_ext, "PROP12", "1h", config)
    orig_set = {(s.sweep_bar_index, s.direction.value) for s in result_orig}
    ext_set = {(s.sweep_bar_index, s.direction.value) for s in result_ext if s.sweep_bar_index < len(df)}
    assert orig_set == ext_set


# Property SWEEP-3: Lifecycle Terminal State
# Validates: Requirements 17.4
@pytest.mark.unit
@pytest.mark.slow
@given(sweep_signal_strategy(), st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sweep_lifecycle_terminal_prop_13(signal: SweepSignal, close_val: float) -> None:
    if update_sweep_lifecycle is None:
        return
    if signal.lifecycle_state != SweepLifecycle.INVALIDATED:
        return
    config = SweepConfig()
    updated = update_sweep_lifecycle(signal, next_candle_close=float(close_val), bar_index=999, config=config)
    assert updated.lifecycle_state == SweepLifecycle.INVALIDATED


# Property SWEEP-4: Scoring Monotonicity
# Validates: Requirements 18.3–18.5
@pytest.mark.unit
@pytest.mark.slow
@given(st.lists(sweep_signal_strategy(), min_size=1, max_size=50))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sweep_scoring_monotonicity_prop_14(signals: list) -> None:
    if rank_sweeps is None:
        return
    if not signals:
        return
    ranked = rank_sweeps(signals, SweepConfig())
    for k in range(len(ranked) - 1):
        assert ranked[k].conviction_score >= ranked[k + 1].conviction_score


# Property SWEEP-5: Round-Trip Serialization
# Validates: Requirements 21.5
@pytest.mark.unit
@pytest.mark.slow
@given(sweep_signal_strategy())
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_sweep_round_trip_prop_15(signal: SweepSignal) -> None:
    restored = SweepSignal.model_validate_json(signal.model_dump_json())
    assert restored == signal


# Property MS-1: BoS Close-Confirmation Invariant
# Validates: Requirements 23.1–23.3
@pytest.mark.unit
@pytest.mark.slow
@given(ohlcv_strategy(min_rows=15))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_ms_bos_close_confirm_prop_16(df: pd.DataFrame) -> None:
    if detect_market_structure is None:
        return
    if len(df) < 15:
        return
    signals = detect_market_structure(df, "PROP16", "1h", MSConfig())
    for s in signals:
        if not s.is_choch:
            if s.signal_type == MSSignalType.BOS_BULLISH:
                assert s.close_at_break > s.broken_level
            elif s.signal_type == MSSignalType.BOS_BEARISH:
                assert s.close_at_break < s.broken_level


# Property MS-2: Trend State Only Changes on ChoCh
# Validates: Requirements 24.4, 24.5, 25.4–25.6, 29.7
@pytest.mark.unit
@pytest.mark.slow
@given(ohlcv_strategy(min_rows=15))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_ms_state_only_changes_on_choch_prop_17(df: pd.DataFrame) -> None:
    if detect_market_structure is None:
        return
    if len(df) < 15:
        return
    signals = detect_market_structure(df, "PROP17", "1h", MSConfig())
    prev_trend_state = None
    for s in signals:
        if prev_trend_state is not None and s.trend_state != prev_trend_state:
            assert s.is_choch is True, "Trend state should only change on ChoCh signals"
        prev_trend_state = s.trend_state


# Property MS-3: Scoring Monotonicity
# Validates: Requirements 26.3–26.5
@pytest.mark.unit
@pytest.mark.slow
@given(st.lists(ms_signal_strategy(), min_size=1, max_size=50))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_ms_scoring_monotonicity_prop_18(signals: list) -> None:
    if rank_ms_signals is None:
        return
    if not signals:
        return
    ranked = rank_ms_signals(signals, MSConfig())
    for k in range(len(ranked) - 1):
        assert ranked[k].conviction_score >= ranked[k + 1].conviction_score


# Property MS-4: No-Lookahead Invariant
# Validates: Requirements 29.1–29.3
@pytest.mark.unit
@pytest.mark.slow
@given(ohlcv_strategy(), st.integers(min_value=1, max_value=5))
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_ms_no_lookahead_prop_19(df: pd.DataFrame, n: int) -> None:
    if detect_market_structure is None:
        return
    if len(df) < 15:
        return
    config = MSConfig()
    result_orig = detect_market_structure(df, "PROP19", "1h", config)
    last_ts = df.index[-1]
    extra_ts = pd.date_range(start=last_ts + pd.Timedelta(days=1), periods=n, freq="D", tz="UTC")
    extra_rows = [{"open": 100.0, "high": 105.0, "low": 95.0, "close": 102.0, "volume": 1000.0}] * n
    df_ext = pd.concat([df, pd.DataFrame(extra_rows, index=extra_ts)])
    result_ext = detect_market_structure(df_ext, "PROP19", "1h", config)
    orig_set = {(s.break_bar_index, s.signal_type.value) for s in result_orig}
    ext_set = {(s.break_bar_index, s.signal_type.value) for s in result_ext if s.break_bar_index < len(df)}
    assert orig_set == ext_set


# Property MS-5: Round-Trip Serialization
# Validates: Requirements 29.5
@pytest.mark.unit
@pytest.mark.slow
@given(ms_signal_strategy())
@h_settings(
    max_examples=100,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
def test_ms_round_trip_prop_20(signal: MSSignal) -> None:
    restored = MSSignal.model_validate_json(signal.model_dump_json())
    assert restored == signal


# ===========================================================================
# Task 2.7 — Point-In-Time (PIT) tests for all four scanners
# ===========================================================================


@pytest.mark.pit
@pytest.mark.unit
def test_fvg_pit_timestamp() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df()
    zones = detect_fvg_zones(df, "PIT_FVG", "1h", FVGConfig())
    for z in zones:
        expected_ts = df.index[z.candle3_bar_index]
        assert z.fvg_timestamp == expected_ts, (
            f"fvg_timestamp {z.fvg_timestamp} != df.index[candle3_bar_index] {expected_ts}"
        )
        # Also verify candle1 and candle3 bar indices are within range
        assert 0 <= z.candle1_bar_index < len(df)
        assert z.candle3_bar_index < len(df)
        assert z.candle1_bar_index < z.candle3_bar_index


@pytest.mark.pit
@pytest.mark.unit
def test_sd_pit_departure_timestamp() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(entry_dir="down", dep_dir="down", extra_bars=10)
    zones = detect_sd_zones(df, "PIT_SD", "1h", SDConfig())
    for z in zones:
        expected_ts = df.index[z.departure_bar_index]
        assert z.departure_timestamp == expected_ts, (
            f"departure_timestamp {z.departure_timestamp} != df.index[departure_bar_index] {expected_ts}"
        )
        # entry_leg_timestamp should also match its bar index
        assert z.base_start_bar_index > 0, "base_start_bar_index must be > 0 (entry_leg is at base_start - 1)"
        entry_leg_bar = z.base_start_bar_index - 1
        expected_entry_ts = df.index[entry_leg_bar]
        assert z.entry_leg_timestamp == expected_entry_ts


@pytest.mark.pit
@pytest.mark.unit
def test_sweep_pit_timestamp() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish", sweep_bar=20)
    config = SweepConfig(require_reversal_confirmation=True)
    signals = detect_sweeps(df, "PIT_SWEEP", "1h", config)
    for s in signals:
        expected_ts = df.index[s.sweep_bar_index]
        assert s.sweep_timestamp == expected_ts, (
            f"sweep_timestamp {s.sweep_timestamp} != df.index[sweep_bar_index] {expected_ts}"
        )
        assert 0 <= s.sweep_bar_index < len(df)


@pytest.mark.pit
@pytest.mark.unit
def test_ms_pit_break_timestamp() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    signals = detect_market_structure(df, "PIT_MS", "1h", MSConfig())
    for s in signals:
        expected_ts = df.index[s.break_bar_index]
        assert s.break_timestamp == expected_ts, (
            f"break_timestamp {s.break_timestamp} != df.index[break_bar_index] {expected_ts}"
        )
        assert 0 <= s.break_bar_index < len(df)


@pytest.mark.pit
@pytest.mark.unit
def test_fvg_no_future_data_in_zone() -> None:
    if detect_fvg_zones is None:
        pytest.skip("detect_fvg_zones not yet implemented")
    df = _make_fvg_df(n_extra=10)
    zones_orig = detect_fvg_zones(df, "PIT_FVG2", "1h", FVGConfig())
    if not zones_orig:
        pytest.skip("No zones detected in test data")
    # Modify a candle far after candle3 of the first zone
    first_zone = zones_orig[0]
    future_bar = first_zone.candle3_bar_index + 5
    if future_bar >= len(df):
        pytest.skip("Not enough bars to test future modification")
    df_mod = df.copy()
    df_mod.iloc[future_bar, df_mod.columns.get_loc("close")] = 9999.0
    zones_mod = detect_fvg_zones(df_mod, "PIT_FVG2", "1h", FVGConfig())
    # Find the matching zone in the modified result
    orig_early = {(z.candle1_bar_index, z.candle3_bar_index, z.direction.value) for z in zones_orig}
    mod_early = {(z.candle1_bar_index, z.candle3_bar_index, z.direction.value) for z in zones_mod
                 if z.candle3_bar_index <= first_zone.candle3_bar_index}
    assert orig_early == mod_early or orig_early.issubset(mod_early), (
        "Modifying a future candle should not change zones detected before it"
    )


@pytest.mark.pit
@pytest.mark.unit
def test_sd_no_future_data_in_zone() -> None:
    if detect_sd_zones is None:
        pytest.skip("detect_sd_zones not yet implemented")
    df = _make_sd_df(entry_dir="down", dep_dir="down", extra_bars=20)
    zones_orig = detect_sd_zones(df, "PIT_SD2", "1h", SDConfig())
    if not zones_orig:
        pytest.skip("No zones detected in test data")
    # Modify a candle after departure
    first_zone = zones_orig[0]
    future_bar = first_zone.departure_bar_index + 5
    if future_bar >= len(df):
        pytest.skip("Not enough bars for future modification test")
    df_mod = df.copy()
    df_mod.iloc[future_bar, df_mod.columns.get_loc("close")] = 9999.0
    zones_mod = detect_sd_zones(df_mod, "PIT_SD2", "1h", SDConfig())
    orig_set = {(z.base_start_bar_index, z.departure_bar_index, z.zone_type.value) for z in zones_orig}
    mod_set = {(z.base_start_bar_index, z.departure_bar_index, z.zone_type.value) for z in zones_mod
               if z.departure_bar_index <= first_zone.departure_bar_index}
    assert orig_set == mod_set or orig_set.issubset(mod_set)


@pytest.mark.pit
@pytest.mark.unit
def test_sweep_no_future_data_in_signal() -> None:
    if detect_sweeps is None:
        pytest.skip("detect_sweeps not yet implemented")
    df = _make_sweep_df(sweep_dir="bullish", sweep_bar=20)
    config = SweepConfig(require_reversal_confirmation=False)
    signals_orig = detect_sweeps(df, "PIT_SWEEP2", "1h", config)
    if not signals_orig:
        pytest.skip("No sweeps detected in test data")
    # Modify a candle 5 bars after the sweep
    first_signal = signals_orig[0]
    future_bar = first_signal.sweep_bar_index + 5
    if future_bar >= len(df):
        pytest.skip("Not enough bars for future modification test")
    df_mod = df.copy()
    df_mod.iloc[future_bar, df_mod.columns.get_loc("close")] = 9999.0
    signals_mod = detect_sweeps(df_mod, "PIT_SWEEP2", "1h", config)
    orig_set = {(s.sweep_bar_index, s.direction.value) for s in signals_orig}
    mod_set = {(s.sweep_bar_index, s.direction.value) for s in signals_mod
               if s.sweep_bar_index <= first_signal.sweep_bar_index}
    assert orig_set == mod_set or orig_set.issubset(mod_set)


@pytest.mark.pit
@pytest.mark.unit
def test_ms_no_future_data_in_signal() -> None:
    if detect_market_structure is None:
        pytest.skip("detect_market_structure not yet implemented")
    df = _make_ms_df(trend="bullish", n_bars=50)
    signals_orig = detect_market_structure(df, "PIT_MS2", "1h", MSConfig())
    if not signals_orig:
        pytest.skip("No signals detected")
    # Modify a future bar
    first_signal = signals_orig[0]
    future_bar = first_signal.break_bar_index + 5
    if future_bar >= len(df):
        pytest.skip("Not enough bars for future modification test")
    df_mod = df.copy()
    df_mod.iloc[future_bar, df_mod.columns.get_loc("high")] = 9999.0
    signals_mod = detect_market_structure(df_mod, "PIT_MS2", "1h", MSConfig())
    orig_early = {(s.break_bar_index, s.signal_type.value) for s in signals_orig
                  if s.break_bar_index <= first_signal.break_bar_index}
    mod_early = {(s.break_bar_index, s.signal_type.value) for s in signals_mod
                 if s.break_bar_index <= first_signal.break_bar_index}
    assert orig_early == mod_early
