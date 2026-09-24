"""
test_data_ingestion.py — Phase B regression tests for DataIngestionPipeline (P0-003).

Verifies:
- Bars fetched exclusively via the injected data client.
- OHLC / volume / duplicate / chronology validation.
- Missing values are dropped, never zero-filled.
- Resumable ingestion via checkpoint.
- Immutable parquet/csv artifact written per (symbol, interval).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.data.ingestion import (
    DataIngestionPipeline,
    IngestionValidationReport,
    normalize_bars,
)

UTC = timezone.utc


def _bar(ts, o, h, low, c, v, **extra):
    d = {"timestamp": ts.isoformat(), "open": o, "high": h, "low": low,
         "close": c, "volume": v}
    d.update(extra)
    return d


def _clean_series(n=10, start=None):
    start = start or datetime(2024, 1, 1, tzinfo=UTC)
    bars = []
    for i in range(n):
        ts = start + timedelta(days=i)
        price = 100.0 + i
        bars.append(_bar(ts, price, price + 2, price - 1, price + 1, 1000 + i,
                         data_confidence=95))
    return bars


# ─── normalize_bars ─────────────────────────────────────────────────────────


def test_normalize_clean_bars():
    rep = IngestionValidationReport(symbol="X", interval="1d")
    df = normalize_bars(_clean_series(5), "X", "1d", rep)
    assert rep.rows_raw == 5
    assert rep.rows_clean == 5
    assert list(df.columns).count("close") == 1
    assert df["data_confidence"].iloc[0] == 95


def test_ohlc_violation_dropped_not_zeroed():
    rep = IngestionValidationReport(symbol="X", interval="1d")
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    # high < low is invalid
    bad = [_bar(ts, 100, 90, 95, 98, 1000)]
    df = normalize_bars(bad, "X", "1d", rep)
    assert rep.ohlc_violations == 1
    assert df.empty  # dropped, not fabricated


def test_missing_volume_dropped():
    rep = IngestionValidationReport(symbol="X", interval="1d")
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    bar = {"timestamp": ts.isoformat(), "open": 100, "high": 101,
           "low": 99, "close": 100.5}  # no volume
    df = normalize_bars([bar], "X", "1d", rep)
    assert df.empty  # missing volume → dropped, never zero


def test_negative_volume_flagged():
    rep = IngestionValidationReport(symbol="X", interval="1d")
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    df = normalize_bars([_bar(ts, 100, 101, 99, 100.5, -5)], "X", "1d", rep)
    assert rep.negative_volume == 1
    assert df.empty


def test_duplicate_timestamps_dropped():
    rep = IngestionValidationReport(symbol="X", interval="1d")
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    bars = [_bar(ts, 100, 101, 99, 100.5, 1000),
            _bar(ts, 100, 101, 99, 100.6, 1001)]
    df = normalize_bars(bars, "X", "1d", rep)
    assert rep.duplicates_dropped == 1
    assert len(df) == 1


def test_gap_detection_daily():
    rep = IngestionValidationReport(symbol="X", interval="1d")
    # Two consecutive business days then skip a week
    d1 = datetime(2024, 1, 1, tzinfo=UTC)   # Mon
    d2 = datetime(2024, 1, 2, tzinfo=UTC)   # Tue
    d3 = datetime(2024, 1, 15, tzinfo=UTC)  # Mon (gap)
    bars = [_bar(d, 100, 101, 99, 100.5, 1000) for d in (d1, d2, d3)]
    df = normalize_bars(bars, "X", "1d", rep)
    assert rep.gaps_detected > 0
    assert len(df) == 3


# ─── Pipeline with mock client ──────────────────────────────────────────────


class _MockClient:
    def __init__(self, series_by_symbol):
        self._data = series_by_symbol
        self.calls = 0

    async def get_historical_ohlcv(self, symbol, exchange="NSE", interval="1d",
                                   from_date=None, to_date=None, pit_date=None):
        self.calls += 1
        return self._data.get(symbol.upper(), [])


@pytest.mark.asyncio
async def test_ingest_writes_artifacts(tmp_path):
    client = _MockClient({"NIFTY": _clean_series(20), "RELIANCE": _clean_series(15)})
    pipeline = DataIngestionPipeline(client, output_root=tmp_path)
    result = await pipeline.ingest(["NIFTY", "RELIANCE"], interval="1d",
                                   from_date="2024-01-01", to_date="2024-02-01")
    assert set(result.symbols_ingested) == {"NIFTY", "RELIANCE"}
    assert result.total_rows == 35
    df = pipeline.load_symbol("NIFTY", "1d")
    assert len(df) == 20


@pytest.mark.asyncio
async def test_ingest_is_resumable(tmp_path):
    client = _MockClient({"NIFTY": _clean_series(20)})
    pipeline = DataIngestionPipeline(client, output_root=tmp_path)
    await pipeline.ingest(["NIFTY"], interval="1d")
    first_calls = client.calls
    # Second run should skip (resume) — no additional fetch.
    await pipeline.ingest(["NIFTY"], interval="1d", resume=True)
    assert client.calls == first_calls  # no re-download


@pytest.mark.asyncio
async def test_failed_symbol_recorded(tmp_path):
    class _FailClient:
        async def get_historical_ohlcv(self, *a, **k):
            raise RuntimeError("data service down")

    pipeline = DataIngestionPipeline(_FailClient(), output_root=tmp_path)
    result = await pipeline.ingest(["BROKEN"], interval="1d")
    assert "BROKEN" in result.symbols_failed
