"""
test_certification_harness.py — data-source honesty + require-live gating (mandate §13,§53,§54,§96).

Exercises the certification harness contract WITHOUT a live data-service:
  - synthetic fallback is explicitly labelled SYNTHETIC
  - --require-live turns unavailability into a hard failure (no synthetic fallback)
  - the harness never fabricates FORWARD_PAPER evidence
"""
from __future__ import annotations

import importlib

import pytest

cert = importlib.import_module("scripts.run_certification")


def test_data_source_classes_are_distinct():
    classes = {
        cert.DataSourceClass.SYNTHETIC,
        cert.DataSourceClass.HISTORICAL_REAL,
        cert.DataSourceClass.LIVE_REAL,
        cert.DataSourceClass.FORWARD_PAPER,
    }
    assert len(classes) == 4


def test_synthetic_fallback_when_live_unavailable(monkeypatch):
    async def _boom(symbols, days=900):
        raise RuntimeError("data-service down")

    monkeypatch.setattr(cert, "_fetch_live", _boom)
    ohlcv, cls = cert._resolve_data(["NIFTY"], require_live=False)
    assert cls == cert.DataSourceClass.SYNTHETIC
    assert "NIFTY" in ohlcv


def test_require_live_hard_fails_when_unavailable(monkeypatch):
    async def _boom(symbols, days=900):
        raise RuntimeError("data-service down")

    monkeypatch.setattr(cert, "_fetch_live", _boom)
    with pytest.raises(cert.LiveDataUnavailableError):
        cert._resolve_data(["NIFTY"], require_live=True)


def test_require_live_hard_fails_on_empty_series(monkeypatch):
    async def _empty(symbols, days=900):
        return {}

    monkeypatch.setattr(cert, "_fetch_live", _empty)
    with pytest.raises(cert.LiveDataUnavailableError):
        cert._resolve_data(["NIFTY"], require_live=True)


def test_real_series_labelled_historical_real(monkeypatch):
    import numpy as np
    import pandas as pd

    async def _real(symbols, days=900):
        idx = pd.date_range("2020-01-01", periods=300, freq="D", tz="UTC")
        return {
            s: pd.DataFrame(
                {"open": np.linspace(100, 120, 300), "high": np.linspace(101, 121, 300),
                 "low": np.linspace(99, 119, 300), "close": np.linspace(100, 120, 300),
                 "volume": np.full(300, 1000.0)},
                index=idx,
            )
            for s in symbols
        }

    monkeypatch.setattr(cert, "_fetch_live", _real)
    _, cls = cert._resolve_data(["NIFTY"], require_live=True)
    assert cls == cert.DataSourceClass.HISTORICAL_REAL
