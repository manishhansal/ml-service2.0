"""
tests/test_data_reliability.py — unit/mock tests for DataReliabilityInvestigator.

All tests are fully mocked — no live data-service2.0 or SentinelPulse calls.

Covers:
  - DataReliabilityReport dataclass: instantiation, to_dict, to_json
  - DataReliabilityInvestigator: verdict logic (PRODUCTION_READY / CONDITIONALLY_READY / NOT_READY)
  - _investigate_symbol: clean symbol, fetch failure, OHLC violations, gap detection
  - _investigate_features: feature quality, leakage detection
  - _investigate_news: coverage, look_ahead_validated, degraded when SentinelPulse None
  - _render_verdict: blockers → NOT_READY, warnings → CONDITIONALLY_READY, clean → PRODUCTION_READY
  - SymbolIngestionResult, FeatureQualityResult, SignalQualityResult, NewsReliabilityResult
  - Threshold constants
"""
from __future__ import annotations

import asyncio
import math
from datetime import date, datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

import os
os.environ.setdefault("ML_SERVICE_API_KEY", "test-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

from src.data.reliability import (
    DataReliabilityInvestigator,
    DataReliabilityReport,
    FeatureQualityResult,
    NewsReliabilityResult,
    SignalQualityResult,
    SymbolIngestionResult,
    _THRESHOLDS,
)

UTC = timezone.utc


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_ohlcv_bars(n: int = 300, symbol: str = "NIFTY") -> list[dict]:
    """Return *n* clean daily OHLCV bars."""
    bars = []
    base = datetime(2023, 1, 2, 10, 0, 0, tzinfo=UTC)
    price = 18000.0
    for i in range(n):
        price += float(np.random.default_rng(i).normal(0, 50))
        price = max(price, 1000.0)
        bars.append({
            "timestamp": (base + pd.Timedelta(days=i)).isoformat(),
            "open":   price - 20,
            "high":   price + 50,
            "low":    price - 60,
            "close":  price,
            "volume": 1_000_000.0,
            "data_confidence": 90,
        })
    return bars


def _make_data_client(bars: list[dict] | None = None, fail: bool = False) -> AsyncMock:
    client = AsyncMock()
    if fail:
        client.get_historical_ohlcv.side_effect = Exception("upstream_down")
        client.get_market_status.side_effect = Exception("upstream_down")
    else:
        client.get_historical_ohlcv.return_value = bars or _make_ohlcv_bars()
        client.get_market_status.return_value = {"isOpen": True, "holidayList": []}
    return client


def _make_sentinel_client(
    reachable: bool = True,
    coverage_pct: float = 1.0,
    look_ahead_pct: float = 100.0,
) -> AsyncMock:
    client = AsyncMock()
    if not reachable:
        client.fetch_market_regime.return_value = None
        client.fetch_news_context.return_value = None
        client.fetch_training_samples.return_value = None
        return client

    client.fetch_market_regime.return_value = {"regime": "BULL"}
    client.fetch_market_context.return_value = {"macro": "positive"}

    ctx = {
        "news_impact_score": 0.5,
        "impact_direction": "BULLISH",
        "impact_confidence": 0.7,
        "sentiment": {"overall": 0.3, "market": 0.2, "company": 0.4, "macro": -0.1, "risk": -0.05},
        "market_regime": "RISK_ON",
        "as_of": "2024-01-01T09:00:00+00:00",
    }
    client.fetch_news_context.return_value = ctx if coverage_pct >= 1.0 else None

    n_samples = 20
    n_valid = int(n_samples * look_ahead_pct / 100)
    samples = [
        {"look_ahead_validated": i < n_valid, "news_impact_score": 0.5,
         "impact_direction": "BULLISH", "impact_confidence": 0.7}
        for i in range(n_samples)
    ]
    client.fetch_training_samples.return_value = {"samples": samples}
    return client


def _run(coro):
    return asyncio.run(coro)


# ── DataReliabilityReport dataclass ──────────────────────────────────────────

class TestDataReliabilityReport:
    def test_instantiation_defaults(self):
        report = DataReliabilityReport(verdict="NOT_READY", verdict_reason="test")
        assert report.verdict == "NOT_READY"
        assert report.symbol_results == []
        assert report.blockers == []
        assert report.warnings == []

    def test_to_dict_has_expected_keys(self):
        report = DataReliabilityReport(verdict="PRODUCTION_READY", verdict_reason="all good")
        d = report.to_dict()
        assert "verdict" in d
        assert "symbol_results" in d
        assert "feature_quality" in d
        assert "signal_quality" in d
        assert "news_reliability" in d
        assert "blockers" in d

    def test_to_json_is_valid_json(self):
        import json
        report = DataReliabilityReport(verdict="NOT_READY", verdict_reason="x")
        report.blockers.append("TEST_BLOCKER")
        j = report.to_json()
        parsed = json.loads(j)
        assert parsed["verdict"] == "NOT_READY"
        assert "TEST_BLOCKER" in parsed["blockers"]

    def test_save_creates_file(self, tmp_path):
        report = DataReliabilityReport(verdict="NOT_READY", verdict_reason="x")
        out = tmp_path / "report.json"
        report.save(out)
        assert out.exists()
        assert out.stat().st_size > 0


# ── Thresholds ────────────────────────────────────────────────────────────────

class TestThresholds:
    def test_ic_mean_min_is_0_03(self):
        assert _THRESHOLDS["ic_mean_min"] == 0.03

    def test_hit_rate_min_is_0_52(self):
        assert _THRESHOLDS["hit_rate_min"] == 0.52

    def test_min_bars_is_252(self):
        assert _THRESHOLDS["min_bars"] == 252

    def test_news_pit_valid_pct_is_100(self):
        assert _THRESHOLDS["news_pit_valid_pct"] == 100.0


# ── _investigate_symbol ───────────────────────────────────────────────────────

class TestInvestigateSymbol:
    def _investigator(self, bars=None, fail=False):
        return DataReliabilityInvestigator(
            data_client=_make_data_client(bars=bars, fail=fail),
            sentinel_client=None,
        )

    def test_clean_300_bars_passes(self):
        inv = self._investigator(bars=_make_ohlcv_bars(300))
        result, df = _run(inv._investigate_symbol("NIFTY", "1d", None, None, "NSE"))
        assert result.bars_clean == 300
        assert result.ohlc_violations == 0
        assert result.bars_clean >= _THRESHOLDS["min_bars"]
        assert result.passes

    def test_fetch_failure_returns_none_df_and_blocker(self):
        inv = self._investigator(fail=True)
        result, df = _run(inv._investigate_symbol("NIFTY", "1d", None, None, "NSE"))
        assert df is None
        assert len(result.blockers) > 0

    def test_insufficient_bars_creates_blocker(self):
        inv = self._investigator(bars=_make_ohlcv_bars(100))
        result, df = _run(inv._investigate_symbol("NIFTY", "1d", None, None, "NSE"))
        assert any("INSUFFICIENT_HISTORY" in b for b in result.blockers)
        assert not result.passes

    def test_ohlc_violations_detected(self):
        bars = _make_ohlcv_bars(300)
        # Corrupt 5 bars: high < low
        for i in range(5):
            bars[i]["high"] = bars[i]["low"] - 1.0
        inv = self._investigator(bars=bars)
        result, df = _run(inv._investigate_symbol("NIFTY", "1d", None, None, "NSE"))
        assert result.ohlc_violations == 5

    def test_confidence_scores_extracted(self):
        bars = _make_ohlcv_bars(300)
        inv = self._investigator(bars=bars)
        result, df = _run(inv._investigate_symbol("NIFTY", "1d", None, None, "NSE"))
        assert result.confidence_mean > 0


# ── _investigate_features ─────────────────────────────────────────────────────

class TestInvestigateFeatures:
    def test_clean_data_passes(self):
        inv = DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=None,
        )
        bars = _make_ohlcv_bars(300)
        df = pd.DataFrame(bars)
        df.columns = [c.lower() for c in df.columns]
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp")
        # Keep only OHLCV columns that FeatureFactory needs
        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        result = inv._investigate_features({"NIFTY": df})
        assert result.n_features > 0
        assert result.n_rows > 0
        # No leakage expected on clean synthetic data
        assert not result.leakage_detected

    def test_empty_frames_returns_blocker(self):
        inv = DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=None,
        )
        result = inv._investigate_features({})
        assert len(result.blockers) > 0


# ── _investigate_news ─────────────────────────────────────────────────────────

class TestInvestigateNews:
    def test_no_sentinel_client_returns_warning_not_blocker(self):
        inv = DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=None,
        )
        result = _run(inv._investigate_news(["NIFTY"]))
        assert result.passes         # no client → warn but don't block
        assert len(result.warnings) > 0
        assert len(result.blockers) == 0

    def test_reachable_sentinel_with_full_coverage(self):
        inv = DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=_make_sentinel_client(reachable=True, coverage_pct=1.0),
        )
        result = _run(inv._investigate_news(["NIFTY", "BANKNIFTY"]))
        assert result.sentinel_reachable
        assert result.coverage_pct > 0

    def test_unreachable_sentinel_warns_not_blocks(self):
        inv = DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=_make_sentinel_client(reachable=False),
        )
        result = _run(inv._investigate_news(["NIFTY"]))
        assert not result.sentinel_reachable
        assert len(result.warnings) > 0
        assert len(result.blockers) == 0

    def test_100pct_look_ahead_valid_no_blocker(self):
        inv = DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=_make_sentinel_client(look_ahead_pct=100.0),
        )
        result = _run(inv._investigate_news(["NIFTY"]))
        assert result.look_ahead_valid_pct == 100.0
        assert not any("PIT" in b for b in result.blockers)

    def test_partial_look_ahead_valid_creates_blocker(self):
        inv = DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=_make_sentinel_client(look_ahead_pct=50.0),
        )
        result = _run(inv._investigate_news(["NIFTY"]))
        assert result.look_ahead_valid_pct < 100.0
        # 50% < 100% threshold → blocker
        assert any("PIT" in b for b in result.blockers)


# ── _render_verdict ───────────────────────────────────────────────────────────

class TestRenderVerdict:
    def _inv(self):
        return DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=None,
        )

    def test_hard_blocker_yields_not_ready(self):
        inv = self._inv()
        report = DataReliabilityReport(verdict="", verdict_reason="")
        report.blockers.append("IC_TOO_LOW: mean IC=0.001")
        verdict, reason = inv._render_verdict(report)
        assert verdict == "NOT_READY"
        assert "IC_TOO_LOW" in reason

    def test_warnings_only_yields_conditionally_ready(self):
        inv = self._inv()
        report = DataReliabilityReport(verdict="", verdict_reason="")
        report.warnings.append("LOW_SHARPE: 0.4")
        report.signal_quality.passes = True
        report.feature_quality.passes = True
        report.symbol_results.append(SymbolIngestionResult(symbol="NIFTY", passes=True))
        verdict, reason = inv._render_verdict(report)
        assert verdict == "CONDITIONALLY_READY"

    def test_no_issues_yields_production_ready(self):
        inv = self._inv()
        report = DataReliabilityReport(verdict="", verdict_reason="")
        report.signal_quality.passes = True
        report.feature_quality.passes = True
        report.signal_quality.ic_mean = 0.05
        report.signal_quality.ic_ir = 2.0
        report.signal_quality.grade = "B"
        report.symbol_results.append(SymbolIngestionResult(symbol="NIFTY", passes=True))
        verdict, reason = inv._render_verdict(report)
        assert verdict == "PRODUCTION_READY"

    def test_fetch_failed_blocker_yields_not_ready(self):
        inv = self._inv()
        report = DataReliabilityReport(verdict="", verdict_reason="")
        report.blockers.append("NIFTY_FETCH_FAILED: timeout")
        verdict, reason = inv._render_verdict(report)
        assert verdict == "NOT_READY"


# ── Grade assignment ──────────────────────────────────────────────────────────

class TestGradeAssignment:
    def _inv(self):
        return DataReliabilityInvestigator(
            data_client=_make_data_client(),
            sentinel_client=None,
        )

    def test_high_ic_and_sharpe_gives_a(self):
        inv = self._inv()
        r = SignalQualityResult(
            ic_mean=0.08, ic_std=0.02, ic_ir=4.0, hit_rate=0.57,
            sharpe_ratio=2.2, max_drawdown=-0.08,
        )
        assert inv._assign_grade(r) == "A"

    def test_minimal_ic_gives_d_or_f(self):
        """IC=0.03 with low ICIR and low Sharpe scores 10+0+0+6+3=19 → F grade."""
        inv = self._inv()
        r = SignalQualityResult(
            ic_mean=0.03, ic_std=0.05, ic_ir=0.6, hit_rate=0.52,
            sharpe_ratio=0.4, max_drawdown=-0.25,
        )
        grade = inv._assign_grade(r)
        # Scoring: IC(10) + ICIR(0) + Sharpe(0) + hit_rate(6) + drawdown(3) = 19 → F
        assert grade in ("D", "F")

    def test_zero_ic_gives_f(self):
        inv = self._inv()
        r = SignalQualityResult()
        assert inv._assign_grade(r) == "F"
