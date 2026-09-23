"""
test_signal_generation_latency.py

Latency benchmark tests for all ml-service2.0 prediction endpoints.

TDD Phase: Written BEFORE model implementations (Phase 4).
Tests are designed to:
1. Verify all endpoints respond (not 404)
2. Measure actual latency using time.perf_counter()
3. Assert p95 latency SLAs once models are implemented
4. Currently acceptable: 503 responses (stubs) within latency bounds

SLA targets (Req 16.1–16.5):
- Single-symbol prediction endpoints: ≤ 50ms p95
- Batch ranking (200 symbols): ≤ 200ms p95
- Meta-decision: ≤ 150ms p95
- Portfolio optimization (50 assets): ≤ 500ms p95
- Analytics endpoints: ≤ 100ms p95

Requirements: Req 16.1, Req 16.2, Req 16.3, Req 16.4, Req 16.5, Req 18.8
"""
from __future__ import annotations

import os
import time
from statistics import median, stdev
from typing import Any

import pytest

os.environ.setdefault("ML_SERVICE_API_KEY", "test-key-for-testing")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-data-key")

# ── Sample request payloads for each endpoint ─────────────────────────────────

_REGIME_PAYLOAD = {
    "nifty_change_pct": 0.5,
    "banknifty_change_pct": 0.8,
    "india_vix": 15.0,
    "nifty_atr_pct": 1.2,
    "nifty_adx": 25.0,
    "advance_decline_ratio": 1.5,
    "market_breadth": 0.6,
    "sector_strength": 0.4,
    "volume_ratio": 1.1,
    "gap_pct": 0.2,
}

_STRATEGY_PAYLOAD = {
    "regime": "bull",
    "symbol": "NIFTY",
    "rsi": 60.0,
    "adx": 25.0,
    "atr_pct": 1.5,
    "volume_ratio": 1.2,
    "vwap_distance_pct": 0.3,
    "bollinger_position": 0.5,
    "trend_strength": 0.4,
    "volatility_rank": 0.6,
    "time_of_day_minutes": 120,
    "iv_regime": "STABLE",
}

_RISK_PAYLOAD = {
    "symbol": "NIFTY",
    "direction": "LONG",
    "entry": 22000.0,
    "stop_loss": 21800.0,
    "target": 22400.0,
    "atr": 200.0,
    "regime": "bull",
    "rsi": 60.0,
    "adx": 25.0,
    "volume_ratio": 1.1,
    "vix": 15.0,
}

_EXECUTION_PAYLOAD = {
    "symbol": "NIFTY",
    "direction": "LONG",
    "entry": 22000.0,
    "current_price": 22150.0,
    "stop_loss": 21800.0,
    "target": 22400.0,
    "unrealized_pnl_pct": 0.68,
    "time_in_trade_minutes": 30,
    "regime": "bull",
    "volume_ratio": 1.1,
    "price_vs_vwap": 0.003,
    "atr": 200.0,
    "momentum": 0.5,
}

_PRICE_REGIME_PAYLOAD = {
    "last_60_bars": [[22000.0, 22100.0, 21900.0, 22050.0, 1000000.0, 0.3, 15.0, 0.9, 50.0]] * 60,
}

_IV_REGIME_PAYLOAD = {
    "data": [[15.0, -0.5, 50000.0, 18.0, 0.5]] * 20,
}

_PORTFOLIO_V2_PAYLOAD = {
    "symbols": ["NIFTY", "BANKNIFTY", "RELIANCE", "HDFC", "TCS"],
    "method": "hrp",
}

_GREEKS_PAYLOAD = {
    "chain": [{"strike": 22000, "ce_ltp": 310.0, "pe_ltp": 150.0, "oi": 1000000}],
    "spot": 22150.0,
    "india_vix": 15.0,
    "expiry_dt": "2025-01-30T15:30:00",
}

_GEX_PAYLOAD = {
    "chain_snapshot": [{"strike": 22000, "ce_gamma": 0.002, "pe_gamma": 0.002, "ce_oi": 1000000, "pe_oi": 800000}],
    "spot": 22150.0,
    "symbol": "NIFTY",
}

_VPIN_PAYLOAD = {
    "symbol": "NIFTY",
    "bars": [{"open": 22000.0, "high": 22100.0, "low": 21900.0, "close": 22050.0, "volume": 100000}] * 50,
}

_VOL_SURFACE_PAYLOAD = {
    "symbol": "NIFTY",
    "snapshots_by_expiry": {
        "2025-01-30": {
            "strikes": [21000.0, 21500.0, 22000.0, 22500.0, 23000.0],
            "ivs": [0.18, 0.16, 0.15, 0.16, 0.18],
            "forward": 22000.0,
            "days_to_expiry": 15.0,
            "atm_iv": 0.15,
        }
    },
}


def _measure_latency_ms(test_client: Any, method: str, path: str, payload: dict) -> float:
    """Measure round-trip latency for a single request in milliseconds."""
    start = time.perf_counter()
    if method == "POST":
        test_client.post(path, json=payload, headers={"X-API-KEY": "test-key-for-testing"})
    else:
        test_client.get(path, headers={"X-API-KEY": "test-key-for-testing"})
    return (time.perf_counter() - start) * 1000


def _compute_p95(latencies: list[float]) -> float:
    """Compute the 95th percentile latency."""
    sorted_latencies = sorted(latencies)
    idx = int(len(sorted_latencies) * 0.95)
    return sorted_latencies[min(idx, len(sorted_latencies) - 1)]


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def latency_client():
    """TestClient for latency tests."""
    import os
    os.environ["ML_SERVICE_API_KEY"] = "test-key-for-testing"
    os.environ["DATA_SERVICE_API_KEY"] = "test-data-key"
    from src.main import app
    from fastapi.testclient import TestClient
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client


# ── Single-symbol endpoint latency tests (p95 ≤ 50ms) ───────────────────────

class TestSingleSymbolEndpointLatency:
    """
    Req 16.1: Single-symbol prediction endpoints must respond within 50ms at p95.

    At the stub stage, these tests verify:
    - Endpoints exist (not 404)
    - Latency measured (will be used to gate promotion once models are implemented)

    TDD: Tests will fail the p95 assertion until Phase 4 models are implemented.
    """

    SINGLE_SYMBOL_ENDPOINTS = [
        ("/v2/predict/regime", _REGIME_PAYLOAD),
        ("/v2/predict/strategy", _STRATEGY_PAYLOAD),
        ("/v2/predict/risk", _RISK_PAYLOAD),
        ("/v2/predict/execution", _EXECUTION_PAYLOAD),
        ("/v2/predict/price-regime", _PRICE_REGIME_PAYLOAD),
        ("/v2/predict/iv-regime", _IV_REGIME_PAYLOAD),
    ]

    @pytest.mark.parametrize("path,payload", SINGLE_SYMBOL_ENDPOINTS)
    def test_endpoint_exists_and_responds(self, latency_client, path: str, payload: dict) -> None:
        """All single-symbol endpoints must respond (not 404)."""
        from fastapi.testclient import TestClient
        r = latency_client.post(
            path,
            json=payload,
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code != 404, f"Endpoint {path} returned 404 — not registered"

    @pytest.mark.parametrize("path,payload", SINGLE_SYMBOL_ENDPOINTS)
    def test_endpoint_latency_under_500ms_stub_phase(
        self,
        latency_client,
        path: str,
        payload: dict,
    ) -> None:
        """
        At stub phase: latency must be under 500ms (generous bound for stubs).
        Will tighten to 50ms p95 once Phase 4 model implementations are complete.

        Note: The p95 ≤ 50ms assertion is a TDD RED test — intentionally commented
        out here and enabled in test_endpoint_p95_latency_sla below once models ship.
        """
        latencies = [
            _measure_latency_ms(latency_client, "POST", path, payload)
            for _ in range(10)
        ]
        p95 = _compute_p95(latencies)

        # Generous stub bound — stubs should be near instant
        assert p95 < 500, (
            f"Endpoint {path} p95 latency={p95:.1f}ms exceeds stub bound of 500ms. "
            "Check for blocking I/O in stub handler."
        )

    @pytest.mark.skip(reason="TDD RED: Enable once Phase 4 model implementations are complete")
    @pytest.mark.parametrize("path,payload", SINGLE_SYMBOL_ENDPOINTS)
    def test_endpoint_p95_latency_sla(
        self,
        latency_client,
        path: str,
        payload: dict,
    ) -> None:
        """
        Req 16.1: Single-symbol endpoints must respond within 50ms p95.

        TDD RED: This test is skipped until Phase 4 models are implemented.
        Unskip this test (remove the @pytest.mark.skip) after Phase 4 deployment.
        """
        latencies = [
            _measure_latency_ms(latency_client, "POST", path, payload)
            for _ in range(100)
        ]
        p95 = _compute_p95(latencies)

        assert p95 < 50.0, (
            f"LATENCY SLA VIOLATION: {path} p95={p95:.1f}ms exceeds 50ms target. "
            f"Median={median(latencies):.1f}ms, Std={stdev(latencies):.1f}ms"
        )


# ── Batch ranking endpoint latency (p95 ≤ 200ms) ─────────────────────────────

class TestBatchRankingLatency:
    """Req 16.2: Batch ranking for 200 symbols must respond within 200ms p95."""

    def _make_ranking_payload(self, n_symbols: int) -> dict:
        """Build a RankingRequest with n_symbols stock features."""
        stocks = []
        for i in range(n_symbols):
            stocks.append({
                "symbol": f"SYM{i:04d}",
                "relative_volume": 1.1 + i * 0.01,
                "atr_expansion": 1.05,
                "momentum_5d": 0.5,
                "momentum_10d": 0.3,
                "vwap_distance_pct": 0.2,
                "ema_stack_score": 0.6,
                "rsi_14": 55.0,
                "macd_histogram": 0.1,
                "adx_14": 22.0,
                "sector_momentum": 0.4,
                "relative_strength_vs_nifty": 1.05,
                "market_breadth": 0.6,
                "gap_pct": 0.1,
            })
        return {"stocks": stocks, "regime": "bull", "top_n": min(n_symbols, 20)}

    def test_batch_ranking_endpoint_exists(self, latency_client) -> None:
        """POST /v2/predict/rankings must exist (not 404)."""
        payload = self._make_ranking_payload(5)
        r = latency_client.post(
            "/v2/predict/rankings",
            json=payload,
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code != 404

    def test_batch_ranking_stub_latency(self, latency_client) -> None:
        """Batch ranking stub must respond within 1 second for 5-symbol batch."""
        payload = self._make_ranking_payload(5)
        latency = _measure_latency_ms(latency_client, "POST", "/v2/predict/rankings", payload)
        assert latency < 1000, f"Batch ranking stub latency={latency:.1f}ms > 1s"

    @pytest.mark.skip(reason="TDD RED: Enable once Phase 4 StockRanker is implemented")
    def test_batch_ranking_200_symbols_p95_sla(self, latency_client) -> None:
        """
        Req 16.2: Batch ranking for 200 symbols must respond within 200ms p95.

        TDD RED: Enable after Phase 4 StockRanker implementation.
        """
        payload = self._make_ranking_payload(200)
        latencies = [
            _measure_latency_ms(latency_client, "POST", "/v2/predict/rankings", payload)
            for _ in range(20)
        ]
        p95 = _compute_p95(latencies)
        assert p95 < 200.0, f"Batch ranking p95={p95:.1f}ms exceeds 200ms SLA"


# ── Meta-decision endpoint latency (p95 ≤ 150ms) ─────────────────────────────

class TestMetaDecisionLatency:
    """Req 16.3: Meta-decision endpoint must respond within 150ms p95."""

    _META_PAYLOAD = {
        "symbol": "NIFTY",
        "regime": "bull",
        "force_refresh_news": False,
    }

    def test_meta_decide_endpoint_exists(self, latency_client) -> None:
        """POST /v2/meta/decide must exist (not 404)."""
        r = latency_client.post(
            "/v2/meta/decide",
            json=self._META_PAYLOAD,
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code != 404

    @pytest.mark.skip(reason="TDD RED: Enable once Phase 10 MetaDecisionEngine is implemented")
    def test_meta_decide_p95_latency_sla(self, latency_client) -> None:
        """
        Req 16.3: Meta-decision must respond within 150ms p95 with cached news context.

        TDD RED: Enable after Phase 10 MetaDecisionEngine implementation.
        """
        latencies = [
            _measure_latency_ms(latency_client, "POST", "/v2/meta/decide", self._META_PAYLOAD)
            for _ in range(50)
        ]
        p95 = _compute_p95(latencies)
        assert p95 < 150.0, f"Meta-decision p95={p95:.1f}ms exceeds 150ms SLA"


# ── Portfolio optimization latency (p95 ≤ 500ms) ─────────────────────────────

class TestPortfolioOptimizationLatency:
    """Req 16.4: Portfolio optimization for ≤50 assets must respond within 500ms p95."""

    def test_portfolio_v2_endpoint_exists(self, latency_client) -> None:
        """POST /v2/predict/portfolio-v2 must exist (not 404)."""
        r = latency_client.post(
            "/v2/predict/portfolio-v2",
            json=_PORTFOLIO_V2_PAYLOAD,
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code != 404

    @pytest.mark.skip(reason="TDD RED: Enable once Phase 4 PortfolioOptimizer is implemented")
    def test_portfolio_50_assets_p95_latency_sla(self, latency_client) -> None:
        """
        Req 16.4: Portfolio optimization for 50 assets must respond within 500ms p95.

        TDD RED: Enable after Phase 4 PortfolioOptimizer implementation.
        """
        symbols = [f"SYM{i:03d}" for i in range(50)]
        payload = {"symbols": symbols, "method": "hrp"}

        latencies = [
            _measure_latency_ms(latency_client, "POST", "/v2/predict/portfolio-v2", payload)
            for _ in range(20)
        ]
        p95 = _compute_p95(latencies)
        assert p95 < 500.0, f"Portfolio optimization p95={p95:.1f}ms exceeds 500ms SLA"


# ── Analytics endpoint latency (p95 ≤ 100ms) ─────────────────────────────────

class TestAnalyticsEndpointLatency:
    """Req 16.5: Analytics endpoints must respond within 100ms p95."""

    ANALYTICS_ENDPOINTS = [
        ("/v2/analytics/greeks", _GREEKS_PAYLOAD),
        ("/v2/analytics/gex", _GEX_PAYLOAD),
        ("/v2/analytics/vpin", _VPIN_PAYLOAD),
        ("/v2/analytics/vol-surface", _VOL_SURFACE_PAYLOAD),
    ]

    @pytest.mark.parametrize("path,payload", ANALYTICS_ENDPOINTS)
    def test_analytics_endpoints_exist(self, latency_client, path: str, payload: dict) -> None:
        """All analytics endpoints must exist (not 404)."""
        r = latency_client.post(
            path,
            json=payload,
            headers={"X-API-KEY": "test-key-for-testing"},
        )
        assert r.status_code != 404

    @pytest.mark.skip(reason="TDD RED: Enable once Phase 5 analytics implementations are complete")
    @pytest.mark.parametrize("path,payload", ANALYTICS_ENDPOINTS)
    def test_analytics_p95_latency_sla(
        self,
        latency_client,
        path: str,
        payload: dict,
    ) -> None:
        """
        Req 16.5: Analytics endpoints must respond within 100ms p95.

        TDD RED: Enable after Phase 5 analytics implementation.
        """
        latencies = [
            _measure_latency_ms(latency_client, "POST", path, payload)
            for _ in range(50)
        ]
        p95 = _compute_p95(latencies)
        assert p95 < 100.0, f"Analytics endpoint {path} p95={p95:.1f}ms exceeds 100ms SLA"
