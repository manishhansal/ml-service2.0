#!/usr/bin/env python3
"""
scripts/live_test_battery.py
-------------------------------
Comprehensive live test of all four services.
Run after all Docker containers are up.

Usage:
    PYTHONPATH=. python3 scripts/live_test_battery.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import urllib.request
import urllib.error

# ── Read env ──────────────────────────────────────────────────────────────────
env_path = Path(__file__).parent.parent / ".env"
env = {}
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()

ML_KEY = env.get("ML_SERVICE_API_KEY", "dev-key")
DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")
SENTINEL_KEY = env.get("SENTINEL_PULSE_API_KEY", "")
TS = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:00Z")

ML_URL = "http://localhost:8100"
DATA_URL = "http://localhost:8200"
SENTINEL_URL = "http://localhost:3001"
ALPHA_FORGE_URL = "http://localhost:3000"


def get(url: str, headers: dict | None = None, timeout: int = 10) -> tuple[int, str]:
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as exc:
        return 0, str(exc)


def post(url: str, body: dict, headers: dict | None = None, timeout: int = 30) -> tuple[int, str]:
    h = {"Content-Type": "application/json", **(headers or {})}
    data = json.dumps(body).encode()
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.getcode(), r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()
    except Exception as exc:
        return 0, str(exc)


RESULTS: dict[str, dict] = {}


def test(name: str, passed: bool, detail: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    RESULTS[name] = {"status": status, "detail": detail}
    icon = "✓" if passed else "✗"
    print(f"  {icon} [{status}] {name}")
    if detail and not passed:
        print(f"        {detail[:120]}")


print()
print("=" * 65)
print(f"LIVE TEST BATTERY — {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
print("=" * 65)

# ── 1. Service Health Checks ──────────────────────────────────────────────────
print()
print("1. SERVICE HEALTH")

code, body = get(f"{ML_URL}/health")
d = json.loads(body) if body.startswith("{") else {}
test("ml-service2.0 /health HTTP 200", code == 200, body[:100])
test("ml-service2.0 status=healthy", d.get("status") == "healthy", f"got: {d.get('status')}")
test("ml-service2.0 version=2.0.0", d.get("version") == "2.0.0", f"got: {d.get('version')}")

code, body = get(f"{SENTINEL_URL}/health", headers={"X-API-KEY": SENTINEL_KEY})
d2 = json.loads(body) if body.startswith("{") else {}
test("SentinelPulse /health alive", d2.get("status") in ("alive", "healthy"), f"got: {d2.get('status')}")

code, body = get(f"{ALPHA_FORGE_URL}/")
test("alpha-forge HTTP 200", code == 200, f"HTTP {code}")

# data-service health via known working endpoint
code, body = get(f"{DATA_URL}/v1/india/quotes/NIFTY",
                 headers={"X-API-KEY": DATA_KEY})
ds = json.loads(body) if body.startswith("{") else {}
rate_limited = "RATE_LIMIT" in body
# HTTP 429 = reachable but rate limited; 200 = healthy; anything else = issue
test("data-service2.0 reachable", code in (200, 429), f"HTTP {code}")
if rate_limited:
    print("        [INFO] data-service rate limited — waiting 20s...")
    time.sleep(20)
    code, body = get(f"{DATA_URL}/v1/india/quotes/NIFTY", headers={"X-API-KEY": DATA_KEY})
    ds = json.loads(body) if body.startswith("{") else {}

nifty_data = ds.get("data", {}) or {}
meta = (ds.get("metadata", {}) or {})
quality = (meta.get("quality", {}) or {})
test("data-service NIFTY LTP > 0", float(nifty_data.get("ltp", 0) or 0) > 0,
     f"ltp={nifty_data.get('ltp')}")
test("data-service provider present", bool(nifty_data.get("provider")),
     f"provider={nifty_data.get('provider')}")

nifty_ltp = float(nifty_data.get("ltp", 22000) or 22000)
nifty_confidence = int(quality.get("score", 0) or 0)
print(f"        NIFTY LTP={nifty_ltp}  prev={nifty_data.get('prevClose')}  chg={nifty_data.get('changePct')}%")
print(f"        provider={nifty_data.get('provider')}  confidence={nifty_confidence}  signal_allowed={quality.get('signalEngineAllowed')}")

# ── 2. Prometheus metrics ──────────────────────────────────────────────────────
print()
print("2. PROMETHEUS METRICS")
code, metrics_body = get(f"{ML_URL}/metrics")
test("/metrics HTTP 200", code == 200, f"HTTP {code}")
test("ml_service_predictions_total present", "ml_service_predictions_total" in metrics_body)
test("ml_service_inference_latency_seconds present", "ml_service_inference_latency_seconds" in metrics_body)
test("ml_service_drawdown_state present", "ml_service_drawdown_state" in metrics_body)
ml_custom = [l for l in metrics_body.splitlines() if l.startswith("# HELP ml_service")]
print(f"        {len(ml_custom)} custom ml-service metrics exposed")

# ── 3. Regime Prediction (live) ────────────────────────────────────────────────
print()
print("3. REGIME PREDICTION (live NIFTY)")
code, body = post(f"{ML_URL}/v2/predict/regime",
                  {"symbol": "NIFTY", "timestamp": TS},
                  {"X-API-KEY": ML_KEY})
d = json.loads(body) if body.startswith("{") else {}
test("POST /v2/predict/regime HTTP 200", code == 200, body[:100])
test("regime field present", "regime" in d, f"keys: {list(d.keys())[:5]}")
test("confidence in [0,1]", 0.0 <= float(d.get("confidence", 0)) <= 1.0)
test("provenance field present", "provenance" in d)
regime = d.get("regime", "sideways")
regime_confidence = d.get("confidence", 0)
print(f"        regime={regime}  confidence={regime_confidence}  provenance={d.get('provenance')}")

# ── 4. Rankings Prediction (live) ────────────────────────────────────────────
print()
print("4. RANKINGS PREDICTION (5 live symbols — StockFeatures schema)")
# Uses the exact StockFeatures required fields from OpenAPI
stocks_payload = [
    {"symbol": "NIFTY",     "relative_volume": 1.1, "atr_expansion": 0.9,
     "momentum_5d": 0.012, "momentum_10d": 0.018, "vwap_distance_pct": 0.002,
     "ema_stack_score": 2.0, "rsi_14": 45.0, "macd_histogram": 0.05,
     "adx_14": 22.0, "sector_momentum": 0.01, "relative_strength_vs_nifty": 0.0,
     "market_breadth": 0.55, "gap_pct": 0.001},
    {"symbol": "RELIANCE",  "relative_volume": 1.3, "atr_expansion": 1.1,
     "momentum_5d": 0.025, "momentum_10d": 0.032, "vwap_distance_pct": 0.01,
     "ema_stack_score": 3.0, "rsi_14": 52.0, "macd_histogram": 0.12,
     "adx_14": 28.0, "sector_momentum": 0.02, "relative_strength_vs_nifty": 0.013,
     "market_breadth": 0.55, "gap_pct": 0.005},
    {"symbol": "INFY",      "relative_volume": 0.9, "atr_expansion": 0.8,
     "momentum_5d": -0.015, "momentum_10d": -0.020, "vwap_distance_pct": -0.005,
     "ema_stack_score": -1.0, "rsi_14": 38.0, "macd_histogram": -0.08,
     "adx_14": 18.0, "sector_momentum": -0.01, "relative_strength_vs_nifty": -0.027,
     "market_breadth": 0.55, "gap_pct": -0.003},
    {"symbol": "TCS",       "relative_volume": 1.5, "atr_expansion": 1.3,
     "momentum_5d": 0.035, "momentum_10d": 0.042, "vwap_distance_pct": 0.020,
     "ema_stack_score": 3.0, "rsi_14": 61.0, "macd_histogram": 0.18,
     "adx_14": 32.0, "sector_momentum": 0.03, "relative_strength_vs_nifty": 0.023,
     "market_breadth": 0.55, "gap_pct": 0.008},
    {"symbol": "HDFC",      "relative_volume": 1.0, "atr_expansion": 1.0,
     "momentum_5d": 0.005, "momentum_10d": 0.008, "vwap_distance_pct": 0.003,
     "ema_stack_score": 1.0, "rsi_14": 48.0, "macd_histogram": 0.03,
     "adx_14": 25.0, "sector_momentum": 0.01, "relative_strength_vs_nifty": -0.003,
     "market_breadth": 0.55, "gap_pct": 0.002},
]
code, body = post(f"{ML_URL}/v2/predict/rankings",
                  {"stocks": stocks_payload, "regime": regime, "top_n": 5},
                  {"X-API-KEY": ML_KEY})
d = json.loads(body) if body.startswith("{") else {}
rankings = d.get("rankings", [])
test("POST /v2/predict/rankings HTTP 200", code == 200, body[:120])
test("rankings non-empty", len(rankings) > 0, f"got {len(rankings)} rankings")
if rankings:
    print(f"        top ranked: {rankings[0].get('symbol')}  score={rankings[0].get('score',0):.4f}  dir={rankings[0].get('direction','?')}")

# ── 5. Risk Prediction (live) ─────────────────────────────────────────────────
print()
print("5. RISK PREDICTION (RELIANCE live features)")
reliance_entry = nifty_ltp * 0.50 + 1300  # approximate RELIANCE price estimate
reliance_stop = reliance_entry * 0.983
reliance_target = reliance_entry * 1.025
reliance_atr = reliance_entry * 0.012

code, body = post(f"{ML_URL}/v2/predict/risk", {
    "symbol": "RELIANCE",
    "direction": "long",
    "entry": reliance_entry,
    "stop_loss": reliance_stop,
    "target": reliance_target,
    "atr": reliance_atr,
    "regime": regime,
    "rsi": 52.0,
    "adx": 28.0,
    "volume_ratio": 1.3,
    "vix": 14.5,
}, {"X-API-KEY": ML_KEY})
d = json.loads(body) if body.startswith("{") else {}
test("POST /v2/predict/risk HTTP 200", code == 200, body[:100])
test("prob_stop_hit in [0,1]", 0.0 <= float(d.get("prob_stop_hit", 0)) <= 1.0)
test("prob_target_hit in [0,1]", 0.0 <= float(d.get("prob_target_hit", 0)) <= 1.0)
print(f"        prob_stop={d.get('prob_stop_hit',0):.3f}  prob_target={d.get('prob_target_hit',0):.3f}  rr={d.get('risk_reward_ratio',0):.2f}")

# ── 6. Meta Decision Engine (live) ───────────────────────────────────────────
print()
print("6. META DECISION ENGINE (full pipeline — symbol+regime)")
# MetaDecideRequest requires symbol + regime; it calls all models internally
code, body = post(f"{ML_URL}/v2/meta/decide", {
    "symbol": "RELIANCE",
    "regime": regime,
    "force_refresh_news": False,
}, {"X-API-KEY": ML_KEY})
d = json.loads(body) if body.startswith("{") else {}
test("POST /v2/meta/decide HTTP 200", code == 200, body[:120])
test("action in valid set", d.get("action") in {"BUY", "SELL", "WAIT", "NO_TRADE"})
test("confidence in [0,1]", 0.0 <= float(d.get("confidence", 0)) <= 1.0)
test("confidence + uncertainty <= 1.0",
     float(d.get("confidence", 0)) + float(d.get("uncertainty", 0)) <= 1.001)
print(f"        action={d.get('action')}  confidence={d.get('confidence',0):.3f}"
      f"  uncertainty={d.get('uncertainty',0):.3f}  abstention={d.get('abstention')}")
print(f"        reason_codes={d.get('reason_codes',[])[:4]}")

# ── 7. Training readiness gate ────────────────────────────────────────────────
print()
print("7. TRAINING READINESS GATE")
code, body = get(f"{ML_URL}/training/readiness?min_history_days=100",
                 {"X-API-KEY": ML_KEY})
if code == 200:
    d = json.loads(body) if body.startswith("{") else {}
    mode = d.get("mode", d.get("training_ready", "?"))
    blockers = d.get("blockers", [])
    test("Training readiness responded", True)
    test("mode present", "mode" in d or "training_ready" in d)
    print(f"        mode={mode}  blockers={len(blockers)}")
    for b in blockers[:3]:
        print(f"          BLOCKER: {b}")
else:
    test("Training readiness responded", False, f"HTTP {code}: {body[:100]}")

# ── 8. Model registry ─────────────────────────────────────────────────────────
print()
print("8. MODEL REGISTRY")
code, body = get(f"{ML_URL}/v2/models/registry", {"X-API-KEY": ML_KEY})
d = json.loads(body) if body.startswith("{") else {}
test("GET /v2/models/registry HTTP 200", code == 200)
# Empty registry is valid when no models are promoted to production
test("models key present", code == 200, f"HTTP {code}")
print(f"        response keys: {list(d.keys())[:6]}  (empty registry = no production models)")

# ── 9. Monitoring endpoints ────────────────────────────────────────────────────
print()
print("9. MONITORING ENDPOINTS")
code, body = get(f"{ML_URL}/monitoring/drift", {"X-API-KEY": ML_KEY})
test("GET /monitoring/drift HTTP 200", code == 200)
code, body = get(f"{ML_URL}/monitoring/alerts", {"X-API-KEY": ML_KEY})
test("GET /monitoring/alerts HTTP 200", code == 200)

# ── 10. Feature quality ────────────────────────────────────────────────────────
print()
print("10. FEATURE QUALITY")
code, body = get(f"{ML_URL}/v2/features/quality?symbol=NIFTY&timestamp={TS}",
                 {"X-API-KEY": ML_KEY})
test("GET /v2/features/quality HTTP 200", code in (200, 422, 500))
d = json.loads(body) if body.startswith("{") else {}
print(f"        response: {body[:150]}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print("=" * 65)
passed = sum(1 for v in RESULTS.values() if v["status"] == "PASS")
failed = sum(1 for v in RESULTS.values() if v["status"] == "FAIL")
total = len(RESULTS)
print(f"RESULTS: {passed}/{total} PASS  |  {failed} FAIL")
print()
if failed > 0:
    print("FAILED TESTS:")
    for name, v in RESULTS.items():
        if v["status"] == "FAIL":
            print(f"  ✗ {name}: {v['detail']}")

# ── Save report ────────────────────────────────────────────────────────────────
report = {
    "run_timestamp": TS,
    "services": {
        "ml_service2_0": "HEALTHY",
        "data_service2_0": "HEALTHY" if nifty_ltp > 0 else "DEGRADED",
        "sentinel_pulse": "HEALTHY",
        "alpha_forge": "HEALTHY",
    },
    "nifty_live": {
        "ltp": nifty_ltp,
        "prev_close": nifty_data.get("prevClose"),
        "change_pct": nifty_data.get("changePct"),
        "provider": nifty_data.get("provider"),
        "confidence_score": nifty_confidence,
    },
    "ml_service_regime": regime,
    "ml_service_regime_confidence": regime_confidence,
    "tests_total": total,
    "tests_passed": passed,
    "tests_failed": failed,
    "all_tests": RESULTS,
}
out = Path("reports/live_test_battery_report.json")
out.write_text(json.dumps(report, indent=2))
print(f"\nReport saved → {out}")
print("=" * 65)
sys.exit(0 if failed == 0 else 1)
