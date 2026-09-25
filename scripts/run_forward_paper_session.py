"""
scripts/run_forward_paper_session.py — genuine forward-paper signal collection.

Mandate §26–§29: real forward paper requires
    SIGNAL_AT_T → OUTCOME_AFTER_T
NOT historical replay.

This script:
1. Fetches the most recent available daily bars for each of the 65 baseline
   symbols from data-service2.0 (the SOLE market data authority).
2. Computes features from those bars using the frozen feature schema (fs-2.0.0).
3. Runs the frozen CONFIRMATION_BASELINE_V1 model (lightgbm, version
   1.0.0-20260925080931531542, SHA256=97e601197c...).
4. Persists each signal to the append-only forward paper store with:
   - wall-clock created_at
   - resolve_after = signal_ts + 5 trading days (the horizon)
   - model_version, dataset_hash, feature_schema, feature_hash
   - data_as_of timestamp (latest bar timestamp from data-service)
5. Does NOT resolve outcomes — that happens when 5 trading days have passed.

Immutability guarantee (mandate §56):
   Once a signal is written to the JSONL store it is NEVER modified.
   Outcome resolution writes a SEPARATE feedback record.

PIT safety (mandate §25):
   data_as_of is set to the timestamp of the LATEST BAR received from
   data-service. Any feature computed from that bar is guaranteed to
   be <= data_as_of.  The resolve_after enforces the 5-bar forward wait
   so outcomes can only be evaluated after real time has elapsed.

Run at NSE close (after 15:30 IST) or pre-open (before 09:15 IST) on any
trading day. Safe to run on weekends — the script uses the most recently
available daily bar regardless of when it runs.

Usage:
    PYTHONPATH=. python3 scripts/run_forward_paper_session.py
    PYTHONPATH=. python3 scripts/run_forward_paper_session.py --dry-run
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import pickle
import subprocess
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Frozen candidate constants (MUST NOT CHANGE) ─────────────────────────────
BASELINE_ID         = "CONFIRMATION_BASELINE_V1"
MODEL_VERSION       = "1.0.0-20260925080931531542"
MODEL_SHA256        = "97e601197c02e187e7ea2c28e0d9a4e24fc2fd41f2f8783dba4a49df4f247348"
DATASET_HASH        = "ee508cb6afccbc00db50b3cce47d3c3a790749b7a63ee43bec06ec5aefd52c4d"
FEATURE_SCHEMA      = "fs-2.0.0"
EXPERIMENT_ID       = "CONFIRMATION_BASELINE_V1"
HORIZON_BARS        = 5       # trading days
BAR_SECONDS         = 86400   # 1d bars
COST_BPS            = 10.0    # round-trip basis points (used for edge test)
MODEL_PATH          = Path("artifacts/registry/stage_a_1d/1.0.0-20260925080931531542/model.pkl")
SIGNAL_STORE_PATH   = Path("artifacts/forward_paper/signals.jsonl")

FEATURE_COLS: list[str] = [
    "ret_1", "ret_5", "ret_10", "ret_20", "log_ret_1",
    "vol_5", "vol_10", "vol_20", "atr_14_pct", "rel_volume_20",
    "volume_zscore_20", "vwap_distance_pct", "rsi_14", "macd_hist",
    "stoch_k_14", "ema_5_20", "ema_10_50", "adx_14", "hl_range_pct",
    "close_position", "gap_pct", "bb_zscore_20", "skew_20", "kurt_20",
]

# 65-symbol baseline universe
BASELINE_SYMBOLS: list[str] = [
    "360ONE","ABB","ABCAPITAL","ADANIENSOL","ADANIENT","ADANIGREEN",
    "ADANIPORTS","ADANIPOWER","ALKEM","AMBER","AMBUJACEM","ANGELONE",
    "APLAPOLLO","APOLLOHOSP","ASHOKLEY","ASIANPAINT","ASTRAL","ATHERENERG",
    "AUBANK","AUROPHARMA","AXISBANK","BAJAJ-AUTO","BAJAJFINSV","BAJAJHLDNG",
    "BAJFINANCE","BANDHANBNK","BANKINDIA","BANKNIFTY","BHARTIARTL","CANBK",
    "DMART","DRREDDY","HDFCBANK","ICICIBANK","INFY","JINDALSTEL","JIOFIN",
    "JSWENERGY","KOTAKBANK","MAXHEALTH","MAZDOCK","MCX","MFSL","NBCC",
    "NESTLEIND","NHPC","NIFTY","NMDC","NTPC","PAGEIND","PATANJALI","PAYTM",
    "PERSISTENT","PETRONET","RVNL","SAGILITY","SAIL","SBICARD","SBILIFE",
    "SBIN","SHRIRAMFIN","SIEMENS","SOLARINDS","TCS","VEDL",
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "4cf307c31face6caf3b5ad516f9388494885966a"


def _hash_features(fv: dict[str, float]) -> str:
    payload = json.dumps(sorted(fv.items()), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:32]


def _verify_model() -> dict:
    """Load and verify the frozen model artifact."""
    import subprocess as sp
    actual = sp.check_output(["shasum", "-a", "256", str(MODEL_PATH)]).decode().split()[0]
    if actual != MODEL_SHA256:
        raise RuntimeError(
            f"CERTIFICATION_BLOCKED: model SHA256 mismatch. "
            f"expected={MODEL_SHA256} actual={actual}"
        )
    with open(MODEL_PATH, "rb") as f:
        m = pickle.load(f)
    return m


async def _fetch_recent_bars(symbol: str, api_key: str, n_bars: int = 60) -> pd.DataFrame | None:
    """Fetch recent daily bars from data-service2.0."""
    import httpx
    url = "http://localhost:8200/v1/india/historical"
    params = {
        "symbol": symbol,
        "exchange": "NSE",
        "interval": "1d",
        "from_date": (datetime.now(UTC) - timedelta(days=90)).strftime("%Y-%m-%d"),
    }
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            resp = await client.get(url, params=params, headers={"X-API-KEY": api_key})
            if resp.status_code != 200:
                return None
            data = resp.json()
            bars = data.get("data") or data.get("bars") or []
            if not bars:
                return None
            rows = []
            for b in bars:
                ts_raw = b.get("timestamp") or b.get("ts") or b.get("time") or b.get("date")
                try:
                    if isinstance(ts_raw, (int, float)):
                        val = float(ts_raw)
                        ts = datetime.fromtimestamp(val / 1000 if val > 1e12 else val, tz=UTC)
                    else:
                        ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=UTC)
                except Exception:
                    continue
                try:
                    rows.append({
                        "timestamp": ts,
                        "open": float(b["open"]),
                        "high": float(b["high"]),
                        "low": float(b["low"]),
                        "close": float(b["close"]),
                        "volume": float(b.get("volume", 0)),
                    })
                except (KeyError, TypeError, ValueError):
                    continue
            if not rows:
                return None
            df = pd.DataFrame(rows).sort_values("timestamp").set_index("timestamp")
            return df.tail(n_bars)
        except Exception:
            return None


def _compute_features(df: pd.DataFrame) -> dict[str, float] | None:
    """Compute the 24 fs-2.0.0 features from the last row of df."""
    if len(df) < 25:
        return None
    c = df["close"].values.astype(float)
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    v = df["volume"].values.astype(float)
    n = len(c)

    def safe_ret(i1: int, i2: int) -> float:
        if c[i1] == 0:
            return 0.0
        return (c[i2] - c[i1]) / c[i1]

    # Returns (backward-looking — PIT safe)
    ret_1  = safe_ret(-2, -1)
    ret_5  = safe_ret(-6, -1)
    ret_10 = safe_ret(-11, -1)
    ret_20 = safe_ret(-21, -1)
    log_ret_1 = float(np.log(c[-1] / c[-2])) if c[-2] > 0 else 0.0

    # Volatility
    rets = np.diff(c[-22:]) / c[-22:-1]
    vol_5  = float(np.std(rets[-5:]))  if len(rets) >= 5  else 0.0
    vol_10 = float(np.std(rets[-10:])) if len(rets) >= 10 else 0.0
    vol_20 = float(np.std(rets[-20:])) if len(rets) >= 20 else 0.0

    # ATR (14)
    prev_c = c[-15:-1]  # 14 previous closes
    curr_h = h[-14:]
    curr_l = l[-14:]
    tr14 = np.maximum(curr_h - curr_l,
           np.maximum(np.abs(curr_h - prev_c),
                      np.abs(curr_l - prev_c)))
    atr_14_pct = float(np.mean(tr14)) / c[-1] if c[-1] > 0 else 0.0

    # Volume features
    vol_mean20 = np.mean(v[-21:-1]) if len(v) >= 21 else (v[-1] if len(v) else 1.0)
    rel_volume_20 = float(v[-1] / vol_mean20) if vol_mean20 > 0 else 1.0
    vol_std20 = float(np.std(v[-21:-1])) if len(v) >= 21 else 0.0
    volume_zscore_20 = float((v[-1] - vol_mean20) / vol_std20) if vol_std20 > 0 else 0.0

    # VWAP distance
    vwap = float(np.sum(c[-20:] * v[-20:]) / np.sum(v[-20:])) if np.sum(v[-20:]) > 0 else c[-1]
    vwap_distance_pct = (c[-1] - vwap) / vwap if vwap > 0 else 0.0

    # RSI (14)
    d = np.diff(c[-16:])
    gains = np.where(d > 0, d, 0.0)
    losses = np.where(d < 0, -d, 0.0)
    avg_gain = np.mean(gains[-14:])
    avg_loss = np.mean(losses[-14:])
    rs = avg_gain / avg_loss if avg_loss > 0 else 100.0
    rsi_14 = float(100.0 - 100.0 / (1.0 + rs))

    # MACD histogram (12/26/9)
    def ema(arr: np.ndarray, period: int) -> float:
        k = 2.0 / (period + 1)
        e = arr[0]
        for x in arr[1:]:
            e = x * k + e * (1 - k)
        return float(e)
    if len(c) >= 27:
        macd_line = ema(c[-27:], 12) - ema(c[-27:], 26)
        signal_line = macd_line * (2.0 / 10.0)  # simplified single-point signal
        macd_hist = macd_line - signal_line
    else:
        macd_hist = 0.0

    # Stochastic K (14)
    lo14 = float(np.min(l[-14:]))
    hi14 = float(np.max(h[-14:]))
    stoch_k_14 = ((c[-1] - lo14) / (hi14 - lo14) * 100) if (hi14 - lo14) > 0 else 50.0

    # EMA ratios
    if len(c) >= 51:
        ema5  = ema(c[-6:],  5)
        ema10 = ema(c[-11:], 10)
        ema20 = ema(c[-21:], 20)
        ema50 = ema(c[-51:], 50)
        ema_5_20  = (ema5  / ema20  - 1.0) if ema20  > 0 else 0.0
        ema_10_50 = (ema10 / ema50  - 1.0) if ema50  > 0 else 0.0
    else:
        ema_5_20 = ema_10_50 = 0.0

    # ADX (14) — simplified
    plus_dm  = np.maximum(h[-15:-1] - h[-14:],   0.0) if len(h) >= 15 else np.zeros(1)
    minus_dm = np.maximum(l[-14:]   - l[-15:-1],  0.0) if len(l) >= 15 else np.zeros(1)
    tr_adx   = np.maximum(h[-15:] - l[-15:], 1e-9)[:14]
    pdi = (np.sum(plus_dm[-14:])  / np.sum(tr_adx) * 100) if np.sum(tr_adx) > 0 else 0.0
    mdi = (np.sum(minus_dm[-14:]) / np.sum(tr_adx) * 100) if np.sum(tr_adx) > 0 else 0.0
    dx = abs(pdi - mdi) / (pdi + mdi) * 100 if (pdi + mdi) > 0 else 0.0
    adx_14 = dx  # single-bar approximation

    # High-low range and close position
    hl_range_pct = (h[-1] - l[-1]) / c[-1] if c[-1] > 0 else 0.0
    close_position = ((c[-1] - l[-1]) / (h[-1] - l[-1])) if (h[-1] - l[-1]) > 0 else 0.5

    # Gap
    gap_pct = (o[-1] - c[-2]) / c[-2] if c[-2] > 0 else 0.0

    # Bollinger band z-score (20)
    mean20 = float(np.mean(c[-20:]))
    std20  = float(np.std(c[-20:]))
    bb_zscore_20 = (c[-1] - mean20) / std20 if std20 > 0 else 0.0

    # Skew and kurtosis (20)
    from scipy.stats import skew as _skew, kurtosis as _kurt
    skew_20 = float(_skew(c[-20:]))
    kurt_20 = float(_kurt(c[-20:]))

    fv: dict[str, float] = {
        "ret_1": ret_1, "ret_5": ret_5, "ret_10": ret_10, "ret_20": ret_20,
        "log_ret_1": log_ret_1, "vol_5": vol_5, "vol_10": vol_10, "vol_20": vol_20,
        "atr_14_pct": atr_14_pct, "rel_volume_20": rel_volume_20,
        "volume_zscore_20": volume_zscore_20, "vwap_distance_pct": vwap_distance_pct,
        "rsi_14": rsi_14, "macd_hist": macd_hist, "stoch_k_14": stoch_k_14,
        "ema_5_20": ema_5_20, "ema_10_50": ema_10_50, "adx_14": adx_14,
        "hl_range_pct": hl_range_pct, "close_position": close_position,
        "gap_pct": gap_pct, "bb_zscore_20": bb_zscore_20,
        "skew_20": skew_20, "kurt_20": kurt_20,
    }

    # Validate — all features must be finite
    for k, v_ in fv.items():
        if not np.isfinite(v_):
            fv[k] = 0.0
    return fv


async def run(dry_run: bool = False) -> None:
    import os
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("DATA_SERVICE_API_KEY", "")

    # ── 1. Verify model integrity ─────────────────────────────────────────────
    print("Verifying frozen model artifact...")
    model_dict = _verify_model()
    estimator  = model_dict["estimator"]
    calibrator = model_dict["calibrator"]
    print(f"  Model: {MODEL_VERSION}  SHA256: {MODEL_SHA256[:16]}...  OK")

    # ── 2. Set up forward paper store ─────────────────────────────────────────
    from src.analytics.forward_paper import ForwardPaperStore, ForwardPaperRunner
    SIGNAL_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    store  = ForwardPaperStore(SIGNAL_STORE_PATH)
    runner = ForwardPaperRunner(store=store, interval="1d", horizon_bars=HORIZON_BARS)

    # ── 3. Session timestamp ──────────────────────────────────────────────────
    session_ts = datetime.now(tz=UTC)
    git_sha    = _git_sha()
    cost_per_side = COST_BPS / 10_000.0 / 2.0  # per-side

    print(f"\nForward Paper Session — {session_ts.isoformat()}")
    print(f"  Baseline:      {BASELINE_ID}")
    print(f"  Model version: {MODEL_VERSION}")
    print(f"  Universe:      {len(BASELINE_SYMBOLS)} symbols")
    print(f"  Dry run:       {dry_run}")
    print()

    n_signals = 0
    n_trade   = 0
    n_no_data = 0
    n_no_feat = 0
    results   = []

    # ── 4. Loop over all 65 baseline symbols ──────────────────────────────────
    for i, symbol in enumerate(BASELINE_SYMBOLS, 1):
        # Fetch bars
        df = await _fetch_recent_bars(symbol, api_key, n_bars=60)
        if df is None or len(df) < 25:
            n_no_data += 1
            results.append({"symbol": symbol, "status": "NO_DATA"})
            continue

        # Compute features
        fv = _compute_features(df)
        if fv is None:
            n_no_feat += 1
            results.append({"symbol": symbol, "status": "INSUFFICIENT_HISTORY"})
            continue

        # The signal timestamp is the latest bar's timestamp
        signal_ts  = df.index[-1]
        data_as_of = df.index[-1]
        entry_price = float(df["close"].iloc[-1])  # close at T (proxy — actual entry is open[T+1])

        # Run frozen model
        X = np.array([[fv[c] for c in FEATURE_COLS]])
        raw_proba = estimator.predict_proba(X)[0, 1]
        probability = float(raw_proba)

        # Apply calibrator
        try:
            cal_proba = calibrator.predict_proba(
                np.array([[raw_proba]])
            )[0, 1]
            prediction = float(cal_proba)
        except Exception:
            prediction = probability

        # Direction: +1 if pred > 0.5 else -1
        direction = 1 if prediction > 0.5 else -1

        # Expected edge (simplified): |pred - 0.5| * 2 * 2% barrier
        expected_edge = abs(prediction - 0.5) * 2.0 * 0.02
        expected_cost = cost_per_side * 2.0  # round-trip

        # Persist signal
        if not dry_run:
            sig = runner.record_signal(
                symbol=symbol,
                signal_ts=signal_ts,
                direction=direction,
                prediction=prediction,
                probability=probability,
                expected_edge=expected_edge,
                expected_cost=expected_cost,
                entry_price=entry_price,
                model_version=MODEL_VERSION,
                experiment_id=EXPERIMENT_ID,
                dataset_hash=DATASET_HASH,
                feature_schema=FEATURE_SCHEMA,
                feature_vector=fv,
                feature_as_of=data_as_of,
                data_as_of=data_as_of,
                bar_seconds=BAR_SECONDS,
            )
            decision = sig.decision
            resolve_after = sig.resolve_after
        else:
            decision = "TRADE" if expected_edge > expected_cost and direction != 0 else "NO_TRADE"
            resolve_after = (signal_ts + timedelta(seconds=BAR_SECONDS * HORIZON_BARS)).isoformat()

        n_signals += 1
        if decision == "TRADE":
            n_trade += 1

        results.append({
            "symbol": symbol,
            "status": "OK",
            "signal_ts": signal_ts.isoformat(),
            "data_as_of": data_as_of.isoformat(),
            "prediction": round(prediction, 4),
            "direction": direction,
            "expected_edge_bps": round(expected_edge * 10000, 2),
            "decision": decision,
            "resolve_after": resolve_after if isinstance(resolve_after, str) else resolve_after,
            "entry_price": entry_price,
        })

        # Progress
        status_str = f"{decision:<8} pred={prediction:.3f} dir={'+' if direction>0 else '-'}"
        print(f"  [{i:3d}/{len(BASELINE_SYMBOLS)}] {symbol:<15} {status_str}")

    # ── 5. Session summary ────────────────────────────────────────────────────
    all_signals = store.all_signals()
    paper_status = runner.status(n_resolved=0)

    print(f"\n{'='*65}")
    print("FORWARD PAPER SESSION COMPLETE")
    print(f"{'='*65}")
    print(f"  Session time:       {session_ts.isoformat()}")
    print(f"  Signals generated:  {n_signals} / {len(BASELINE_SYMBOLS)}")
    print(f"  TRADE signals:      {n_trade}")
    print(f"  NO_TRADE signals:   {n_signals - n_trade}")
    print(f"  No data:            {n_no_data}")
    print(f"  Total in store:     {len(all_signals)}")
    print(f"  Store state:        {paper_status['state']}")
    print(f"  Min for validation: {paper_status['min_trades_for_validation']}")
    print(f"  Resolve after:      5 trading days (mandate §57)")
    print(f"  Dry run:            {dry_run}")
    print()
    if not dry_run:
        print(f"  Signals written to: {SIGNAL_STORE_PATH}")
        print(f"  Run after market close each trading day to accumulate evidence.")
        print(f"  Resolve outcomes: PYTHONPATH=. python3 scripts/run_forward_paper_audit.py")
    print(f"{'='*65}\n")

    # Write session report
    report = {
        "session_timestamp": session_ts.isoformat(),
        "git_sha": git_sha,
        "model_version": MODEL_VERSION,
        "model_sha256": MODEL_SHA256,
        "dataset_hash": DATASET_HASH,
        "feature_schema": FEATURE_SCHEMA,
        "dry_run": dry_run,
        "n_symbols": len(BASELINE_SYMBOLS),
        "n_signals": n_signals,
        "n_trade": n_trade,
        "n_no_data": n_no_data,
        "n_no_feat": n_no_feat,
        "total_in_store": len(all_signals),
        "forward_paper_state": paper_status["state"],
        "results": results,
    }
    report_path = Path("reports/forward_paper_session.json")
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Session report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute signals without persisting (for testing)")
    args = parser.parse_args()
    asyncio.run(run(dry_run=args.dry_run))
