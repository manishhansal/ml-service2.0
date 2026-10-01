#!/usr/bin/env python3
"""
autorun_till_close.py
=====================
Single unified automation that runs from NOW until NSE market close (15:30 IST).

Every 5 minutes:
  • Fetches live quotes for all 218 symbols
  • Scores all 218 with LightGBM fs-3.0.0
  • Calculates live P&L for all forward paper positions
  • Appends a structured JSON record to the session log
  • Prints a clean dashboard to stdout

At 15:30 IST (market close):
  1. Final quote snapshot
  2. Ingest Sep-28 close bars into all parquets
  3. Resolve all due forward paper signals
  4. Run SignalPromotionEngine on resolved outcomes
  5. Update LIVE_SESSION_REPORT.md with final numbers
  6. Print comprehensive end-of-day summary

Usage:
    PYTHONPATH=. python3 -W ignore scripts/autorun_till_close.py
"""
from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import time
import urllib.request
import urllib.error
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

warnings.filterwarnings("ignore")

# Silence all debug/info from structlog/src modules BEFORE importing anything from src
import logging
logging.disable(logging.WARNING)
for name in ("structlog", "src", "root", ""):
    logging.getLogger(name).setLevel(logging.CRITICAL)
os.environ["LOG_LEVEL"] = "ERROR"

# Patch structlog to suppress debug/info
try:
    import structlog
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
    )
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
import numpy as np
import pandas as pd

# ── Phil-inspired integrations ────────────────────────────────────────────────
# Imported lazily inside functions to avoid startup cost; explicit imports here
# so linters and type checkers see them and to document the dependency surface.
try:
    from src.analytics.forecast_ledger import ForecastLedger
    from src.analytics.counterfactual_ledger import CounterfactualLedger
    from src.analytics.score_threshold_sweep import ScoreThresholdSweep
    from src.analytics.feature_weight_manager import FeatureWeightManager
    from src.analytics.reversal_detector import ReversalDetector
    _PHIL_IMPORTS_OK = True
except Exception as _e:
    _PHIL_IMPORTS_OK = False
    print(f"[warn] Phil integrations not available: {_e}", file=sys.stderr)

# ── SentinelPulse news damper ─────────────────────────────────────────────────
try:
    from src.clients.sentinel_pulse import SentinelPulseClient
    from src.features.families.news import news_features_from_context
    _SENTINEL_OK = True
except Exception as _se:
    _SENTINEL_OK = False
    print(f"[warn] SentinelPulse not available: {_se}", file=sys.stderr)

# ── Symbol IC Tracker ─────────────────────────────────────────────────────────
try:
    from src.analytics.symbol_ic_tracker import SymbolICTracker
    _symbol_ic_tracker = SymbolICTracker()
    _IC_TRACKER_OK = True
except Exception as _ite:
    _symbol_ic_tracker = None
    _IC_TRACKER_OK = False
    print(f"[warn] SymbolICTracker not available: {_ite}", file=sys.stderr)

# Session-level cache: one SentinelPulseClient per autorun session (reconnects once)
_sentinel_client: "SentinelPulseClient | None" = None


async def _fetch_news_batch(symbols: list[str]) -> dict[str, dict]:
    """
    Batch-fetch SentinelPulse context for market + per-asset (NSE: prefixed).

    Returns:
        {
            "MARKET": { market-level news_features dict },
            "RELIANCE": { asset-level news_features dict },
            ...
        }
    Cache: SentinelPulseClient has 90s LRU cache, so repeat calls within a
    5-min cycle are effectively free.  Network latency = 1 market call +
    len(symbols) × ~100ms, run concurrently via asyncio.gather.
    """
    global _sentinel_client
    if not _SENTINEL_OK:
        return {}

    # Lazy-init client
    if _sentinel_client is None:
        _sentinel_client = SentinelPulseClient()
        await _sentinel_client.connect()

    try:
        # 1. Market context (always)
        market_ctx = await _sentinel_client._fetch_with_retry(
            "/api/v1/alphaforge/context/market"
        )
        events_raw = await _sentinel_client._fetch_with_retry(
            "/api/v1/ml/training/events", params={"limit": "50"}
        )
        events_list: list[dict] = (
            events_raw if isinstance(events_raw, list) else []
        )

        # 2. Per-asset context for each symbol — run concurrently
        async def _asset(sym: str) -> tuple[str, dict | None]:
            r = await _sentinel_client._fetch_with_retry(
                f"/api/v1/alphaforge/context/asset/NSE:{sym}"
            )
            return sym, r

        asset_results = await asyncio.gather(
            *[_asset(s) for s in symbols], return_exceptions=False
        )

        # 3. Build result dict
        out: dict[str, dict] = {}
        # Market-level entry (shared base for all symbols)
        market_feats = news_features_from_context(
            market_ctx=market_ctx, asset_ctx=None, events=events_list
        )
        out["MARKET"] = market_feats

        # Per-asset: override asset_sentiment from symbol-specific context
        for sym, asset_ctx in asset_results:
            feats = news_features_from_context(
                market_ctx=market_ctx,
                asset_ctx=asset_ctx,
                events=events_list,
            )
            out[sym] = feats

        return out

    except Exception as exc:
        print(f"[warn] SentinelPulse batch fetch failed: {exc}", file=sys.stderr)
        return {}


def apply_news_damper(
    scores: list[dict],
    news_ctx: dict[str, dict],
    now: "datetime",
) -> tuple[list[dict], int, list[str]]:
    """
    Apply SentinelPulse news sentiment to dampen conflicting signals.

    Dampening rules (all require news data to be present):
      Rule 1 — Market-regime conflict:
        If news_regime_score < -0.5 (bear) AND signal is LONG with grade C/D
        → neutralize  (strong market headwind overrides weak LONG calls)

      Rule 2 — Per-asset news conflict:
        If news_asset_sentiment < -0.25 AND ML signal is LONG
        → neutralize  (asset-specific negative news)
        If news_asset_sentiment > +0.25 AND ML signal is SHORT
        → neutralize  (asset-specific positive news)

      Rule 3 — High-importance event:
        If news_event_importance > 0.55 AND news direction conflicts with ML
        → reduce conviction (set score toward 0.5 by 30%)

    Args:
        scores:    List of scored signal dicts (mutated in-place).
        news_ctx:  {symbol: {feature: float}} from _fetch_news_batch().
        now:       Current IST datetime for logging.

    Returns:
        (modified_scores, n_dampened, reason_list)
    """
    if not news_ctx:
        return scores, 0, []

    market_feats = news_ctx.get("MARKET", {})
    mkt_regime   = float(market_feats.get("news_regime_score", 0.0))
    mkt_sent     = float(market_feats.get("news_market_sentiment", 0.0))
    n_dampened   = 0
    reasons: list[str] = []

    def _conviction_grade(score: float) -> str:
        dist = abs(score - 0.5)
        if dist >= 0.30: return "A"
        if dist >= 0.20: return "B"
        if dist >= 0.10: return "C"
        return "D"

    for s in scores:
        if s.get("direction") == 0:
            continue  # already neutral

        sym     = s["symbol"]
        ml_dir  = s["direction"]
        score   = float(s["score"])
        grade   = _conviction_grade(score)

        # Get asset-specific news (fall back to market-level)
        afeats      = news_ctx.get(sym, market_feats)
        asset_sent  = float(afeats.get("news_asset_sentiment", 0.0))
        evt_imp     = float(afeats.get("news_event_importance", 0.0))

        damped = False
        reason = ""

        # ── Rule 1: Market regime conflict (weak LONG in bear regime) ─────
        if mkt_regime < -0.5 and ml_dir == 1 and grade in ("C", "D"):
            s["direction"] = 0
            s["news_dimmed"] = f"BEAR_REGIME_{mkt_regime:+.2f}"
            damped = True
            reason = f"R1:bear_regime({mkt_regime:+.2f})"

        # ── Rule 2: Asset-level news conflict ─────────────────────────────
        elif asset_sent < -0.25 and ml_dir == 1:
            s["direction"] = 0
            s["news_dimmed"] = f"NEGATIVE_NEWS_{asset_sent:+.3f}"
            damped = True
            reason = f"R2:asset_bearish({asset_sent:+.3f})"

        elif asset_sent > +0.25 and ml_dir == -1:
            s["direction"] = 0
            s["news_dimmed"] = f"POSITIVE_NEWS_{asset_sent:+.3f}"
            damped = True
            reason = f"R2:asset_bullish({asset_sent:+.3f})"

        # ── Rule 3: High-importance event pulls score toward 0.5 ─────────
        elif evt_imp > 0.55:
            news_dir = 1 if asset_sent > 0.05 else (-1 if asset_sent < -0.05 else 0)
            if news_dir != 0 and news_dir != ml_dir:
                # Pull score 30% toward 0.5 (soften, don't flip)
                s["score"] = round(score + 0.30 * (0.5 - score), 4)
                s["news_dimmed"] = f"HIGH_EVENT_{evt_imp:.2f}"
                damped = True
                reason = f"R3:high_event({evt_imp:.2f},pull)"

        if damped:
            n_dampened += 1
            reasons.append(f"{sym}:{reason}")

    if n_dampened:
        regime_label = (
            f"BEAR({mkt_regime:+.2f})" if mkt_regime < -0.3
            else f"BULL({mkt_regime:+.2f})" if mkt_regime > 0.3
            else "NEUTRAL"
        )
        print(
            f"[{now.strftime('%H:%M')}] 📰 SentinelPulse damped {n_dampened} signals  "
            f"market={mkt_sent:+.2f} regime={regime_label}"
        )
        if len(reasons) <= 5:
            for r in reasons:
                print(f"   {r}")

    return scores, n_dampened, reasons

# ── Constants ─────────────────────────────────────────────────────────────────
BASE         = Path(__file__).parent.parent
PARQUET_DIR  = BASE / "data/1d/1d"
SESSION_DIR  = BASE / "artifacts/live_session"
SESSION_LOG  = SESSION_DIR / "autorun_log.jsonl"
REPORT_PATH  = BASE / "reports/LIVE_SESSION_REPORT.md"
COST_BPS     = 27.65  # equity round-trip
SAMPLE_MINS  = 0.5    # 30-second cycle — async fetch makes this feasible

SESSION_DIR.mkdir(parents=True, exist_ok=True)
LATEST_SCORES_PATH = SESSION_DIR / "latest_scores.json"

# ── Load env ──────────────────────────────────────────────────────────────────
env = {}
env_path = BASE / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, _, v = line.partition("=")
            env[k.strip()] = v.strip()
DATA_KEY = env.get("DATA_SERVICE_API_KEY", "")
DATA_URL = env.get("DATA_SERVICE_2_URL", "http://localhost:8200")

# ── Time helpers ──────────────────────────────────────────────────────────────
def ist_now() -> datetime:
    return datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)

def mins_to_close() -> float:
    now = ist_now()
    close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return (close - now).total_seconds() / 60

def market_open() -> bool:
    now = ist_now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return (9, 15) <= t <= (15, 31)

# ── Data fetching ─────────────────────────────────────────────────────────────
# Module-level cache for last-known NIFTY quote (avoids showing "UNAVAILABLE")
_last_nifty_quote: dict = {}

def get_quote(sym: str) -> dict | None:
    global _last_nifty_quote
    url = f"{DATA_URL}/v1/india/quotes/{sym}"
    req = urllib.request.Request(url, headers={"X-API-KEY": DATA_KEY})
    try:
        with urllib.request.urlopen(req, timeout=4) as r:
            d = json.loads(r.read().decode())
            dd = d.get("data", {}) or {}
            if dd.get("ltp") is not None:
                if sym == "NIFTY":
                    _last_nifty_quote = dd   # cache for fallback
                return dd
    except Exception:
        pass
    # Fallback: return last known value for NIFTY to avoid dashboard gaps
    if sym == "NIFTY" and _last_nifty_quote:
        return _last_nifty_quote
    return None

def get_all_quotes(symbols: list[str]) -> dict[str, dict]:
    quotes = {}
    for sym in symbols:
        q = get_quote(sym)
        if q:
            quotes[sym] = q
        time.sleep(0.12)  # ~8 req/sec, within 500/60s limit
    return quotes


# ── Async concurrent quote fetch (production-grade) ──────────────────────────
# Replaces the sequential get_all_quotes() which took ~37s for 218 symbols.
# With 8 concurrent requests (= 8 req/s) we respect the 500/60s rate limit
# while cutting total fetch time from 37s → ~2s (19× speedup).

async def _fetch_quotes_async(symbols: list[str]) -> dict[str, dict]:
    """Fetch all quotes concurrently with a rate-limit semaphore."""
    global _last_nifty_quote
    CONCURRENCY = 8      # 8 concurrent = 8 req/s ≤ 500/60s service limit
    semaphore = asyncio.Semaphore(CONCURRENCY)
    results: dict[str, dict] = {}

    try:
        import httpx as _httpx
        async with _httpx.AsyncClient(
            headers={"X-API-KEY": DATA_KEY},
            timeout=_httpx.Timeout(4.0),
            limits=_httpx.Limits(max_connections=CONCURRENCY + 4,
                                  max_keepalive_connections=CONCURRENCY),
        ) as client:
            async def _one(sym: str) -> None:
                async with semaphore:
                    try:
                        r = await client.get(f"{DATA_URL}/v1/india/quotes/{sym}")
                        if r.status_code == 200:
                            dd = r.json().get("data", {}) or {}
                            if dd.get("ltp") is not None:
                                if sym == "NIFTY":
                                    _last_nifty_quote = dd
                                results[sym] = dd
                    except Exception:
                        pass

            await asyncio.gather(*[_one(s) for s in symbols])
    except ImportError:
        # httpx not available — fall back to sequential
        for sym in symbols:
            q = get_quote(sym)
            if q:
                results[sym] = q
            time.sleep(0.08)

    # NIFTY cache fallback
    if "NIFTY" not in results and _last_nifty_quote:
        results["NIFTY"] = _last_nifty_quote
    return results


def get_all_quotes_fast(symbols: list[str]) -> dict[str, dict]:
    """Concurrent async quote fetch — drop-in replacement for get_all_quotes().

    Falls back to sequential urllib on any asyncio error.
    """
    try:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(_fetch_quotes_async(symbols))
    except Exception:
        # Last-resort fallback
        return get_all_quotes(symbols)

# ── Model loading ─────────────────────────────────────────────────────────────
def load_model():
    for pattern in ["artifacts/expanded_lgbm/*/model.pkl",
                     "artifacts/registry/stage_a_1d/*/model.pkl"]:
        paths = sorted(BASE.glob(pattern))
        if paths:
            with open(paths[-1], "rb") as f:
                p = pickle.load(f)
            # Reconstruct FeatureNormalizer from its serialised state dict.
            normalizer = None
            raw_state = p.get("normalizer_state")
            if isinstance(raw_state, dict) and raw_state:
                try:
                    from src.features.normalizer import FeatureNormalizer
                    normalizer = FeatureNormalizer.from_dict(raw_state)
                except Exception:
                    pass
            elif raw_state is not None:
                normalizer = raw_state
            return (p["estimator"], p["feature_names"],
                    normalizer, p.get("feature_schema_version","?"))
    return None, [], None, "?"


def load_ensemble() -> dict | None:
    """
    Load multi-horizon ensemble manifest and both H1 + H5 models.

    Returns a dict:
        {"h1": (est, feat_names, norm), "h5": (est, feat_names, norm), "manifest": {...}}
    or None if the ensemble manifest doesn't exist.
    """
    manifest_path = BASE / "artifacts/expanded_lgbm/ensemble_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        models: dict[str, tuple] = {}
        for m in manifest.get("models", []):
            h   = m.get("horizon")
            pkl = m.get("pkl_path")
            if not pkl or not Path(pkl).exists():
                continue
            with open(pkl, "rb") as f:
                p = pickle.load(f)
            norm = None
            raw_state = p.get("normalizer_state")
            if isinstance(raw_state, dict) and raw_state:
                try:
                    from src.features.normalizer import FeatureNormalizer
                    norm = FeatureNormalizer.from_dict(raw_state)
                except Exception:
                    pass
            models[f"h{h}"] = (p["estimator"], p["feature_names"], norm)
        if len(models) >= 2:
            models["manifest"] = manifest
            return models
    except Exception as exc:
        print(f"[warn] Ensemble load failed: {exc}", file=sys.stderr)
    return None


def score_symbol_ensemble(
    sym: str,
    ensemble: dict,
    live_ltp: float | None = None,
) -> dict | None:
    """
    Score a symbol using the multi-horizon ensemble.
    Only emit a signal if ALL horizon models agree on direction.

    Args:
        sym:      NSE symbol.
        ensemble: Output of load_ensemble().
        live_ltp: Live LTP for partial-bar injection.

    Returns:
        Signal dict with "direction" set to:
          +1 if all models LONG, -1 if all models SHORT, 0 if disagreement.
    """
    manifest   = ensemble.get("manifest", {})
    thresholds = manifest.get("thresholds", {})
    signals: list[dict] = []

    for key, value in ensemble.items():
        if not isinstance(key, str) or not key.startswith("h"):
            continue
        if not isinstance(value, tuple) or len(value) != 3:
            continue
        est, feat_names, norm = value
        h_signal = score_symbol(sym, est, feat_names, norm, live_ltp=live_ltp)
        if h_signal is None:
            return None
        signals.append({"key": key, **h_signal})

    if not signals:
        return None

    # Agreement: all horizons must vote same direction
    directions = [s["direction"] for s in signals]
    if all(d == 1  for d in directions):
        final_dir = 1
    elif all(d == -1 for d in directions):
        final_dir = -1
    else:
        final_dir = 0   # disagreement → WAIT

    # Use the H5 score as the canonical score (longer horizon, more stable)
    h5_signal = next((s for s in signals if s["key"] == "h5"), signals[-1])
    return {
        **h5_signal,
        "direction":       final_dir,
        "ensemble_agree":  final_dir != 0,
        "h_signals":       {s["key"]: s["score"] for s in signals},
    }

def score_symbol(sym: str, estimator, feat_names, normalizer=None, live_ltp: float | None = None) -> dict | None:
    pf = PARQUET_DIR / f"{sym}.parquet"
    if not pf.exists():
        return None
    try:
        from src.features.expanded_factory import ExpandedFeatureFactory
        from src.features.factory import FeatureFactory
        df = pd.read_parquet(pf)
        df.columns = [c.lower() for c in df.columns]
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # ── INTRADAY FIX (Gap 1+2 from Sep-30 analysis) ─────────────────────
        # Append a synthetic partial bar using the live LTP so the model sees
        # today's intraday move, not just yesterday's close.
        # Without this: TCS was called SHORT all day because yesterday's EOD showed
        # bearish momentum.  With this: today's +2% LTP updates ret_1/momentum_5d
        # and the model correctly identifies IT stocks as LONG intraday.
        if live_ltp is not None and live_ltp > 0 and len(df) > 0:
            ist_now = datetime.now(tz=timezone.utc) + timedelta(hours=5, minutes=30)
            # Today's bar timestamp = today's midnight IST = (today - 1d) 18:30 UTC
            # Strip tzinfo before pd.Timestamp to avoid ValueError in Python 3.12
            # when passing tz-aware datetime with tz= kwarg simultaneously.
            ist_midnight_naive = ist_now.replace(
                hour=0, minute=0, second=0, microsecond=0, tzinfo=None
            )
            today_bar_ts = pd.Timestamp(ist_midnight_naive, tz="UTC") - timedelta(hours=5, minutes=30)
            if today_bar_ts > df.index.max():
                prev_close = float(df["close"].iloc[-1]) if "close" in df.columns else live_ltp
                # Build partial bar: open=prev_close, close=live_ltp
                # high/low reflect the intraday range so far
                partial_data: dict[str, float] = {
                    "open":   prev_close,
                    "high":   max(prev_close, live_ltp),
                    "low":    min(prev_close, live_ltp),
                    "close":  live_ltp,
                    "volume": float(df["volume"].iloc[-1]) if "volume" in df.columns else 0.0,
                }
                partial_row = pd.DataFrame(partial_data, index=[today_bar_ts])
                # Only keep columns that exist in the parquet to avoid column mismatch
                keep_cols = [c for c in df.columns if c in partial_row.columns]
                if keep_cols:
                    partial_row = partial_row[keep_cols]
                    df = pd.concat([df[keep_cols], partial_row])
                else:
                    df = pd.concat([df, partial_row])

        factory = ExpandedFeatureFactory() if len(feat_names) > 24 else FeatureFactory()
        features, _ = factory.build(df)
        last = features.iloc[-1].fillna(0)

        # ── Market features (nifty_ret_1d/5d/20d) ────────────────────────────
        # If the model was trained with NIFTY market returns, compute them now
        # from the on-disk NIFTY parquet so inference matches training.
        market_feat_vals: dict[str, float] = {}
        if any(c.startswith("nifty_ret") for c in feat_names):
            try:
                nifty_pf = PARQUET_DIR / "NIFTY.parquet"
                if nifty_pf.exists():
                    ndf = pd.read_parquet(nifty_pf)
                    nclose = ndf["close"].astype(float).dropna()
                    market_feat_vals["nifty_ret_1d"]  = float(nclose.pct_change(1).iloc[-1]) if len(nclose) > 1  else 0.0
                    market_feat_vals["nifty_ret_5d"]  = float(nclose.pct_change(5).iloc[-1]) if len(nclose) > 5  else 0.0
                    market_feat_vals["nifty_ret_20d"] = float(nclose.pct_change(20).iloc[-1]) if len(nclose) > 20 else 0.0
            except Exception:
                pass

        # ── Intraday features (Group G from 5m bars) ─────────────────────────
        # If the model was trained with intraday features, compute them from
        # today's 5m bars (if available on disk).
        intraday_feat_vals: dict[str, float] = {}
        if any(c.startswith("intraday_") for c in feat_names):
            try:
                pf_5m = BASE / "data" / "5m" / "5m" / f"{sym}.parquet"
                if pf_5m.exists():
                    from src.features.families.intraday import (  # noqa: PLC0415
                        compute_intraday_features_for_date,
                    )
                    df_5m_full = pd.read_parquet(pf_5m)
                    if df_5m_full.index.tz is None:
                        df_5m_full.index = df_5m_full.index.tz_localize("UTC")
                    # Filter to today's bars
                    from datetime import date as _date  # noqa: PLC0415
                    today_utc = pd.Timestamp(_date.today()).tz_localize("UTC")
                    df_today_5m = df_5m_full[df_5m_full.index.normalize() == today_utc]
                    if len(df_today_5m) >= 6:
                        intraday_feat_vals = compute_intraday_features_for_date(
                            df_today_5m,
                            is_partial=(live_ltp is not None),
                        )
            except Exception:
                pass

        # Build feature vector — use 0.0 fallback for missing features
        X = np.array([[
            intraday_feat_vals.get(c,
                market_feat_vals.get(c,
                    last.get(c, 0.0)))
            for c in feat_names
        ]])
        if normalizer is not None:
            try:
                X_df = pd.DataFrame(X, columns=feat_names)
                X = normalizer.transform(X_df).to_numpy(dtype=float)
            except Exception:
                pass
        score = float(estimator.predict(X)[0])
        # data_date: use today if we injected a live partial bar, else last bar
        if live_ltp and live_ltp > 0:
            try:
                data_date = str(today_bar_ts.date())
            except NameError:
                data_date = str(df.index.max().date())
        else:
            data_date = str(df.index.max().date())
        return {"symbol": sym, "score": round(score, 4), "direction": 1 if score > 0.5 else -1,
                "data_date": data_date, "live_bar": live_ltp is not None and live_ltp > 0}
    except Exception:
        return None


def score_all(estimator, feat_names, normalizer, live_quotes: dict | None = None) -> list[dict]:
    """Score all 218 symbols.  Pass live_quotes to enable intraday partial-bar feature update."""
    results = []
    for pf in sorted(PARQUET_DIR.glob("*.parquet")):
        sym  = pf.stem
        ltp  = None
        if live_quotes:
            q   = live_quotes.get(sym)
            ltp = q.get("ltp") if isinstance(q, dict) else None
        r = score_symbol(pf.stem, estimator, feat_names, normalizer, live_ltp=ltp)
        if r:
            results.append(r)
    return results

# ── P&L calculation ──────────────────────────────────────────────────────────
def load_fp_signals() -> dict[str, dict]:
    """Load forward-paper signals, preferring the most recently scored entry per symbol.

    FIX P0-SIGNAL-001: Original first-occurrence-wins logic could load a stale
    signal from signals.jsonl (score=null, wrong direction) and ignore a fresher,
    scored signal in signals_v2.jsonl. Now selects the entry with the most recent
    created_at timestamp AND a valid score, falling back to any entry if necessary.
    """
    # Collect ALL signals keyed by symbol, keeping the best per symbol
    all_sigs: dict[str, list[dict]] = {}
    for sp in [BASE/"artifacts/forward_paper/signals.jsonl",
               BASE/"artifacts/forward_paper/signals_v2.jsonl"]:
        if sp.exists():
            for line in sp.read_text().splitlines():
                if line.strip():
                    try:
                        s = json.loads(line)
                        sym = s["symbol"]
                        if sym not in all_sigs:
                            all_sigs[sym] = []
                        all_sigs[sym].append(s)
                    except Exception:
                        pass

    # Per symbol: prefer entry with valid score and latest created_at timestamp
    sigs: dict[str, dict] = {}
    for sym, entries in all_sigs.items():
        # Scored entries (those with a numeric score) are preferred
        scored = [e for e in entries if e.get("score") is not None]
        pool = scored if scored else entries
        # Among candidates, take the most recently created
        best = max(pool, key=lambda e: e.get("created_at", e.get("signal_ts", "")))
        sigs[sym] = best
    return sigs

# Load data quality flags
def load_excluded_symbols() -> set[str]:
    flag_path = BASE / "artifacts/data_quality_flags.json"
    if not flag_path.exists():
        return set()
    try:
        flags = json.loads(flag_path.read_text())
        return {f["symbol"] for f in flags}
    except Exception:
        return set()

def calc_pnl(fp_signals: dict, live_quotes: dict, excluded: set) -> dict:
    results = []
    for sym, sig in fp_signals.items():
        if sym in excluded:
            continue
        direction = sig.get("direction", 0)
        if direction == 0:
            continue
        # Get entry price
        entry = sig.get("entry_price")
        if entry is None:
            pf = PARQUET_DIR / f"{sym}.parquet"
            if pf.exists():
                df = pd.read_parquet(pf)
                # Skip zero-column parquets (corrupted during backfill)
                if df.empty or len(df.columns) == 0:
                    continue
                # Normalise column names to lowercase (some parquets use 'Close'/'Open')
                df.columns = [c.lower() for c in df.columns]
                sig_ts = pd.Timestamp(sig.get("signal_ts", sig.get("created_at", "")))
                if sig_ts.tzinfo is None:
                    sig_ts = sig_ts.tz_localize("UTC")
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                future = df[df.index > sig_ts]
                if len(future) > 0 and "open" in future.columns:
                    entry = float(future["open"].iloc[0])
                elif len(df) > 0:
                    entry = float(df["close"].iloc[-1])
        if not entry or entry <= 0:
            continue
        ltp_data = live_quotes.get(sym, {})
        ltp = ltp_data.get("ltp")
        if ltp is None:
            continue
        ltp = float(ltp)
        # Outlier filter: >50% mismatch = data issue
        if abs(ltp - entry) / max(entry, 1e-6) > 0.50:
            continue
        gross = direction * (ltp - entry) / entry * 100
        net   = gross - (COST_BPS / 100)
        results.append({
            "symbol": sym, "direction": direction,
            "entry": round(entry, 2), "ltp": round(ltp, 2),
            "gross_pct": round(gross, 3), "net_pct": round(net, 3),
            "chg_today_pct": float(ltp_data.get("changePct", 0) or 0),
        })
    if not results:
        return {"n": 0, "mean_net": 0, "win_rate": 0, "short_mean": 0, "long_mean": 0, "positions": []}
    arr = pd.DataFrame(results)
    wins = arr[arr["net_pct"] > 0]
    short_arr = arr[arr["direction"] == -1]
    long_arr  = arr[arr["direction"] == 1]
    return {
        "n": len(arr),
        "mean_net": round(float(arr["net_pct"].mean()), 4),
        "win_rate": round(len(wins) / len(arr) * 100, 1),
        "short_mean": round(float(short_arr["net_pct"].mean()), 4) if len(short_arr) else 0,
        "long_mean":  round(float(long_arr["net_pct"].mean()), 4) if len(long_arr) else 0,
        "best":  arr.loc[arr["net_pct"].idxmax(), ["symbol","net_pct"]].to_dict(),
        "worst": arr.loc[arr["net_pct"].idxmin(), ["symbol","net_pct"]].to_dict(),
        "n_short": len(short_arr), "n_long": len(long_arr),
        "positions": results,
    }

# ── Post-close actions ────────────────────────────────────────────────────────
def post_close_ingest():
    print("\n[close] Ingesting Sep 28 closing bars...")
    result = subprocess.run(
        ["python3", "-W", "ignore", str(BASE/"scripts/ingest_all_outdated.py")],
        capture_output=True, text=True, cwd=str(BASE),
        env={**os.environ, "PYTHONPATH": str(BASE)}, timeout=300,
    )
    out = result.stdout.strip()
    if out:
        print(out[-400:])
    return out

def post_close_resolve():
    print("\n[close] Resolving forward paper signals...")
    result = subprocess.run(
        ["python3", "-W", "ignore", str(BASE/"scripts/resolve_forward_paper.py")],
        capture_output=True, text=True, cwd=str(BASE),
        env={**os.environ, "PYTHONPATH": str(BASE)}, timeout=120,
    )
    out = result.stdout.strip()
    if out:
        print(out[-600:])
    return out

def post_close_promotion():
    print("\n[close] Running SignalPromotionEngine...")
    result = subprocess.run(
        ["python3", "-W", "ignore", str(BASE/"scripts/run_signal_promotion.py")],
        capture_output=True, text=True, cwd=str(BASE),
        env={**os.environ, "PYTHONPATH": str(BASE)}, timeout=60,
    )
    out = result.stdout.strip()
    if out:
        print(out[-400:])
    return out

# ── Dashboard printing ────────────────────────────────────────────────────────
def print_dashboard(sample_n: int, now: datetime, nifty: dict | None, scores: list,
                    pnl: dict, mins_left: float) -> None:
    n_long  = sum(1 for s in scores if s["direction"] == 1)
    n_short = sum(1 for s in scores if s["direction"] == -1)
    nifty_ltp = nifty.get("ltp","?") if nifty else "UNAVAILABLE"
    nifty_chg = nifty.get("changePct", 0) if nifty else 0

    bar_pnl  = "▓" * max(0, min(20, int(pnl.get("short_mean",0)*20))) if pnl["n"] > 0 else ""
    bar_long = "░" * max(0, min(20, int(abs(pnl.get("long_mean",0))*10))) if pnl["n"] > 0 else ""

    print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  LIVE SESSION #{sample_n:02d}  │  {now.strftime('%H:%M IST')}  │  {mins_left:.0f}min to close      ║
╠══════════════════════════════════════════════════════════════════════╣
║  MARKET     │ NIFTY {nifty_ltp} ({nifty_chg:+.2f}%)                             ║
║  SIGNALS    │ {len(scores)} scored │ LONG={n_long} SHORT={n_short} (bias={n_short/max(len(scores),1)*100:.0f}% SHORT)  ║
╠══════════════════════════════════════════════════════════════════════╣
║  LIVE P&L   │ {pnl['n']} positions tracked                                 ║
║  Mean net   │ {pnl.get('mean_net',0):+.3f}%                                          ║
║  Win rate   │ {pnl.get('win_rate',0):.1f}%                                             ║
║  SHORT avg  │ {pnl.get('short_mean',0):+.3f}% ({pnl.get('n_short',0)} positions) {bar_pnl}   ║
║  LONG avg   │ {pnl.get('long_mean',0):+.3f}% ({pnl.get('n_long',0)} positions) {bar_long}  ║""")
    if pnl["n"] > 0 and "best" in pnl:
        print(f"║  Best trade │ {pnl['best']['symbol']} {pnl['best']['net_pct']:+.3f}%                              ║")
        print(f"║  Worst      │ {pnl['worst']['symbol']} {pnl['worst']['net_pct']:+.3f}%                              ║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝")

    # Top signals
    top_long  = sorted([s for s in scores if s["direction"] == 1],  key=lambda x: -x["score"])[:5]
    top_short = sorted([s for s in scores if s["direction"] == -1], key=lambda x:  x["score"])[:5]
    long_str  = ", ".join(f'{s["symbol"]}({s["score"]:.3f})' for s in top_long)
    short_str = ", ".join(f'{s["symbol"]}({s["score"]:.3f})' for s in top_short)
    print(f"\n  TOP LONG  (score->1.0):  {long_str}")
    print(f"  TOP SHORT (score->0.0):  {short_str}")


# ── Main loop ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"  AUTORUN TILL CLOSE  |  {ist_now().strftime('%Y-%m-%d %H:%M IST')}")
    print(f"  NSE closes in {mins_to_close():.0f} minutes (15:30 IST)")
    print(f"  Sampling every {SAMPLE_MINS} minutes")
    print("=" * 70)

    # Load model once
    # load_model() already reconstructs the FeatureNormalizer from its dict state.
    # The 3rd return value IS the live normalizer object (or None if reconstruction failed).
    # Do NOT call FeatureNormalizer.from_dict() again here — that is the bug.
    estimator, feat_names, normalizer, schema = load_model()
    if estimator is None:
        print("ERROR: No model found. Check artifacts/expanded_lgbm/")
        sys.exit(1)

    print(f"  Model: {schema} | {len(feat_names)} features | normalizer: {'yes' if normalizer else 'no'}")

    # ── Try loading multi-horizon ensemble ────────────────────────────────────
    _ensemble = load_ensemble()
    if _ensemble:
        horizons = [k for k in _ensemble if k.startswith("h")]
        print(f"  Ensemble: {horizons} horizons loaded ✓  (agreement filter ACTIVE)")
    else:
        print("  Ensemble: single H5 model (run train_multihorizon_ensemble.py to enable)")

    # ── Phil integration: initialise components ───────────────────────────────
    _forecast_ledger    = ForecastLedger()    if _PHIL_IMPORTS_OK else None
    _cfactual_ledger    = CounterfactualLedger() if _PHIL_IMPORTS_OK else None
    _weight_mgr         = FeatureWeightManager() if _PHIL_IMPORTS_OK else None
    _reversal_detector  = ReversalDetector(override_threshold=0.60) if _PHIL_IMPORTS_OK else None
    session_date        = ist_now().strftime("%Y-%m-%d")
    if _PHIL_IMPORTS_OK:
        print(f"  Phil integrations: ForecastLedger ✓ | CounterfactualLedger ✓ | FeatureWeightManager ✓ | ReversalDetector ✓")

    # Load forward paper signals and exclusions
    fp_signals = load_fp_signals()
    excluded   = load_excluded_symbols()
    universe   = [pf.stem for pf in sorted(PARQUET_DIR.glob("*.parquet"))]
    key_syms   = ["NIFTY","BANKNIFTY","RELIANCE","HDFCBANK","ICICIBANK",
                  "INFY","TCS","KOTAKBANK","AXISBANK","BHARTIARTL",
                  "SBIN","LT","MARUTI","WIPRO","TITAN","NTPC","ONGC",
                  "BAJFINANCE","HINDUNILVR","ADANIENT"]

    print(f"  Universe: {len(universe)} | FP positions: {len(fp_signals)} | Excluded: {excluded}")
    print("  Running...\n")

    # Session state
    all_samples: list[dict] = []
    sample_n   = 0
    post_close_done = False

    while True:
        now     = ist_now()
        mins    = mins_to_close()

        # ── MARKET CLOSED & POST-CLOSE ─────────────────────────────────────
        if mins <= 0 and not post_close_done:
            print(f"\n{'='*70}")
            print(f"  MARKET CLOSED — {now.strftime('%H:%M IST')}  Running post-close actions...")
            print(f"{'='*70}")

            # Final quote snapshot
            print("\n[close] Final quote snapshot...")
            final_quotes = get_all_quotes_fast(key_syms + list(fp_signals.keys())[:50])

            # Post-close actions
            ingest_out  = post_close_ingest()
            resolve_out = post_close_resolve()
            promo_out   = post_close_promotion()

            # Final P&L with fresh quotes
            final_scores = score_all(estimator, feat_names, normalizer)
            final_pnl    = calc_pnl(fp_signals, final_quotes, excluded)

            # ── Symbol IC tracker: record today's outcomes ───────────────────
            if _IC_TRACKER_OK and _symbol_ic_tracker is not None and final_pnl.get("positions"):
                ic_outcomes = []
                for pos in final_pnl.get("positions", []):
                    sym = pos.get("symbol")
                    net = pos.get("net_pct")
                    # Find the model score for this symbol
                    score_entry = next((s for s in final_scores if s["symbol"] == sym), None)
                    if score_entry and net is not None and sym:
                        ic_outcomes.append({
                            "symbol":           sym,
                            "score":            score_entry["score"],
                            "realized_return":  float(net) / 100.0,   # convert % to decimal
                            "date":             session_date,
                        })
                if ic_outcomes:
                    _symbol_ic_tracker.record_batch(ic_outcomes)
                    ic_sum = _symbol_ic_tracker.summary()
                    print(
                        f"\n[close] SymbolIC: recorded {len(ic_outcomes)} outcomes | "
                        f"mean_ic={ic_sum['mean_ic']}  dead={ic_sum['n_dead']} symbols"
                    )
                    if ic_sum.get("bottom5"):
                        worst = ic_sum["bottom5"][:3]
                        print(f"         Worst IC: " +
                              "  ".join(f"{w['symbol']}={w['ic']:.3f}" for w in worst))

            # ── Phil: resolve all 218 forecasts with final realized returns ──
            if _forecast_ledger is not None and final_pnl.get("positions"):
                realized_returns = {
                    p["symbol"]: p["net_pct"]
                    for p in final_pnl.get("positions", [])
                }
                brier_report = _forecast_ledger.resolve_session(
                    session_date=session_date,
                    realized_returns=realized_returns,
                )
                bd = brier_report.get("brier_delta", 0)
                print(f"\n[close] ForecastLedger: brier_delta={bd:+.6f} "
                      f"({'BEATING market' if bd < 0 else 'behind market'}) | "
                      f"n={brier_report.get('n_resolved', 0)} resolved")

            # ── Phil: score threshold sweep (update optimal threshold) ───────
            if _PHIL_IMPORTS_OK and final_pnl.get("positions"):
                sweep = ScoreThresholdSweep()
                resolved_outcomes = [
                    {"symbol": p["symbol"], "net_pct": p["net_pct"],
                     "direction": p["direction"]}
                    for p in final_pnl.get("positions", [])
                ]
                sweep_report = sweep.run(
                    scores=final_scores, resolved_outcomes=resolved_outcomes,
                    date=session_date,
                )
                sweep.save_report(
                    sweep_report,
                    BASE / "reports" / f"score_threshold_sweep_{session_date}.json",
                )
                if sweep_report.optimal_threshold > 0:
                    print(f"[close] ScoreThresholdSweep: optimal threshold={sweep_report.optimal_threshold:.2f} "
                          f"win_rate={sweep_report.optimal_win_rate:.1%}")
                    # Auto-update feature_weights.json with sweep result
                    if _weight_mgr is not None:
                        _weight_mgr.update_from_sweep(
                            optimal_threshold=sweep_report.optimal_threshold,
                            win_rate=sweep_report.optimal_win_rate,
                            evidence=sweep_report.reasoning,
                        )

            # ── Phil: grade any blocked signals (counterfactual ledger) ──────
            if _cfactual_ledger is not None and final_pnl.get("positions"):
                realized_for_cf = {
                    p["symbol"]: p["net_pct"] for p in final_pnl.get("positions", [])
                }
                cf_report = _cfactual_ledger.resolve(
                    session_date=session_date,
                    realized_returns=realized_for_cf,
                )
                if cf_report.get("n", 0) > 0:
                    print(f"[close] CounterfactualLedger: {cf_report['n']} blocked signals graded | "
                          f"verdict={cf_report.get('overall_verdict','N/A')}")

            # Final sample
            nifty_q = final_quotes.get("NIFTY", {})
            final_sample = {
                "timestamp":   now.isoformat(),
                "is_close":    True,
                "nifty_ltp":   nifty_q.get("ltp"),
                "nifty_chg":   nifty_q.get("changePct", 0),
                "n_scored":    len(final_scores),
                "n_long":      sum(1 for s in final_scores if s["direction"] == 1),
                "n_short":     sum(1 for s in final_scores if s["direction"] == -1),
                "pnl":         final_pnl,
            }
            all_samples.append(final_sample)
            with SESSION_LOG.open("a") as f:
                f.write(json.dumps(final_sample) + "\n")

            print_dashboard(sample_n + 1, now, nifty_q, final_scores, final_pnl, 0)

            # ── AlphaForge: write final close snapshot ───────────────────────
            _write_latest_scores(
                scores=final_scores, session_date=session_date, schema=schema,
                nifty_chg=nifty_q.get("changePct", 0), nifty_ltp=nifty_q.get("ltp"),
                market_open=False, pnl=final_pnl, beta_hedge={},
            )

            # Summary
            print(f"\n{'='*70}")
            print("  END-OF-DAY SUMMARY")
            print(f"{'='*70}")
            print(f"  Session samples: {len(all_samples)}")
            print(f"  Market move: NIFTY {nifty_q.get('changePct',0):+.2f}%")
            print(f"  Final P&L: {final_pnl.get('mean_net',0):+.4f}% mean net | {final_pnl.get('win_rate',0):.1f}% win rate")
            print(f"  SHORT avg: {final_pnl.get('short_mean',0):+.4f}% | LONG avg: {final_pnl.get('long_mean',0):+.4f}%")
            n_pos = final_pnl.get("n_short", 0) + final_pnl.get("n_long", 0)
            if n_pos > 0:
                pos_side = "SHORT" if final_pnl.get("n_short",0) > final_pnl.get("n_long",0) else "LONG"
                print(f"  Model bias: {pos_side} ({final_pnl.get('n_short',0)} SHORT / {final_pnl.get('n_long',0)} LONG)")
            print(f"\n  All files saved:")
            print(f"    • {SESSION_LOG}")
            print(f"    • {REPORT_PATH}")
            print(f"    • artifacts/forward_paper/outcomes.jsonl")
            print(f"    • artifacts/live_session/session_summary.json")

            # Save session summary
            summary = {
                "session_date": now.strftime("%Y-%m-%d"),
                "n_samples": len(all_samples),
                "final_nifty_chg": nifty_q.get("changePct", 0),
                "final_pnl": final_pnl,
                "all_samples": all_samples,
            }
            (SESSION_DIR/"session_summary.json").write_text(json.dumps(summary, indent=2, default=str))

            # Update live report
            _append_close_to_report(now, nifty_q, final_pnl, len(all_samples))

            post_close_done = True
            print(f"\n  Done! Session complete. All logs saved.")
            break

        # ── MARKET CLOSED — no more work ──────────────────────────────────
        if mins <= 0 and post_close_done:
            break

        # ── MARKET OPEN — regular sample ──────────────────────────────────
        sample_n += 1

        # Fetch quotes — key symbols first (fast, for NIFTY display)
        print(f"[{now.strftime('%H:%M')}] Sample #{sample_n} | {mins:.0f}min left | Fetching {len(key_syms)} quotes...", end="", flush=True)
        live_quotes = get_all_quotes_fast(key_syms)
        nifty_q = live_quotes.get("NIFTY", {})
        print(f" NIFTY={nifty_q.get('ltp','?')} ({nifty_q.get('changePct',0):+.2f}%)")

        # ── INTRADAY FIX: fetch ALL FP quotes BEFORE scoring ─────────────────
        # This enables live-LTP partial-bar in score_symbol() so the model sees
        # TODAY'S momentum instead of being frozen on yesterday's close.
        # (Sep-30 finding: TCS called SHORT all day because yesterday's EOD was
        # bearish; today's +2% LTP would have correctly updated momentum features.)
        print(f"[{now.strftime('%H:%M')}] Fetching live LTPs for 218 symbols (intraday update)...", end="", flush=True)
        all_fp_quotes_prefetch = get_all_quotes_fast(list(fp_signals.keys()))
        live_quotes.update(all_fp_quotes_prefetch)
        n_live = sum(1 for q in live_quotes.values() if isinstance(q, dict) and q.get("ltp"))
        print(f" {n_live} live LTPs received")

        # GAP-1: verified symbols = those with a real-time live LTP from the data service.
        # Only these symbols can have their signals tracked intraday.
        # Used to bias LONG selection toward trackable large-caps (GAP-4) and to mark
        # signal verification status in latest_scores.json for the tracker.
        verified_syms: set[str] = {
            sym for sym, q in live_quotes.items()
            if isinstance(q, dict) and q.get("ltp") is not None and float(q.get("ltp", 0) or 0) > 0
        }

        # Score all 218 WITH live LTPs (intraday partial bar)
        print(f"[{now.strftime('%H:%M')}] Scoring 218 symbols...", end="", flush=True)
        if _ensemble:
            # Multi-horizon ensemble: score each symbol with H1 + H5, keep only agreements
            scores_raw = score_all(estimator, feat_names, normalizer, live_quotes=live_quotes)
            scores = []
            n_agree = n_disagree = 0
            for s in scores_raw:
                ens_result = score_symbol_ensemble(s["symbol"], _ensemble,
                                                   live_ltp=live_quotes.get(s["symbol"], {}).get("ltp") if live_quotes else None)
                if ens_result is None:
                    scores.append(s)   # fallback to single-model
                elif ens_result["ensemble_agree"]:
                    scores.append(ens_result)
                    n_agree += 1
                else:
                    # disagreement → force neutral (keep in list but direction=0)
                    s["direction"] = 0
                    s["ensemble_disagree"] = True
                    scores.append(s)
                    n_disagree += 1
            if n_agree + n_disagree > 0:
                print(f" done ({len(scores)} scored, ensemble agree={n_agree} disagree={n_disagree})")
            else:
                print(f" done ({len(scores)} scored)")
        else:
            scores = score_all(estimator, feat_names, normalizer, live_quotes=live_quotes)
            print(f" done ({len(scores)} scored)")

        # GAP-1: tag each signal with whether it has a live LTP (trackable intraday)
        for s in scores:
            s["has_live_ltp"] = s["symbol"] in verified_syms

        # ── Cross-sectional rank-based direction assignment ────────────────
        # Root cause of "all LONG no SHORT" issue: when model score distribution
        # shifts (e.g., mean=0.63 today), absolute threshold (>0.55 LONG / <0.45 SHORT)
        # fails — everything becomes LONG. Fix: rank scores cross-sectionally and
        # reassign direction based on top/bottom decile. This is how institutional
        # quant models work (IC is a cross-sectional metric, not absolute).
        if scores:
            raw_scores = sorted([(s["score"], s["symbol"]) for s in scores], reverse=True)
            n = len(raw_scores)
            # Top 15% → LONG, Bottom 15% → SHORT, middle 70% → neutral (direction=0)
            SIGNAL_PCT = 0.15
            n_each = max(5, int(n * SIGNAL_PCT))
            top_syms    = {sym for _, sym in raw_scores[:n_each]}
            bottom_syms = {sym for _, sym in raw_scores[-n_each:]}
            for s in scores:
                sym = s["symbol"]
                if sym in top_syms:
                    s["direction"] = 1
                elif sym in bottom_syms:
                    s["direction"] = -1
                else:
                    s["direction"] = 0
            n_reassigned = len(top_syms) + len(bottom_syms)
            score_mean = sum(s["score"] for s in scores) / n
            score_min  = min(s["score"] for s in scores)
            score_max  = max(s["score"] for s in scores)
            print(
                f"[{now.strftime('%H:%M')}] CrossSectional: score range=[{score_min:.3f},{score_max:.3f}] "
                f"mean={score_mean:.3f}  assigned {n_each}L/{n_each}S from {n} symbols"
            )
        if _IC_TRACKER_OK and _symbol_ic_tracker is not None:
            scores, n_ic_suppressed = _symbol_ic_tracker.apply_dead_filter(scores, verbose=True)
            if n_ic_suppressed == 0:
                ic_summary = _symbol_ic_tracker.summary()
                if ic_summary["n_dead"] > 0:
                    print(f"[{now.strftime('%H:%M')}] SymbolIC: {ic_summary['n_dead']} dead symbols on cooldown")

        # ── Symbol IC dead-symbol filter ──────────────────────────────────────
        # Skip symbols where rolling 60-trade IC < -0.05 (chronically wrong)
        # Sep-30 finding: APOLLOHOSP (A-grade LONG, 0.639) dropped -8.8% — a
        # corporate event (earnings/news) the model can't predict from EOD bars.
        # Now we flag any symbol with an extreme intraday move (>4%) vs ML call.
        EVENT_THRESHOLD = 4.0  # % move that constitutes an event signal
        score_map = {s["symbol"]: s for s in scores}
        event_alerts: list[dict] = []
        for sym, q in live_quotes.items():
            if not isinstance(q, dict):
                continue
            chg = q.get("changePct")
            if chg is None or abs(chg) < EVENT_THRESHOLD:
                continue
            ml = score_map.get(sym)
            if ml is None:
                continue
            ml_dir = ml.get("direction", 0)
            actual_dir = 1 if chg > 0 else -1
            if ml_dir != 0 and ml_dir != actual_dir:
                # High conviction wrong call — likely a corporate event
                event_alerts.append({
                    "symbol":    sym,
                    "ml_dir":    "LONG" if ml_dir == 1 else "SHORT",
                    "ml_grade":  ml.get("conviction", "?"),
                    "ml_score":  ml.get("score", 0.5),
                    "actual_chg": round(chg, 2),
                    "alert_type": "EVENT_RISK",
                })
        if event_alerts:
            print(f"[{now.strftime('%H:%M')}] ⚠  EVENT RISK ALERTS ({len(event_alerts)}):")
            for a in event_alerts[:3]:
                print(f"   {a['symbol']:15s} ML={a['ml_dir']:5s} {a['ml_grade']} "
                      f"| actual={a['actual_chg']:+.1f}% | CORPORATE EVENT LIKELY")

        # ── Gap 5 Fix: Dynamic threshold based on intraday volatility ─────────
        # Sep-30 finding: 125 signals were filtered at a static threshold.
        # When market is highly directional, lower threshold lets more signals through.
        if _weight_mgr is not None:
            # Compute universe average absolute move from live quotes
            all_chgs = [q.get("changePct") for q in live_quotes.values()
                        if isinstance(q, dict) and q.get("changePct") is not None]
            if all_chgs:
                avg_abs_move = sum(abs(c) for c in all_chgs) / len(all_chgs)
                # High volatility day → lower threshold → more signals pass
                # Low volatility day → higher threshold → fewer, higher quality signals
                if avg_abs_move > 1.5:      # High vol: >1.5% avg move
                    _weight_mgr.set_threshold(0.05)   # admit score distance ≥ 0.05 from 0.5
                elif avg_abs_move > 0.8:    # Normal
                    _weight_mgr.set_threshold(0.10)
                else:                       # Low vol
                    _weight_mgr.set_threshold(0.15)

        # ── Weight manager: filter by conviction threshold ────────────────────────
        # GAP-5 fix: weight manager now runs BEFORE sector dampening. Previously it
        # ran after, resetting direction=0 for threshold violations and wiping
        # sector_boost/sector_dimmed flags that the final cross-sectional would then
        # preserve as 0 instead of the sector-intended direction.
        nifty_chg_now = float(nifty_q.get("changePct", 0) or 0)
        if _weight_mgr is not None and scores:
            regime = _weight_mgr.detect_regime(nifty_chg=nifty_chg_now)
            scores, _wt_summary = _weight_mgr.apply(scores, regime=regime, nifty_chg=nifty_chg_now)
            n_filtered = _wt_summary.get("n_filtered", 0)
            if n_filtered > 0:
                print(f"[{now.strftime('%H:%M')}] Feature weights: regime={regime}, "
                      f"{n_filtered} signals filtered (sector_dim={_wt_summary.get('n_sector_dimmed',0)} "
                      f"threshold={_wt_summary.get('n_threshold',0)})")

        # ── Sector intraday regime dampening (now runs AFTER weight manager) ─────
        # Sep-30 finding: IT sector rallied +2% all day but model had IT stocks SHORT.
        # Now detect intraday sector direction and dampen conflicting signals.
        IT_SYMS    = {"TCS","INFY","HCLTECH","WIPRO","TECHM","COFORGE","PERSISTENT","OFSS","TATAELXSI","KPITTECH"}
        PHARMA_SYMS = {"SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","LUPIN","AUROPHARMA","GLENMARK","ZYDUSLIFE","ALKEM"}
        AUTO_SYMS  = {"MARUTI","HEROMOTOCO","TVSMOTOR","BAJAJ-AUTO","EICHERMOT","M&M"}
        BANK_SYMS  = {"HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","INDUSINDBK","BANDHANBNK","FEDERALBNK","IDFCFIRSTB"}
        CEMENT_SYMS = {"ULTRACEMCO","AMBUJACEM","ACC","SHREECEM","RAMCOCEM","DALMIACEMT","JKCEMENT"}

        def _sector_avg_chg(syms: set) -> float | None:
            vals = [live_quotes.get(s, {}).get("changePct")
                    for s in syms if isinstance(live_quotes.get(s), dict)
                    and live_quotes[s].get("changePct") is not None]
            return sum(vals) / len(vals) if vals else None

        it_chg      = _sector_avg_chg(IT_SYMS)
        pharma_chg  = _sector_avg_chg(PHARMA_SYMS)
        auto_chg    = _sector_avg_chg(AUTO_SYMS)
        bank_chg    = _sector_avg_chg(BANK_SYMS)
        cement_chg  = _sector_avg_chg(CEMENT_SYMS)

        SECTOR_THRESHOLD = 1.0   # % sector-avg move to trigger dampening
        n_sector_dampened = 0
        n_sector_boosted  = 0
        for s in scores:
            sym = s["symbol"]
            chg_now = None
            if sym in IT_SYMS:        chg_now = it_chg
            elif sym in PHARMA_SYMS:  chg_now = pharma_chg
            elif sym in AUTO_SYMS:    chg_now = auto_chg
            elif sym in BANK_SYMS:    chg_now = bank_chg
            elif sym in CEMENT_SYMS:  chg_now = cement_chg
            if chg_now is None:
                continue
            # Pass 1: neutralize conflicting directional signals
            if chg_now > SECTOR_THRESHOLD and s.get("direction") == -1:
                s["direction"] = 0
                s["sector_dimmed"] = f"SECTOR_UP_{chg_now:+.1f}pct"
                n_sector_dampened += 1
            elif chg_now < -SECTOR_THRESHOLD and s.get("direction") == 1:
                s["direction"] = 0
                s["sector_dimmed"] = f"SECTOR_DN_{chg_now:+.1f}pct"
                n_sector_dampened += 1
            # Pass 2: boost neutral → directional when sector moves hard
            # Runs as a separate if (not elif) so a stock that was just dampened
            # LONG→neutral in Pass 1 can still be boosted to SHORT here if the
            # sector is down hard AND the stock itself is also falling.
            if (chg_now < -(SECTOR_THRESHOLD * 1.5) and s.get("direction") == 0):
                sym_q = live_quotes.get(sym, {}) if live_quotes else {}
                sym_chg = float(sym_q.get("changePct", 0) or 0) if isinstance(sym_q, dict) else 0.0
                if sym_chg < -(SECTOR_THRESHOLD * 0.5):
                    s["direction"] = -1
                    s["sector_boost"] = f"SECTOR_CONF_DN_{chg_now:+.1f}pct"
                    n_sector_boosted += 1
            elif (chg_now > (SECTOR_THRESHOLD * 1.5) and s.get("direction") == 0):
                sym_q = live_quotes.get(sym, {}) if live_quotes else {}
                sym_chg = float(sym_q.get("changePct", 0) or 0) if isinstance(sym_q, dict) else 0.0
                if sym_chg > (SECTOR_THRESHOLD * 0.5):
                    s["direction"] = 1
                    s["sector_boost"] = f"SECTOR_CONF_UP_{chg_now:+.1f}pct"
                    n_sector_boosted += 1

        if n_sector_dampened or n_sector_boosted:
            it_str  = f"IT={it_chg:+.1f}%"      if it_chg     is not None else ""
            pha_str = f"PHA={pharma_chg:+.1f}%"  if pharma_chg is not None else ""
            aut_str = f"AUTO={auto_chg:+.1f}%"   if auto_chg   is not None else ""
            bnk_str = f"BANK={bank_chg:+.1f}%"   if bank_chg   is not None else ""
            cem_str = f"CEMENT={cement_chg:+.1f}%" if cement_chg is not None else ""
            sectors_str = " ".join(s for s in [it_str, pha_str, aut_str, bnk_str, cem_str] if s)
            print(f"[{now.strftime('%H:%M')}] Sector: dampened={n_sector_dampened} boosted={n_sector_boosted} "
                  f"({sectors_str})")

        # ── Individual stock-level dampening removed from here — see AFTER LTP override ──

        # ── Reversal detection: scan for oversold bounces / overbought drops ──
        if _reversal_detector is not None and scores:
            reversals = _reversal_detector.scan([s["symbol"] for s in scores])
            strong = [r for r in reversals if r.reversal_score >= 0.60]
            if strong:
                setups = _reversal_detector.top_setups(reversals, n=3)
                ob = setups["oversold_bounce"]; od = setups["overbought_drop"]
                if ob or od:
                    print(f"[{now.strftime('%H:%M')}] ReversalDetector: {len(strong)} "
                          f"setups (oversold={len(ob)} overbought={len(od)})")
                    for s in ob[:2]:
                        print(f"   BOUNCE {s['symbol']:12}: score={s['score']:.2f}  {s['evidence']}")
                    for s in od[:2]:
                        print(f"   DROP   {s['symbol']:12}: score={s['score']:.2f}  {s['evidence']}")
            scores = _reversal_detector.merge_with_momentum(scores, reversals)
            n_rev = sum(1 for s in scores if s.get("reversal_override"))
            if n_rev:
                print(f"[{now.strftime('%H:%M')}] Reversal overrides: {n_rev} signals flipped")

        # ── SentinelPulse news-sentiment signal damper ─────────────────────────
        # Fetch market + per-asset news context and apply 3 dampening rules:
        #   R1: strong bear regime → neutralize weak (C/D grade) LONG signals
        #   R2: bearish asset news  → neutralize LONG; bullish asset → neutralize SHORT
        #   R3: high-importance event conflicting with ML → pull score 30% toward 0.5
        #
        # Only scored symbols with non-neutral directions are sent to SentinelPulse
        # (reduces API calls to ~30-50 vs full 285). The client has 90s LRU cache,
        # so this adds <2 seconds per cycle when cache is warm.
        _news_dampened = 0
        if _SENTINEL_OK:
            try:
                # Symbols that have an active directional signal
                active_syms = [s["symbol"] for s in scores if s.get("direction", 0) != 0]
                if active_syms:
                    try:
                        # Create a fresh event loop each cycle to avoid
                        # "Event loop is closed" error from prior asyncio.run() calls
                        import asyncio as _aio
                        try:
                            loop = _aio.get_event_loop()
                            if loop.is_closed():
                                loop = _aio.new_event_loop()
                                _aio.set_event_loop(loop)
                        except RuntimeError:
                            loop = _aio.new_event_loop()
                            _aio.set_event_loop(loop)
                        news_ctx = loop.run_until_complete(_fetch_news_batch(active_syms))
                    except Exception as _sp_loop_exc:
                        news_ctx = {}
                        print(f"[{now.strftime('%H:%M')}] [warn] SP event loop: {_sp_loop_exc}",
                              file=sys.stderr)
                    scores, _news_dampened, _news_reasons = apply_news_damper(
                        scores, news_ctx, now
                    )
                    if _news_dampened == 0:
                        mkt_regime = float(news_ctx.get("MARKET", {}).get("news_regime_score", 0))
                        mkt_sent   = float(news_ctx.get("MARKET", {}).get("news_market_sentiment", 0))
                        print(
                            f"[{now.strftime('%H:%M')}] 📰 SentinelPulse OK: "
                            f"mkt_sent={mkt_sent:+.2f} regime={mkt_regime:+.2f} "
                            f"no dampening needed"
                        )
            except Exception as _sp_exc:
                print(
                    f"[{now.strftime('%H:%M')}] [warn] SentinelPulse damper failed: {_sp_exc}",
                    file=sys.stderr,
                )

        # ── Cross-sectional rank-based direction — FINAL assignment ────────
        # Runs after ALL filters (sector, threshold, event, reversal, SP damper).
        # Ranks the REMAINING signals and picks top-15% LONG / bottom-15% SHORT.
        # This ensures we ALWAYS have balanced L+S signals regardless of the
        # model's absolute score distribution on any given day.
        # GAP-4 fix: use a separate ranking for LONG selection that gives a
        # +0.03 score bonus to verified (live-LTP) symbols, biasing LONGs toward
        # large-caps we can actually price and track intraday.
        _all_for_rank = scores   # rank ALL (including pre-filtered neutral)
        if _all_for_rank:
            n_rank = len(_all_for_rank)
            SIGNAL_PCT = 0.15
            n_each = max(5, int(n_rank * SIGNAL_PCT))
            VERIFIED_BONUS = 0.03   # boost for live-priced symbols in LONG ranking

            # LONG ranking: verified symbols get a score bonus → prefer trackable LONGs
            long_sorted = sorted(
                [(s["score"] + (VERIFIED_BONUS if s.get("has_live_ltp") else 0.0), s["symbol"])
                 for s in _all_for_rank],
                reverse=True,
            )
            # SHORT ranking: no bonus — use raw model score for all symbols
            short_sorted = sorted(
                [(s["score"], s["symbol"]) for s in _all_for_rank],
            )
            top_syms    = {sym for _, sym in long_sorted[:n_each]}
            bottom_syms = {sym for _, sym in short_sorted[:n_each]}
            for s in scores:
                if (s.get("news_dimmed") or s.get("ensemble_disagree")
                        or s.get("sector_boost") or s.get("sector_dimmed")
                        or s.get("stock_dampened") or s.get("reversal_override")):
                    continue   # don't override explicit damper/boost decisions
                sym = s["symbol"]
                if sym in top_syms:
                    s["direction"] = 1
                elif sym in bottom_syms:
                    s["direction"] = -1
                else:
                    s["direction"] = 0
            score_mean = sum(s["score"] for s in scores) / n_rank
            score_min  = min(s["score"] for s in scores)
            score_max  = max(s["score"] for s in scores)
            n_verified_long = sum(
                1 for s in scores
                if s["direction"] == 1 and s.get("has_live_ltp")
            )
            print(
                f"[{now.strftime('%H:%M')}] CrossSectional(final): "
                f"range=[{score_min:.3f},{score_max:.3f}] mean={score_mean:.3f}  "
                f"{n_each}L (verified={n_verified_long}) + {n_each}S from {n_rank} ranked"
            )
        else:
            score_mean = 0.5   # fallback if no scores

        # ── Live-LTP momentum override ─────────────────────────────────────────
        # After all filters, any symbol in live_quotes that moved ≥1.5% today
        # AND whose ML score aligns with the direction → force a signal.
        # These override direction=0 but do NOT override explicit dampers.
        LIVE_MOVE_THRESHOLD = 1.5   # GAP-3: lowered from 2.0% — catches INFY +2% moves
        n_ltp_overrides = 0
        sym_map = {s["symbol"]: s for s in scores}
        for sym, q in live_quotes.items():
            s = sym_map.get(sym)
            if s is None:
                continue   # not in scoring universe
            if s.get("news_dimmed") or s.get("stock_dampened"):
                continue   # explicit suppression — never override
            if s.get("direction", 0) != 0:
                continue   # already has a signal
            ltp_c = float(q.get("changePct", 0) or 0)
            if abs(ltp_c) < LIVE_MOVE_THRESHOLD:
                continue   # move too small
            # sector_dimmed guard: only block if dimming direction CONFLICTS with LTP move
            sd = s.get("sector_dimmed", "")
            if sd and "UP" in sd and ltp_c < 0:
                continue   # sector is up, stock falling — conflicting, skip
            if sd and "DN" in sd and ltp_c > 0:
                continue   # sector is down, stock rising — conflicting, skip
            # GAP-3: use dynamically computed score_mean (not hardcoded 0.628)
            if ltp_c > 0 and s["score"] >= score_mean:
                s["direction"] = 1
                s["ltp_override"] = f"LIVE_MOVE_{ltp_c:+.1f}pct"
                n_ltp_overrides += 1
            elif ltp_c < 0 and s["score"] < score_mean:
                s["direction"] = -1
                s["ltp_override"] = f"LIVE_MOVE_{ltp_c:+.1f}pct"
                n_ltp_overrides += 1
        if n_ltp_overrides:
            print(f"[{now.strftime('%H:%M')}] LTP override: {n_ltp_overrides} signals added from live moves")

        # ── Individual stock-level dampening (GAP-2) — runs LAST, final word ─────
        # Suppress SHORT on a stock that is individually up >1.5% today, and LONG
        # on a stock that is individually down >1.5%. Runs after the final CS and
        # LTP override so it has the last word before signals are written.
        # This is why it runs here and NOT earlier in the pipeline.
        STOCK_DAMP_THRESHOLD = 1.5   # % individual move to suppress conflicting signal
        n_stock_dampened = 0
        for s in scores:
            if s.get("news_dimmed") or s.get("sector_boost"):
                continue   # don't override explicit directional overrides
            sym = s["symbol"]
            sym_q = live_quotes.get(sym, {})
            if not isinstance(sym_q, dict):
                continue
            stock_chg = float(sym_q.get("changePct", 0) or 0)
            if s.get("direction") == -1 and stock_chg > STOCK_DAMP_THRESHOLD:
                # Shorting a stock that is rising — suppress
                s["direction"] = 0
                s["stock_dampened"] = f"STOCK_UP_{stock_chg:+.1f}pct"
                n_stock_dampened += 1
            elif s.get("direction") == 1 and stock_chg < -STOCK_DAMP_THRESHOLD:
                # Going long on a stock that is falling — suppress
                s["direction"] = 0
                s["stock_dampened"] = f"STOCK_DN_{stock_chg:+.1f}pct"
                n_stock_dampened += 1
        if n_stock_dampened:
            print(f"[{now.strftime('%H:%M')}] Stock-level dampened: {n_stock_dampened} signals")

        # Sep-30 finding: LONG book avg -2.5% while SHORT avg +1.35%.  Root cause:
        # the model scores LONGs and SHORTs independently — when markets drift up
        # the SHORT book partially hedges by design, but LONGs accumulate market
        # beta exposure with no offsetting hedge.  Net result: LONG P&L ≈ alpha - β×market.
        #
        # Fix: After final signal processing, compute net LONG exposure and recommend
        # a NIFTY futures SHORT of equivalent notional × avg_beta to flatten beta.
        # This converts the LONG book from "raw stock" → "pure alpha" exposure.
        #
        # Implementation: we add a synthetic "NIFTY_SHORT_HEDGE" signal to the scores
        # list.  This signal is NOT emitted as a trade recommendation — it is logged
        # as a HEDGE signal in the session snapshot and dashboard for operator review.
        # Actual hedge sizing requires knowing position sizes (out of scope for signal engine).

        active_longs  = [s for s in scores if s.get("direction") == 1]
        active_shorts = [s for s in scores if s.get("direction") == -1]
        n_long        = len(active_longs)
        n_short       = len(active_shorts)
        net_long      = n_long - n_short      # net directional exposure (unit positions)

        # Compute portfolio-level beta from live quotes:
        # beta_i ≈ corr(stock_i, NIFTY) × (σ_stock / σ_nifty)
        # For simplicity use 60-day rolling beta approximation from EOD data.
        # Fallback: assume avg beta = 1.0 for all F&O stocks (close to empirical avg).
        AVG_PORTFOLIO_BETA = 1.05   # empirical avg for NSE F&O large-caps

        # NIFTY 1% ≈ equivalent notional in NIFTY futures (1 lot = 50 NIFTY units)
        # We express hedge as a fractional NIFTY position, not hard lots.
        nifty_hedge_units = net_long * AVG_PORTFOLIO_BETA   # how many NIFTY "position units" to short

        beta_hedge: dict = {
            "n_long":            n_long,
            "n_short":           n_short,
            "net_long_exposure": net_long,
            "avg_portfolio_beta": AVG_PORTFOLIO_BETA,
            "nifty_hedge_units": round(nifty_hedge_units, 2),
            "hedge_direction":   "SHORT" if nifty_hedge_units > 0 else ("LONG" if nifty_hedge_units < 0 else "FLAT"),
            "nifty_ltp":         float(nifty_q.get("ltp") or 0),
            "nifty_chg_pct":     round(nifty_chg_now, 3),
        }

        if net_long > 2:
            # Meaningful net long exposure → recommend NIFTY SHORT hedge
            print(
                f"[{now.strftime('%H:%M')}] 🔷 Beta-Neutral Hedge: "
                f"net_long={net_long}  NIFTY SHORT {nifty_hedge_units:.1f} units "
                f"(beta={AVG_PORTFOLIO_BETA})  NIFTY={nifty_q.get('ltp','?')} ({nifty_chg_now:+.2f}%)"
            )
        elif net_long < -2:
            # Net short book → recommend NIFTY LONG hedge
            print(
                f"[{now.strftime('%H:%M')}] 🔷 Beta-Neutral Hedge: "
                f"net_short={abs(net_long)}  NIFTY LONG {abs(nifty_hedge_units):.1f} units  "
                f"NIFTY={nifty_q.get('ltp','?')} ({nifty_chg_now:+.2f}%)"
            )
        else:
            # Near market-neutral — no hedge needed
            beta_hedge["nifty_hedge_units"] = 0.0
            beta_hedge["hedge_direction"]   = "FLAT"

        # ── Phil: log ALL 218 forecasts (not just 26 tracked positions) ───────
        if _forecast_ledger is not None:
            n_logged = _forecast_ledger.record_session(
                scores=scores, session_date=session_date,
                nifty_chg=nifty_chg_now, model_version=schema,
            )
            if sample_n == 1:  # only print on first sample
                print(f"[{now.strftime('%H:%M')}] ForecastLedger: {n_logged} forecasts logged")

        # P&L calculation — reuse the quotes we already fetched above
        # (avoids a second 218-symbol fetch round-trip)
        all_fp_quotes = {sym: live_quotes[sym] for sym in fp_signals if sym in live_quotes}
        # Top-up any symbols that weren't in the prefetch
        missing = [sym for sym in fp_signals if sym not in live_quotes]
        if missing:
            extra = get_all_quotes_fast(missing)
            live_quotes.update(extra)
            all_fp_quotes.update(extra)
        pnl = calc_pnl(fp_signals, live_quotes, excluded)

        # ── AlphaForge: write latest_scores.json for UI consumption ──────────
        _write_latest_scores(
            scores=scores, session_date=session_date, schema=schema,
            nifty_chg=nifty_chg_now, nifty_ltp=nifty_q.get("ltp"),
            market_open=True, pnl=pnl, beta_hedge=beta_hedge,
        )

        # ── Share live_quotes with signal_tracker ─────────────────────────────
        # The tracker reads this file instead of making its own API calls,
        # so it doesn't compete for the 500-req/60s rate limit.
        try:
            lq_path = SESSION_DIR / "live_quotes.json"
            lq_tmp  = lq_path.with_suffix(".tmp")
            lq_tmp.write_text(json.dumps({
                "generated_at": datetime.now(tz=timezone.utc).isoformat(),  # true UTC
                "quotes": {sym: dict(q) for sym, q in live_quotes.items()
                           if isinstance(q, dict)},
            }))
            lq_tmp.replace(lq_path)
        except Exception as _lq_err:
            pass  # non-fatal

        # Dashboard
        print_dashboard(sample_n, now, nifty_q, scores, pnl, mins)

        # Save sample
        sample = {
            "timestamp":  now.isoformat(),
            "sample_n":   sample_n,
            "nifty_ltp":  nifty_q.get("ltp"),
            "nifty_chg":  nifty_q.get("changePct", 0),
            "n_scored":   len(scores),
            "n_long":     sum(1 for s in scores if s["direction"] == 1),
            "n_short":    sum(1 for s in scores if s["direction"] == -1),
            "top_long":   sorted([s for s in scores if s["direction"]==1],  key=lambda x: -x["score"])[:5],
            "top_short":  sorted([s for s in scores if s["direction"]==-1], key=lambda x:  x["score"])[:5],
            "pnl":        pnl,
            "beta_hedge": beta_hedge,
            "news_dampened": _news_dampened,
        }
        all_samples.append(sample)
        with SESSION_LOG.open("a") as f:
            f.write(json.dumps(sample) + "\n")

        # Wait until next sample
        sleep_secs = min(SAMPLE_MINS * 60, max(30, int(mins * 60) - 60))
        print(f"\n  Next sample in {sleep_secs//60}m{sleep_secs%60:02d}s  ({now.strftime('%H:%M')} → {(now+timedelta(seconds=sleep_secs)).strftime('%H:%M')} IST)")
        time.sleep(sleep_secs)

    print("\nAutorun complete.")


def _write_latest_scores(
    scores: list[dict],
    session_date: str,
    schema: str,
    nifty_chg: float,
    nifty_ltp: float | None,
    market_open: bool,
    pnl: dict | None = None,
    beta_hedge: dict | None = None,
) -> None:
    """Atomic snapshot of all scored symbols written after every cycle.

    AlphaForge reads this file via GET /v2/signals/latest — always fresh,
    never requires replaying the session log.  Written via a temp file so
    any concurrent reader never sees a partial write.
    """
    try:
        def _conviction(score: float) -> str:
            dist = abs(score - 0.5)
            if dist >= 0.40: return "S"
            if dist >= 0.30: return "A"
            if dist >= 0.20: return "B"
            if dist >= 0.10: return "C"
            return "D"

        # Rank by conviction (distance from 0.5); ties broken by symbol
        ranked = sorted(scores, key=lambda s: abs(s["score"] - 0.5), reverse=True)
        # Build per-symbol live P&L lookup for the UI
        pnl_by_sym: dict[str, dict] = {}
        if pnl:
            for pos in pnl.get("positions", []):
                pnl_by_sym[pos["symbol"]] = {
                    "net_pct":    pos.get("net_pct"),
                    "gross_pct":  pos.get("gross_pct"),
                    "entry":      pos.get("entry"),
                    "ltp":        pos.get("ltp"),
                    "chg_today_pct": pos.get("chg_today_pct"),
                }

        signals_out = []
        for i, s in enumerate(ranked):
            entry = {
                "symbol":       s["symbol"],
                "score":        s["score"],
                "direction":    s["direction"],   # +1 LONG | -1 SHORT | 0 neutral
                "data_date":    s.get("data_date", ""),
                "rank":         i + 1,
                "conviction":   _conviction(s["score"]),
                "has_live_ltp": s.get("has_live_ltp", False),
                # Diagnostic flags — present only when set
                **({"ltp_override":      s["ltp_override"]}      if s.get("ltp_override")      else {}),
                **({"sector_boost":      s["sector_boost"]}      if s.get("sector_boost")      else {}),
                **({"sector_dimmed":     s["sector_dimmed"]}     if s.get("sector_dimmed")     else {}),
                **({"stock_dampened":    s["stock_dampened"]}    if s.get("stock_dampened")    else {}),
                **({"news_dimmed":       s["news_dimmed"]}       if s.get("news_dimmed")       else {}),
                **({"reversal_override": s["reversal_override"]} if s.get("reversal_override") else {}),
                **({"filter_reason":     s["filter_reason"]}     if s.get("filter_reason")     else {}),
            }
            if s["symbol"] in pnl_by_sym:
                entry["live_pnl"] = pnl_by_sym[s["symbol"]]
            signals_out.append(entry)

        snapshot = {
            "session_date": session_date,
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "market_open":  market_open,
            "model_version": schema,
            "n_scored":     len(signals_out),
            "n_long":       sum(1 for s in signals_out if s["direction"] == 1),
            "n_short":      sum(1 for s in signals_out if s["direction"] == -1),
            "nifty_chg":    round(nifty_chg, 4),
            "nifty_ltp":    nifty_ltp,
            "session_pnl":  {
                "mean_net":   pnl.get("mean_net", 0) if pnl else None,
                "win_rate":   pnl.get("win_rate", 0) if pnl else None,
                "n_positions": pnl.get("n", 0) if pnl else 0,
            },
            # Beta-neutral hedge recommendation for operator / AlphaForge UI
            "beta_hedge": beta_hedge or {},
            "signals": signals_out,
        }
        tmp = LATEST_SCORES_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(snapshot, default=str))
        tmp.replace(LATEST_SCORES_PATH)
    except Exception as e:
        print(f"[warn] Could not write latest_scores.json: {e}", file=sys.stderr)


def _append_close_to_report(now: datetime, nifty_q: dict, pnl: dict, n_samples: int) -> None:
    """Append end-of-day close summary to LIVE_SESSION_REPORT.md."""
    try:
        existing = REPORT_PATH.read_text() if REPORT_PATH.exists() else ""
        close_section = f"""

---

## CLOSE-OF-DAY UPDATE — {now.strftime('%H:%M IST')}

### Market Close Summary

| Metric | Value |
|--------|-------|
| Session samples | {n_samples} |
| NIFTY close | {nifty_q.get('ltp','?')} ({nifty_q.get('changePct',0):+.2f}%) |
| SHORT mean P&L | {pnl.get('short_mean',0):+.4f}% |
| LONG mean P&L | {pnl.get('long_mean',0):+.4f}% |
| Mean net P&L | {pnl.get('mean_net',0):+.4f}% |
| Win rate | {pnl.get('win_rate',0):.1f}% |
| Positions tracked | {pnl.get('n',0)} |

*Report auto-updated at {now.strftime('%H:%M IST')} by autorun_till_close.py*
"""
        REPORT_PATH.write_text(existing.rstrip() + close_section)
    except Exception as e:
        print(f"[warn] Could not update report: {e}")


if __name__ == "__main__":
    main()
