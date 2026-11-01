"""Pre-market SMC signal report for NSE F&O — any trading day.

Automatically detects the latest available data date and generates signals
for the next trading day.  Run the evening before market opens.

Usage:
    python scripts/monday_signals.py
"""
from __future__ import annotations
import os, sys, logging
from datetime import date, datetime, timezone, timedelta

os.environ.setdefault("ML_SERVICE_API_KEY", "test")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://x")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://x")
sys.path.insert(0, "/Users/manishkumar/Desktop/ml-service2.0")

import structlog
logging.basicConfig(level=logging.ERROR)
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))

import pandas as pd
from pathlib import Path

from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.scanner import ObScanner
from src.scanners.fair_value_gap.config import FVGConfig
from src.scanners.fair_value_gap.scanner import FVGScanner
from src.scanners.supply_demand.config import SDConfig
from src.scanners.supply_demand.scanner import SDScanner
from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.scanner import SweepScanner
from src.scanners.market_structure.config import MSConfig
from src.scanners.market_structure.scanner import MSScanner
from src.scanners.quality import compute_market_direction, passes_liquidity_filter
from src.scanners.quality import DEFAULT_BIAS_SMA, DEFAULT_MIN_VALUE_CRORE

IST         = timezone(timedelta(hours=5, minutes=30))
DATA        = Path("/Users/manishkumar/Desktop/ml-service2.0/data/1d/1d")
MS_BARS     = 250   # limit MS to last ~1 year for speed
LOOKBACK_TD = 20    # how many trading days to include in RECENT window
MAX_PROX    = 10.0  # skip signals where price has moved >10% away from entry


# ── helpers ───────────────────────────────────────────────────────────────────

def next_trading_day(d: date) -> date:
    """Next Mon–Fri trading day after d (handles weekends)."""
    delta = 3 if d.weekday() == 4 else (2 if d.weekday() == 5 else 1)
    return d + timedelta(days=delta)


def build_recent_set(latest: date, n: int = LOOKBACK_TD) -> set[str]:
    """Return ISO date strings for the last n Mon–Fri days ending on latest."""
    result: set[str] = set()
    d = latest
    while len(result) < n:
        if d.weekday() < 5:
            result.add(str(d))
        d -= timedelta(days=1)
    return result


def get_signal_date(item: object) -> date | None:
    for attr in ("ob_timestamp", "bos_timestamp", "fvg_timestamp",
                 "departure_timestamp", "sweep_timestamp", "break_timestamp"):
        ts = getattr(item, attr, None)
        if ts:
            return ts.astimezone(IST).date()
    return None


def fmt_row(item: object, d: date) -> str:
    sym  = item.symbol                                      # type: ignore[attr-defined]
    dirn = (getattr(item, "direction", None)
            or getattr(item, "signal_type", None))
    dv   = getattr(dirn, "value", str(dirn)) if dirn else "?"
    e    = getattr(item, "intraday_entry_price",  0.0) or 0.0
    t    = getattr(item, "intraday_target_price", 0.0) or 0.0
    s    = getattr(item, "intraday_stop_price",   0.0) or 0.0
    rr   = getattr(item, "intraday_rr_ratio",     0.0) or 0.0
    pr   = getattr(item, "proximity_pct",       999.0) or 999.0
    ac   = "✅" if getattr(item, "is_actionable",  False) else "  "
    inv  = " [X]"    if getattr(item, "invalidated",  False) else ""
    chch = " ★ChoCh" if getattr(item, "is_choch",     False) else ""
    return (
        f"  {ac} {sym:<14} {dv:<22} "
        f"Entry:{e:>8.2f}  Tgt:{t:>8.2f}  Stp:{s:>8.2f}  "
        f"R:R:{rr:.1f}x  {pr:.1f}%  {d}{chch}{inv}"
    )


# ── load data ─────────────────────────────────────────────────────────────────

dfs: dict[str, pd.DataFrame] = {}
for p in sorted(DATA.glob("*.parquet")):
    sym = p.stem
    df  = pd.read_parquet(p).drop(columns=["provider"], errors="ignore")
    df  = df[["open", "high", "low", "close", "volume"]].dropna()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if len(df) >= 50:
        dfs[sym] = df

print(f"Loaded {len(dfs)} symbols")

# Detect latest IST data date from the most recent parquet bar
sample      = next(iter(dfs.values()))
latest_ist  = sample.index[-1].astimezone(IST).date()
next_td     = next_trading_day(latest_ist)
RECENT      = build_recent_set(latest_ist, LOOKBACK_TD)

DAY_NAMES   = ["Monday", "Tuesday", "Wednesday", "Thursday",
               "Friday", "Saturday", "Sunday"]
next_day    = DAY_NAMES[next_td.weekday()]

# ── market breadth stats ──────────────────────────────────────────────────────

md      = compute_market_direction(dfs)
above   = eligible = 0
for df in dfs.values():
    if len(df) < DEFAULT_BIAS_SMA:
        continue
    sma = df["close"].rolling(DEFAULT_BIAS_SMA, min_periods=DEFAULT_BIAS_SMA).mean()
    if pd.notna(sma.iloc[-1]):
        eligible += 1
        if float(df["close"].iloc[-1]) > float(sma.iloc[-1]):
            above += 1
breadth_pct  = round(above / eligible * 100) if eligible else 0
liquid_count = sum(1 for df in dfs.values()
                   if passes_liquidity_filter(df, DEFAULT_MIN_VALUE_CRORE))

# ── header ────────────────────────────────────────────────────────────────────

W = 90
print()
print("=" * W)
print(f"  {next_day.upper()} {next_td.strftime('%B %d, %Y').upper()} — INTRADAY SIGNALS")
print(f"  (Data: {latest_ist} IST close  |  RECENT window: last {LOOKBACK_TD} trading days"
      f"  |  Max proximity: {MAX_PROX:.0f}%)")
print("=" * W)
print(f"  Market Direction : {md}  ({breadth_pct}% of {eligible} symbols above SMA{DEFAULT_BIAS_SMA})")
print(f"  Liquidity filter : {liquid_count}/{len(dfs)} qualify (>₹{DEFAULT_MIN_VALUE_CRORE:.0f}cr avg daily value)")
print("=" * W)

# ── scanners ──────────────────────────────────────────────────────────────────

dfs_ms = {
    sym: df.iloc[-MS_BARS:] if len(df) > MS_BARS else df
    for sym, df in dfs.items()
}

scanners = [
    ("ORDER BLOCK",      ObScanner(OrderBlockConfig()),   dfs,    "zones"),
    ("FAIR VALUE GAP",   FVGScanner(FVGConfig()),          dfs,    "zones"),
    ("SUPPLY & DEMAND",  SDScanner(SDConfig()),             dfs,    "zones"),
    ("LIQUIDITY SWEEP",  SweepScanner(SweepConfig()),       dfs,    "signals"),
    ("MARKET STRUCTURE", MSScanner(MSConfig()),             dfs_ms, "signals"),
]

total = 0
for name, scanner, data_dict, attr in scanners:
    result = scanner.scan(data_dict, timeframe="1d")
    items  = getattr(result, attr, [])

    # Filter 1: signal must fall within the RECENT window
    # Filter 2: proximity_pct must be <= MAX_PROX (not a stale entry far from price)
    recent = [
        (i, get_signal_date(i))
        for i in items
        if (d := get_signal_date(i)) and str(d) in RECENT
        and (getattr(i, "proximity_pct", 999.0) or 999.0) <= MAX_PROX
    ]

    # Sort: actionable first, then by conviction / combined score
    recent.sort(key=lambda x: (
        not getattr(x[0], "is_actionable", False),
        -(getattr(x[0], "conviction_score", None)
          or getattr(x[0], "combined_score", None) or 0),
    ))

    total += len(recent)
    print(f"\n  ── {name}  ({len(recent)} signals) " + "─" * (60 - len(name)))
    if not recent:
        print("  (no signals within proximity threshold)")
    else:
        for item, d in recent[:25]:
            print(fmt_row(item, d))

# ── footer ────────────────────────────────────────────────────────────────────

print()
print("=" * W)
print(f"  TOTAL: {total} signals within {MAX_PROX:.0f}% of entry for"
      f" {next_day} {next_td}  |  Market: {md} ({breadth_pct}% above SMA{DEFAULT_BIAS_SMA})")
print(f"  ✅ = price currently within 2% of entry (trade ready at open)")
print("=" * W)
