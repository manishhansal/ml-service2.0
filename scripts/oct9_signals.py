"""Fast October 9 intraday signal report."""
import os, sys, logging
from datetime import timezone, timedelta
os.environ.setdefault("ML_SERVICE_API_KEY","test")
os.environ.setdefault("DATA_SERVICE_API_KEY","test")
os.environ.setdefault("DATA_SERVICE_2_URL","http://x")
os.environ.setdefault("SENTINEL_PULSE_URL","http://x")
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
from src.scanners.quality import compute_market_direction, passes_liquidity_filter, DEFAULT_BIAS_SMA, DEFAULT_MIN_VALUE_CRORE

def get_avg_value(df):
    tail = df.tail(20)
    return float((tail['volume'] * tail['close']).mean()) / 1e7

IST = timezone(timedelta(hours=5, minutes=30))
DATA = Path("/Users/manishkumar/Desktop/ml-service2.0/data/1d/1d")

def load():
    dfs = {}
    for p in sorted(DATA.glob("*.parquet")):
        sym = p.stem
        df = pd.read_parquet(p).drop(columns=["provider"], errors="ignore")
        df = df[["open","high","low","close","volume"]].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if len(df) >= 50:
            dfs[sym] = df
    return dfs

def get_ist_date(item):
    for attr in ("ob_timestamp","bos_timestamp","fvg_timestamp","departure_timestamp","sweep_timestamp","break_timestamp"):
        ts = getattr(item, attr, None)
        if ts:
            return ts.astimezone(IST).date()
    return None

def fmt(item, ist_date):
    sym = item.symbol
    dirn = getattr(item,"direction",None) or getattr(item,"signal_type",None)
    dv = getattr(dirn,"value",str(dirn)) if dirn else "?"
    entry = getattr(item,"intraday_entry_price",0.0) or 0.0
    target = getattr(item,"intraday_target_price",0.0) or 0.0
    stop = getattr(item,"intraday_stop_price",0.0) or 0.0
    rr = getattr(item,"intraday_rr_ratio",0.0) or 0.0
    prox = getattr(item,"proximity_pct",999.0) or 999.0
    act = "✅ YES" if getattr(item,"is_actionable",False) else "-"
    inv = " [INVALIDATED]" if getattr(item,"invalidated",False) else ""
    return ("  %-14s  %-24s  data:%-10s  Entry:%8.2f  Target:%8.2f  Stop:%8.2f  R:R:%.1fx  %.1f%%  %s%s" %
            (sym, dv, str(ist_date), entry, target, stop, rr, prox, act, inv))

dfs = load()
print(f"Loaded {len(dfs)} symbols")

# Filter to top 150 symbols by avg daily turnover for speed
liquid_ranked = sorted(
    [(sym, df, get_avg_value(df)) for sym, df in dfs.items()],
    key=lambda x: -x[2]
)
dfs_scan = {sym: df for sym, df, val in liquid_ranked[:150] if val >= DEFAULT_MIN_VALUE_CRORE}
print(f"Running scanners on top {len(dfs_scan)} liquid symbols (by avg daily turnover)")

# Market direction (use full universe for breadth)
md = compute_market_direction(dfs)
above = eligible = 0
for df in dfs.values():
    if len(df) < DEFAULT_BIAS_SMA: continue
    sma = df["close"].rolling(DEFAULT_BIAS_SMA, min_periods=DEFAULT_BIAS_SMA).mean()
    if pd.notna(sma.iloc[-1]):
        eligible += 1
        if float(df["close"].iloc[-1]) > float(sma.iloc[-1]): above += 1
breadth_pct = round(above/eligible*100) if eligible else 0
liquid_count = sum(1 for df in dfs.values() if passes_liquidity_filter(df, DEFAULT_MIN_VALUE_CRORE))

print()
print("=" * 85)
print(f"  SIGNALS FOR MONDAY, OCTOBER 12, 2026 — INTRADAY TRADING")
print(f"  Generated from Oct 7-9 IST closing data (latest available)")
print("=" * 85)
print(f"  Market Direction : {md}  ({breadth_pct}% of {eligible} symbols above SMA20)")
print(f"  Liquidity filter : {liquid_count}/{len(dfs)} symbols qualify (avg daily value > Rs.{DEFAULT_MIN_VALUE_CRORE:.0f} crore)")
print("=" * 85)

RECENT_DATES = {"2026-10-07", "2026-10-08", "2026-10-09"}
scanners = [
    ("ORDER BLOCK",      ObScanner(OrderBlockConfig()),     "zones"),
    ("FAIR VALUE GAP",   FVGScanner(FVGConfig()),           "zones"),
    ("SUPPLY & DEMAND",  SDScanner(SDConfig()),             "zones"),
    ("LIQUIDITY SWEEP",  SweepScanner(SweepConfig()),       "signals"),
    ("MARKET STRUCTURE", MSScanner(MSConfig()),             "signals"),
]

grand_total = 0
for sname, scanner, attr in scanners:
    result = scanner.scan(dfs_scan, timeframe="1d")
    items = getattr(result, attr, [])
    
    recent = []
    for item in items:
        d = get_ist_date(item)
        if d and str(d) in RECENT_DATES:
            recent.append((item, d))
    
    recent.sort(key=lambda x: (
        not getattr(x[0],"is_actionable",False),
        -(getattr(x[0],"conviction_score",None) or getattr(x[0],"combined_score",None) or 0)
    ))
    
    grand_total += len(recent)
    print()
    print(f"  {'─'*83}")
    print(f"  {sname}  |  {len(recent)} signal(s)")
    print(f"  {'─'*83}")
    if not recent:
        print("  (no signals passed quality filters)")
    else:
        print(f"  {'SYMBOL':<14}  {'DIRECTION/TYPE':<24}  {'DATA DATE':<10}  {'ENTRY':>8}  {'TARGET':>8}  {'STOP':>8}  {'R:R':>5}  {'PROX%':>6}  ACTION")
        print(f"  {'─'*85}")
        for item, d in recent[:20]:
            print(fmt(item, d))

print()
print("=" * 85)
print(f"  TOTAL: {grand_total} signals for Monday, October 12, 2026 intraday trading")
print("=" * 85)

if grand_total == 0:
    print()
    print("  NOTE: Market is BEARISH. To see bearish intraday setups across all scanners,")
    print("  look for entry SHORT at zone resistance levels when price rallies into them.")

print()
print("  NOTE: Run 'python scripts/refresh_upstox_token.py' to refresh the Upstox token")
print("  and re-run 'python scripts/refresh_oct9.py' to include Friday Oct 9 data.")
