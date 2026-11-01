"""Last-trading-day signal report for all 5 SMC scanners on real NSE F&O data.

Shows only high-confidence, trend-aligned, liquid signals.  Also detects
multi-scanner confluence and validates previous-day signals against today's close.

Timestamps are displayed in IST (UTC+5:30).  Parquet candles close at
18:30:00 UTC = 00:00:00 IST next calendar day, so the IST date is the
correct "trading day" label.

Usage:
    python scripts/last_day_signals.py
    python scripts/last_day_signals.py --date 2025-07-10
"""
from __future__ import annotations
import argparse
import datetime
import os
import sys
from datetime import timezone, timedelta

IST = timezone(timedelta(hours=5, minutes=30))

os.environ.setdefault("ML_SERVICE_API_KEY", "test")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://x")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://x")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import pandas as pd
from pathlib import Path
from typing import Any

import logging
import structlog

# Suppress verbose scanner debug output
logging.basicConfig(level=logging.WARNING)
structlog.configure(
    wrapper_class=structlog.make_filtering_bound_logger(logging.WARNING),
)

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
from src.scanners.quality import (
    compute_market_direction,
    passes_liquidity_filter,
    DEFAULT_BIAS_SMA,
    DEFAULT_MIN_VALUE_CRORE,
)

DATA = Path(PROJECT_ROOT) / "data" / "1d" / "1d"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_all_symbols() -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    for p in sorted(DATA.glob("*.parquet")):
        sym = p.stem
        df = pd.read_parquet(p).drop(columns=["provider"], errors="ignore")
        df = df[["open", "high", "low", "close", "volume"]].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        if len(df) >= 50:
            dfs[sym] = df
    return dfs


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_timestamp(item: Any):
    for attr in ("ob_timestamp", "bos_timestamp", "fvg_timestamp",
                 "departure_timestamp", "sweep_timestamp", "break_timestamp"):
        ts = getattr(item, attr, None)
        if ts is not None:
            return ts
    return None


def get_timestamp_ist(item: Any):
    """Return the signal's IST calendar date, or None if no timestamp found."""
    ts = get_timestamp(item)
    if ts is None:
        return None
    return ts.astimezone(IST).date()


def get_dirn_val(item: Any) -> str:
    dirn = getattr(item, "direction", None) or getattr(item, "signal_type", None)
    return (getattr(dirn, "value", str(dirn)).lower()) if dirn else ""


def fmt_item(item: Any, confluence_tag: str = "") -> str:
    sym    = item.symbol
    dirn   = (getattr(item, "direction", None)
              or getattr(item, "signal_type", None)
              or getattr(item, "zone_type", None))
    dirn_str   = getattr(dirn, "value", str(dirn)) if dirn else "?"
    entry      = getattr(item, "intraday_entry_price", 0.0) or 0.0
    target     = getattr(item, "intraday_target_price", 0.0) or 0.0
    stop       = getattr(item, "intraday_stop_price", 0.0) or 0.0
    rr         = getattr(item, "intraday_rr_ratio", 0.0) or 0.0
    prox       = getattr(item, "proximity_pct", 999.0) or 999.0
    actionable = getattr(item, "is_actionable", False)
    action_str = "YES" if actionable else "-"
    row = "  %-14s  %-22s  %8.2f  %8.2f  %8.2f  %4.1fx  %6.1f%%  %-5s" % (
        sym, dirn_str, entry, target, stop, rr, prox, action_str)
    if confluence_tag:
        row += "  " + confluence_tag
    return row


def scan_all(
    scanner: Any,
    syms: dict[str, pd.DataFrame],
    timeframe: str = "1d",
) -> list[Any]:
    """Run a scanner and return all returned items."""
    result = scanner.scan(syms, timeframe=timeframe)
    return getattr(result, "zones", None) or getattr(result, "signals", [])


def on_date(items: list[Any], target_date: Any) -> list[Any]:
    """Filter items by signal timestamp converted to IST date."""
    return [i for i in items if get_timestamp_ist(i) == target_date]


def sort_by_quality(items: list[Any]) -> list[Any]:
    """Actionable signals first, then descending R:R ratio."""
    return sorted(
        items,
        key=lambda x: (
            not getattr(x, "is_actionable", False),
            -(getattr(x, "intraday_rr_ratio", 0.0) or 0.0),
        ),
    )


def find_confluence_symbols(results_by_scanner: dict[str, list[Any]]) -> set[str]:
    """Return symbols that appear in 2 or more scanners."""
    scanner_syms = {sname: {i.symbol for i in items}
                    for sname, items in results_by_scanner.items()}
    all_syms_sets = list(scanner_syms.values())
    if not all_syms_sets:
        return set()
    universe: set[str] = set().union(*all_syms_sets)
    return {
        sym for sym in universe
        if sum(1 for syms in scanner_syms.values() if sym in syms) >= 2
    }


def check_outcome(
    signal: Any,
    all_syms: dict[str, pd.DataFrame],
    outcome_date: Any,
) -> bool | None:
    """True = signal direction was correct on outcome_date, False = wrong, None = N/A."""
    df = all_syms.get(signal.symbol)
    if df is None:
        return None
    mask = [d == outcome_date for d in df.index.date]
    today_rows = df[mask]
    if len(today_rows) == 0:
        return None
    close_today = float(today_rows["close"].iloc[-1])
    entry = getattr(signal, "intraday_entry_price", 0.0) or 0.0
    if entry == 0.0:
        return None
    dv = get_dirn_val(signal)
    is_bullish = "bull" in dv or "demand" in dv
    return (close_today > entry) if is_bullish else (close_today < entry)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── CLI args ──────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Last-day SMC signal report (dates in IST)")
    parser.add_argument(
        "--date", type=str, default=None,
        help="Target IST trading date YYYY-MM-DD (default: latest available)")
    args, _ = parser.parse_known_args()

    print("Loading NSE F&O symbols from data/1d/1d/ ...")
    all_syms = load_all_symbols()
    if not all_syms:
        print("No symbols found. Check data/1d/1d/ path.")
        return

    # Convert the last candle's UTC timestamp → IST to get the true trading date
    sample_df = next(iter(all_syms.values()))
    last_ts_ist = sample_df.index[-1].astimezone(IST)
    last_date_ist = last_ts_ist.date()

    # Determine target date (IST)
    if args.date:
        target_date = datetime.date.fromisoformat(args.date)
    else:
        target_date = last_date_ist

    # Previous IST date (for validation section)
    prev_ts_ist = (sample_df.index[-2].astimezone(IST)
                   if len(sample_df) >= 2 else None)
    prev_date = prev_ts_ist.date() if prev_ts_ist is not None else None

    print("Loaded %d symbols  |  Latest IST data: %s  |  Target date: %s" % (
        len(all_syms), last_date_ist, target_date))

    # ── Next trading day (IST) ────────────────────────────────────────────────
    # Friday (weekday 4) → skip to Monday (+3 days); otherwise +1 day.
    days_ahead = 3 if target_date.weekday() == 4 else 1
    next_trading = target_date + datetime.timedelta(days=days_ahead)

    # ── Market breadth and liquidity stats ────────────────────────────────────
    market_direction = compute_market_direction(all_syms)

    above = eligible = 0
    for df in all_syms.values():
        if len(df) < DEFAULT_BIAS_SMA:
            continue
        sma = df["close"].rolling(DEFAULT_BIAS_SMA, min_periods=DEFAULT_BIAS_SMA).mean()
        if pd.notna(sma.iloc[-1]):
            eligible += 1
            if float(df["close"].iloc[-1]) > float(sma.iloc[-1]):
                above += 1

    breadth_pct  = round(above / eligible * 100) if eligible > 0 else 0
    liquid_count = sum(1 for df in all_syms.values()
                       if passes_liquidity_filter(df, DEFAULT_MIN_VALUE_CRORE))

    print()
    print("=" * 80)
    print("  Generated from %s (IST) closing data → Signals to watch on %s intraday" % (
        target_date, next_trading))
    print("=" * 80)
    print("  Market Direction: %s (%d%% of %d symbols above SMA%d)" % (
        market_direction, breadth_pct, eligible, DEFAULT_BIAS_SMA))
    print("  Liquidity filter: %d/%d symbols qualify (avg daily value > \u20b9%.0f crore)" % (
        liquid_count, len(all_syms), DEFAULT_MIN_VALUE_CRORE))
    print("  Data from: %s (IST)  |  Signals to watch on: %s" % (target_date, next_trading))
    print("  Showing quality-filtered signals (trend-aligned, R:R \u2265 1.0 for zones, \u2265 1.5 for MS/Sweep, liquid)")
    print("=" * 80)

    # ── Run each scanner ONCE with quality filters ON ─────────────────────────
    scanner_defs = [
        ("Order Block",      ObScanner(OrderBlockConfig())),
        ("Fair Value Gap",   FVGScanner(FVGConfig())),
        ("Supply & Demand",  SDScanner(SDConfig())),
        ("Liquidity Sweep",  SweepScanner(SweepConfig())),
        ("Market Structure", MSScanner(MSConfig())),
    ]

    all_items_by_scanner: dict[str, list[Any]] = {}
    for sname, scanner in scanner_defs:
        all_items_by_scanner[sname] = scan_all(scanner, all_syms)

    # Items on target_date for each scanner
    today_by_scanner: dict[str, list[Any]] = {
        sname: on_date(items, target_date)
        for sname, items in all_items_by_scanner.items()
    }

    # ── Multi-scanner confluence detection ────────────────────────────────────
    confluence_syms = find_confluence_symbols(today_by_scanner)

    # ── Print per-scanner signal tables ──────────────────────────────────────
    grand_total = 0
    for sname, items in today_by_scanner.items():
        sorted_items = sort_by_quality(items)
        grand_total += len(sorted_items)

        print()
        print("-" * 80)
        print("  %-20s  |  %d signal(s) on %s (IST)" % (sname.upper(), len(sorted_items), target_date))
        print("-" * 80)

        if not sorted_items:
            print("  (no signals passed quality filters on this date)")
        else:
            print("  %-14s  %-22s  %8s  %8s  %8s  %5s  %7s  %-5s" % (
                "SYMBOL", "DIRECTION / TYPE", "ENTRY", "TARGET", "STOP", "R:R", "PROX%", "ACTION?"))
            print("  " + "-" * 90)
            for item in sorted_items[:30]:
                ctag = "[CONFLUENCE]" if item.symbol in confluence_syms else ""
                print(fmt_item(item, ctag))

    print()
    print("=" * 80)
    print("  TOTAL: %d quality-filtered signals across all scanners on %s (IST)" % (grand_total, target_date))
    if confluence_syms:
        print("  Multi-scanner confluence: %s" % ", ".join(sorted(confluence_syms)))
    print("=" * 80)

    # ── Compact quality-filter breakdown (fast bypass for non-FVG scanners) ───
    # FVG is excluded here (slow scanner, R:R always <1.5 for zone-based signals).
    fast_bypass_defs = [
        ("Order Block",     ObScanner(OrderBlockConfig(
            require_trend_alignment=False, min_daily_value_crore=0.0,
            min_rr_ratio=0.0, min_volume_score=0.0))),
        ("Supply & Demand", SDScanner(SDConfig(
            require_trend_alignment=False, min_daily_value_crore=0.0,
            min_rr_ratio=0.0, fresh_zones_only=False))),
        ("Liquidity Sweep", SweepScanner(SweepConfig(
            require_trend_alignment=False, min_daily_value_crore=0.0,
            min_rr_ratio=0.0, min_equal_levels_count=1))),
        ("Market Structure", MSScanner(MSConfig(
            require_trend_alignment=False, min_daily_value_crore=0.0,
            min_rr_ratio=0.0, min_break_magnitude=0.0))),
    ]

    raw_all_by_scanner: dict[str, list[Any]] = {}
    raw_counts_fast: dict[str, int] = {}
    for sname, rscan in fast_bypass_defs:
        items_raw = scan_all(rscan, all_syms)
        raw_all_by_scanner[sname] = items_raw
        raw_counts_fast[sname] = len(on_date(items_raw, target_date))

    fvg_note = "R:R \u2265 1.0 (zone formula; bypass not run)"
    print()
    print("  Filter breakdown on %s (IST):" % target_date)
    print("  %-22s  %6s  %6s  %s" % ("Scanner", "Passed", "Raw", "Main filter reason"))
    print("  " + "-" * 78)
    for sname, items in today_by_scanner.items():
        if sname in raw_counts_fast:
            raw_n = raw_counts_fast[sname]
            if sname == "Order Block":
                note = "R:R \u2265 1.0 for OB zones (zone-height formula)"
            elif sname == "Supply & Demand":
                note = "R:R \u2265 1.0 for S&D zones (zone-height formula)"
            elif sname == "Liquidity Sweep":
                note = "trend alignment, R:R \u2265 1.5, min_equal_levels_count=%d" % SweepConfig().min_equal_levels_count
            else:
                note = "trend alignment, R:R \u2265 1.5, min_break_magnitude=%.1f" % MSConfig().min_break_magnitude
        else:
            raw_n = 0
            note = fvg_note
        print("  %-22s  %6d  %6d  %s" % (sname, len(items), raw_n, note))
    print()

    # ── Signal validation: prev_date → target_date ───────────────────────────
    # Uses bypass scan results (already computed above) to check prev_date signals.
    # Bypass results include signals from all dates in the detection window.
    if prev_date is None:
        return

    print()
    print("=" * 80)
    print("  SIGNAL VALIDATION: %s signals \u2192 %s outcome (IST)" % (prev_date, target_date))
    print("  (Comparing filtered vs raw signals from previous session)")
    print("=" * 80)

    # Filtered prev_date signals come from the main scan result
    filt_prev: list[Any] = []
    for items in all_items_by_scanner.values():
        filt_prev.extend(on_date(items, prev_date))

    # Raw prev_date signals from bypass results (fast scanners only, FVG excluded)
    raw_prev: list[Any] = []
    for items in raw_all_by_scanner.values():
        raw_prev.extend(on_date(items, prev_date))

    def hit_rate(signals: list[Any]) -> tuple[int, int]:
        h = t = 0
        for sig in signals:
            out = check_outcome(sig, all_syms, target_date)
            if out is not None:
                t += 1
                if out:
                    h += 1
        return h, t

    filt_hits, filt_total = hit_rate(filt_prev)
    raw_hits,  raw_total  = hit_rate(raw_prev)

    if filt_total == 0 and raw_total == 0:
        print("  No previous-session signals found to validate.")
    else:
        print()
        print("  %-40s  %s" % ("Mode", "Hit rate (close moved in signal direction)"))
        print("  " + "-" * 72)
        if filt_total > 0:
            fp = filt_hits / filt_total * 100
            print("  %-40s  %d/%d = %.1f%%" % ("Quality-filtered (MS + Sweep + OB + SD)", filt_hits, filt_total, fp))
        else:
            print("  %-40s  no signals (strict filters)" % "Quality-filtered")
        if raw_total > 0:
            rp = raw_hits / raw_total * 100
            print("  %-40s  %d/%d = %.1f%%" % ("Raw bypass (no filters applied)", raw_hits, raw_total, rp))
            if filt_total > 0 and raw_total > 0:
                diff = fp - rp
                sign = "+" if diff >= 0 else ""
                print()
                print("  \u2192 Quality filter %s hit rate by %s%.1f pp" % (
                    "improved" if diff >= 0 else "reduced", sign, diff))
    print()


if __name__ == "__main__":
    main()
