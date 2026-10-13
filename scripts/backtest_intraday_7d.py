"""
backtest_intraday_7d.py — SMC intraday scanner backtest on last 7 trading days.

Methodology
───────────
1. Run all 5 scanners ONCE on the full F&O universe.
2. For each trading day D in the backtest window [last 7 bars]:
   a. Determine which signals were actionable on day D:
      proximity_pct(D) = abs(close[D] - entry) / close[D] * 100 <= 1.5%
   b. Apply intraday quality gates for day D:
      - Volume gate : today's volume >= 75% of 20-day avg
      - Volatility  : ATR(14)/price in [0.5%, 8.0%]
      - Trend aligned to market direction on day D
   c. Evaluate same-day intraday outcome using bar D's OHLC:
      Bearish short: WIN if Low[D]  <= target | LOSS if High[D] >= stop
      Bullish long : WIN if High[D] >= target | LOSS if Low[D]  <= stop
      (Both hit → check opening direction → conservative LOSS if ambiguous)
      (Neither   → OPEN = signal did not trigger same day)

This simulates a pre-market scan (using yesterday's close) then same-day
intraday execution: enter on a limit at the entry price, exit at target
or stop within the same trading session.

Usage:
    python scripts/backtest_intraday_7d.py
    python scripts/backtest_intraday_7d.py --days 14
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.ERROR)

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("ML_SERVICE_API_KEY",   "backtest")
os.environ.setdefault("DATA_SERVICE_API_KEY", "backtest")
os.environ.setdefault("DATA_SERVICE_2_URL",   "http://localhost:8200")
os.environ.setdefault("SENTINEL_PULSE_URL",   "http://localhost:3001")

import pandas as pd
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR))

from src.scanners.order_block.config   import OrderBlockConfig
from src.scanners.order_block.scanner  import ObScanner
from src.scanners.fair_value_gap.config  import FVGConfig
from src.scanners.fair_value_gap.scanner import FVGScanner
from src.scanners.supply_demand.config   import SDConfig
from src.scanners.supply_demand.scanner  import SDScanner
from src.scanners.liquidity_sweep.config  import SweepConfig
from src.scanners.liquidity_sweep.scanner import SweepScanner
from src.scanners.market_structure.config  import MSConfig
from src.scanners.market_structure.scanner import MSScanner

from src.scanners.quality import (
    compute_market_direction, compute_symbol_trend,
    passes_liquidity_filter, passes_intraday_volume,
    passes_intraday_volatility, passes_quality_filters,
    is_signal_trend_aligned,
)
from src.scanners.data_loader import load_1d_data
from src.scanners.universe    import FO_UNIVERSE

IST = timezone(timedelta(hours=5, minutes=30))
W   = 80
ACTIONABLE_PCT = 1.5   # same as live scanner


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_direction_info(item: Any) -> tuple[str, bool]:
    """Return (direction_value_str, is_bullish)."""
    dirn = getattr(item, "direction", None) or getattr(item, "signal_type", None)
    dv   = getattr(dirn, "value", str(dirn)).lower() if dirn else ""
    return dv, ("bull" in dv or "demand" in dv)


def proximity_on_day(entry: float, close: float) -> float:
    if close <= 0:
        return 999.0
    return abs(close - entry) / close * 100.0


def evaluate_intraday(
    is_bull: bool,
    entry: float,
    target: float,
    stop: float,
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
) -> tuple[str, float]:
    """
    Evaluate trade outcome on the signal day's OHLC.

    For a pre-market signal (entry = limit order at price `entry`):
    - We assume the order fills if price touches the entry level intraday.
    - After fill, evaluate whether target or stop is hit first.
    - If entry is never touched (gap past target or never reached): adjust.

    Returns (outcome, pnl_pct) where outcome = 'WIN' | 'LOSS' | 'OPEN'.
    """
    if entry <= 0 or target <= 0 or stop <= 0:
        return "OPEN", 0.0

    risk   = abs(entry - stop)
    reward = abs(target - entry)

    # Check if the limit entry would have been filled today
    if is_bull:
        entry_filled = bar_low <= entry   # price dipped to our buy limit
    else:
        entry_filled = bar_high >= entry  # price rallied to our sell limit

    if not entry_filled:
        return "OPEN", 0.0  # entry never reached today

    # Entry was filled — now check target and stop
    if is_bull:
        hit_target = bar_high >= target
        hit_stop   = bar_low  <= stop
    else:
        hit_target = bar_low  <= target
        hit_stop   = bar_high >= stop

    if hit_target and hit_stop:
        # Both hit same candle — check open direction for gap resolution
        if is_bull:
            if bar_open >= target:   # opened above target (gap up past target)
                return "WIN", reward / entry * 100
        else:
            if bar_open <= target:   # opened below target (gap down past target)
                return "WIN", reward / entry * 100
        return "LOSS", -risk / entry * 100   # conservative: assume stop first

    if hit_target:
        return "WIN",  reward / entry * 100
    if hit_stop:
        return "LOSS", -risk  / entry * 100

    return "OPEN", 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Extract all signals from scan results
# ─────────────────────────────────────────────────────────────────────────────

def extract_signals(scanner_name: str, result: Any) -> list[dict]:
    items: list[Any] = (getattr(result, "zones", None)
                        or getattr(result, "signals", []) or [])
    out = []
    for item in items:
        if getattr(item, "invalidated", False):
            continue
        sym  = item.symbol
        dv, is_bull = get_direction_info(item)

        entry  = getattr(item, "intraday_entry_price",  0.0) or 0.0
        target = getattr(item, "intraday_target_price", 0.0) or 0.0
        stop   = getattr(item, "intraday_stop_price",   0.0) or 0.0
        rr     = getattr(item, "intraday_rr_ratio",     0.0) or 0.0

        if entry <= 0 or target <= 0 or stop <= 0:
            continue

        out.append({
            "scanner":  scanner_name,
            "symbol":   sym,
            "direction": dv,
            "is_bull":  is_bull,
            "entry":    entry,
            "target":   target,
            "stop":     stop,
            "rr":       rr,
            "choch":    getattr(item, "is_choch", False),
            "lc_state": str(getattr(getattr(item, "lifecycle_state", ""), "value", "")),
            "_item":    item,   # keep for quality filter
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main backtest
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7,  help="Trading days to backtest")
    args = parser.parse_args()

    print("=" * W)
    print("  SMC SCANNER — 7-DAY INTRADAY WALK-FORWARD BACKTEST")
    print(f"  Methodology: pre-market scan → same-day intraday execution")
    print(f"  Signal gate : actionable (entry within {ACTIONABLE_PCT}%) + vol + vol + trend")
    print(f"  Run: {datetime.now(IST).strftime('%Y-%m-%d %H:%M IST')}")
    print("=" * W)

    # ── Load data ──────────────────────────────────────────────────────────────
    print("\nLoading F&O universe...")
    all_dfs = load_1d_data(force=True)
    print(f"  {len(all_dfs)} symbols")

    # Reference series: NIFTY for date indexing
    ref = all_dfs["NIFTY"]
    n   = len(ref)

    # Backtest window: last args.days bars (we evaluate ON those bars)
    # We need at least 1 prior bar for "yesterday" quality checks
    # Window: bars [n-days .. n-1] = last args.days trading bars
    bt_bars = list(range(max(1, n - args.days), n))
    bt_dates = [ref.index[i].astimezone(IST).date() for i in bt_bars]

    print(f"  Window: {bt_dates[0]} → {bt_dates[-1]}")
    print(f"  Days  : {', '.join(str(d) for d in bt_dates)}\n")

    # ── Run all 5 scanners on FULL history ────────────────────────────────────
    print("Running 5 scanners on full universe...")
    dfs_ms = {s: df.iloc[-250:] if len(df) > 250 else df for s, df in all_dfs.items()}

    raw_signals: list[dict] = []
    scanner_defs = [
        ("OrderBlock",   ObScanner(OrderBlockConfig()).scan(all_dfs, "1d")),
        ("FVG",          FVGScanner(FVGConfig()).scan(all_dfs, "1d")),
        ("S&D",          SDScanner(SDConfig()).scan(all_dfs, "1d")),
        ("LiqSweep",     SweepScanner(SweepConfig()).scan(all_dfs, "1d")),
        ("MktStructure", MSScanner(MSConfig()).scan(dfs_ms, "1d")),
    ]
    for name, result in scanner_defs:
        sigs = extract_signals(name, result)
        raw_signals.extend(sigs)
        print(f"  {name:<14}: {len(sigs)} signals extracted")

    print(f"  Total raw     : {len(raw_signals)} signals across all scanners\n")

    # ── Per-day evaluation ────────────────────────────────────────────────────
    print("Evaluating signals per day...")
    all_trades: list[dict] = []
    day_summaries: list[dict] = []

    for bar_idx in bt_bars:
        bar_date = ref.index[bar_idx].astimezone(IST).date()
        prev_idx = bar_idx - 1   # yesterday's data for quality checks

        # Market direction uses data through yesterday
        dfs_prev = {s: df.iloc[:prev_idx+1] for s, df in all_dfs.items()
                    if prev_idx < len(df)}
        market_dir = compute_market_direction(dfs_prev)

        day_trades: list[dict] = []
        seen_sym_scanner: set[tuple] = set()   # deduplicate per (sym, scanner, dir)

        for sig in raw_signals:
            sym  = sig["symbol"]
            df   = all_dfs.get(sym)
            if df is None or bar_idx >= len(df) or prev_idx >= len(df):
                continue

            df_prev = df.iloc[:prev_idx+1]  # data through yesterday
            bar     = df.iloc[bar_idx]      # today's OHLC for evaluation

            entry   = sig["entry"]
            close_y = float(df_prev["close"].iloc[-1])   # yesterday's close

            # ── Gate 1: actionable on yesterday's close ──────────────────────
            prox = proximity_on_day(entry, close_y)
            if prox > ACTIONABLE_PCT:
                continue

            # ── Gate 2: liquidity (historical avg) ───────────────────────────
            if not passes_liquidity_filter(df_prev):
                continue

            # ── Gate 3: volume (today's volume vs avg) ───────────────────────
            # Use today's bar since we're evaluating same-day
            df_today = df.iloc[:bar_idx+1]
            if not passes_intraday_volume(df_today, symbol=sym):
                continue

            # ── Gate 4: volatility (ATR range) ───────────────────────────────
            if not passes_intraday_volatility(df_today, symbol=sym):
                continue

            # ── Gate 5: trend alignment ──────────────────────────────────────
            sym_trend  = compute_symbol_trend(df_prev)
            is_bull    = sig["is_bull"]
            if not is_signal_trend_aligned(is_bull, sym_trend, market_dir):
                continue

            # ── Gate 6: min R:R >= 1.0 ──────────────────────────────────────
            if sig["rr"] < 1.0:
                continue

            # ── Dedup: one trade per (sym, scanner, direction) per day ────────
            key = (sym, sig["scanner"], sig["direction"])
            if key in seen_sym_scanner:
                continue
            seen_sym_scanner.add(key)

            # ── Evaluate intraday outcome ────────────────────────────────────
            o_bar, h_bar, l_bar, c_bar = (
                float(bar["open"]),  float(bar["high"]),
                float(bar["low"]),   float(bar["close"]),
            )
            outcome, pnl = evaluate_intraday(
                is_bull, entry, sig["target"], sig["stop"],
                o_bar, h_bar, l_bar, c_bar,
            )

            trade = {
                **{k: v for k, v in sig.items() if k != "_item"},
                "date":       bar_date,
                "close_y":    close_y,
                "prox_y":     prox,
                "market_dir": market_dir,
                "outcome":    outcome,
                "pnl_pct":    pnl,
            }
            day_trades.append(trade)

        w = sum(1 for t in day_trades if t["outcome"] == "WIN")
        l = sum(1 for t in day_trades if t["outcome"] == "LOSS")
        o = sum(1 for t in day_trades if t["outcome"] == "OPEN")
        day_summaries.append({
            "date": bar_date, "market_dir": market_dir,
            "total": len(day_trades), "wins": w, "losses": l, "opens": o,
        })
        all_trades.extend(day_trades)
        print(f"  {bar_date} ({bar_date.strftime('%a')}) | {market_dir:<8} | "
              f"{len(day_trades):3d} signals | W={w} L={l} OPEN={o}")

    # ─────────────────────────────────────────────────────────────────────────
    # RESULTS
    # ─────────────────────────────────────────────────────────────────────────
    if not all_trades:
        print("\nNo qualifying signals in window (all filtered out).")
        return

    wins    = [t for t in all_trades if t["outcome"] == "WIN"]
    losses  = [t for t in all_trades if t["outcome"] == "LOSS"]
    opens   = [t for t in all_trades if t["outcome"] == "OPEN"]
    evaled  = len(wins) + len(losses)
    total   = len(all_trades)

    win_r   = len(wins) / evaled * 100 if evaled else 0
    avg_win = sum(t["pnl_pct"] for t in wins)   / len(wins)   if wins   else 0
    avg_los = sum(t["pnl_pct"] for t in losses) / len(losses) if losses else 0
    avg_rr  = sum(t["rr"]      for t in all_trades) / total   if total  else 0
    exp_val = (win_r/100 * avg_win) + ((1 - win_r/100) * avg_los) if evaled else 0

    print()
    print("=" * W)
    print("  OVERALL RESULTS")
    print("=" * W)
    print(f"  Total signals  : {total}  (evaluated={evaled}, open/no-fill={len(opens)})")
    print(f"  Win rate       : {win_r:.1f}%  ({len(wins)} wins / {evaled} evaluated)")
    print(f"  Avg win        : +{avg_win:.2f}%")
    print(f"  Avg loss       : {avg_los:.2f}%")
    print(f"  Avg R:R        : {avg_rr:.2f}x")
    print(f"  Expected value : {exp_val:+.3f}% per trade")
    if wins:
        b = max(wins, key=lambda x: x["pnl_pct"])
        print(f"  Best trade     : {b['symbol']} [{b['scanner']}] {b['direction']} +{b['pnl_pct']:.2f}% on {b['date']}")
    if losses:
        w2 = min(losses, key=lambda x: x["pnl_pct"])
        print(f"  Worst trade    : {w2['symbol']} [{w2['scanner']}] {w2['direction']} {w2['pnl_pct']:.2f}% on {w2['date']}")

    # ── PER-SCANNER ────────────────────────────────────────────────────────────
    print()
    print("─" * W)
    print(f"  {'Scanner':<14} {'Sigs':>5} {'W':>4} {'L':>4} {'Open':>5} {'Win%':>7} "
          f"{'AvgWin':>8} {'AvgLoss':>9} {'AvgRR':>7}")
    print("  " + "─" * (W - 2))
    for sname in ["OrderBlock", "FVG", "S&D", "LiqSweep", "MktStructure"]:
        sg = [t for t in all_trades if t["scanner"] == sname]
        if not sg:
            print(f"  {sname:<14}    —")
            continue
        sw  = [t for t in sg if t["outcome"] == "WIN"]
        sl  = [t for t in sg if t["outcome"] == "LOSS"]
        so  = [t for t in sg if t["outcome"] == "OPEN"]
        se  = len(sw) + len(sl)
        wr  = len(sw) / se * 100 if se else 0
        aw  = sum(t["pnl_pct"] for t in sw) / len(sw) if sw else 0
        al  = sum(t["pnl_pct"] for t in sl) / len(sl) if sl else 0
        ar  = sum(t["rr"]      for t in sg) / len(sg)
        print(f"  {sname:<14} {len(sg):>5} {len(sw):>4} {len(sl):>4} {len(so):>5} "
              f"{wr:>6.0f}% {aw:>+8.2f}% {al:>+9.2f}% {ar:>7.2f}x")

    # ── PER-DAY ─────────────────────────────────────────────────────────────
    print()
    print("─" * W)
    print(f"  {'Date':<12} {'Day':<9} {'Market':<9} {'Sigs':>5} {'W':>4} {'L':>4} "
          f"{'Open':>5} {'Win%':>6} {'Symbols'}")
    print("  " + "─" * (W - 2))
    for ds in day_summaries:
        sg = [t for t in all_trades if t["date"] == ds["date"]]
        syms = sorted(set(t["symbol"] for t in sg if t["outcome"] != "OPEN"))
        wr = ds["wins"] / (ds["wins"]+ds["losses"]) * 100 if (ds["wins"]+ds["losses"]) else 0
        sym_str = ", ".join(syms[:5]) + ("..." if len(syms) > 5 else "")
        print(f"  {str(ds['date']):<12} {ds['date'].strftime('%a'):<9} "
              f"{ds['market_dir']:<9} {ds['total']:>5} {ds['wins']:>4} {ds['losses']:>4} "
              f"{ds['opens']:>5} {wr:>5.0f}%  {sym_str}")

    # ── TOP TRADES ────────────────────────────────────────────────────────────
    if wins:
        print()
        print("─" * W)
        print("  TOP WINNING TRADES")
        print("─" * W)
        print(f"  {'Date':<10} {'Symbol':<13} {'Scanner':<12} {'Direction':<22} "
              f"{'R:R':>5} {'P&L':>7} {'Prox':>6}")
        print("  " + "─" * (W - 2))
        for t in sorted(wins, key=lambda x: x["pnl_pct"], reverse=True)[:10]:
            star = " ★" if t.get("choch") else ""
            print(f"  {str(t['date']):<10} {t['symbol']:<13} {t['scanner']:<12} "
                  f"{t['direction']:<22} {t['rr']:>5.1f}x {t['pnl_pct']:>+7.2f}%"
                  f" {t['prox_y']:>5.1f}%{star}")

    if losses:
        print()
        print("─" * W)
        print("  LOSING TRADES")
        print("─" * W)
        print(f"  {'Date':<10} {'Symbol':<13} {'Scanner':<12} {'Direction':<22} "
              f"{'R:R':>5} {'P&L':>7} {'Prox':>6}")
        print("  " + "─" * (W - 2))
        for t in sorted(losses, key=lambda x: x["pnl_pct"])[:8]:
            print(f"  {str(t['date']):<10} {t['symbol']:<13} {t['scanner']:<12} "
                  f"{t['direction']:<22} {t['rr']:>5.1f}x {t['pnl_pct']:>+7.2f}%"
                  f" {t['prox_y']:>5.1f}%")

    # ── ANALYSIS ───────────────────────────────────────────────────────────────
    print()
    print("─" * W)
    print("  ANALYSIS")
    print("─" * W)

    # ChoCh vs BOS breakdown
    for label, sg in [
        ("ChoCh signals ", [t for t in all_trades if t.get("choch")]),
        ("BoS signals   ", [t for t in all_trades if not t.get("choch") and t["scanner"] == "MktStructure"]),
        ("Zone signals  ", [t for t in all_trades if t["scanner"] in ("OrderBlock","FVG","S&D","LiqSweep")]),
    ]:
        if not sg:
            continue
        sw = [t for t in sg if t["outcome"] == "WIN"]
        sl = [t for t in sg if t["outcome"] == "LOSS"]
        se = len(sw) + len(sl)
        wr = len(sw)/se*100 if se else 0
        ar = sum(t["rr"] for t in sg)/len(sg)
        print(f"  {label}: {len(sg):3d} | Win {wr:5.0f}% ({len(sw)}/{se}) | Avg R:R {ar:.2f}x")

    print()
    # Proximity buckets
    print("  Proximity at entry (yesterday's close vs entry):")
    for lo, hi, label in [(0,0.5,"0–0.5%"), (0.5,1.0,"0.5–1.0%"), (1.0,1.5,"1.0–1.5%")]:
        sg = [t for t in all_trades if lo <= t["prox_y"] < hi]
        if sg:
            sw = sum(1 for t in sg if t["outcome"] == "WIN")
            se = sw + sum(1 for t in sg if t["outcome"] == "LOSS")
            wr = sw/se*100 if se else 0
            print(f"    {label:<10}: {len(sg):3d} signals | Win {wr:4.0f}% ({sw}/{se})")

    print()
    # Market direction
    for mdir in ["BEARISH", "BULLISH", "NEUTRAL"]:
        sg = [t for t in all_trades if t["market_dir"] == mdir]
        if sg:
            sw = sum(1 for t in sg if t["outcome"] == "WIN")
            se = sw + sum(1 for t in sg if t["outcome"] == "LOSS")
            wr = sw/se*100 if se else 0
            print(f"  {mdir:<8} market: {len(sg):3d} signals | Win {wr:4.0f}% ({sw}/{se})")

    # ── VERDICT ────────────────────────────────────────────────────────────────
    print()
    print("=" * W)
    if evaled == 0:
        print("  VERDICT: INSUFFICIENT DATA — no signals triggered intraday")
        print("  All signals were OPEN (entry limit never touched during the day)")
        print("  → Review: entry levels may need wider proximity threshold")
    else:
        verdict = ("STRONG ✅" if win_r >= 60 else
                   ("PASS ✅"  if win_r >= 55 else
                   ("MARGINAL" if win_r >= 45 else "FAIL ❌")))
        print(f"  VERDICT: {verdict}  — {win_r:.1f}% win rate on {evaled} evaluated trades")
        print(f"  Edge vs random : {win_r-50:+.1f}pp")
        print(f"  Expected value : {exp_val:+.3f}% per trade  (avg win {avg_win:.2f}%, avg loss {avg_los:.2f}%)")
        if exp_val > 0:
            print(f"  Positive EV — scanner has profitable intraday edge on F&O universe")
        else:
            print(f"  Negative EV — review filter thresholds or market conditions")
    print("=" * W)


if __name__ == "__main__":
    main()
