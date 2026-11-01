"""
backtest_scanners_7d.py — 7-day walk-forward backtest for all SMC scanners.

Usage:
    python scripts/backtest_scanners_7d.py

What it does
────────────
1. Loads real 1d OHLCV parquet data for up to 50 F&O symbols.
2. Runs all 5 scanners (OB, FVG, S&D, Liquidity Sweep, Market Structure).
3. Filters signals to the last 7 bars of each symbol's history
   (Oct 1–7 2026 based on the available dataset).
4. Evaluates each signal by checking what happened in the next 1–3 bars.
5. Prints a detailed per-scanner report with signal stats, quality scores,
   and any bugs / exceptions encountered.

Quality score convention
────────────────────────
  +1 = signal respected / confirmed
  -1 = signal immediately invalidated
   0 = untested within the evaluation window (last bar, no future data yet)
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

# ── Project root on path ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Minimal env vars required before src imports ──────────────────────────────
os.environ.setdefault("ML_SERVICE_API_KEY", "test-backtest-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-backtest-key")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://localhost:3001")

import pandas as pd
import numpy as np

# ── Scanner imports ───────────────────────────────────────────────────────────
from src.scanners.order_block.config import OrderBlockConfig
from src.scanners.order_block.scanner import ObScanner
from src.scanners.order_block.schemas import OBZoneDirection

from src.scanners.fair_value_gap.config import FVGConfig
from src.scanners.fair_value_gap.scanner import FVGScanner
from src.scanners.fair_value_gap.schemas import FVGDirection

from src.scanners.supply_demand.config import SDConfig
from src.scanners.supply_demand.scanner import SDScanner
from src.scanners.supply_demand.schemas import SDDirection

from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.scanner import SweepScanner
from src.scanners.liquidity_sweep.schemas import SweepDirection

from src.scanners.market_structure.config import MSConfig
from src.scanners.market_structure.scanner import MSScanner
from src.scanners.market_structure.schemas import MSSignalType

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "data" / "1d" / "1d"
MAX_SYMBOLS = 50
SIGNAL_WINDOW = 7          # report signals only from last N bars
EVAL_BARS = 3              # look-ahead bars for quality evaluation
MIN_HISTORY_BARS = 50      # skip symbol if fewer bars available


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_symbols(max_symbols: int = MAX_SYMBOLS) -> dict[str, pd.DataFrame]:
    """
    Load up to *max_symbols* parquet files from the 1d data directory.

    Drops the 'provider' column, drops rows with NaN OHLC, and ensures
    a UTC-aware DatetimeIndex.  Symbols with fewer than MIN_HISTORY_BARS
    rows are skipped.
    """
    dfs: dict[str, pd.DataFrame] = {}
    paths = sorted(DATA_DIR.glob("*.parquet"))[:max_symbols]
    print(f"[DATA] Scanning {len(paths)} parquet files in {DATA_DIR.relative_to(PROJECT_ROOT)}")

    for path in paths:
        symbol = path.stem
        try:
            df = pd.read_parquet(path)
            df = df.drop(columns=["provider"], errors="ignore")
            df = df[["open", "high", "low", "close", "volume"]].dropna(
                subset=["open", "high", "low", "close"]
            )
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            if len(df) >= MIN_HISTORY_BARS:
                dfs[symbol] = df
        except Exception as exc:
            print(f"  [WARN] Failed to load {symbol}: {exc}")

    return dfs


# ═════════════════════════════════════════════════════════════════════════════
# BAR INDEX RESOLUTION
# ═════════════════════════════════════════════════════════════════════════════

def get_signal_bar_idx(item: Any, df: pd.DataFrame) -> int | None:
    """
    Return the integer bar index at which a signal / zone was confirmed.

    Each scanner stores the confirmation point differently:
    ─ FVGZone        → candle3_bar_index   (3rd candle of the gap pattern)
    ─ SDZone         → departure_bar_index  (bar where price departed the base)
    ─ SweepSignal    → sweep_bar_index
    ─ MSSignal       → break_bar_index
    ─ OBZone         → no bar index field; resolved from bos_timestamp (BOS
                       confirmation timestamp) via DatetimeIndex lookup.

    Returns None when the bar cannot be resolved (signal predates the df, etc.).
    """
    # All scanners except OB carry an explicit integer bar index ──────────────
    for attr in (
        "candle3_bar_index",    # FVGZone
        "departure_bar_index",  # SDZone
        "sweep_bar_index",      # SweepSignal
        "break_bar_index",      # MSSignal
    ):
        val = getattr(item, attr, None)
        if val is not None:
            idx = int(val)
            return idx if 0 <= idx < len(df) else None

    # OBZone: resolve via bos_timestamp (confirmed BOS bar), then ob_timestamp
    for ts_attr in ("bos_timestamp", "ob_timestamp"):
        ts = getattr(item, ts_attr, None)
        if ts is None:
            continue
        try:
            # searchsorted returns the insertion point; clamp to valid range
            pos = int(df.index.searchsorted(ts, side="left"))
            # Accept if the timestamp at pos is within one day of the target
            if pos < len(df):
                delta_sec = abs((df.index[pos] - ts).total_seconds())
                if delta_sec < 86_400:          # within 24 h → valid match
                    return pos
            # Try one position back (searchsorted can overshoot by 1 slot)
            if pos > 0:
                pos_prev = pos - 1
                delta_sec = abs((df.index[pos_prev] - ts).total_seconds())
                if delta_sec < 86_400:
                    return pos_prev
        except Exception:
            pass

    return None


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL QUALITY EVALUATORS
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_ob_quality(zone: Any, df: pd.DataFrame, signal_bar_idx: int) -> int:
    """
    Order Block quality.

    Bullish OB: quality = +1 if price re-enters the zone (low ≤ ob_high AND
    high ≥ ob_low) within EVAL_BARS.  quality = -1 if a close prints below
    ob_low (zone invalidated).

    Bearish OB: mirror logic.
    """
    future = df.iloc[signal_bar_idx + 1 : signal_bar_idx + 1 + EVAL_BARS]
    if future.empty:
        return 0

    ob_high = float(zone.ob_high)
    ob_low  = float(zone.ob_low)

    for _, row in future.iterrows():
        low_k   = float(row["low"])
        high_k  = float(row["high"])
        close_k = float(row["close"])

        if zone.direction == OBZoneDirection.BULLISH:
            if low_k <= ob_high and high_k >= ob_low:
                return 1   # price re-entered OB zone (respected)
            if close_k < ob_low:
                return -1  # closed below zone (invalidated)
        else:  # BEARISH
            if high_k >= ob_low and low_k <= ob_high:
                return 1
            if close_k > ob_high:
                return -1

    return 0


def evaluate_fvg_quality(zone: Any, df: pd.DataFrame, signal_bar_idx: int) -> int:
    """
    Fair Value Gap quality.

    +1 if price partially fills the gap (enters the gap zone) within EVAL_BARS.
    -1 if price moves further away from the gap (confirms the imbalance was NOT
       pulled back — technically a negative for FVG traders expecting a fill).
    """
    future = df.iloc[signal_bar_idx + 1 : signal_bar_idx + 1 + EVAL_BARS]
    if future.empty:
        return 0

    fvg_high = float(zone.fvg_high)
    fvg_low  = float(zone.fvg_low)

    for _, row in future.iterrows():
        low_k  = float(row["low"])
        high_k = float(row["high"])

        if low_k <= fvg_high and high_k >= fvg_low:
            return 1   # entered gap

        if zone.direction == FVGDirection.BULLISH and high_k < fvg_low:
            return -1  # gapped even further away (bearish continuation)
        if zone.direction == FVGDirection.BEARISH and low_k > fvg_high:
            return -1  # gapped further up (bullish continuation)

    return 0


def evaluate_sd_quality(zone: Any, df: pd.DataFrame, signal_bar_idx: int) -> int:
    """
    Supply & Demand zone quality.

    Demand zone: +1 if no close below zone_low within EVAL_BARS (zone intact).
                 -1 if any close breaks below zone_low immediately.
    Supply zone: mirror logic.
    """
    future = df.iloc[signal_bar_idx + 1 : signal_bar_idx + 1 + EVAL_BARS]
    if future.empty:
        return 0

    zone_high = float(zone.zone_high)
    zone_low  = float(zone.zone_low)

    for _, row in future.iterrows():
        close_k = float(row["close"])
        if zone.direction == SDDirection.DEMAND:
            if close_k < zone_low:
                return -1   # demand zone broken
        else:  # SUPPLY
            if close_k > zone_high:
                return -1   # supply zone broken

    return 1   # zone survived all evaluated bars → intact


def evaluate_sweep_quality(signal: Any, df: pd.DataFrame, signal_bar_idx: int) -> int:
    """
    Liquidity Sweep quality.

    Bullish sweep (swept lows): +1 if ALL of the next 2 closes stay above
    swept_level (reversal confirmed).  -1 if ANY close breaks back below.

    Bearish sweep: mirror logic.
    """
    future = df.iloc[signal_bar_idx + 1 : signal_bar_idx + 3]   # 2 bars
    if future.empty:
        return 0

    swept = float(signal.swept_level)
    closes = [float(c) for c in future["close"].values]
    if not closes:
        return 0

    if signal.direction == SweepDirection.BULLISH:
        if all(c > swept for c in closes):
            return 1
        if any(c < swept for c in closes):
            return -1
    else:  # BEARISH
        if all(c < swept for c in closes):
            return 1
        if any(c > swept for c in closes):
            return -1

    return 0


def evaluate_ms_quality(signal: Any, df: pd.DataFrame, signal_bar_idx: int) -> int:
    """
    Market Structure quality.

    BOS_BULLISH / CHOCH_BULLISH: +1 if the last close of the next 2 bars is
    above broken_level, -1 otherwise.  Mirror for bearish variants.
    """
    future = df.iloc[signal_bar_idx + 1 : signal_bar_idx + 3]   # 2 bars
    if future.empty:
        return 0

    broken = float(signal.broken_level)
    closes  = [float(c) for c in future["close"].values]
    if not closes:
        return 0

    if signal.signal_type in (MSSignalType.BOS_BULLISH, MSSignalType.CHOCH_BULLISH):
        return 1 if closes[-1] > broken else -1
    else:
        return 1 if closes[-1] < broken else -1


# Map scanner name → quality evaluator
QUALITY_FN = {
    "OrderBlock":   evaluate_ob_quality,
    "FVG":          evaluate_fvg_quality,
    "SupplyDemand": evaluate_sd_quality,
    "LiqSweep":     evaluate_sweep_quality,
    "MktStructure": evaluate_ms_quality,
}


# ═════════════════════════════════════════════════════════════════════════════
# MAIN BACKTEST RUNNER
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest() -> None:
    print("=" * 72)
    print("  SMC Scanner  —  7-Day Walk-Forward Backtest")
    print(f"  Data directory : {DATA_DIR.relative_to(PROJECT_ROOT)}")
    print(f"  Run at         : {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print("=" * 72)

    # ── Load data ─────────────────────────────────────────────────────────────
    all_symbols = load_symbols(max_symbols=MAX_SYMBOLS)
    print(f"[DATA] Loaded {len(all_symbols)} symbols with ≥{MIN_HISTORY_BARS} bars\n")
    if not all_symbols:
        print("ERROR: No symbols loaded — check DATA_DIR path.")
        return

    # Show the backtest date window using the first symbol as a reference
    sample_df   = next(iter(all_symbols.values()))
    last_7_idx  = sample_df.tail(SIGNAL_WINDOW)
    cutoff_date = last_7_idx.index[0].date()
    last_date   = last_7_idx.index[-1].date()
    print(f"[WINDOW] Last {SIGNAL_WINDOW} trading days: {cutoff_date}  →  {last_date}\n")

    # ── Instantiate scanners ──────────────────────────────────────────────────
    scanners: dict[str, tuple[Any, str]] = {
        "OrderBlock":   (ObScanner(OrderBlockConfig()),  "1d"),
        "FVG":          (FVGScanner(FVGConfig()),        "1d"),
        "SupplyDemand": (SDScanner(SDConfig()),          "1d"),
        "LiqSweep":     (SweepScanner(SweepConfig()),    "1d"),
        "MktStructure": (MSScanner(MSConfig()),          "1d"),
    }

    # ── Per-scanner stats container ───────────────────────────────────────────
    results: dict[str, dict] = {}

    for scanner_name, (scanner, tf) in scanners.items():
        print(f"{'─' * 72}")
        print(f"  ▶  {scanner_name}")

        total_signals   = 0
        quality_scores: list[int] = []
        bugs_found:     list[str] = []
        per_symbol_cnt: dict[str, int] = {}
        sample_signals: list[str] = []   # human-readable examples (up to 3)

        for symbol, full_df in all_symbols.items():
            if len(full_df) < MIN_HISTORY_BARS:
                continue

            try:
                # ── Run scanner on FULL history (gives ATR / swing context) ──
                result = scanner.scan({symbol: full_df}, timeframe=tf)

                # ── Pull the zone / signal list ───────────────────────────────
                items: list[Any] = (
                    getattr(result, "zones",   None)
                    or getattr(result, "signals", [])
                    or []
                )

                # ── Filter to last SIGNAL_WINDOW bars ─────────────────────────
                cutoff_idx = len(full_df) - SIGNAL_WINDOW
                recent_items: list[tuple[Any, int]] = []

                for item in items:
                    bar_idx = get_signal_bar_idx(item, full_df)
                    if bar_idx is not None and bar_idx >= cutoff_idx:
                        # Skip already-invalidated signals (MS detector marks failed BoS/ChoCh)
                        if getattr(item, "invalidated", False):
                            continue
                        recent_items.append((item, bar_idx))

                total_signals        += len(recent_items)
                per_symbol_cnt[symbol] = len(recent_items)

                # ── Evaluate quality ──────────────────────────────────────────
                eval_fn = QUALITY_FN[scanner_name]
                for item, bar_idx in recent_items:
                    try:
                        q = eval_fn(item, full_df, bar_idx)
                        quality_scores.append(q)
                    except Exception as exc:
                        quality_scores.append(0)
                        bugs_found.append(
                            f"  [BUG:quality] {symbol} bar={bar_idx}: "
                            f"{type(exc).__name__}: {exc}"
                        )

                    # Collect a few sample signals for the report
                    if len(sample_signals) < 3:
                        ts   = getattr(item, "ob_timestamp", None) \
                            or getattr(item, "fvg_timestamp", None) \
                            or getattr(item, "departure_timestamp", None) \
                            or getattr(item, "sweep_timestamp", None) \
                            or getattr(item, "break_timestamp", None)
                        dirn_raw = getattr(item, "direction",    None) \
                            or getattr(item, "signal_type",  None) \
                            or getattr(item, "zone_type",    "?")
                        # Extract the plain string value from Enum if needed
                        dirn = getattr(dirn_raw, "value", str(dirn_raw))
                        ts_str = ts.strftime("%Y-%m-%d") if ts else "?"
                        sample_signals.append(
                            f"    {symbol:>12}  bar={bar_idx:<4}  "
                            f"date={ts_str}  dir={dirn}"
                        )

            except Exception as exc:
                tb = traceback.format_exc().strip().splitlines()[-1]
                bugs_found.append(
                    f"  [BUG:scan]  {symbol}: {type(exc).__name__}: {exc}  |  {tb}"
                )

        # ── Print per-scanner report ──────────────────────────────────────────
        respected = quality_scores.count(1)
        broken    = quality_scores.count(-1)
        untested  = quality_scores.count(0)
        total_q   = len(quality_scores)
        n_active  = sum(1 for v in per_symbol_cnt.values() if v > 0)

        print(f"  Signals in last {SIGNAL_WINDOW} days : {total_signals}")
        print(f"  Symbols with signals       : {n_active} / {len(all_symbols)}")

        if total_q > 0:
            r_pct = respected / total_q * 100
            b_pct = broken    / total_q * 100
            u_pct = untested  / total_q * 100
            print(
                f"  Quality breakdown          : "
                f"✅ {respected} respected ({r_pct:.0f}%)  "
                f"❌ {broken} broken ({b_pct:.0f}%)  "
                f"⏳ {untested} untested ({u_pct:.0f}%)"
            )
            evaluated = respected + broken
            if evaluated > 0:
                print(
                    f"  Respect rate (excl. open) : "
                    f"{respected / evaluated * 100:.0f}%  "
                    f"({respected}/{evaluated} evaluated)"
                )
        else:
            print("  Quality breakdown          : no signals in window")

        if sample_signals:
            print("  Sample signals:")
            for s in sample_signals:
                print(s)

        if bugs_found:
            print(f"  ⚠️  BUGS / EXCEPTIONS ({len(bugs_found)}):")
            for b in bugs_found[:8]:
                print(b)
            if len(bugs_found) > 8:
                print(f"    … and {len(bugs_found) - 8} more")

        results[scanner_name] = {
            "total_signals": total_signals,
            "n_active_symbols": n_active,
            "quality_scores": quality_scores,
            "bugs": bugs_found,
        }
        print()

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"{'=' * 72}")
    print("  SUMMARY")
    print(f"{'─' * 72}")
    print(
        f"  {'Scanner':<16}"
        f"{'Signals':>8}"
        f"{'Symbols':>9}"
        f"{'Respect%':>10}"
        f"{'Bugs':>7}"
    )
    print(f"  {'─' * 16}{'─' * 8}{'─' * 9}{'─' * 10}{'─' * 7}")

    for name, r in results.items():
        q          = r["quality_scores"]
        total_q    = len(q)
        evaluated  = q.count(1) + q.count(-1)
        respect_pct = (
            f"{q.count(1) / evaluated * 100:.0f}%"
            if evaluated > 0
            else "N/A"
        )
        print(
            f"  {name:<16}"
            f"{r['total_signals']:>8}"
            f"{r['n_active_symbols']:>9}"
            f"{respect_pct:>10}"
            f"{len(r['bugs']):>7}"
        )

    print(f"{'=' * 72}")
    total_bugs = sum(len(r["bugs"]) for r in results.values())
    if total_bugs == 0:
        print("  ✅  No bugs found across all scanners.")
    else:
        print(f"  ⚠️   {total_bugs} total bugs found — see per-scanner details above.")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_backtest()
