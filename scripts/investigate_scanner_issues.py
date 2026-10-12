"""
investigate_scanner_issues.py
─────────────────────────────
Drills into broken LiqSweep and MktStructure signals found by the 7-day
backtest and probes the three known potential bugs in each scanner.

Run:
    .venv/bin/python scripts/investigate_scanner_issues.py
"""
from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

# ── project root ───────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("ML_SERVICE_API_KEY", "test-backtest-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "test-backtest-key")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://localhost:3001")

import pandas as pd
import numpy as np

from src.scanners.liquidity_sweep.config import SweepConfig
from src.scanners.liquidity_sweep.scanner import SweepScanner
from src.scanners.liquidity_sweep.schemas import SweepDirection

from src.scanners.market_structure.config import MSConfig
from src.scanners.market_structure.scanner import MSScanner
from src.scanners.market_structure.schemas import MSSignalType

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR       = PROJECT_ROOT / "data" / "1d" / "1d"
MAX_SYMBOLS    = 50
SIGNAL_WINDOW  = 7
EVAL_BARS      = 2    # look-ahead bars used for quality check
MIN_HISTORY    = 50
SEP            = "─" * 72


# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════════

def load_symbols(max_symbols: int = MAX_SYMBOLS) -> dict[str, pd.DataFrame]:
    dfs: dict[str, pd.DataFrame] = {}
    paths = sorted(DATA_DIR.glob("*.parquet"))[:max_symbols]
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
            if len(df) >= MIN_HISTORY:
                dfs[symbol] = df
        except Exception as exc:
            print(f"  [WARN] {symbol}: {exc}")
    return dfs


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL QUALITY HELPERS (same logic as backtest)
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_sweep_quality(signal: Any, df: pd.DataFrame, bar_idx: int) -> int:
    future = df.iloc[bar_idx + 1 : bar_idx + 1 + EVAL_BARS]
    if future.empty:
        return 0
    swept  = float(signal.swept_level)
    closes = [float(c) for c in future["close"].values]
    if not closes:
        return 0
    if signal.direction == SweepDirection.BULLISH:
        if all(c > swept for c in closes):
            return 1
        if any(c < swept for c in closes):
            return -1
    else:
        if all(c < swept for c in closes):
            return 1
        if any(c > swept for c in closes):
            return -1
    return 0


def evaluate_ms_quality(signal: Any, df: pd.DataFrame, bar_idx: int) -> int:
    future = df.iloc[bar_idx + 1 : bar_idx + 1 + EVAL_BARS]
    if future.empty:
        return 0
    broken = float(signal.broken_level)
    closes = [float(c) for c in future["close"].values]
    if not closes:
        return 0
    if signal.signal_type in (MSSignalType.BOS_BULLISH, MSSignalType.CHOCH_BULLISH):
        return 1 if closes[-1] > broken else -1
    else:
        return 1 if closes[-1] < broken else -1


def fmt_ohlcv(row: pd.Series, label: str) -> str:
    return (
        f"  {label:<14} O={row['open']:>10.2f}  H={row['high']:>10.2f}"
        f"  L={row['low']:>10.2f}  C={row['close']:>10.2f}"
        f"  V={row['volume']:>12.0f}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# SWEEP EDGE-CASE PROBES
# ═════════════════════════════════════════════════════════════════════════════

def probe_sweep_edge_cases(
    dfs: dict[str, pd.DataFrame],
    config: SweepConfig,
) -> None:
    """
    Check three known potential bugs in the Liquidity Sweep scanner.

    Bug 1 — Volume bypass when avg_vol==0.
    Bug 2 — Sweep zone half: ATR==0 fallback (0.5% price).
    Bug 3 — Equal-levels overcount via rolling-min series.
    """
    print()
    print("═" * 72)
    print("  PROBE: Liquidity Sweep known edge cases")
    print("═" * 72)

    # ── Bug 1: avg_vol == 0 ───────────────────────────────────────────────────
    print()
    print("  [BUG-1] Volume confirmation bypass (avg_vol_20 == 0)")
    print(SEP)
    zero_vol_symbols: list[tuple[str, int]] = []
    for symbol, df in dfs.items():
        avg_vol = df["volume"].rolling(window=config.avg_volume_window, min_periods=1).mean()
        zero_mask = avg_vol == 0
        cnt = int(zero_mask.sum())
        if cnt:
            zero_vol_symbols.append((symbol, cnt))

    if zero_vol_symbols:
        print(f"  Found {len(zero_vol_symbols)} symbol(s) with ≥1 bar of avg_vol==0:")
        for sym, cnt in zero_vol_symbols[:10]:
            print(f"    {sym}: {cnt} bars with avg_vol==0")
        print("  ⚠  Any sweep on those bars passes volume check unconditionally!")
    else:
        print("  ✅  No symbols have avg_vol_20 == 0 in this dataset — bypass never triggers.")

    # ── Bug 2: ATR == 0 fallback ──────────────────────────────────────────────
    print()
    print("  [BUG-2] Sweep zone half — ATR==0 fallback (0.5% price)")
    print(SEP)
    atr_zero_symbols: list[tuple[str, int]] = []
    for symbol, df in dfs.items():
        prev_close = df["close"].shift(1)
        tr = pd.concat([
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"]  - prev_close).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(window=config.atr_window, min_periods=1).mean()
        cnt = int((atr == 0).sum())
        if cnt:
            atr_zero_symbols.append((symbol, cnt))

    if atr_zero_symbols:
        print(f"  Found {len(atr_zero_symbols)} symbol(s) with ≥1 bar of ATR==0:")
        for sym, cnt in atr_zero_symbols[:10]:
            sample_price = float(dfs[sym]["close"].median())
            fallback_half = abs(sample_price * 0.005)
            print(f"    {sym}: {cnt} bars ATR=0, median price ₹{sample_price:.2f}"
                  f" → fallback zone_half ≈ ₹{fallback_half:.2f}")
        print("  ℹ  0.5% fallback on NSE stocks is reasonable; not a bug in logic.")
    else:
        print("  ✅  No ATR==0 bars in this dataset — fallback never triggers.")

    # ── Bug 3: equal-levels overcount ─────────────────────────────────────────
    print()
    print("  [BUG-3] Equal-levels count — potential overcount via rolling-min series")
    print(SEP)
    overcount_examples: list[str] = []
    for symbol, df in dfs.items():
        sw_low = df["low"].rolling(window=config.swing_lookback, min_periods=1).min()
        # A rolling min stays constant across consecutive bars whenever a persistent
        # low dominates the window. Count runs of identical rolling-min values.
        for k in range(config.swing_lookback, len(df)):
            current = float(sw_low.iloc[k])
            # Count how many of the preceding swing_lookback bars share this value
            run_len = 0
            for j in range(max(0, k - config.swing_lookback), k):
                if abs(float(sw_low.iloc[j]) - current) < 1e-9:
                    run_len += 1
            if run_len >= config.swing_lookback - 1:
                # The entire lookback window has the same rolling min — count will
                # be inflated to swing_lookback, not the number of *actual* distinct lows.
                bar_date = df.index[k].date()
                overcount_examples.append(
                    f"    {symbol} bar={k} ({bar_date}): sw_low plateau len={run_len+1}"
                    f"  → eq_count={run_len+1} (only 1 distinct low in window)"
                )
                if len(overcount_examples) >= 5:
                    break
        if len(overcount_examples) >= 5:
            break

    if overcount_examples:
        print(f"  Found cases where rolling-min plateau inflates equal_levels_count:")
        for ex in overcount_examples:
            print(ex)
        print()
        print("  ⚠  GENUINE BUG: _count_equal_levels counts sw_low/sw_high *series*")
        print("     values, not confirmed pivot points.  When the rolling window")
        print("     carries the same minimum for N bars, eq_count = N instead of 1.")
        print("     Fix: count only CONFIRMED swing lows (i.e., bars where sw_low")
        print("     changes), not every bar whose rolling-min equals the same value.")
    else:
        print("  ✅  No runaway plateaus detected in this sample.")


# ═════════════════════════════════════════════════════════════════════════════
# MARKET STRUCTURE EDGE-CASE PROBES
# ═════════════════════════════════════════════════════════════════════════════

def probe_ms_edge_cases(
    dfs: dict[str, pd.DataFrame],
    config: MSConfig,
) -> None:
    """
    Check two known potential issues in the Market Structure scanner.

    Issue 1 — None anchors after ChoCh if no swings seen yet.
    Issue 2 — BoS anchor reuse after anchor_sh advances to same price.
    """
    print()
    print("═" * 72)
    print("  PROBE: Market Structure known edge cases")
    print("═" * 72)

    # We instrument detect_market_structure manually to trace anchor state.
    # Instead of monkey-patching, we replicate a mini version of the detector
    # loop to count None-anchor states and repeated-level BoS occurrences.

    # ── Issue 1: anchor is None when ChoCh fires ──────────────────────────────
    print()
    print("  [MS-1] anchor_sh / anchor_sl = None after ChoCh reset")
    print(SEP)

    from src.scanners.market_structure.state_manager import TrendStateTracker
    from src.scanners.market_structure.schemas import TrendState

    null_anchor_incidents: list[str] = []

    for symbol, df in list(dfs.items())[:50]:
        eff_lookback = config.tf_1d_swing_lookback   # using 1d effective config
        sw_h = df["high"].rolling(window=eff_lookback, min_periods=1).max()
        sw_l = df["low"].rolling(window=eff_lookback, min_periods=1).min()

        tracker = TrendStateTracker()
        last_sh: float | None = None
        last_sl: float | None = None
        anchor_sh: float | None = None
        anchor_sl: float | None = None

        for k in range(eff_lookback, len(df)):
            high_k  = float(df["high"].iloc[k])
            low_k   = float(df["low"].iloc[k])
            close_k = float(df["close"].iloc[k])

            is_swing_high = abs(high_k - float(sw_h.iloc[k])) < 1e-9
            is_swing_low  = abs(low_k  - float(sw_l.iloc[k])) < 1e-9

            if is_swing_high:
                tracker.register_swing_high(k, high_k)
                last_sh = high_k
                if anchor_sh is None:
                    anchor_sh = high_k
            if is_swing_low:
                tracker.register_swing_low(k, low_k)
                last_sl = low_k
                if anchor_sl is None:
                    anchor_sl = low_k

            if tracker.trend_state == TrendState.UNDEFINED:
                tracker.update_trend_state()
                if tracker.trend_state != TrendState.UNDEFINED:
                    anchor_sh = last_sh
                    anchor_sl = last_sl
                continue

            # Simulate ChoCh reset
            if tracker.trend_state == TrendState.BULLISH and anchor_sl is not None:
                if close_k < anchor_sl:
                    tracker.on_choch_bearish()
                    # After ChoCh, the real code does:
                    anchor_sh = last_sh   # could be None!
                    anchor_sl = last_sl   # could be None!
                    if anchor_sh is None or anchor_sl is None:
                        null_anchor_incidents.append(
                            f"  {symbol} bar={k} ({df.index[k].date()})"
                            f" ChoCh_BEARISH → anchor_sh={anchor_sh}, anchor_sl={anchor_sl}"
                        )
            elif tracker.trend_state == TrendState.BEARISH and anchor_sh is not None:
                if close_k > anchor_sh:
                    tracker.on_choch_bullish()
                    anchor_sh = last_sh
                    anchor_sl = last_sl
                    if anchor_sh is None or anchor_sl is None:
                        null_anchor_incidents.append(
                            f"  {symbol} bar={k} ({df.index[k].date()})"
                            f" ChoCh_BULLISH → anchor_sh={anchor_sh}, anchor_sl={anchor_sl}"
                        )

    if null_anchor_incidents:
        print(f"  Found {len(null_anchor_incidents)} case(s) where anchors are None after ChoCh:")
        for inc in null_anchor_incidents[:5]:
            print(inc)
        print("  ⚠  NULL ANCHOR: Scan continues without valid anchors — no signals")
        print("     fire until the next swing is detected.  Low-risk (silent miss)")
        print("     but anchor_sh=None after ChoCh is an implicit NoneType guard")
        print("     already present (`if anchor_sh is not None`).  Safe as-is.")
    else:
        print("  ✅  No None-anchor ChoCh incidents found in this sample.")

    # ── Issue 2: BoS anchor re-fires on same swing level ──────────────────────
    print()
    print("  [MS-2] BoS anchor reuse — same swing high/low triggers repeated BoS")
    print(SEP)

    refire_incidents: list[str] = []
    for symbol, df in list(dfs.items())[:50]:
        eff_lookback = config.tf_1d_swing_lookback
        sw_h = df["high"].rolling(window=eff_lookback, min_periods=1).max()
        sw_l = df["low"].rolling(window=eff_lookback, min_periods=1).min()

        tracker2 = TrendStateTracker()
        last_sh2: float | None = None
        last_sl2: float | None = None
        anchor_sh2: float | None = None
        anchor_sl2: float | None = None
        prev_anchor_sh: float | None = None
        used_levels: set[tuple[float, str]] = set()

        for k in range(eff_lookback, len(df)):
            high_k  = float(df["high"].iloc[k])
            low_k   = float(df["low"].iloc[k])
            close_k = float(df["close"].iloc[k])

            is_swing_high = abs(high_k - float(sw_h.iloc[k])) < 1e-9
            is_swing_low  = abs(low_k  - float(sw_l.iloc[k])) < 1e-9

            if is_swing_high:
                tracker2.register_swing_high(k, high_k)
                last_sh2 = high_k
                if anchor_sh2 is None:
                    anchor_sh2 = high_k
            if is_swing_low:
                tracker2.register_swing_low(k, low_k)
                last_sl2 = low_k
                if anchor_sl2 is None:
                    anchor_sl2 = low_k

            if tracker2.trend_state == TrendState.UNDEFINED:
                tracker2.update_trend_state()
                if tracker2.trend_state != TrendState.UNDEFINED:
                    anchor_sh2 = last_sh2
                    anchor_sl2 = last_sl2
                continue

            if tracker2.trend_state == TrendState.BULLISH and anchor_sh2 is not None:
                key = (anchor_sh2, "BOS_BULLISH")
                if close_k > anchor_sh2 and key not in used_levels:
                    used_levels.add(key)
                    tracker2.on_bos()
                    prev_anchor_sh = anchor_sh2
                    anchor_sh2 = last_sh2  # advance anchor
                    # If last_sh2 == prev_anchor (no new swing high observed),
                    # the new anchor is identical to old. The key *will* differ
                    # because the old key was added to used_levels, but if last_sh2
                    # is a NEW price that is STILL below close_k, BoS fires again
                    # on the very next bar.
                    if anchor_sh2 is not None and anchor_sh2 == prev_anchor_sh:
                        refire_incidents.append(
                            f"  {symbol} bar={k} ({df.index[k].date()})"
                            f" BOS_BULLISH anchor unchanged at {anchor_sh2:.2f}"
                            f" — next bar will re-fire on same price"
                        )

            elif tracker2.trend_state == TrendState.BEARISH and anchor_sl2 is not None:
                key = (anchor_sl2, "BOS_BEARISH")
                if close_k < anchor_sl2 and key not in used_levels:
                    used_levels.add(key)
                    tracker2.on_bos()
                    prev_anchor_sl = anchor_sl2
                    anchor_sl2 = last_sl2
                    if anchor_sl2 is not None and anchor_sl2 == prev_anchor_sl:
                        refire_incidents.append(
                            f"  {symbol} bar={k} ({df.index[k].date()})"
                            f" BOS_BEARISH anchor unchanged at {anchor_sl2:.2f}"
                            f" — next bar will re-fire on same price"
                        )

    if refire_incidents:
        print(f"  Found {len(refire_incidents)} case(s) of repeated BoS on same anchor level:")
        for inc in refire_incidents[:10]:
            print(inc)
        print()
        print("  ⚠  GENUINE BUG: When no new swing forms between two BoS bars, the")
        print("     anchor advances to last_swing_high_price which equals old anchor_sh.")
        print("     The (level, 'BOS_BULLISH') key was already consumed, so")
        print("     used_levels dedup prevents a literal duplicate — but a different")
        print("     bar at the SAME price produces a new key and fires again.")
        print("     Fix: only advance anchor when last_swing_{high/low}_price")
        print("     is strictly DIFFERENT from the current anchor_sh/anchor_sl.")
    else:
        print("  ✅  No same-anchor BoS refire found in this sample.")


# ═════════════════════════════════════════════════════════════════════════════
# BROKEN SIGNAL DIAGNOSIS
# ═════════════════════════════════════════════════════════════════════════════

def diagnose_broken_signals(dfs: dict[str, pd.DataFrame]) -> None:
    print()
    print("═" * 72)
    print("  DIAGNOSIS: Broken LiqSweep and MktStructure signals")
    print("═" * 72)

    sweep_scanner = SweepScanner(SweepConfig())
    ms_scanner    = MSScanner(MSConfig())
    cutoff_bar    = -SIGNAL_WINDOW   # last N bars

    for scanner_name, scanner, eval_fn in [
        ("LiqSweep",     sweep_scanner, evaluate_sweep_quality),
        ("MktStructure", ms_scanner,    evaluate_ms_quality),
    ]:
        print()
        print(f"  ── {scanner_name} broken signals ──────────────────────────────────")

        broken_count = 0
        respected_count = 0
        untested_count = 0

        for symbol, df in dfs.items():
            try:
                result  = scanner.scan({symbol: df}, timeframe="1d")
                items   = getattr(result, "signals", []) or []
                cutoff_idx = len(df) + cutoff_bar  # len(df) - SIGNAL_WINDOW

                for item in items:
                    bar_idx_attr = (
                        "sweep_bar_index" if scanner_name == "LiqSweep"
                        else "break_bar_index"
                    )
                    bar_idx = getattr(item, bar_idx_attr, None)
                    if bar_idx is None or bar_idx < cutoff_idx:
                        continue

                    quality = eval_fn(item, df, bar_idx)

                    if quality == 1:
                        respected_count += 1
                        continue
                    if quality == 0:
                        untested_count += 1
                        continue

                    # quality == -1  →  broken
                    broken_count += 1

                    # ── Print full diagnostics ────────────────────────────────
                    ts_attr = (
                        "sweep_timestamp" if scanner_name == "LiqSweep"
                        else "break_timestamp"
                    )
                    ts = getattr(item, ts_attr, None)
                    dir_raw = getattr(item, "direction", None) or getattr(
                        item, "signal_type", "?"
                    )
                    direction = getattr(dir_raw, "value", str(dir_raw))

                    if scanner_name == "LiqSweep":
                        level = float(item.swept_level)
                        level_label = "swept_level"
                    else:
                        level = float(item.broken_level)
                        level_label = "broken_level"

                    print()
                    print(f"  ❌  {symbol}  |  {scanner_name}")
                    print(
                        f"      bar_idx={bar_idx}"
                        f"  date={ts.date() if ts else '?'}"
                        f"  direction={direction}"
                        f"  {level_label}={level:.2f}"
                    )

                    # Signal bar OHLCV
                    sig_row = df.iloc[bar_idx]
                    print(fmt_ohlcv(sig_row, f"signal bar[{bar_idx}]"))

                    # Next 2 bars
                    for offset in range(1, EVAL_BARS + 1):
                        idx2 = bar_idx + offset
                        if idx2 < len(df):
                            next_row = df.iloc[idx2]
                            label = f"+{offset} bar[{idx2}]"
                            print(fmt_ohlcv(next_row, label))
                            c = float(next_row["close"])
                            if scanner_name == "LiqSweep":
                                rel = "ABOVE" if c > level else "BELOW"
                                print(f"  {'':14}  close {rel} swept_level ({level:.2f})")
                            else:
                                rel = "ABOVE" if c > level else "BELOW"
                                print(f"  {'':14}  close {rel} broken_level ({level:.2f})")
                        else:
                            print(f"  +{offset} bar: out of range")

                    # Extra context for LiqSweep
                    if scanner_name == "LiqSweep":
                        avg_vol = df["volume"].rolling(
                            window=20, min_periods=1
                        ).mean()
                        avg_v  = float(avg_vol.iloc[bar_idx])
                        vol_k  = float(df["volume"].iloc[bar_idx])
                        vol_ratio = vol_k / avg_v if avg_v > 0 else float("inf")
                        prev_close = df["close"].shift(1)
                        tr = pd.concat([
                            df["high"] - df["low"],
                            (df["high"] - prev_close).abs(),
                            (df["low"]  - prev_close).abs(),
                        ], axis=1).max(axis=1)
                        atr = tr.rolling(14, min_periods=1).mean()
                        atr_k = float(atr.iloc[bar_idx])
                        print(
                            f"  {'':14}  avg_vol={avg_v:.0f}"
                            f"  vol={vol_k:.0f}"
                            f"  vol_ratio={vol_ratio:.2f}"
                            f"  ATR={atr_k:.2f}"
                        )
                        eq_cnt = getattr(item, "equal_levels_count", "?")
                        print(f"  {'':14}  equal_levels_count={eq_cnt}")

                    # Extra context for MktStructure
                    else:
                        sig_type = getattr(item, "signal_type", "?")
                        swing_cnt = getattr(item, "swing_count", "?")
                        bm = getattr(item, "break_magnitude", "?")
                        print(
                            f"  {'':14}  signal_type={sig_type.value if hasattr(sig_type,'value') else sig_type}"
                            f"  swing_count={swing_cnt}"
                            f"  break_magnitude={bm:.3f}" if isinstance(bm, float) else ""
                        )

            except Exception as exc:
                import traceback as tb
                print(f"  [ERROR] {symbol} / {scanner_name}: {exc}")
                tb.print_exc()

        print()
        print(f"  {scanner_name} summary: "
              f"✅ {respected_count} respected  "
              f"❌ {broken_count} broken  "
              f"⏳ {untested_count} untested")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 72)
    print("  Scanner Investigation — Broken Signal Analysis")
    print("=" * 72)

    print(f"\n[DATA] Loading up to {MAX_SYMBOLS} symbols from {DATA_DIR.relative_to(PROJECT_ROOT)} …")
    dfs = load_symbols(MAX_SYMBOLS)
    print(f"[DATA] Loaded {len(dfs)} symbols\n")

    if not dfs:
        print("ERROR: No symbols loaded.")
        return

    sweep_config = SweepConfig()
    ms_config    = MSConfig()

    # 1. Diagnose broken signals with full OHLCV printout
    diagnose_broken_signals(dfs)

    # 2. Probe Liquidity Sweep edge cases
    probe_sweep_edge_cases(dfs, sweep_config)

    # 3. Probe Market Structure edge cases
    probe_ms_edge_cases(dfs, ms_config)

    print()
    print("=" * 72)
    print("  Investigation complete.")
    print("=" * 72)


if __name__ == "__main__":
    main()
