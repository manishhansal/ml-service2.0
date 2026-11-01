#!/usr/bin/env python3
"""
_settle_outage.py — Settle / expire forward-paper signals that went
unresolved during the Oct 14 – Nov 1, 2026 service outage.

Resolves 68 open signals from 2026-10-12/13 using the backfilled parquets.
Already-resolved signals are skipped (idempotent).

Usage:
    PYTHONPATH=. python3 scripts/_settle_outage.py [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BASE        = Path(__file__).parent.parent
PARQUET_DIR = BASE / "data/1d/1d"
SIGNALS_V2  = BASE / "artifacts/forward_paper/signals_v2.jsonl"
OUTCOMES    = BASE / "artifacts/forward_paper/outcomes.jsonl"
COST_FRAC   = 27.65 / 10_000.0   # round-trip equity cost


def _load_ohlcv(sym: str) -> pd.DataFrame | None:
    pf = PARQUET_DIR / f"{sym}.parquet"
    if not pf.exists():
        return None
    df = pd.read_parquet(pf)
    df.columns = [c.lower() for c in df.columns]
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _resolve_signal(sig: dict, ohlcv: pd.DataFrame) -> dict | None:
    """
    Entry at open[T+1], exit at open[T+1+horizon_bars].
    Falls back to close[last] if exit bar not yet available.
    Returns outcome dict or None.
    """
    signal_ts_str = sig.get("signal_ts", "")
    horizon       = int(sig.get("horizon_bars", 5))
    direction     = int(sig.get("direction", 1))
    symbol        = sig.get("symbol", "")

    try:
        signal_ts = pd.Timestamp(signal_ts_str)
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("UTC")
    except Exception:
        return None

    idx = ohlcv.index
    before_mask = idx <= signal_ts
    if not before_mask.any():
        return None

    signal_bar_pos = int(np.where(before_mask)[0][-1])
    entry_pos      = signal_bar_pos + 1
    exit_pos       = signal_bar_pos + 1 + horizon
    partial        = exit_pos >= len(ohlcv)

    if entry_pos >= len(ohlcv):
        return None   # entry bar not yet available

    entry_price = float(ohlcv["open"].iloc[entry_pos])
    if pd.isna(entry_price) or entry_price <= 0:
        return None

    if partial:
        exit_pos   = len(ohlcv) - 1
        exit_price = float(ohlcv["close"].iloc[exit_pos])
    else:
        exit_price = float(ohlcv["open"].iloc[exit_pos])

    if pd.isna(exit_price) or exit_price <= 0:
        return None

    # Sanity guard: implausible return (±30%) → DATA_ERROR
    raw_ret = (exit_price - entry_price) / entry_price
    if direction == -1:
        raw_ret = -raw_ret
    if abs(raw_ret) > 0.30:
        return {
            "symbol":           symbol,
            "signal_id":        sig.get("signal_id", ""),
            "direction":        direction,
            "signal_ts":        signal_ts_str,
            "entry_price":      entry_price,
            "exit_price":       exit_price,
            "realized_return":  raw_ret,
            "net_return":       raw_ret - COST_FRAC,
            "win":              False,
            "status":           "DATA_ERROR",
            "partial":          partial,
            "resolved_at":      datetime.now(tz=timezone.utc).isoformat(),
            "outage_recovery":  True,
        }

    net_ret = raw_ret - COST_FRAC
    return {
        "symbol":           symbol,
        "signal_id":        sig.get("signal_id", ""),
        "direction":        direction,
        "signal_ts":        signal_ts_str,
        "entry_price":      entry_price,
        "exit_price":       exit_price,
        "realized_return":  round(raw_ret, 6),
        "net_return":       round(net_ret, 6),
        "win":              net_ret > 0,
        "status":           "SETTLED" if not partial else "EXPIRED",
        "partial":          partial,
        "resolved_at":      datetime.now(tz=timezone.utc).isoformat(),
        "outage_recovery":  True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Load already-resolved signal IDs from outcomes to stay idempotent
    resolved_ids: set[str] = set()
    if OUTCOMES.exists():
        for line in OUTCOMES.read_text().splitlines():
            if line.strip():
                try:
                    o = json.loads(line)
                    sid = o.get("signal_id") or o.get("symbol", "")
                    if sid:
                        resolved_ids.add(sid)
                except Exception:
                    pass
    print(f"Already resolved: {len(resolved_ids)} signals")

    # Load unresolved signals from signals_v2.jsonl
    pending: list[dict] = []
    if SIGNALS_V2.exists():
        for line in SIGNALS_V2.read_text().splitlines():
            if line.strip():
                try:
                    s = json.loads(line)
                    sid = s.get("signal_id") or s.get("symbol", "")
                    # Only resolve signals from Oct 12/13 (outage-era open positions)
                    created = s.get("created_at", "")[:10]
                    if created >= "2026-10-12" and sid not in resolved_ids:
                        pending.append(s)
                except Exception:
                    pass
    print(f"Pending (Oct 12-13 unresolved): {len(pending)} signals")

    if not pending:
        print("Nothing to resolve — already up to date.")
        return

    settled = expired = data_errors = skipped = 0
    new_outcomes: list[dict] = []

    for sig in pending:
        sym = sig.get("symbol", "")
        ohlcv = _load_ohlcv(sym)
        if ohlcv is None:
            print(f"  [SKIP] {sym}: no parquet")
            skipped += 1
            continue

        outcome = _resolve_signal(sig, ohlcv)
        if outcome is None:
            print(f"  [SKIP] {sym}: could not resolve (entry bar missing)")
            skipped += 1
            continue

        status  = outcome["status"]
        net_pct = outcome["net_return"] * 100
        print(f"  [{status}] {sym:15s}  net={net_pct:+.2f}%  entry={outcome['entry_price']:.2f}  exit={outcome['exit_price']:.2f}")

        new_outcomes.append(outcome)
        if status == "SETTLED":
            settled += 1
        elif status == "EXPIRED":
            expired += 1
        else:
            data_errors += 1

    print()
    print(f"Results: SETTLED={settled}  EXPIRED={expired}  DATA_ERROR={data_errors}  SKIP={skipped}")

    # Promotion-engine summary (informational)
    all_tradeable = [o for o in new_outcomes if o["status"] in ("SETTLED", "EXPIRED")]
    if all_tradeable:
        wins      = sum(1 for o in all_tradeable if o.get("win"))
        win_rate  = wins / len(all_tradeable) * 100
        mean_net  = sum(o["net_return"] for o in all_tradeable) / len(all_tradeable) * 100
        print(f"Outage batch: n={len(all_tradeable)}  win_rate={win_rate:.1f}%  mean_net={mean_net:+.2f}%")

    if args.dry_run:
        print("[DRY RUN] No files written.")
        return

    if new_outcomes:
        with open(OUTCOMES, "a") as f:
            for o in new_outcomes:
                f.write(json.dumps(o) + "\n")
        print(f"Appended {len(new_outcomes)} outcomes → {OUTCOMES}")
    else:
        print("No new outcomes to write.")


if __name__ == "__main__":
    main()
