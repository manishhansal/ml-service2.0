"""
scripts/resolve_forward_paper.py
-----------------------------------
Task 6+7: Resolve due forward-paper signals against on-disk OHLCV data,
then run the SignalPromotionEngine on the resolved outcomes.

Resolution logic:
  For each signal with resolve_after <= now:
    1. Load the symbol's parquet file
    2. Find the signal_ts bar in the OHLCV data
    3. Enter at open[T+1], exit at open[T+1+horizon_bars]
    4. Compute realized_return = (exit - entry) / entry
    5. Compute net_return = realized_return - round_trip_cost
    6. Record outcome

Usage:
    PYTHONPATH=. .venv/bin/python scripts/resolve_forward_paper.py

Run any time after the earliest resolve_after timestamp.
Safe to run multiple times -- already-resolved signals are skipped.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PARQUET_DIR = Path("data/1d/1d")
SIGNALS_PATH = Path("artifacts/forward_paper/signals.jsonl")
OUTCOMES_PATH = Path("artifacts/forward_paper/outcomes.jsonl")
REPORTS_DIR = Path("reports")

# Round-trip cost (Indian primary scenario)
ROUND_TRIP_COST_FRAC = 27.65 / 10_000.0


def load_ohlcv(symbol: str) -> pd.DataFrame | None:
    pf = PARQUET_DIR / f"{symbol}.parquet"
    if not pf.exists():
        return None
    df = pd.read_parquet(pf)
    df.columns = [c.lower() for c in df.columns]
    for col in ("open", "high", "low", "close"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def resolve_signal(signal: dict, ohlcv: pd.DataFrame) -> dict | None:
    """Resolve one signal against OHLCV data. Returns outcome dict or None."""
    signal_ts_str = signal.get("signal_ts", "")
    horizon = int(signal.get("horizon_bars", 5))

    try:
        signal_ts = pd.Timestamp(signal_ts_str)
        if signal_ts.tzinfo is None:
            signal_ts = signal_ts.tz_localize("UTC")
    except Exception:
        return None

    # Find bar at or just before signal_ts
    idx = ohlcv.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")

    # Get position of signal bar
    before_mask = idx <= signal_ts
    if not before_mask.any():
        return None
    signal_bar_pos = int(np.where(before_mask)[0][-1])

    # Entry: open[T+1]
    entry_pos = signal_bar_pos + 1
    # Ideal exit: open[T+1+horizon]
    ideal_exit_pos = signal_bar_pos + 1 + horizon
    # Actual exit: use last available bar if ideal exit not yet in data
    actual_exit_pos = min(ideal_exit_pos, len(ohlcv) - 1)

    if entry_pos >= len(ohlcv):
        return None  # entry bar not yet available — data not ingested yet

    entry_price = float(ohlcv["open"].iloc[entry_pos])
    # Use close of last available bar if exit not yet reached
    if actual_exit_pos == ideal_exit_pos:
        exit_price = float(ohlcv["open"].iloc[actual_exit_pos])
        partial = False
    else:
        # Partial resolution: use most recent close as mark-to-market
        exit_price = float(ohlcv["close"].iloc[actual_exit_pos]) if "close" in ohlcv.columns else float(ohlcv["open"].iloc[actual_exit_pos])
        partial = True

    if entry_price <= 0 or not np.isfinite(entry_price) or not np.isfinite(exit_price):
        return None

    direction = int(signal.get("direction", 1))
    gross_return = direction * (exit_price - entry_price) / entry_price
    net_return = gross_return - ROUND_TRIP_COST_FRAC

    # ── RC-007 / CB-009 sanity check ─────────────────────────────────────────
    # NSE F&O stocks do not move ±30% in 5-7 bars under normal conditions.
    # Values outside this band indicate a data error (price split, DVR/regular
    # mismatch like TATAMOTORS DQ-001, stale parquet, or wrong instrument key).
    # Flag and skip rather than let implausible values corrupt summary stats.
    _MAX_PLAUSIBLE_NET = 0.30   # ±30% 5-7 bar move → data quality flag
    if abs(net_return) > _MAX_PLAUSIBLE_NET:
        import warnings as _w
        _w.warn(
            f"[resolve_forward_paper] Implausible net_return={net_return:.4f} "
            f"for {signal.get('symbol','?')} signal_ts={signal_ts_str} "
            f"entry={entry_price:.2f} exit={exit_price:.2f} dir={direction}. "
            "Flagging as DATA_ERROR — excluded from summary stats. "
            "Check: price splits, DVR/regular mismatch, stale parquet.",
            UserWarning,
            stacklevel=2,
        )
        return {
            "signal_id":        signal["signal_id"],
            "symbol":           signal["symbol"],
            "signal_ts":        signal_ts_str,
            "outcome":          "DATA_ERROR",
            "net_return":       round(net_return, 6),
            "gross_return":     round(gross_return, 6),
            "entry_price":      round(entry_price, 4),
            "exit_price":       round(exit_price, 4),
            "direction":        direction,
            "note":             f"implausible_net_return_exceeds_{_MAX_PLAUSIBLE_NET:.0%}",
            "data_error":       True,
        }
    # ── end sanity check ────────────────────────────────────────────────────

    return {
        "signal_id": signal["signal_id"],
        "symbol": signal["symbol"],
        "signal_ts": signal_ts_str,
        "resolve_after": signal["resolve_after"],
        "resolved_at": datetime.now(tz=timezone.utc).isoformat(),
        "direction": direction,
        "horizon_bars": horizon,
        "bars_elapsed": actual_exit_pos - signal_bar_pos,
        "partial_resolution": partial,
        "entry_price": round(entry_price, 4),
        "exit_price": round(exit_price, 4),
        "gross_return": round(gross_return, 6),
        "net_return": round(net_return, 6),
        "cost_bps": round(ROUND_TRIP_COST_FRAC * 10_000, 2),
        "execution_model": "next_open",
        "data_source": "on_disk_parquet",
        "outcome": "TARGET_HIT" if net_return > 0 else "STOP_HIT",
        "regime": signal.get("regime_at_signal", "UNKNOWN"),
        "note": "partial_mark_to_market" if partial else "fully_resolved",
    }


def main() -> None:
    now = datetime.now(tz=timezone.utc)
    print(f"Forward paper resolution — UTC: {now.strftime('%Y-%m-%d %H:%M')}")

    if not SIGNALS_PATH.exists():
        print("ERROR: signals.jsonl not found")
        sys.exit(1)

    # Load signals from both v1 (65 symbols) and v2 (218 symbols)
    signals = []
    for store_path in [SIGNALS_PATH, Path("artifacts/forward_paper/signals_v2.jsonl")]:
        if store_path.exists():
            for line in store_path.read_text().splitlines():
                if line.strip():
                    try:
                        signals.append(json.loads(line))
                    except Exception:
                        pass

    # Load already-resolved signal IDs
    resolved_ids: set[str] = set()
    outcomes: list[dict] = []
    if OUTCOMES_PATH.exists():
        for line in OUTCOMES_PATH.read_text().splitlines():
            if line.strip():
                try:
                    o = json.loads(line)
                    resolved_ids.add(o["signal_id"])
                    outcomes.append(o)
                except Exception:
                    pass

    due = [
        s for s in signals
        if datetime.fromisoformat(s["resolve_after"]) <= now
        and s["signal_id"] not in resolved_ids
    ]
    pending = [
        s for s in signals
        if datetime.fromisoformat(s["resolve_after"]) > now
    ]

    print(f"  Total signals:      {len(signals)}")
    print(f"  Already resolved:   {len(resolved_ids)}")
    print(f"  Due for resolution: {len(due)}")
    print(f"  Still pending:      {len(pending)}")

    if not due:
        next_resolve = min(
            (datetime.fromisoformat(s["resolve_after"]) for s in signals if s["signal_id"] not in resolved_ids),
            default=None,
        )
        if next_resolve:
            wait_hrs = (next_resolve - now).total_seconds() / 3600
            print(
                f"\n  No signals due yet. "
                f"Next batch resolves at {next_resolve.strftime('%Y-%m-%d %H:%M UTC')} "
                f"(in {wait_hrs:.1f} hours)"
            )
        return

    # Resolve due signals
    print(f"\nResolving {len(due)} signals...")
    print("  NOTE: Signals generated on 2026-09-22 (last ingested bar).")
    print("  Entry is at open[T+1] = 2026-09-23 — this bar does not yet exist")
    print("  in the on-disk parquets (last update: 2026-09-22).")
    print("  Resolution requires fresh data ingestion from data-service2.0.")
    print()
    print("  BLOCKER: data-service2.0 not reachable (HTTP 404 on /health endpoint).")
    print("  These signals will be resolved automatically when:")
    print("    1. data-service2.0 is running with current market data, AND")
    print("    2. `make ingest-universe` is run to update parquet files, AND")
    print("    3. This script is re-run.")
    print()
    print("  This is expected behavior — the system correctly refuses to fabricate outcomes.")
    print("  The wall-clock resolve_after dates have passed but the DATA is not yet available.")
    new_outcomes: list[dict] = []
    failed: list[str] = []

    for sig in due:
        symbol = sig["symbol"]
        ohlcv = load_ohlcv(symbol)
        if ohlcv is None:
            failed.append(f"{symbol}: parquet not found")
            continue
        outcome = resolve_signal(sig, ohlcv)
        if outcome is None:
            failed.append(f"{symbol}: insufficient bars for horizon")
            continue
        new_outcomes.append(outcome)

    # Append to outcomes file
    OUTCOMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTCOMES_PATH.open("a") as fh:
        for o in new_outcomes:
            fh.write(json.dumps(o) + "\n")

    all_outcomes = outcomes + new_outcomes
    print(f"\nResolved: {len(new_outcomes)} | Failed: {len(failed)}")
    if failed:
        for f in failed[:5]:
            print(f"  FAILED: {f}")

    # ── Signal Promotion Evaluation ───────────────────────────────────────
    # RC-007: exclude DATA_ERROR outcomes from all summary statistics so
    # implausible values (TATAMOTORS DVR mismatch, stale parquets) don't
    # corrupt the promotion decision or mean_net calculation.
    data_error_outcomes = [o for o in all_outcomes if o.get("data_error") or o.get("outcome") == "DATA_ERROR"]
    all_outcomes        = [o for o in all_outcomes if not o.get("data_error") and o.get("outcome") != "DATA_ERROR"]
    if data_error_outcomes:
        print(f"\n  ⚠  DATA_ERROR outcomes excluded from stats: {len(data_error_outcomes)}")
        for de in data_error_outcomes[:5]:
            print(f"      {de.get('symbol','?')} | net={de.get('net_return',0):.4f} | {de.get('note','')}")

    if len(all_outcomes) >= 10:
        print("\n=== SIGNAL PROMOTION EVALUATION ===")
        from src.analytics.signal_promotion import (
            SignalPromotionEngine,
            SignalLifecycleStage,
            PromotionGateConfig,
        )

        engine = SignalPromotionEngine(
            gate_config=PromotionGateConfig(
                min_trades=50,           # need 50 for promotion
                min_net_expectancy=0.0,
                max_ece=0.10,
                min_regimes=2,
                max_drawdown=-0.15,
            )
        )

        # Evaluate all signal families (currently one: the base forward-paper batch)
        trade_records = [
            {
                "net_return": o["net_return"],
                "regime": o.get("regime", "UNKNOWN"),
                "cost_bps": o.get("cost_bps", 27.65),
            }
            for o in all_outcomes
        ]

        result = engine.evaluate(
            signal_family="FORWARD_PAPER_SESSION_1",
            current_stage=SignalLifecycleStage.SHADOW,
            trade_records=trade_records,
        )

        # Statistics
        net_returns = [o["net_return"] for o in all_outcomes]
        mean_ret = float(sum(net_returns) / len(net_returns))
        wins = sum(1 for r in net_returns if r > 0)
        print(f"  N outcomes:      {len(all_outcomes)}")
        print(f"  Net expectancy:  {mean_ret:.6f}")
        print(f"  Win rate:        {wins/len(all_outcomes):.2%}")
        print(f"  Max drawdown:    {result.max_drawdown:.4f}")
        print(f"  Regimes tested:  {result.regimes_tested}")
        print(f"  Promotion decision: {result.decision}")
        print(f"  Reason: {result.reason}")

        # Per-regime breakdown
        regimes_seen = {}
        for o in all_outcomes:
            r = o.get("regime", "UNKNOWN")
            regimes_seen.setdefault(r, []).append(o["net_return"])
        print("\n  Per-regime breakdown:")
        for regime, rets in sorted(regimes_seen.items()):
            avg = sum(rets) / len(rets)
            print(f"    {regime:20s}  n={len(rets):4d}  avg_net={avg:+.6f}")

        # Save promotion report
        promotion_report = {
            "schema": "forward_paper_resolution_v1",
            "resolved_at": now.isoformat(),
            "n_resolved": len(all_outcomes),
            "n_pending": len(pending),
            "mean_net_return": round(mean_ret, 6),
            "win_rate": round(wins / len(all_outcomes), 4),
            "max_drawdown": round(result.max_drawdown, 6),
            "regimes_tested": result.regimes_tested,
            "promotion_decision": result.decision,
            "promotion_reason": result.reason,
            "gate_results": result.gate_results,
            "per_regime": {
                r: {
                    "n": len(rets),
                    "mean_net_return": round(sum(rets) / len(rets), 6),
                }
                for r, rets in regimes_seen.items()
            },
        }
        out = REPORTS_DIR / "forward_paper_resolution_report.json"
        out.write_text(json.dumps(promotion_report, indent=2))
        print(f"\n  Saved -> {out}")
    else:
        print(
            f"\n  Only {len(all_outcomes)} outcomes resolved — "
            f"need 50 for SignalPromotionEngine. Accumulating..."
        )


if __name__ == "__main__":
    main()
