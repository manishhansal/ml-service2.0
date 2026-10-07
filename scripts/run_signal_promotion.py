#!/usr/bin/env python3
"""
scripts/run_signal_promotion.py
────────────────────────────────
Run SignalPromotionEngine on v2c live session outcomes from PostgreSQL.

Fixes applied (ISSUE-11 through ISSUE-14):
  ISSUE-11: Derive regime from nifty_chg_pct instead of UNKNOWN string
  ISSUE-12: Use futures cost 7.26bps (not equity 27.65bps)
  ISSUE-13: Include all settled positions (intraday horizon, not 5-bar filter)
  ISSUE-14: Drawdown computed on mean-daily-return series, not concatenated signals

Data source: PostgreSQL positions + sessions tables (v2c live data)
             Falls back to historical_outcomes (pre-v2c forward paper data)
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

import numpy as np
from src.analytics.signal_promotion import (
    SignalPromotionEngine,
    SignalLifecycleStage,
    PromotionGateConfig,
)
from src.data.signal_db import get_db

REPORTS_DIR = BASE / "reports" / "gates"
COST_BPS_FUTURES = 7.26  # ISSUE-12: use futures cost (v2c authorized strategy)


def infer_regime(nifty_chg_pct: float | None) -> str:
    """ISSUE-11: infer regime from NIFTY change instead of storing 'UNKNOWN'."""
    if nifty_chg_pct is None:
        return "UNKNOWN"
    if nifty_chg_pct > 0.3:
        return "BULL"
    if nifty_chg_pct < -0.3:
        return "BEAR"
    return "SIDEWAYS"


def build_trade_records(db) -> list[dict]:
    """
    Build trade records from PostgreSQL.

    Primary: live positions (sessions + positions tables)
    Fallback: historical_outcomes (pre-v2c forward paper data, only fully resolved)

    ISSUE-12: Override cost_bps to futures (7.26bps) for all records.
    ISSUE-13: No horizon filter — include all settled positions.
    ISSUE-14: Return individual trade returns, not sequential equity curve.
    """
    records: list[dict] = []

    # ── Live positions from sessions table ───────────────────────────────────
    sessions = db.execute_query("""
        SELECT s.session_date, s.nifty_chg_pct,
               p.symbol, p.direction, p.final_return_pct,
               p.gross_return_pct, p.status
        FROM positions p
        JOIN sessions s ON p.session_date = s.session_date
        WHERE p.status IN ('SETTLED_WIN','SETTLED_LOSS')
          AND p.final_return_pct IS NOT NULL
        ORDER BY s.session_date, p.symbol
    """)

    for row in sessions:
        regime = infer_regime(row.get("nifty_chg_pct"))
        records.append({
            "net_return":  float(row["final_return_pct"]) / 100.0,
            "regime":      regime,
            "cost_bps":    COST_BPS_FUTURES,
            "session_date": row["session_date"],
            "symbol":      row["symbol"],
            "direction":   row["direction"],
            "source":      "live",
        })

    # ── Historical outcomes (pre-v2c) ─────────────────────────────────────────
    # Only include fully resolved (not partial) with futures-equivalent cost
    historical = db.execute_query("""
        SELECT symbol, direction, net_return, regime, bars_elapsed, partial
        FROM historical_outcomes
        WHERE outcome IN ('TARGET_HIT','STOP_HIT')
          AND partial = FALSE
          AND net_return IS NOT NULL
          AND net_return BETWEEN -0.30 AND 0.30
        ORDER BY resolved_at
    """)

    for row in historical:
        # Adjust cost from equity (27.65bps) to futures (7.26bps)
        adjusted_net = float(row["net_return"]) + (27.65 - COST_BPS_FUTURES) / 10_000.0
        regime = row.get("regime") or "UNKNOWN"
        records.append({
            "net_return":   adjusted_net,
            "regime":       regime,
            "cost_bps":     COST_BPS_FUTURES,
            "session_date": None,
            "symbol":       row["symbol"],
            "direction":    row["direction"],
            "source":       "historical",
        })

    return records


def compute_drawdown_correctly(trade_records: list[dict]) -> float:
    """
    ISSUE-14: Compute drawdown from daily mean returns, not concatenated signal P&L.

    The old approach stacked all 218 signal returns as an equity curve, which created
    an artificial 52.8% drawdown (as if you entered all 218 sequentially).

    Correct approach: group by session_date, compute mean session return, then
    compute max drawdown on that daily return series.
    """
    from collections import defaultdict
    daily_rets: dict[str, list[float]] = defaultdict(list)

    for r in trade_records:
        sd = r.get("session_date")
        if sd and r.get("net_return") is not None:
            daily_rets[sd].append(r["net_return"])

    if not daily_rets:
        return 0.0

    sorted_dates = sorted(daily_rets.keys())
    daily_means  = [np.mean(daily_rets[d]) for d in sorted_dates]

    # Compute equity curve and max drawdown
    equity = np.cumprod(1.0 + np.array(daily_means))
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / (running_max + 1e-9)
    return float(np.min(drawdowns))


def main() -> None:
    db = get_db()
    records = build_trade_records(db)

    live_records = [r for r in records if r["source"] == "live"]
    hist_records = [r for r in records if r["source"] == "historical"]

    print(f"Trade records loaded: {len(records)} total")
    print(f"  Live (v2c sessions):       {len(live_records)}")
    print(f"  Historical (pre-v2c):      {len(hist_records)}")

    if not records:
        print("No settled outcomes — cannot run promotion engine")
        return

    net_rets = [r["net_return"] for r in records]
    wins     = sum(1 for r in net_rets if r > 0)
    win_rate = wins / len(records)
    mean_ret = float(np.mean(net_rets))

    print(f"Win rate:        {win_rate:.1%} ({wins}/{len(records)})")
    print(f"Mean net return: {mean_ret:+.6f}")

    # Regime distribution (ISSUE-11)
    regime_counts: dict[str, int] = {}
    for r in records:
        regime_counts[r["regime"]] = regime_counts.get(r["regime"], 0) + 1
    print(f"Regimes: {regime_counts}")

    # Correct drawdown (ISSUE-14)
    max_dd = compute_drawdown_correctly(records)
    print(f"Max drawdown (daily series): {max_dd:+.4f}")
    print()

    # ── Run promotion engine ──────────────────────────────────────────────────
    engine = SignalPromotionEngine(
        gate_config=PromotionGateConfig(
            min_trades=10,
            min_net_expectancy=0.0,
            max_ece=0.10,
            min_regimes=1,           # relax for preliminary
            max_drawdown=-0.20,
        )
    )

    # Pass records with corrected cost/regime (ISSUE-11, 12, 13)
    engine_records = [
        {"net_return": r["net_return"],
         "regime":     r["regime"],
         "cost_bps":   r["cost_bps"]}
        for r in records
    ]

    result = engine.evaluate(
        signal_family="V2C_LIVE_SESSIONS",
        current_stage=SignalLifecycleStage.SHADOW,
        trade_records=engine_records,
    )

    # Override drawdown with correct calculation
    result_max_dd = max_dd

    print(f"Promotion decision: {result.decision}")
    print(f"Reason:             {result.reason}")
    print()
    print("Gate results:")
    for gate, status in result.gate_results.items():
        print(f"  {gate}: {status}")

    # Re-evaluate drawdown gate with corrected value
    dd_gate_pass = result_max_dd > -0.20
    print(f"  G6_DRAWDOWN (corrected): {'PASS' if dd_gate_pass else f'FAIL (dd={result_max_dd:.4f})'}")

    # ── Save report ───────────────────────────────────────────────────────────
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "run_timestamp":            datetime.now(tz=timezone.utc).isoformat(),
        "model":                    "v2c",
        "cost_model":               f"futures {COST_BPS_FUTURES}bps (ISSUE-12 fix)",
        "n_records":                len(records),
        "n_live":                   len(live_records),
        "n_historical":             len(hist_records),
        "win_rate":                 round(win_rate, 4),
        "mean_net_return":          round(mean_ret, 6),
        "max_drawdown_corrected":   round(result_max_dd, 6),
        "regime_distribution":      regime_counts,
        "regimes_tested":           list(regime_counts.keys()),
        "preliminary_decision":     result.decision,
        "preliminary_reason":       result.reason,
        "gate_results":             {
            **result.gate_results,
            "G6_DRAWDOWN_CORRECTED": "PASS" if dd_gate_pass else f"FAIL (dd={result_max_dd:.4f})",
        },
        "fixes_applied": [
            "ISSUE-11: regime derived from nifty_chg_pct (not UNKNOWN string)",
            "ISSUE-12: cost_bps=7.26 futures (was 27.65 equity)",
            "ISSUE-13: all settled positions included (no horizon filter)",
            "ISSUE-14: drawdown on daily mean-return series (not concatenated signals)",
        ],
        "data_source": "PostgreSQL positions + historical_outcomes tables",
    }

    out = REPORTS_DIR / "signal_promotion_preliminary.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
