#!/usr/bin/env python3
"""
Run SignalPromotionEngine on all available forward paper outcomes.
Also produces the preliminary promotion report.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone

from src.analytics.signal_promotion import (
    SignalPromotionEngine, SignalLifecycleStage, PromotionGateConfig,
)

OUTCOMES_PATH = Path("artifacts/forward_paper/outcomes.jsonl")
REPORTS_DIR = Path("reports")


def main():
    outcomes = []
    if OUTCOMES_PATH.exists():
        outcomes = [json.loads(l) for l in OUTCOMES_PATH.read_text().splitlines() if l.strip()]

    print(f"Forward paper outcomes available: {len(outcomes)}")
    if not outcomes:
        print("No outcomes yet — cannot run promotion engine")
        return

    import numpy as np
    net_rets = [o["net_return"] for o in outcomes]
    wins = sum(1 for r in net_rets if r > 0)
    print(f"Win rate: {wins}/{len(outcomes)} = {wins/len(outcomes):.1%}")
    print(f"Mean net return: {np.mean(net_rets):.6f}")
    print(f"Partial resolutions: {sum(1 for o in outcomes if o.get('partial_resolution'))}")
    print()

    # ── Run with preliminary thresholds ──────────────────────────────
    engine = SignalPromotionEngine(
        gate_config=PromotionGateConfig(
            min_trades=10,           # preliminary; full gate requires 50
            min_net_expectancy=0.0,
            max_ece=0.10,
            min_regimes=1,           # relax to 1 for preliminary
            max_drawdown=-0.20,
        )
    )
    trade_records = [
        {"net_return": o["net_return"],
         "regime": o.get("regime", "UNKNOWN"),
         "cost_bps": o.get("cost_bps", 27.65)}
        for o in outcomes
    ]
    result = engine.evaluate(
        signal_family="FORWARD_PAPER_SESSION_1",
        current_stage=SignalLifecycleStage.SHADOW,
        trade_records=trade_records,
    )
    print(f"Preliminary promotion decision: {result.decision}")
    print(f"Reason: {result.reason}")
    print()
    print("Gate results:")
    for gate, status in result.gate_results.items():
        print(f"  {gate}: {status}")

    # ── Save report ────────────────────────────────────────────────────
    report = {
        "run_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "n_outcomes": len(outcomes),
        "n_partial_resolutions": sum(1 for o in outcomes if o.get("partial_resolution")),
        "win_rate": round(wins / len(outcomes), 4),
        "mean_net_return": round(float(np.mean(net_rets)), 6),
        "max_drawdown": round(result.max_drawdown, 6),
        "regimes_tested": result.regimes_tested,
        "preliminary_decision": result.decision,
        "preliminary_reason": result.reason,
        "gate_results": result.gate_results,
        "outcomes": [o for o in outcomes],
        "note": (
            "Preliminary assessment with only 9 partial (mark-to-market) outcomes. "
            "Full promotion requires 50+ fully resolved outcomes (final exit price). "
            "Second batch of 49 signals resolves at 2026-09-28 18:30 UTC. "
            "All outcomes are partial — entry at open[Sep 23], exit at close[Sep 24] "
            "(not the intended open[Sep 30] final exit). True P&L unknown until Sep 30."
        ),
    }
    out = REPORTS_DIR / "signal_promotion_preliminary.json"
    out.write_text(json.dumps(report, indent=2, default=str))
    print()
    print(f"Saved -> {out}")
    print()
    print("STATUS: INSUFFICIENT_EVIDENCE — 9 partial outcomes (need 50 fully resolved)")
    print("NEXT:   Wait for 2026-09-28 18:30 UTC batch (49 signals) → resolve → re-run")


if __name__ == "__main__":
    main()
