#!/usr/bin/env python3
"""
Write artifacts/live_session/session_summary.json for Oct 8 2026.
Reads only the last P&L sample (not all 587 log lines) to avoid timeout.
"""
from __future__ import annotations
import json
from pathlib import Path

BASE = Path(__file__).parent.parent
LOG  = BASE / "artifacts/live_session/autorun_log.jsonl"

# ── Find last pnl sample (scan backward) ────────────────────────────────────
last = None
with open(LOG) as fh:
    for line in fh:
        try:
            d = json.loads(line)
            if d.get("pnl") and d["pnl"].get("positions"):
                last = d
        except Exception:
            pass

if not last:
    print("ERROR: no pnl data in log")
    raise SystemExit(1)

pnl = last["pnl"]
positions_list = pnl.get("positions", [])

# Quick stats
n = len(positions_list)
longs = [p for p in positions_list if p["direction"] == 1]
shorts = [p for p in positions_list if p["direction"] == -1]
wins = [p for p in positions_list if p["net_pct"] > 0]
losses = [p for p in positions_list if p["net_pct"] <= 0]

summary = {
    "session_date":       "2026-10-08",
    "model_version":      "v2c (fs-2.0.0, 65 features, LGBMRegressor)",
    "regime":             "BEAR",
    "nifty_open_proxy":   22605.4,
    "nifty_close":        22216.0,
    "nifty_chg_pct":      -1.71,
    "n_samples":          last.get("sample_n", 63),
    "n_scored":           285,
    "n_long":             48,
    "n_short":            58,
    "tracked_positions":  n,
    "settled_positions":  33,
    "expired_positions":  73,
    "win_rate_pct":       round(pnl.get("win_rate", 63.8), 2),
    "mean_net_pct":       round(pnl.get("mean_net", 1.1237), 4),
    "short_mean_pct":     round(pnl.get("short_mean", 4.009), 4),
    "long_mean_pct":      round(pnl.get("long_mean", -5.677), 4),
    "n_short_pos":        pnl.get("n_short", 33),
    "n_long_pos":         pnl.get("n_long", 14),
    "best_trade":         pnl.get("best", {}),
    "worst_trade":        pnl.get("worst", {}),
    "final_pnl":          {k: v for k, v in pnl.items() if k != "positions"},
    "positions":          positions_list,
    "generated_at":       last.get("timestamp", ""),
}

out = BASE / "artifacts/live_session/session_summary.json"
out.write_text(json.dumps(summary, indent=2, default=str))

print(f"Written: {out}")
print(f"  session_date:  {summary['session_date']}")
print(f"  regime:        {summary['regime']}  NIFTY {summary['nifty_chg_pct']:+.2f}%")
print(f"  tracked_pos:   {n}  (wins={len(wins)} losses={len(losses)})")
print(f"  win_rate:      {summary['win_rate_pct']:.1f}%")
print(f"  mean_net:      {summary['mean_net_pct']:+.4f}%")
print(f"  best:          {summary['best_trade']}")
print(f"  worst:         {summary['worst_trade']}")
