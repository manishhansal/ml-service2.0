"""Run settlement for today's positions with actual close prices."""
import sys, json
import pandas as pd
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
PARQUET_DIR = BASE / "data/1d/1d"

from src.analytics.signal_ledger import SignalLedger

# Build final quotes from parquets (last close) + live quotes
quotes = {}
for pf in PARQUET_DIR.glob("*.parquet"):
    try:
        df = pd.read_parquet(str(pf), columns=["close"])
        if len(df) > 0:
            quotes[pf.stem] = {"ltp": float(df["close"].iloc[-1])}
    except Exception:
        pass
lq_raw = json.load(open(BASE / "artifacts/live_session/live_quotes.json"))
for sym, q in lq_raw.get("quotes", {}).items():
    if isinstance(q, dict) and q.get("ltp") and float(q["ltp"]) > 0:
        quotes[sym] = {"ltp": float(q["ltp"])}

ledger  = SignalLedger()
session = "2026-10-06"
settled = ledger.settle_session(session, final_quotes=quotes)

wins   = [p for p in settled if p["status"] == "SETTLED_WIN"]
losses = [p for p in settled if p["status"] == "SETTLED_LOSS"]
rets   = [p["final_return_pct"] for p in wins + losses if p.get("final_return_pct") is not None]
mean_r = sum(rets) / len(rets) if rets else 0.0
win_rt = len(wins) / len(settled) if settled else 0.0

print(f"\nINTRADAY SETTLEMENT — {session}  (NIFTY +0.72%)")
print("=" * 60)
print(f"Total settled:   {len(settled)}")
print(f"WIN:  {len(wins)} ({win_rt:.0%})  |  LOSS: {len(losses)} ({1-win_rt:.0%})")
print(f"Mean net return: {mean_r:+.3f}%  (after 7.26bps futures cost)")
print()
all_pos = sorted(wins + losses, key=lambda p: p.get("final_return_pct") or 0, reverse=True)
print(f"{'Symbol':<14} {'Dir':>6} {'Score':>7} {'Entry':>8} {'Exit':>8} {'Return':>8}")
print("-" * 58)
for p in all_pos:
    d   = "LONG " if p["direction"] == 1 else "SHORT"
    ep  = f"{p.get('entry_price',0):.1f}" if p.get("entry_price") else "—"
    xp  = f"{p.get('exit_price',0):.1f}"  if p.get("exit_price") else "—"
    ret = f"{p['final_return_pct']:+.2f}%" if p.get("final_return_pct") is not None else "—"
    mk  = "✅" if p["status"] == "SETTLED_WIN" else "❌"
    print(f"{p['symbol']:<14} {d:>6} {p['score']:>7.4f} {ep:>8} {xp:>8} {ret:>8} {mk}")

# Also save updated report
from pathlib import Path
REPORTS_DIR = BASE / "reports" / "live"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
report_path = REPORTS_DIR / f"signal_ledger_{session}.md"
report_path.write_text(ledger.markdown_report(session))
print(f"\nReport saved: {report_path.relative_to(BASE)}")
