"""Check portfolio backtest report."""
import json
from pathlib import Path

r = json.load(open(Path(__file__).parent.parent / "artifacts/portfolio_backtest/portfolio_backtest_report.json"))
mb = r["mode_b_portfolio"]
ma = r["mode_a_isolated"]
comp = r["comparison"]

print("=== MODE B (portfolio-constrained) ===")
for k in ["n_trades","win_rate","mean_net_pnl","total_net_pnl","total_return_pct","sharpe","max_drawdown","profit_factor"]:
    print(f"  {k}: {mb.get(k)}")
print()
print("=== MODE A (isolated) ===")
for k in ["n_trades","win_rate","mean_net_pnl","total_net_pnl","total_return_pct","sharpe"]:
    print(f"  {k}: {ma.get(k)}")
print()
print("=== COMPARISON ===")
for k, v in comp.items():
    if k not in ("mode_a_isolated", "mode_b_portfolio"):
        print(f"  {k}: {v}")
print()
initial = mb.get("initial_capital", 1e6)
net = mb.get("total_net_pnl", 0)
ret = mb.get("total_return_pct", 0)
print(f"Reconciliation: initial={initial:,.0f} + net_pnl={net:,.0f} = {initial+net:,.0f}")
print(f"total_return implies final_equity={initial*(1+ret/100):,.0f}")
