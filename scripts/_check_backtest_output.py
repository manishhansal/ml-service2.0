"""Check backtest output files."""
import pandas as pd
import json
from pathlib import Path

ROOT = Path(__file__).parent.parent

sig = pd.read_csv(ROOT / "artifacts/backtest_7d/backtest_signals.csv")
print("backtest_signals.csv shape:", sig.shape)
print("net_pnl mean:", round(float(sig["net_pnl"].mean()), 4), "%")
print("net_pnl positive rate:", round(float((sig["net_pnl"] > 0).mean()) * 100, 1), "%")
print("win_rate:", round(float((sig["outcome"] == "WIN").mean()) * 100, 1), "%")

print()
with open(ROOT / "artifacts/backtest_7d/backtest_report.json") as f:
    report = json.load(f)
print("mean_net_pnl:", report["performance"].get("mean_net_pnl"), "%")
print("win_rate:", report["performance"].get("win_rate"))
print("sharpe:", report["performance"].get("sharpe"))
print("verdict:", report.get("acceptance_verdict"))

print()
opp = pd.read_csv(ROOT / "artifacts/backtest_7d/profitable_opportunities.csv")
print("Opportunities shape:", opp.shape)
by_dir = opp.groupby("direction")["net_return_7d_pct"].agg(["mean","count"])
print("By direction:\n", by_dir)

print()
cov = report.get("coverage", {})
print("Coverage:")
for k, v in cov.items():
    print(f"  {k}: {v}")

print()
decay = report.get("signal_decay", {})
print("Signal decay (mean net % per day):")
for d in range(1, 8):
    day_info = decay.get(f"day_{d}", {})
    print(f"  T+{d}: mean={day_info.get('mean_net_pct', 'N/A')}%  win_rate={day_info.get('win_rate', 'N/A')}")

print()
bl = json.load(open(ROOT / "artifacts/backtest_7d/baseline_comparison.json"))
print("Baseline comparison:")
for k, v in bl.items():
    if isinstance(v, dict) and "win_rate" in v:
        print(f"  {k}: win_rate={v['win_rate']}  mean_net={v.get('mean_net_pct','?')}%")
    elif isinstance(v, dict):
        for kk, vv in v.items():
            if isinstance(vv, dict) and "win_rate" in vv:
                print(f"  {kk}: win_rate={vv.get('win_rate','?')}  mean_net={vv.get('mean_net_pct','?')}%")

print()
sym_perf = pd.read_csv(ROOT / "artifacts/backtest_7d/performance_by_symbol.csv")
print("Top 5 symbols by mean_net_pct:")
print(sym_perf.head())
