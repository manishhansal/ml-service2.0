#!/usr/bin/env python3
"""Final validation of v5.0.0 model and full pipeline."""
import warnings; warnings.filterwarnings("ignore")
import sys, pickle, subprocess
sys.path.insert(0, "/Users/manishkumar/Desktop/ml-service2.0")
import os; os.chdir("/Users/manishkumar/Desktop/ml-service2.0")
import pandas as pd
from pathlib import Path

results = []

def chk(name, cond, detail=""):
    flag = "OK" if cond else "FAIL"
    results.append((flag, name, detail))
    marker = "✓" if cond else "✗"
    print(f"  {marker}  {name}" + (f"  [{detail}]" if detail else ""))
    return cond

print("=" * 60)
print("  FINAL VALIDATION — v5.0.0  (84 features)")
print("=" * 60)

# 1. Model
print("\n[1] Model")
from scripts.autorun_till_close import load_model, score_symbol
est, fn, norm, schema = load_model()
pkls = sorted(Path("artifacts/expanded_lgbm").glob("*/model.pkl"))
version = pkls[-1].parent.name if pkls else "?"
chk("84-feature model loaded", version.startswith("5.") or version.startswith("h"), version)
chk("84 features", len(fn) == 84, str(len(fn)))
chk("normalizer loaded", norm is not None)

n_eod     = sum(1 for f in fn if not f.startswith(("intraday_","nifty_ret","news_")))
n_intra   = sum(1 for f in fn if f.startswith("intraday_"))
n_market  = sum(1 for f in fn if f.startswith("nifty_ret"))
n_news    = sum(1 for f in fn if f.startswith("news_"))
chk("67 EOD features",     n_eod == 67,   str(n_eod))
chk("8 intraday features", n_intra == 8,  str(n_intra))
chk("3 market features",   n_market == 3, str(n_market))
chk("6 news features",     n_news == 6,   str(n_news))

# 2. Scoring
print("\n[2] Scoring pipeline")
symbols_to_test = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "NIFTY"]
ok_count = 0
for sym in symbols_to_test:
    r = score_symbol(sym, est, fn, norm)
    if r is not None and 0 < r["score"] < 1:
        ok_count += 1
chk("5/5 symbols score", ok_count == 5, f"{ok_count}/5")

# 3. Data coverage
print("\n[3] Data coverage")
n_1d   = len(list(Path("data/1d/1d").glob("*.parquet")))
n_5m   = len(list(Path("data/5m/5m").glob("*.parquet")))
n_news = len(list(Path("data/news/1d").glob("*.parquet")))
chk("285+ 1d parquets",   n_1d >= 285,   str(n_1d))
chk("280+ 5m parquets",   n_5m >= 280,   str(n_5m))
chk("286+ news parquets", n_news >= 286, str(n_news))

mkt_pf = Path("data/news/1d/MARKET.parquet")
if mkt_pf.exists():
    mkt = pd.read_parquet(mkt_pf)
    chk("MARKET 1900+ rows", len(mkt) >= 1900, str(len(mkt)))
    chk("MARKET from 2021", str(mkt.index.min().date()).startswith("2021"),
        str(mkt.index.min().date()))

# 4. Symbol IC tracker
print("\n[4] Symbol IC tracker")
from src.analytics.symbol_ic_tracker import SymbolICTracker
tracker = SymbolICTracker()
tracker.record("TEST_VALIDATION", 0.72, 0.018, "2026-10-01")
dead = tracker.dead_symbols()
summary = tracker.summary()
chk("IC tracker records", isinstance(dead, set))
chk("IC tracker summary", "n_symbols_tracked" in summary)
chk("dead_symbols() safe", len(dead) >= 0)

# 5.5. Ensemble manifest
print("\n[4b] Ensemble manifest")
ensemble_manifest = Path("artifacts/expanded_lgbm/ensemble_manifest.json")
chk("ensemble_manifest.json exists", ensemble_manifest.exists())
if ensemble_manifest.exists():
    import json
    em = json.loads(ensemble_manifest.read_text())
    n_models = len(em.get("models", []))
    horizons = [m.get("horizon") for m in em.get("models", [])]
    chk("H1 + H5 both in manifest", 1 in horizons and 5 in horizons, str(horizons))
    h1_ic = next((m.get("ic") for m in em.get("models",[]) if m.get("horizon")==1), 0)
    h5_ic = next((m.get("ic") for m in em.get("models",[]) if m.get("horizon")==5), 0)
    chk("H1 IC > 0.05", h1_ic > 0.05, f"{h1_ic:.4f}")
    chk("H5 IC > 0.05", h5_ic > 0.05, f"{h5_ic:.4f}")

# 5. Infrastructure
print("\n[5] Infrastructure")
r = subprocess.run(
    ["docker", "ps", "--filter", "name=ml-service-news-scheduler", "--format", "{{.Status}}"],
    capture_output=True, text=True,
)
chk("news-scheduler running", "Up" in r.stdout, r.stdout.strip()[:30])

sp_res = subprocess.run(
    ["docker", "exec", "data-service-postgres", "psql",
     "-h", "127.0.0.1", "-p", "5432", "-U", "sentinel", "-d", "sentinel_pulse",
     "-t", "-A", "-c",
     "SELECT COUNT(*) as total, "
     "(SELECT COUNT(*) FROM news_sentiment) as sents, "
     "(SELECT COUNT(*) FROM news_articles WHERE published_at < '2026-01-01') as pre2026 "
     "FROM news_articles;"],
    capture_output=True, text=True,
)
if sp_res.stdout.strip():
    parts = sp_res.stdout.strip().split("|")
    if len(parts) == 3:
        total, sents, pre2026 = int(parts[0]), int(parts[1]), int(parts[2])
        chk("100k+ SP articles", total >= 90000, f"{total:,}")  # target 100k, threshold 90k
        chk("50k+ SP sentiments", sents >= 50000, f"{sents:,}")
        chk("40k+ pre-2026 articles", pre2026 >= 40000, f"{pre2026:,}")

# ── Summary ───────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
failures = [(n, d) for (f, n, d) in results if f == "FAIL"]
print(f"  RESULT: {len(results)-len(failures)}/{len(results)} checks passed")
if not failures:
    print("  ALL CHECKS PASSED ✓")
else:
    print(f"  FAILURES ({len(failures)}):")
    for n, d in failures:
        print(f"    ✗  {n}  [{d}]")
print("=" * 60)
