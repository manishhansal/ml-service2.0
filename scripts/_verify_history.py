# -*- coding: utf-8 -*-
"""Verify History tab fixes are working end-to-end."""
import json
import urllib.request

env = {}
for line in open(".env").readlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

API_KEY  = env.get("ML_SERVICE_API_KEY", env.get("API_KEY", ""))
BASE_URL = "http://localhost:8100"
SEP      = "=" * 65


def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}", headers={"X-API-KEY": API_KEY})
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


# ── 1. History endpoint ───────────────────────────────────────────────────────
print(SEP)
print("  GET /v2/signals/history  (Oct-1)")
print(SEP)
h    = get("/v2/signals/history?date=2026-10-01")
s    = h.get("stats", {})
recs = h.get("records", [])

print(f"  n_total    : {s.get('n_total')}   (was 218)")
print(f"  n_won      : {s.get('n_won')}    (was 0)")
print(f"  n_lost     : {s.get('n_lost')}   (was 1)")
print(f"  n_open     : {s.get('n_open')}   (was 217)")
print(f"  n_resolved : {s.get('n_resolved')}")
print(f"  win_rate   : {s.get('win_rate')}%")
print(f"  mean_net   : {s.get('mean_net')}%")
print(f"  records[]  : {h.get('n')} entries  (limit=500)")

print("\n  First 5 records (should be won/lost, not open):")
for rec in recs[:5]:
    net_str = f"{rec.get('net_pct'):+.3f}%" if rec.get("net_pct") is not None else "—"
    print(f"    {rec['symbol']:14s}  status={rec.get('status','?'):6s}  "
          f"dir={rec.get('direction')}  net={net_str}")

won  = s.get("n_won", 0)
lost = s.get("n_lost", 0)
pass1 = won > 0 or lost > 0
print(f"\n  PASS resolved>0: {pass1}  ({won} won, {lost} lost)")

# ── 2. Latest — market closed ─────────────────────────────────────────────────
print()
print(SEP)
print("  GET /v2/signals/latest  (market closed)")
print(SEP)
lt = get("/v2/signals/latest")
print(f"  market_open : {lt.get('market_open')}")
print(f"  n_long      : {lt.get('n_long')}   (should be 0)")
print(f"  n_short     : {lt.get('n_short')}  (should be 0)")
print(f"  signals[]   : {len(lt.get('signals', []))} entries")
pass2 = lt.get("n_long") == 0 and lt.get("n_short") == 0
print(f"\n  PASS closed=no signals: {pass2}")

# ── 3. Movers ──────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  GET /v2/signals/movers?limit=3")
print(SEP)
mv = get("/v2/signals/movers?limit=3")
nifty = mv.get("nifty")
if nifty:
    print(f"  NIFTY  : {nifty.get('ltp')} ({nifty.get('changePct', 0):+.2f}%)")
print(f"  stale  : {mv.get('stale')}")
print(f"  gainers: {[g['symbol'] for g in mv.get('gainers', [])]}")
print(f"  losers : {[lo['symbol'] for lo in mv.get('losers', [])]}")
pass3 = len(mv.get("gainers", [])) > 0
print(f"\n  PASS movers has data: {pass3}")

# ── Summary ────────────────────────────────────────────────────────────────────
print()
print(SEP)
passed = sum([pass1, pass2, pass3])
print(f"  {passed}/3 checks passed")
if passed == 3:
    print("  ALL GOOD — refresh AlphaForge History tab")
