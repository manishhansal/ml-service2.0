# -*- coding: utf-8 -*-
"""Smoke test: /signals/latest, /signals/movers, WebSocket push."""
import json
import sys
import urllib.request

env = {}
for line in open(".env").readlines():
    if "=" in line and not line.strip().startswith("#"):
        k, _, v = line.partition("=")
        env[k.strip()] = v.strip()

API_KEY = env.get("ML_SERVICE_API_KEY", env.get("API_KEY", ""))
BASE    = "http://localhost:8100"
HDR     = {"X-API-KEY": API_KEY}
SEP     = "-" * 60


def get(path):
    req = urllib.request.Request(f"{BASE}{path}", headers=HDR)
    with urllib.request.urlopen(req, timeout=8) as r:
        return json.loads(r.read().decode())


# ── /signals/latest ───────────────────────────────────────────────────────────
print(SEP)
print("  GET /v2/signals/latest")
print(SEP)
d = get("/v2/signals/latest")
n_long  = d.get("n_long", "?")
n_short = d.get("n_short", "?")
n_neut  = d.get("n_neutral", "?")
age     = d.get("age_seconds")
stale   = d.get("stale")

print(f"  n_long    : {n_long}   (should be ~40, NOT 241)")
print(f"  n_short   : {n_short}   (should be ~35, NOT 44)")
print(f"  n_neutral : {n_neut}  (filtered, not sent to UI)")
print(f"  age_secs  : {age:.1f}s" if age else "  age_secs  : n/a")
print(f"  stale     : {stale}")
print(f"  signals[] : {len(d.get('signals',[]))} entries")

ok_long  = isinstance(n_long, int) and n_long < 100
ok_short = isinstance(n_short, int) and n_short < 100
ok_no_neutral_leaked = all(s.get("direction") in (1,-1) for s in d.get("signals",[]))
print(f"\n  PASS long_count<100:    {ok_long}")
print(f"  PASS short_count<100:   {ok_short}")
print(f"  PASS no neutral leaked: {ok_no_neutral_leaked}")

# ── /signals/movers ────────────────────────────────────────────────────────────
print()
print(SEP)
print("  GET /v2/signals/movers?limit=5")
print(SEP)
m = get("/v2/signals/movers?limit=5")
gainers = m.get("gainers", [])
losers  = m.get("losers", [])
nifty   = m.get("nifty")
m_age   = m.get("age_seconds")
m_stale = m.get("stale")

print(f"  stale     : {m_stale}")
print(f"  age_secs  : {m_age:.1f}s" if m_age else "  age_secs  : n/a")
if nifty:
    print(f"  NIFTY     : {nifty.get('ltp')} ({nifty.get('changePct',0):+.2f}%)")
print(f"\n  Top 5 gainers:")
for g in gainers:
    print(f"    {g['symbol']:14s}  ltp={g['ltp']:9.2f}  chg={g['changePct']:+.2f}%")
print(f"\n  Top 5 losers:")
for l in losers:
    print(f"    {l['symbol']:14s}  ltp={l['ltp']:9.2f}  chg={l['changePct']:+.2f}%")

ok_movers = len(gainers) > 0 and len(losers) > 0
print(f"\n  PASS movers returned data: {ok_movers}")

# ── /health ────────────────────────────────────────────────────────────────────
print()
print(SEP)
print("  GET /health")
print(SEP)
h = get("/health")
print(f"  status: {h.get('status')}")

# ── /signals/latest age ────────────────────────────────────────────────────────
print()
print(SEP)
print("  Stale threshold check (should be 2min now)")
print(SEP)
age_ok = age is not None and age < 120
print(f"  age={age:.1f}s  stale={stale}  threshold=120s")
print(f"  PASS within 2-min threshold: {age_ok}")

print()
print("=" * 60)
passed = sum([ok_long, ok_short, ok_no_neutral_leaked, ok_movers, h.get("status")=="healthy", age_ok])
print(f"  {passed}/6 checks passed")
if passed < 6:
    sys.exit(1)
