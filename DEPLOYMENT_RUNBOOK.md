# DEPLOYMENT RUNBOOK
**AlphaForge ml-service2.0 — Shadow → Production Path**
**Created:** 2026-09-28 (post G12 approval) | **Last updated:** 2026-09-30
**Model:** `expanded_lgbm` v`1.0.0-20260928053134956099` | Stage: **SHADOW**

---

## CURRENT STATE

```
CHALLENGER (complete) → SHADOW (active, 2026-09-28) → PRODUCTION (target: 2026-10-15)
```

The model is **live in shadow mode**: it scores all 218 NSE F&O symbols every 5 minutes
during market hours and logs paper P&L. No real capital is deployed.

**Live session record (as of 2026-09-29 close):**

| Date | NIFTY | SHORT P&L | WIN RATE | Status |
|------|-------|-----------|----------|--------|
| Sep 28 | −1.52% | +0.655% net | 80% (16/20) | ✓ |
| Sep 29 | −0.42% | +0.945% net | 80% (16/20) | ✓ |
| **2-day avg** | **−0.97%** | **+0.800% net** | **80% (32/40)** | p<0.001 |

---

## ⚠️  CRITICAL OPERATIONS: UPSTOX TOKEN ROTATION

> **Must read before any data operation.** Token expiry caused 10 days of stale data
> (Sep 19–29). This section documents the complete recovery procedure.

### Token Lifecycle

Upstox OAuth tokens expire **daily at 22:00 UTC** (03:30 IST next day).
They must be manually refreshed every day via OAuth — there is no automatic rotation.

```
Token valid window: ~04:00 UTC (login IST morning) → 22:00 UTC same day
Data-service EOD catchup runs at: 19:00 IST = 13:30 UTC (within valid window ✓)
```

### Daily Token Rotation Checklist

```bash
# Step 1: Get a fresh token
# → Log in at https://api.upstox.com/v2/login
# → Copy the new access_token from the OAuth callback URL

# Step 2: Update the .env file in data-service2.0
nano /Users/manishkumar/Desktop/data-service2.0/.env
# Edit line: UPSTOX_ACCESS_TOKEN=<new_token>

# Step 3: CRITICAL — force-recreate containers (NOT docker restart!)
# docker restart keeps the OLD env vars baked into the container.
# docker compose up --force-recreate re-reads .env.
cd /Users/manishkumar/Desktop/data-service2.0
docker compose up -d --force-recreate api worker scheduler

# Step 4: Verify new token loaded in container (tokens should match)
docker inspect data-service-api | grep "UPSTOX_ACCESS_TOKEN" | cut -c1-100

# Step 5: Verify token is valid and not expired
make -C /Users/manishkumar/Desktop/ml-service2.0 check-token

# Step 6: If running after EOD catchup already passed (19:00 IST), manually backfill
make -C /Users/manishkumar/Desktop/ml-service2.0 fast-backfill-today
```

### ⚠️  Critical: `docker restart` vs `docker compose up --force-recreate`

| Command | Env vars | Use when |
|---------|----------|----------|
| `docker restart <container>` | **Keeps old baked-in env** | Quick restart (no .env changes) |
| `docker compose up -d --force-recreate` | **Re-reads .env** | After any .env change (token, config) |

**Always use `--force-recreate` after updating `.env`.** Using `docker restart` after a token
update will leave the container running with the expired token.

### Token Expiry Symptoms

- `make check-token` reports `EXPIRED`
- Data freshness check shows last bars at the expiry date
- `docker logs data-service-worker` shows `upstox_historical_ohlcv_v3_request` returning 0 candles
- `ohlcv_catchup` loop loads instruments every 60s but never fetches bars

### Data Recovery After Token Expiry

```bash
# 1. Refresh token + recreate containers (see checklist above)

# 2. Fast-backfill recent bars (5d lookback, 8 parallel workers)
make fast-backfill

# 3. For a specific date range only (2d lookback, 10 workers)
make fast-backfill-today

# 4. Verify freshness
cd /Users/manishkumar/Desktop/ml-service2.0
PYTHONPATH=. python3 scripts/fast_backfill_recent.py --skip-backfill --days 1 --workers 1
# Expect: "Sep 29 IST bar: 62 symbols, Sep 28 IST bar: 150 symbols, Stale: 6"
```

### Known Chronically Stale Symbols (6)

These symbols have no recent coverage from either Angel One or Upstox CDN and are
**excluded from live signals** by the feature factory:

| Symbol | Last Date | Reason |
|--------|-----------|--------|
| LTM | 2022-10-24 | New listing, no CDN coverage yet |
| MIDCPNIFTY | 2022-10-24 | New listing |
| PGEL | 2022-10-24 | New listing |
| TMPV | 2022-10-24 | New listing |
| GVT&D | 2026-09-23 | No Angel One / Upstox mapping |
| M&M | 2026-09-23 | Instrument key mismatch |

---

---

## PHASE 1 — SHADOW PERIOD (Oct 1–14, 2026)

### Daily Operations (automated)

| Time IST | Action | Script/Command | Expected Output |
|----------|--------|---------------|----------------|
| 08:45 | Ingest previous day's close bars | `make ingest-universe` | New parquet rows per symbol |
| 09:15 | Session start — scoring begins | `scripts/autorun_till_close.py` | 218 symbols scored every 5 min |
| 15:35 | Session close — P&L snapshot | auto via autorun script | session_summary.json updated |
| 16:00 | Drift check | `curl http://localhost:8100/drift/check` | drift_severity ≠ HIGH |
| 16:15 | Shadow P&L report | `scripts/track_live_pnl.py` | SHORT/LONG/net P&L |

### Weekly Operations (every Friday)

```bash
# 1. Rebuild IC term structure
PYTHONPATH=. python3 scripts/run_g6_robustness_test.py

# 2. Regime alpha matrix update
PYTHONPATH=. python3 -c "from src.analytics.regime_alpha_matrix import RegimeAlphaMatrix; ..."

# 3. Alpha decay check
PYTHONPATH=. python3 -c "from src.analytics.alpha_decay import AlphaDecayDetector; ..."

# 4. Full test suite
python3 -m pytest tests/ --no-cov -q
```

### Shadow Health Checks

The model remains in SHADOW if **all** of the following hold:
- [ ] 3-day rolling IC > 0 (not decaying)
- [ ] No consecutive 5+ loss days in paper book
- [ ] Drift monitor severity ≠ HIGH
- [ ] Paper book cumulative drawdown < 5%
- [ ] Data service healthy (`curl http://localhost:8200/health`)

**Auto-demotion triggers** (system reverts to `stage_a_1d` baseline):
```
ic_3day_rolling_below_0
consecutive_loss_days_gt_5
drift_severity_high
drawdown_gt_5pct
```

---

## PHASE 2 — FORWARD PAPER RESOLUTION (Sep 30, 2026)

### Current Status (as of 2026-09-30 01:00 UTC)

G10 resolve ran. All 85 resolved outcomes are **partial** (only 2 market days of data):

| Metric | Value | Gate | Result |
|--------|-------|------|--------|
| N outcomes | 85 | ≥ 50 | PASS |
| Win rate | 50.6% (43/85) | > 50% | MARGINAL |
| Net expectancy | −0.00423 | > 0 | FAIL |
| Max drawdown | −51.93% | > −20% | FAIL |
| Status | **INSUFFICIENT_EVIDENCE** | — | PRELIMINARY REJECT |

**Why PRELIMINARY REJECT is expected here:**
All 85 resolutions are partial — signals were generated Sep 24 IST with a 5-day horizon.
The ideal exit is Oct 1 IST open. We only have data through Sep 28 IST (2 market days).
The −51.93% max drawdown is dominated by **POLICYBZR LONG −32.46%** — a genuine
corporate event crash on Sep 24 IST (open 1697→close 1207, volume 27M vs normal 800K).

**Full resolution timeline:**

| Date | Action | Expected |
|------|--------|---------|
| Sep 30 | Market closes, Sep 30 bars settle | T+3 available for most signals |
| Oct 1 open | T+1+5 bar available for Sep 24 signals | Full resolution for signals.jsonl (65) |
| Oct 2-3 open | Full resolution for signals_v2.jsonl (153) | All 218 signals fully resolved |
| Oct 3 EOD | **Re-run G10+G11 with complete data** | Final promotion verdict |

### Step-by-step (Sep 30 EOD + Oct 1-3)

```bash
cd /Users/manishkumar/Desktop/ml-service2.0

# Daily after market close: refresh data + re-resolve
make check-token           # Verify Upstox token still valid
make fast-backfill-today   # Pull today's bars into parquets
make forward-paper-resolve # Resolve newly-due signals (incremental, safe to re-run)
make signal-promote        # Re-run promotion engine with updated outcomes

# Oct 3 (all signals fully resolved): definitive G10+G11 result
cat reports/signal_promotion_preliminary.json | python3 -m json.tool | grep -E '"decision"|"win_rate"|"net_expectancy"'
```

### Success Criteria (final, Oct 3)

| Gate | Criterion | Notes |
|------|-----------|-------|
| G10 | n_outcomes ≥ 50 | 218 outcomes expected ✓ |
| G10 | mean_net_return > 0 | IC=0.376 implies ~+0.4% |
| G11 | G2_NET_EXPECTANCY PASS | net expectancy > 0 |
| G11 | G6_DRAWDOWN PASS | max_dd > −20% |
| G11 | decision = PROMOTE | Required for phase 3 |

### If G10+G11 FAIL (final verdict)

```bash
# 1. Check outcome distribution
python3 -c "
import json; outcomes = [json.loads(l) for l in open('artifacts/forward_paper/outcomes.jsonl')]
import numpy as np; rets = [o['net_return'] for o in outcomes]
print('win', sum(1 for r in rets if r>0), '/', len(rets))
print('mean', np.mean(rets))
for o in sorted(outcomes, key=lambda x: x['net_return'])[:5]:
    print(o['symbol'], o['net_return'], o.get('partial_resolution'))
"
# 2. If negative expectancy is concentrated in LONG signals: enable LONG dimmer
# 3. If data issues: fix parquets, delete outcomes.jsonl, re-run resolve
# 4. If genuine alpha decay: remain in SHADOW, retrain with Sep-Oct data
```

---

## PHASE 3 — SHADOW → PRODUCTION (Oct 15, 2026)

### Prerequisites checklist

```
□ G10: forward paper ≥ 50 outcomes with positive mean net  [Oct 3 — all 218 fully resolved]
□ G11: SignalPromotionEngine decision = PROMOTE             [Oct 3]
□ 14-day shadow monitoring: no demotion triggers fired      [Oct 1-14]
□ Shadow Sharpe ≥ 0.5 (14-day paper P&L)                  [Oct 14]
□ No drift alerts during shadow period                      [Oct 1-14]
□ Sep 28-30 bars ingested and verified                     [✓ Done 2026-09-30]
□ POLICYBZR event investigated (corporate crash Sep 24)    [Oct 1-3]
□ TATAMOTORS DQ-001 instrument fix applied                 [Oct 1-3]
□ Beta-neutral LONG overlay tested                         [Oct 1-3]
□ LONG dimmer for MILD_BEAR regime added                   [Oct 1-3]
□ fs-4.0.0 model retrain (67 features incl. Group F)      [Oct 1-7]
□ Final human approval for PRODUCTION deployment            [Oct 15]
```

### Promotion script

```bash
# When all prerequisites are met:
PYTHONPATH=. python3 scripts/promote_to_production.py \
  --approver "head_of_quant" \
  --note "14-day shadow clean, G10+G11 pass, beta overlay live" \
  --confirm-production
```

*(This script to be created when promotion is due.)*

### Production execution model

| Item | Value |
|------|-------|
| Instrument | NSE F&O Futures (not equity) |
| Cost assumption | 8.5bps round-trip |
| Portfolio type | Concentrated 5% long-short |
| n positions | 11 LONG + 11 SHORT = 22 total |
| Rebalancing | Weekly (every 5 bars) |
| Position sizing | Equal weight within each leg |
| Max gross exposure | 100% of allocated AUM |
| Stop-loss | DrawdownManager HALTED at −10% |
| Execution | DMA limit orders (not market orders) |
| Universe | 218 NSE F&O symbols (TATAMOTORS excluded until DQ-001 fixed) |

### Capital scaling

| Phase | AUM | Expected Net | Notes |
|-------|-----|-------------|-------|
| Shadow | ₹0 (paper) | +0.655%/signal → ~+34% ann. | IC=0.376, SHORT only |
| Production (pilot) | ₹1 Cr | ~+12-16% ann. at 8.5bps | Concentrated 5% |
| Production (full) | ₹10 Cr | ~+10-14% ann. | Capacity check needed |

---

## PHASE 4 — CONTINUOUS IMPROVEMENT

### Sep 30+ Improvements (post-resolution)

```bash
# Retrain with Sep 28-30 data included
PYTHONPATH=. python3 scripts/run_lgbm_conc_backtest.py

# Run multi-horizon analysis to find optimal holding period
PYTHONPATH=. python3 -c "
from src.labels.multi_horizon import MultiHorizonLabelFactory
factory = MultiHorizonLabelFactory(horizons=[1,3,5,10,21])
# ... build for full universe
"
```

### Oct 1-3: Risk Improvements

**Beta-neutral LONG overlay:**
```python
# Target: compute 60-day rolling beta for each LONG position
# Hedge: NIFTY futures SHORT = sum(LONG_weight × beta_i)
# Expected: eliminates -2.14% LONG P&L from market beta on bear days
# Turns today's portfolio from +0.01% → ~+0.35% on Sep 28 equivalent
```

**Sector-regime filter (v4.0 features):**
```python
# Add to ExpandedFeatureFactory:
# sector_defensive_flag × nifty_drawdown_3d
# Expected: eliminates DRREDDY/ASIANPAINT/MARUTI SHORT misses
# Impact: SHORT win rate 80% → ~90%+ on high-correlation bear days
```

### Retraining schedule

| Trigger | Action |
|---------|--------|
| Weekly IC drop > 0.05 | Alert; no retrain yet |
| Monthly | Retrain on rolling 5yr window |
| IC 3-day rolling < 0 | Emergency retrain |
| Regime shift detected | Retrain with regime-weighted labels |
| Market microstructure change | Full pipeline review |

---

## APPENDIX A — KEY COMMANDS

### Daily Operations

```bash
# Token check + freshness report
make check-token

# Fast backfill after token refresh (last 5 days, 8 workers)
make fast-backfill

# Fast backfill for today only (last 2 days, 10 workers, urgent recovery)
make fast-backfill-today

# Full 218-symbol refresh (slower, more thorough)
make refresh-data
```

### Data-service Container Management

```bash
# Check all container status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# ⚠️  AFTER .env changes: MUST use force-recreate, NOT docker restart
cd /Users/manishkumar/Desktop/data-service2.0
docker compose up -d --force-recreate api worker scheduler

# Check which env vars are loaded in a container
docker inspect data-service-api | grep -E '"CATCHUP_EOD|CATCHUP_CONCURRENCY|UPSTOX_ACCESS_TOKEN"'

# View data-service worker activity
docker logs data-service-worker --tail 50 | grep -E "candle|catchup|error"

# Check if EOD catchup job is registered
docker logs data-service-scheduler --tail 20 | grep -E "eod|catchup|job"
```

### Market Session

```bash
# Shadow scoring session (interactive, runs till 15:30 IST)
PYTHONPATH=. python3 scripts/autorun_till_close.py

# Reversal scan
make reversal-scan

# Threshold sweep (Phil integration)
make threshold-sweep
```

### Forward Paper / Promotion Gates

```bash
# G10: Resolve forward paper signals
make forward-paper-resolve
# or:
PYTHONPATH=. .venv/bin/python scripts/resolve_forward_paper.py

# G11: Run signal promotion engine
make signal-promote
# or:
PYTHONPATH=. .venv/bin/python scripts/run_signal_promotion.py

# View promotion result
cat reports/signal_promotion_preliminary.json | python3 -m json.tool | grep -E '"decision"|"win_rate"|"net_expectancy"'
```

### Health Checks

```bash
# ml-service
curl http://localhost:8100/health

# data-service API
curl -H "X-API-KEY: dev-key-local-1" http://localhost:8200/v1/india/historical?symbol=NIFTY&interval=1d | python3 -m json.tool | tail -10

# Full test suite
python3 -m pytest tests/ --no-cov -q
```

### Model Introspection

```bash
# G6 robustness check
PYTHONPATH=. python3 scripts/run_g6_robustness_test.py

# Data freshness by IST date
python3 scripts/fast_backfill_recent.py --skip-backfill --skip-api --days 1 --workers 1

# Forward paper outcome analysis
python3 -c "
import json, numpy as np
from pathlib import Path
outcomes = [json.loads(l) for l in Path('artifacts/forward_paper/outcomes.jsonl').read_text().splitlines() if l.strip()]
rets = [o['net_return'] for o in outcomes]
print('N:', len(rets), '| Win:', sum(1 for r in rets if r>0), '| Mean:', round(np.mean(rets)*100,3), '%')
for o in sorted(outcomes, key=lambda x: x['net_return'])[:3]:
    print(' WORST:', o['symbol'], round(o['net_return']*100,2), '%', 'partial='+str(o.get('partial_resolution',False)))
"
```

---

## APPENDIX B — DEMOTION PROCEDURE

If the model needs to be demoted from SHADOW back to CHALLENGER:

```bash
# 1. Update metadata
python3 -c "
import json
from pathlib import Path
p = Path('artifacts/expanded_lgbm/1.0.0-20260928053134956099/metadata.json')
m = json.loads(p.read_text())
m['stage'] = 'challenger'
p.write_text(json.dumps(m, indent=2))
print('Demoted to challenger')
"

# 2. Revert deployment mode
# Edit .env: DEPLOYMENT_MODE=research (or paper)

# 3. Restart service
docker restart ml-service2-api

# 4. Log demotion
echo '{"event":"DEMOTED_TO_CHALLENGER","ts":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'","reason":"..."}' \
  >> artifacts/promotion_audit.jsonl
```

---

*Runbook created: 2026-09-28 | Last updated: 2026-09-30 01:05 UTC*
*Owner: AlphaForge Quant Engineering*
*Review: Oct 3 (G10+G11 final), Oct 15 (production promotion decision)*
