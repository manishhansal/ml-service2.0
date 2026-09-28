# DEPLOYMENT RUNBOOK
**AlphaForge ml-service2.0 — Shadow → Production Path**
**Created:** 2026-09-28 (post G12 approval)
**Model:** `expanded_lgbm` v`1.0.0-20260928053134956099` | Stage: **SHADOW**

---

## CURRENT STATE

```
CHALLENGER (complete) → SHADOW (active, 2026-09-28) → PRODUCTION (target: 2026-10-15)
```

The model is **live in shadow mode**: it scores all 218 NSE F&O symbols every 5 minutes
during market hours and logs paper P&L. No real capital is deployed.

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

### Step-by-step

```bash
# 09:30 IST — Sep 30 market opens; prices available for resolution
cd /Users/manishkumar/Desktop/ml-service2.0

# Step 1: Ingest Sep 28-29 bars (needed for exit prices)
PYTHONPATH=. python3 scripts/fast_ingest.py
# Expected: ~436 new bars (218 symbols × 2 days)

# Step 2: Resolve all 218 forward paper signals
PYTHONPATH=. python3 scripts/resolve_forward_paper.py
# Expected: outcomes.jsonl grows from 13 → 218 entries
# Exit price = open[Sep 30] for 5-bar horizon signals

# Step 3: Run SignalPromotionEngine (G11)
PYTHONPATH=. python3 scripts/run_signal_promotion.py
# Expected: PROMOTE decision if win_rate > 50% and mean_net > 0
# G10 gate: n_outcomes >= 50 ✓ (218 outcomes)
# G11 gate: engine decision = PROMOTE

# Step 4: Check results
cat reports/signal_promotion_preliminary.json | python3 -m json.tool | grep "decision\|win_rate\|mean_net"
```

### Success Criteria

| Gate | Criterion | Required |
|------|-----------|---------|
| G10 | n_outcomes ≥ 50 | 218 outcomes expected ✓ |
| G10 | mean_net_return > 0 | IC=0.376 implies ~+0.4% |
| G11 | G2_NET_EXPECTANCY PASS | mean net > 0 |
| G11 | G1_TRADE_COUNT PASS | n ≥ 50 ✓ |
| G11 | Overall decision = PROMOTE | Required for production |

### If G10+G11 FAIL

1. Investigate: `python3 -c "import json; [print(o['symbol'], o['net_return']) for o in [json.loads(l) for l in open('artifacts/forward_paper/outcomes.jsonl')]]"`
2. Check for data quality issues (TATAMOTORS, wrong exit prices)
3. If signal is genuinely negative: remain in SHADOW, schedule retraining with updated data
4. If data issue: fix data, re-resolve, re-run promotion

---

## PHASE 3 — SHADOW → PRODUCTION (Oct 15, 2026)

### Prerequisites checklist

```
□ G10: forward paper ≥ 50 outcomes with positive mean net  [Sep 30]
□ G11: SignalPromotionEngine decision = PROMOTE             [Sep 30]
□ 14-day shadow monitoring: no demotion triggers fired      [Oct 1-14]
□ Shadow Sharpe ≥ 0.5 (14-day paper P&L)                  [Oct 14]
□ No drift alerts during shadow period                      [Oct 1-14]
□ Sep 28+29+30 bars ingested and verified                  [Oct 1]
□ TATAMOTORS DQ-001 instrument fix applied                 [Oct 1-3]
□ Beta-neutral LONG overlay tested                         [Oct 1-3]
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

```bash
# Health checks
curl http://localhost:8100/health
curl http://localhost:8200/health

# Shadow scoring (manual)
PYTHONPATH=. python3 scripts/autorun_till_close.py

# P&L tracking
PYTHONPATH=. python3 scripts/track_live_pnl.py

# Data ingestion
PYTHONPATH=. python3 scripts/fast_ingest.py

# Forward paper resolution (Sep 30)
PYTHONPATH=. python3 scripts/resolve_forward_paper.py

# Signal promotion (after G10)
PYTHONPATH=. python3 scripts/run_signal_promotion.py

# G6 robustness check
PYTHONPATH=. python3 scripts/run_g6_robustness_test.py

# Retraining
PYTHONPATH=. python3 scripts/run_lgbm_conc_backtest.py

# Full test suite
python3 -m pytest tests/ --no-cov -q

# Promote to shadow (already done)
PYTHONPATH=. python3 scripts/promote_to_shadow.py --approver "..." --note "..."
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

*Runbook created: 2026-09-28*
*Owner: AlphaForge Quant Engineering*
*Review: Oct 15 (production promotion decision)*
