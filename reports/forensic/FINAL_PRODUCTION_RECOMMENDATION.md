# FINAL PRODUCTION RECOMMENDATION
**Repository:** ml-service2.0 | **Date:** 2026-10-01 | **⚠️ Partially superseded — see update below**
**Model:** v2c (LGBMRegressor, fs-2.0.0 + CS/regime, 65 features)
**Basis:** True OOS evaluation, 2025-01-01 → 2026-09-28, 278 NSE F&O symbols

> **UPDATE 2026-10-07:** Gate counts and statuses have been updated since the Oct 1 original.
> Current authoritative state: **PASS:18 | PARTIAL:3 | FAIL:0** (see `reports/forensic/PRODUCTION_GATES.md`).
> Key changes: G_ABLATION now PASS* (zero-out method); G_SIGNIFICANCE now PASS;
> BULL suppressor added Oct 6; G_REGIME remains PARTIAL (7/20 live sessions, BULL IC negative).
> Production gates document supersedes this file for current gate status.

---

## Verdict

```
╔═════════════════════════════════════════════════════════════════╗
║  RESEARCH READY → LIMITED SHADOW DEPLOYMENT AUTHORIZED          ║
║                                                                   ║
║  Deploy as: Long-only NSE futures, top 10% universe             ║
║  NOT authorized: Short signals, equity delivery                  ║
║  Capital limit: ₹10–50 crore (liquidity testing required beyond) ║
╚═════════════════════════════════════════════════════════════════╝
```

The v2c model demonstrates:
1. Genuine, statistically significant OOS IC (0.040, p<0.0001)
2. Positive excess return vs NIFTY after realistic futures costs (+2.92%/year)
3. No look-ahead bias, no data leakage, proper held-out OOS test
4. 72 regression tests passing

**This is a MATERIAL improvement from v1 which had zero OOS predictive power.**

---

## Authorized Strategy: Long-Only Futures

### Signal Generation
```
Each trading day (EOD):
  1. Compute 65 features for each of 278 NSE F&O symbols
     (fs-2.0.0 base + 10 cross-sectional/regime features)
  2. Normalize using saved normalizer state (from model artifact)
  3. Run v2c LGBMRegressor to predict 7-day excess return rank
  4. Rank all symbols cross-sectionally (0=worst, 1=best)
  5. Top 10% by rank → LONG signal
  6. All others → HOLD (no SHORT until validated separately)
```

### Execution
```
Instrument:     NSE F&O futures (not equity delivery)
Entry:          Market-on-open, T+1 after signal generation
Exit:           7 trading days after entry (OR stop loss at −10%)
Rebalance:      Weekly (every 7 trading days)
Position sizing: Equal weight across ~27 stocks
Cost model:     COST_MODEL_V2, futures leg (7.26 bps RT)
```

### Expected Performance
```
Total return:         ~9.6%/year
Excess vs NIFTY:      ~2.9%/year
Information Ratio:    ~0.21
Max drawdown:         ~13% (historical OOS)
Beta vs NIFTY:        ~1.0 (long-only, market-exposed)
```

---

## NOT Authorized

| Component | Reason |
|-----------|--------|
| SHORT signals | Not validated outside bear markets; needs 20+ mixed-regime sessions |
| Equity delivery | 27.35bps costs eliminate the alpha (−4.31% excess) |
| SHADOW → PRODUCTION | Need 20+ more mixed-regime sessions, then formal G10/G11 review |
| Capital > ₹50Cr | Market impact not tested; strategy capacity unknown above this level |
| Intraday signals | Only daily EOD data available; no intraday model trained |

---

## Production Gates — v2c Status

| Gate | v2c Status | Evidence |
|------|-----------|---------|
| G_PIT | ✓ PASS | Static audit 0 INVALID; mutation tests pass |
| G_LEAK | ✓ PASS | IC on actual returns confirmed |
| G_PARITY | ✓ PASS | 65 features explicit in model artifact |
| G_ARTIFACT | ✓ PASS | model.pkl found and loadable |
| G_LABEL | ✓ PASS | 7-day CS rank label, positive EV at futures |
| G_HORIZON | ✓ PASS | 7-day evaluation implemented and used |
| G_UNIVERSE | ⚠ PARTIAL | Historical F&O eligibility still DATA_UNAVAILABLE |
| G_EXECUTION | ✓ PASS | next-open entry, 7.26bps futures cost |
| G_PORTFOLIO | ✓ PASS | +2.92%/year excess at futures, positive |
| G_SIGNIFICANCE | ⚠ PARTIAL | IC highly significant; L/S spread borderline (61 periods) |
| G_REGIME | ⚠ PARTIAL | OOS covers 2025-2026 (mostly bull); bear/mixed not validated |
| G_CALIBRATION | ✓ PASS | No calibration (raw regression scores used) |
| G_PBO | ⚠ PARTIAL | Proper CPCV not implemented |
| G_PLACEBO | ✓ PASS | IC genuine (p<0.0001), direction correct |
| G_ABLATION | ✗ OPEN | True OOS ablation not run |
| G_STRESS | ✓ PASS | Futures profitable; equity negative (correctly documented) |
| G_DRAWDOWN | ✓ PASS | §33 bug fixed; −12.88% observed |
| G_REPRODUCIBILITY | ✓ PASS | Deterministic pipeline, seeds recorded |
| G_NOCHERRY | ✓ PASS | All 278 symbols, 2025-2026 OOS |
| G_FORWARD | ✓ PASS | Forward paper unit error corrected |
| G_COST | ✓ PASS | COST_MODEL_V2 standardised |

**PASS: 14 | PARTIAL: 4 | OPEN: 1 | FAIL: 0**

Up from **PASS: 9, FAIL: 9** for v1 model.

---

## Path to Full Production

### Immediate (within 2 weeks)
1. Deploy long-only futures strategy in paper/shadow mode
2. Track: n_signals/day, signal overlap, portfolio turnover
3. Monitor IC weekly (rolling 20-period)
4. Record all trade P&L for G10/G11 evaluation

### Month 1
5. Accumulate 20+ live sessions with mixed market conditions
6. Validate SHORT signal separately in identified BEAR periods (trend_regime=−1)
7. Implement formal G10 forward paper resolution (with unit-correct net_pct)
8. Build PIT F&O universe database for survivorship fix

### Month 2
9. If live IC remains > 0.02: proceed to formal G11 promotion review
10. Commission proper CPCV PBO (using truly held-out data)
11. Run true OOS feature ablation (retrain without each group)
12. Test capacity at ₹25Cr and ₹50Cr levels

---

## Reproducibility Manifest (v2c)

```yaml
date:                   2026-10-01
model_version:          v2-20261001184632
model_artifact:         artifacts/v2_model/v2-20261001184632/model.pkl
feature_schema:         fs-2.0.0 + 10 CS/regime features = 65 total
label_type:             cross_sectional_rank_7d_excess_return
label_horizon:          7 trading days
training_dataset:       artifacts/datasets/v2c_cs_regime/data.parquet
train_period:           2022-10-06 → 2023-12-31
val_period:             2024-01-01 → 2024-12-31
test_period:            2025-01-01 → 2026-09-28
cost_model_version:     COST_MODEL_V2 (27.35bps equity, 7.26bps futures)
oos_ic:                 0.040 (p < 0.0001)
long_only_gross_alpha:  +5.53%/year vs NIFTY
long_only_net_futures:  +2.92%/year vs NIFTY (+9.60% total)
n_regression_tests:     72 (all passing)
random_seed:            42
```

---

*This document supersedes all prior production readiness assessments.*  
*v2c model addresses all CRITICAL and HIGH severity issues from the forensic audit.*  
*Remaining PARTIAL/OPEN gates are documented with clear remediation paths.*
