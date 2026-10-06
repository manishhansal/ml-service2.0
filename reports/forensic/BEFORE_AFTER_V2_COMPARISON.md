# BEFORE / AFTER COMPARISON — v1 → v2c
**Repository:** ml-service2.0 | **Date:** 2026-10-01  
**Basis:** True M1 evaluation (actual model, OOS 2025-01-01 → 2026-09-28)  
**No proxy labels used in this comparison**

---

## Model Summary

| Dimension | v1 (Shadow, REJECTED) | v2c (Current Best) | Δ |
|-----------|----------------------|-------------------|---|
| Model type | LGBMClassifier | LGBMRegressor | Changed |
| Label | 5-bar ±2% triple-barrier | 7-day CS rank excess return | Changed |
| Features | 55 (fs-2.0.0) | 65 (55 + 10 CS/regime) | +10 |
| Calibration | IsotonicWrapper (broken) | None (raw regression) | Removed |
| Normalizer at inference | MISSING | `load_state()` fixed | Fixed |
| Direction method | score > 0.5 (degenerates) | CS rank percentile | Changed |
| Cost assumption | 10 bps (pipeline), 27.65 bps (labels) | 27.35 bps (COST_MODEL_V2) | Standardised |
| True OOS test | Never performed | 2025-01-01+ | Added |

---

## Predictive Quality (True OOS)

| Metric | v1 | v2c | Δ |
|--------|----|-----|---|
| IC (Spearman, raw score vs label) | **−0.001** | **+0.040** | **+0.041** |
| IC p-value | 0.659 | < 0.0001 | SIGNIFICANT |
| Direction correct? | NO (inverted) | YES | Fixed |
| Permutation test | p = 0.67 (fail) | p < 0.0001 (pass) | Fixed |
| In-sample IC (training CV) | 0.41 | 0.40 | Similar |
| OOS IC degradation ratio | **414×** | **10×** | Improved |

The v1 model had a **direction inversion** — predicting the wrong way in OOS. v2c correctly predicts relative outperformers with genuine IC of 0.040.

---

## Signal Quality (7-Day, OOS)

| Metric | v1 (M1) | v2c (M1) | Δ |
|--------|---------|---------|---|
| True L/S spread (top 10% vs bottom 10%) | Unknown | **+0.417%/period** | New |
| Open-to-close IC | −0.001 | +0.026 | +0.027 |
| Gap IC (overnight adverse selection) | Unknown | +0.020 | Measured |
| Win rate (label-based, cross-sectional) | 47.0% | **59.3%** | +12.3pp |

---

## Trading Performance (Long-Only, OOS 2025+)

| Metric | v1 | v2c | Δ |
|--------|----|-----|---|
| Long portfolio return | Not measured | **+12.22%/year** | New |
| NIFTY benchmark | +6.68%/year | +6.68%/year | — |
| Gross alpha vs NIFTY | Undefined | **+5.53%/year** | New |
| Net excess (equity 27.35bps) | −0.090%/trade | **−4.31%/year** | ✗ |
| Net excess (futures 7.26bps) | Not viable | **+2.92%/year** | **✓ PROFITABLE** |
| Net total return (futures) | Not viable | **+9.60%/year** | **✓ PROFITABLE** |
| Information Ratio | None | **0.21** | New |
| Max drawdown (long-only, futures) | — | −12.88% | Measured |

---

## Portfolio Execution (Mode B, M1)

| Metric | v1 (Mode B) | v2c (Mode B, long-only futures) | Δ |
|--------|-------------|--------------------------------|---|
| Total return | +5.83% (18 months, proxy) | +9.60%/year (M1, futures) | M1 vs proxy |
| n trades (18 months) | 36 (from 998 signals) | ~60 (weekly, 27 stocks) | Structured |
| Win rate | 55.6% (filtered, proxy) | 47.5% (excess vs NIFTY) | Changed metric |
| Drawdown limit | −9.20% (§33 bug fixed) | −12.88% | Clear |

---

## Statistical Significance

| Test | v1 | v2c | Assessment |
|------|----|----|-----------|
| OOS IC permutation test | p=0.67 FAIL | p<0.0001 PASS | Genuine signal |
| Label shuffle placebo | IC indistinguishable from null | IC 0.040, significant | Signal confirmed |
| Direction inversion test | Inverted better than original | Original better | Direction fixed |
| L/S spread significance | None | t≈1.65 (borderline) | More data needed |

---

## Cost Sensitivity

| Cost (bps RT) | v1 net | v2c net excess | Viable? |
|--------------|--------|----------------|--------|
| 0 (theoretical) | Broken | +5.53%/year | Yes |
| 7.26 (NSE futures) | N/A | +2.92%/year | **YES** |
| 15 (institutional equity) | N/A | −1.86%/year | Borderline |
| 27.35 (retail equity) | −0.090%/trade | −4.31%/year | No |

**v2c is viable only with NSE futures execution (7.26bps RT).**

---

## Root Cause Resolution

| Root Cause (from audit) | v1 | v2c | Status |
|------------------------|----|----|--------|
| RC-001: Walk-forward CV inflated IC | OPEN | Partially fixed (OOS split) | PARTIAL |
| RC-002: Backtest searched `.lgb` not `.pkl` | OPEN | **FIXED** | ✓ |
| RC-003: Calibration destroyed variance | OPEN | **FIXED** (removed) | ✓ |
| RC-004: 2 bear-market days only | OPEN | OOS extends to 2025+ | IMPROVED |
| RC-005: PBO simplified formula | OPEN | Documented | OPEN |
| RC-006: Feature schema mismatch | OPEN | **FIXED** (65 features explicit) | ✓ |
| RC-007: Forward paper unit error | OPEN | **FIXED** (×100 bug) | ✓ |
| RC-008: Normalizer load_state() missing | OPEN | **FIXED** | ✓ |
| RC-009: Cost inconsistency | OPEN | **FIXED** (COST_MODEL_V2) | ✓ |
| RC-010: Survivorship bias | OPEN | Still open | OPEN |
| RC-011: Score direction degenerate | OPEN | **FIXED** (CS rank) | ✓ |
| RC-012: Direction inversion in OOS | OPEN | **FIXED** (v2c regressor) | ✓ |

**Fixed: 8/12 root causes | Open: 4/12**

---

## Test Coverage

| Test Suite | v1 | v2c | Tests Added |
|-----------|----|----|------------|
| 7-day engine | 19 | 19 | — |
| Sprint 3 alpha signal | 13 | 13 | — |
| Portfolio engine | 25 | 25 | — |
| Static leakage audit | 4 | 4 | — |
| v2 pipeline tests | 0 | **11** | +11 |
| **Total** | **61** | **72** | **+11** |

All 72 tests pass (0 failures).

---

## Maximum Honest OOS Performance

```
Without leakage, with genuine OOS evaluation (2025-01-01+):

v1 model:
  IC = −0.001 (not significant)
  Strategy: NOT VIABLE

v2c model (best achieved):
  IC = +0.040 (p<0.0001, genuine signal)
  Long-only strategy (top 10%, weekly, futures):
    Gross alpha vs NIFTY:  +5.53%/year
    Net total return:      +9.60%/year
    Net excess vs NIFTY:   +2.92%/year
    Info Ratio:             0.21
    Max drawdown:          −12.88%
    
  At equity costs:        NOT PROFITABLE (−4.31% excess)
  At futures costs:       PROFITABLE (+2.92% excess) ✓

The honest ceiling with current features (pure technical):
  IC = 0.04–0.06 (typical for mature equity technical factors)
  This translates to ~3–5%/year excess at futures costs.
  
To achieve meaningfully higher returns (>10%/year excess):
  → Add fundamental features (P/B, ROE, earnings growth)
  → Add earnings momentum (PEAD) signals
  → Add sector-rotation signals using macro data
```
