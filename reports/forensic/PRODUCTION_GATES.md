# PRODUCTION GATES — FINAL ASSESSMENT
**Repository:** ml-service2.0 | **Date:** 2026-10-01  
**Model:** expanded_lgbm v1.0.0-20260928053134956099  
**Evaluation mode:** M1 (actual model) where possible, otherwise documented

---

## Gate Results

| Gate | Description | Status | Evidence |
|------|-------------|--------|---------|
| **G_PIT** | No look-ahead bias | ✓ PASS | Static audit: 0 INVALID; mutation tests pass |
| **G_LEAK** | No feature leakage | ✓ PASS | |r| < 0.15 all features vs forward returns |
| **G_PARITY** | Training/inference schema match | ⚠ PARTIAL | Features match; normalizer load_state() missing |
| **G_ARTIFACT** | Real model artifact loaded | ✓ PASS | model.pkl found; discovery bug fixed |
| **G_LABEL** | Label economically valid | ✗ FAIL | Old labels: EV = −0.310%/trade; new labels partially fixed |
| **G_HORIZON** | 7-trading-day alignment | ✗ FAIL | Training on 5-bar labels; 7-day evaluation exists but is new |
| **G_UNIVERSE** | Historical universe validated | ✗ FAIL | HistoricalUniverse returns DATA_UNAVAILABLE for most dates |
| **G_EXECUTION** | Realistic execution | ✓ PASS | next_open fills; NSE calendar; 26.4bps costs |
| **G_PORTFOLIO** | Portfolio-level P&L positive | ✓ MARGINAL | +5.83% over 18 months, 36 trades — insufficient sample |
| **G_SIGNIFICANCE** | Statistical significance | ✗ FAIL | OOS IC = −0.001, p=0.659; permutation p=0.67 |
| **G_REGIME** | Multi-regime robustness | ✗ FAIL | Only BEAR regime tested live (2 days); Oct 1 collapse = 23.8% |
| **G_CALIBRATION** | Probability calibration valid | ✗ FAIL | ECE=0.000 is in-sample artifact; calibrator destroys OOS variance |
| **G_PBO** | PBO analysis valid | ✗ FAIL | Old PBO = 0.000 used fold-count not CPCV; true OOS collapse |
| **G_PLACEBO** | Placebo tests pass | ✗ FAIL | Direction inversion: IC(1−score) > IC(score); IC p=0.659 |
| **G_ABLATION** | Feature ablation OOS | ✗ FAIL | Only static importance used; no OOS retrain ablation |
| **G_STRESS** | Cost stress test | ✓ PARTIAL | Mode B positive at base costs; not tested at 2× |
| **G_DRAWDOWN** | Drawdown limits enforced | ✓ PASS (fixed) | §33 bug fixed; max DD = −9.20% within 10% limit |
| **G_REPRODUCIBILITY** | Same config → same results | ✓ PASS | Deterministic pipeline; seeds recorded |
| **G_NOCHERRY** | No cherry-picking | ✓ PASS | All signals, all dates, all symbols included |
| **G_FORWARD** | Forward paper reconciled | ✓ PASS (fixed) | Unit error corrected; Oct 1 win rate 23.8% is real |
| **G_COST** | Single cost model | ✓ PASS | COST_MODEL_V2.md standardizes all evaluations |

### Gate Summary

```
PASS:     G_PIT, G_LEAK, G_ARTIFACT, G_EXECUTION, G_DRAWDOWN,
          G_REPRODUCIBILITY, G_NOCHERRY, G_FORWARD, G_COST
          = 9 PASS

PARTIAL:  G_PARITY, G_PORTFOLIO, G_STRESS
          = 3 PARTIAL

FAIL:     G_LABEL, G_HORIZON, G_UNIVERSE, G_SIGNIFICANCE,
          G_REGIME, G_CALIBRATION, G_PBO, G_PLACEBO, G_ABLATION
          = 9 FAIL
```

**9 FAIL / 9 PASS / 3 PARTIAL → NOT PRODUCTION READY**

---

## Most Critical Fails

### G_SIGNIFICANCE — CRITICAL
True OOS IC = −0.001 (p=0.659). The model has no statistically significant predictive power. This single gate failure is sufficient to block all deployment.

### G_PLACEBO — CRITICAL
Direction inversion outperforms the model in OOS. The model learned the wrong pattern.

### G_REGIME — CRITICAL
Only 2 consecutive bear-market days of live testing. No multi-regime validation exists. Oct 1 collapse confirms regime-specific bias.

### G_PBO — HIGH
The claimed PBO = 0.000 was computed with an invalid formula. The proper OOS test confirms overfitting (IC degrades from 0.114 to −0.001 = 114× degradation).

---

## Path to Passing All Gates

To pass the 9 currently failing gates:

1. **G_SIGNIFICANCE** + **G_PLACEBO**: Requires retraining a model that demonstrates OOS IC > 0.005 on a genuinely held-out test period. Current model fails.

2. **G_REGIME**: Requires 20+ live trading days across 4 regimes.

3. **G_LABEL** + **G_HORIZON**: New 7-day asymmetric labels (`src/labels/seven_day.py`) + retraining.

4. **G_UNIVERSE**: Build complete historical F&O eligibility database.

5. **G_CALIBRATION**: Use temperature scaling on a true holdout, not isotonic on validation.

6. **G_PBO**: Implement proper CPCV with truly held-out data.

7. **G_ABLATION**: Retrain with each feature group removed, evaluate OOS IC.
