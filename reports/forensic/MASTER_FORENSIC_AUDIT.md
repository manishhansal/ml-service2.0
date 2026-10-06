# MASTER FORENSIC AUDIT
**Repository:** ml-service2.0  
**Date:** 2026-10-01  
**Mode:** Institutional full-spectrum audit  
**Standard:** Evidence-first; no fabricated performance

---

## Executive Verdict

```
╔══════════════════════════════════════════════════════════════════════════╗
║  VERDICT: NOT PRODUCTION READY — NOT EVEN SHADOW READY                  ║
║                                                                           ║
║  The model has ZERO demonstrated out-of-sample predictive power.         ║
║  All prior performance claims are based on within-training cross-         ║
║  validation or label-proxy evaluation, not true held-out OOS results.   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Section 1 — Critical Findings

### F-001 [CRITICAL] Model Artifact Discovery Bug

**Finding:** The 7-day backtest runner (`scripts/run_7d_backtest.py`) searched for `*.lgb` files. The actual model artifact is `model.pkl` (Python pickle of a dict). The runner silently fell back to label-proxy (P0) mode and reported those results as if they were model (M1) results.

**Impact:** Every performance number from `ML_7_DAY_BACKTEST_REPORT.md` and all downstream certification documents is P0 (proxy), not M1 (model).

**Fix:** Updated `load_model()` to search for `*/model.pkl`. Added `--mode` flag. Hard fail on `MODEL_ARTIFACT_REQUIRED` when M1 is requested but artifact is absent.

**Status:** FIXED in this audit cycle.

---

### F-002 [CRITICAL] True OOS IC Is Zero

**Finding:** The claimed IC = 0.4136 was computed on within-training walk-forward folds. The true held-out OOS IC (January 2025 onwards) is:

| Metric | Claimed | True OOS (M1) |
|--------|---------|---------------|
| IC (Spearman) | 0.4136 | −0.001 (p=0.659) |
| Binary accuracy | ~69–80% | 52.3% |
| Win rate (net) | 80% | 47.0% |
| EV/trade | +0.8% net | −0.090% (dataset labels) |
| Directional correctness | Correct | INVERTED (1−score > score) |

The model predicts the **wrong direction** in OOS. The in-sample to OOS IC degradation is 0.115.

**Root cause:** Classic financial ML overfitting. Walk-forward CV within the training period inflates IC. The model learned patterns that do not generalize beyond the training window.

---

### F-003 [CRITICAL] Calibration Destroys Information

**Finding:** The `_IsotonicWrapper` calibrator was fit on training/validation data to minimize in-sample ECE. The result: all OOS scores are compressed to a narrow band (std = 0.010 vs raw std = 0.029). The claimed ECE = 0.000 is trivially achieved in-sample by isotonic overfitting — not evidence of genuine calibration.

**Impact:** The calibrated scores are nearly constant, making threshold-based signal generation unreliable.

---

### F-004 [CRITICAL] PBO = 0.000 Is Invalid

**Finding:** PBO was computed as the fraction of CV folds with negative Sharpe. With 0/5 negative-Sharpe folds, PBO = 0.000. This is NOT a valid Probability of Backtest Overfitting estimate. Proper CPCV PBO with C(n,k) combinatorics is required.

**Impact:** The G5 gate ("PBO < 0.50 PASS") was incorrectly certified.

---

### F-005 [CRITICAL] 80% Win Rate Based on 2 Bear-Market Days

**Finding:** The Sep 28-29 live sessions occurred on two consecutive NIFTY down-days (−1.52%, −0.42%). SHORT alpha in a falling market is expected and does not prove cross-sectional alpha. Aggregate 3-session win rate (Sep 29 + Sep 30 + Oct 1) = 46.0% (below 50%).

**Impact:** All G7 (regime robustness), G9 (net Sharpe), and G12 (human approval) decisions were made on this evidence.

---

### F-006 [HIGH] Feature Schema Version Mismatch

**Finding:** The SHADOW model is `fs-2.0.0` (55 features from FeatureFactory). Current training pipeline uses `fs-4.0.0` (91 total, 67 feature columns). All 55 model features ARE present in `fs-4.0.0` datasets as a subset.

**Impact:** The model cannot be retrained directly against the new dataset without explicitly selecting the 55 original features. Inference works when the correct subset is provided.

---

### F-007 [HIGH] Forward Paper net_pct Values Implausible

**Finding:** Oct 1 forward paper mean_net = −28.548% for 5-bar trades. This is physically implausible for NSE F&O stocks. Root cause: unknown — likely a unit-conversion error (fraction reported as percentage) or a sign inversion in the `ForecastLedger` update.

**Impact:** The G10 (forward paper ≥ 50 outcomes with positive mean) and G11 (signal promotion) decisions cannot be trusted until this is reconciled.

---

### F-008 [HIGH] Cost Inconsistency Across Documents

Six different round-trip cost assumptions appear across reports: 8.5, 10, 14, 26.4, 27.35, 27.65 bps. This creates incomparable performance numbers.

---

### F-009 [MEDIUM] Survivorship Bias — Historical Universe Not Validated

**Finding:** The training universe uses today's F&O eligible symbols. `HistoricalUniverse` and `FnOStateStore` return `DATA_UNAVAILABLE` for most historical dates. The PIT universe validation is a placeholder, not a functioning guard.

---

### F-010 [MEDIUM] Normalizer State Not Loaded at Inference

**Finding:** `FeatureNormalizer.load_state()` method does not exist. The normalizer state is stored in the model pkl but cannot be restored at inference time. The model was thus run without its trained normalization.

---

## Section 2 — Code Fixes Applied

| Fix | File | Status |
|-----|------|--------|
| Model discovery: .pkl not .lgb | `scripts/run_7d_backtest.py` | ✓ FIXED |
| Hard fail MODEL_ARTIFACT_REQUIRED | `scripts/run_7d_backtest.py` | ✓ FIXED |
| --mode flag (m1/proxy/auto) | `scripts/run_7d_backtest.py` | ✓ FIXED |
| Label TIME_EXPIRY cost-adjusted threshold | `src/labels/seven_day.py` | ✓ FIXED |
| Label 7-day horizon (was 5-bar) | `src/labels/seven_day.py` | ✓ FIXED |
| Asymmetric 2:1 R:R barrier | `src/labels/seven_day.py` | ✓ FIXED |
| SevenDayBacktestEngine (new) | `src/backtest/seven_day_engine.py` | ✓ NEW |
| Portfolio engine (new) | `src/backtest/portfolio_engine.py` | ✓ NEW |
| Leakage audit path fix | `src/features/leakage_validator.py` | ✓ FIXED |

---

## Section 3 — What Must Change Before Production

### Mandatory (P0 Blockers)
1. Retrain model with genuine OOS held-out test (not just walk-forward CV)
2. Fix calibration: use temperature scaling or platt scaling on a true hold-out set, not isotonic on validation data
3. Require minimum 20 mixed-regime trading days with positive aggregate win rate before any promotion
4. Implement proper CPCV PBO
5. Fix `FeatureNormalizer.load_state()` to restore training-time normalization at inference

### Required Before Shadow
6. Reconcile forward paper net_pct units
7. Build PIT F&O universe for survivorship control
8. Standardize cost model (single version used everywhere)
9. Run true OOS IC test per fold; reject any model with true OOS IC < 0.005

---

## Section 4 — Maximum Honest OOS Performance

With the current model and feature set, the maximum leakage-free OOS performance demonstrated is:

```
IC (Spearman, raw score):     −0.001 (not statistically significant, p=0.659)
IC (calibrated score):        +0.009 (barely significant, p=0.002)
Binary accuracy:              52.3%
Directional win rate (M1):    47.0% (below 50%)
Economic precision (net):     49.3% (below random)
Expected value (M1, 7-day):  −0.090%/trade (negative)
Coverage of opportunity set:   6.2%

The model FAILS every production gate that requires OOS evidence.
```

The 47.0% win rate with +0.416% mean net P&L in the 7-day backtest (M1) is NOT alpha — it arises from the 2:1 asymmetric barrier (6% target vs 3% stop), not from the model's directional predictions.

---

*Generated: 2026-10-01 | See companion: ROOT_CAUSE_REGISTER.md*
