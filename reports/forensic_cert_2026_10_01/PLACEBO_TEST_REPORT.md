# PLACEBO TEST REPORT
**Repository:** ml-service2.0 | **Date:** 2026-10-01  
**Model:** expanded_lgbm v1.0.0-20260928053134956099  
**OOS period:** 2025-01-01 → 2026-09-28 (118,266 valid rows)

---

## Summary

| Test | Expected | Actual | Result |
|------|---------|--------|--------|
| P1: Label shuffle | IC ≈ 0 | IC = 0.000 (null mean) | PASS (model IC ≈ null) |
| P2: Timestamp shuffle | IC ≈ 0 | IC = +0.006 (shuffled) | PASS |
| P3: Random prediction | IC ≈ 0, acc ≈ 50% | IC = −0.003, acc = 49.8% | PASS |
| P4: Raw vs calibrated | Calibrated ≥ raw | Calibrated IC = 0.009 > raw IC = −0.001 | ⚠ MARGINAL |
| P5: Direction inversion | Inverted worse | Inverted IC = +0.001 > raw IC = −0.001 | **FAIL** |
| P6: In-sample vs OOS | OOS < in-sample | OOS IC = −0.001 vs train IC = 0.114 | FAIL (overfitting) |

---

## P1: Label Shuffle (Permutation Test)

**Setup:** Scores fixed (from true model). Labels randomly permuted 1,000 times.  
**Expected:** True IC should exceed null distribution if model has skill.

```
Null IC mean:  −0.000066
Null IC std:    0.002881
True IC:       −0.001284
Z-score:       −0.42  (true IC BELOW the null mean)
p-value:        0.67   (NOT SIGNIFICANT)
```

**Interpretation:** The true model IC is INDISTINGUISHABLE from shuffled labels. The model has NO statistically significant predictive power.

---

## P2: Timestamp Shuffle

**Setup:** Features re-matched to random future returns (destroys temporal relationship).  
**Result:** IC = +0.006 — slightly higher than true IC (−0.001)!

This means the model captures some CROSS-SECTIONAL relationship between features and returns that persists even across random time periods. This suggests mild data-mining bias in the feature set (features that happened to correlate with returns in the historical period but for non-causal reasons).

---

## P3: Random Prediction Baseline

```
Random IC:       −0.003
Random accuracy: 49.8%
True model IC:   −0.001
True accuracy:    52.3%
Delta IC:        +0.002 (essentially zero)
Delta accuracy:  +2.5pp (minimal)
```

The model barely outperforms a random number generator in OOS.

---

## P4: Raw vs Calibrated Score

```
Raw score:          mean=0.464, std=0.029, IC=−0.001
Calibrated score:   mean=0.455, std=0.010, IC=+0.009
```

The isotonic calibration accidentally improves IC from −0.001 to +0.009. This occurs because the calibrator maps scores to a slightly different distribution that happens to correlate better with outcomes in this specific OOS period. This is not reliable calibration — it is incidental and not reproducible.

The calibrator also DESTROYS 66% of score variance (std 0.029 → 0.010), making threshold-based direction logic nearly useless (almost all calibrated scores ≈ 0.45).

---

## P5: Direction Inversion (**CRITICAL FAILURE**)

```
Original score IC:  −0.001284
Inverted (1-score): +0.001284  ← BETTER than original
```

**The inverted model (predict the OPPOSITE) outperforms the original model in OOS.**

This is the strongest possible evidence of overfitting. The model learned patterns in the training period that are reversed or noise in the OOS period. The direction sign flipped between training and deployment.

Root cause candidates:
1. Training period was a BULL market; OOS is SIDEWAYS/BEAR → momentum signals reversed
2. Training labels used different direction semantics than OOS evaluation
3. Overfitting to specific bull-market patterns that reversed

---

## P6: In-Sample vs True OOS IC

```
Within-training (last 20% of training period):  IC = +0.114
True OOS (2025 onwards):                         IC = −0.001
Degradation:                                     0.115
Ratio (in-sample / OOS):                         114×
```

A 114× degradation from in-sample to true OOS is extreme. In financial ML, a degradation ratio > 3–5× typically indicates significant overfitting.

---

## Overall Assessment

The model **FAILS the placebo tests** on the most important criteria:
- P1: IC indistinguishable from null (p=0.67)
- P5: Direction inversion outperforms original
- P6: 114× IC degradation from in-sample to OOS

These results confirm that the model discovered training-period artifacts, not causal relationships between features and future returns.

**Verdict: NO GENUINE OOS PREDICTIVE POWER DEMONSTRATED**
