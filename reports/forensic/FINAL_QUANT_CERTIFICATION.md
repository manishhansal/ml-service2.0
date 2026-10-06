# FINAL QUANTITATIVE CERTIFICATION
**Repository:** ml-service2.0  
**Audit Date:** 2026-10-01  
**Auditor:** Kiro Institutional Forensic System  
**Model:** expanded_lgbm v1.0.0-20260928053134956099  
**Evaluation Standard:** M1 (actual model) — no proxy substitution

---

## Executive Verdict

```
╔═══════════════════════════════════════════════════════════════════════════╗
║  CERTIFICATION: NOT READY — RESEARCH ONLY                                ║
║                                                                            ║
║  The model demonstrates zero statistically significant out-of-sample      ║
║  predictive power. All prior performance claims were based on within-      ║
║  training cross-validation or label-proxy evaluation. The model is NOT    ║
║  production ready, NOT shadow ready, and must not be used to deploy        ║
║  capital until the fundamental overfitting issues are resolved.            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## Section 1 — The 58 Questions (Summary Answers)

### What is the model predicting?
LGBMClassifier predicts probability that a 5-bar ±2% triple-barrier target is hit before the stop. **This is not a 7-day directional predictor.**

### Is the target economically valid?
**No.** Old labels: EV = −0.310%/trade (negative). Improved 7-day labels exist in `src/labels/seven_day.py` but the model was not trained on them.

### Is the target aligned to 7 trading days?
**No.** The model uses 5-bar labels. The 7-day backtest engine exists but was evaluated on the old model.

### Is the model predicting direction or ranking?
It outputs a probability [0,1] used as a cross-sectional ranking score.

### Is the score a probability or ranking score?
Nominally a probability (LGBMClassifier), but the calibration destroys variance (std=0.010). Used in practice as a ranking score.

### Can the model identify profitable opportunities OOS?
**No.** OOS IC = −0.001 (p=0.659). Coverage = 6.2%.

### How many profitable opportunities existed?
5,929 in the OOS period (20 symbols, 18 months).

### How many did the model capture?
368 (6.2%).

### How many profitable opportunities did it miss?
5,561 (93.8%).

### What is economic precision?
49.3% — near random.

### What is true OOS expectancy?
−0.090%/trade (negative, using dataset label returns).

### What is portfolio-level net return?
+5.83% over 18 months from 36 trades (Mode B M1). Insufficient sample to claim significance.

### What is maximum drawdown?
−9.20% (Mode B M1, within 10% policy limit after §33 fix).

### What happens during regime transitions?
Model collapses. Sep 28-29 (BEAR) = 80% win rate. Oct 1 (RECOVERY) = 23.8%.

### What happens under conservative costs?
Not tested with M1. Proxy-mode stress shows positive results, but proxy ≠ model.

### What happens if the top feature group is removed?
Not tested with OOS retrain ablation (only static importance computed).

### What happens under placebo labels?
Model IC = −0.001 matches shuffled IC distribution (p=0.659). **No information above random.**

### What is the proper PBO?
CPCV within training = 0.000. But true OOS collapse (IC: 0.114 → −0.001) confirms overfitting.

### Is performance statistically significant?
**No.** OOS IC p-value = 0.659. Direction inversion has better IC than original.

### Is performance concentrated in a few symbols?
Yes. 10 defensive stocks drive positive IC. 54.1% of symbols have negative IC.

### Is performance concentrated in a few dates?
2 consecutive bear market days explain live performance claim.

### Does the model work outside bear markets?
**No OOS evidence it works at all**, including bear markets (SIDEWAYS regime in 2025+ = IC −0.001).

### Does it work for both LONG and SHORT?
Labels are LONG-only semantics. SHORT signals are inverted LONG predictions. Side-aware labels not implemented.

### Does it survive realistic execution?
Portfolio P&L is positive (+5.83%) but only 36 trades. Not meaningful evidence.

### What is the true capacity?
Unknown. At 36 trades/18 months, capacity question is academic.

### Is the strategy production-ready?
**No.** 9/21 production gates FAIL.

### If not, what gates remain?
G_SIGNIFICANCE, G_PLACEBO, G_REGIME, G_LABEL, G_HORIZON, G_UNIVERSE, G_CALIBRATION, G_PBO, G_ABLATION

### What is the maximum honest OOS performance demonstrated?
```
IC (Spearman, OOS 2025+):     −0.001 (NOT SIGNIFICANT)
Directional accuracy:         47.0% (below 50%)
Economic precision:           49.3% (below random)
Expected value/trade:         −0.090% (negative)
Portfolio return (18 months): +5.83% from 36 trades (INSUFFICIENT SAMPLE)

The maximum leakage-free, out-of-sample performance certified:
  → ZERO DEMONSTRATED ALPHA
  → NOT production ready in any capacity tier
```

---

## Section 2 — True Model Performance (M1)

### Classification
| Metric | Value |
|--------|-------|
| IC (Spearman raw) | −0.001 |
| IC (calibrated) | +0.009 |
| Binary accuracy | 52.3% |
| Permutation p-value | 0.659 (NOT SIGNIFICANT) |
| Direction correct? | NO — inverted IC (+0.001) > forward IC (−0.001) |

### Signal Quality
| Metric | Value |
|--------|-------|
| Directional win rate | 47.0% |
| Precision | 49.3% |
| Coverage | 6.2% |
| EV/trade | −0.090% |
| Profitable opp capture | 6.2% |

### Portfolio (Mode B M1)
| Metric | Value |
|--------|-------|
| Total return (18 months) | +5.83% |
| Trades | 36 |
| Win rate (filtered) | 55.6% |
| Max drawdown | −9.20% |
| Sharpe | 6.07 |
| Significance | LOW (36 trades) |

---

## Section 3 — Bugs Fixed in This Audit

| Bug | Fix | Impact |
|----|-----|--------|
| Model discovery searched *.lgb | Fixed to *.pkl | All prior M1 backtests were actually P0 |
| Portfolio drawdown side-effect | Removed mutation from property | Max DD corrected |
| Forward paper ×100 unit error | Corrected in analysis script | Oct 1 = −0.285%, not −28.5% |
| Label TIME_EXPIRY threshold | Fixed to use net cost | Label quality improved |
| 55 feature selection at inference | Explicitly uses model_dict["feature_names"] | Parity fix |

---

## Section 4 — Remaining Limitations

1. **Fundamental**: Model has no demonstrated OOS alpha. Retrain required with proper held-out test.
2. **Survivorship bias**: Historical F&O universe not validated. Affect unknown but potentially 5-10% of rows.
3. **Normalizer**: `load_state()` method missing. Model runs without training-time normalization.
4. **Label direction**: No side-aware LONG/SHORT labels. SHORT signals are semantically incorrect.
5. **PBO**: Proper CPCV PBO requires truly held-out data — not achievable with current data structure until retrain.

---

## Section 5 — Reproducibility Manifest

```yaml
audit_date:            2026-10-01
model_version:         1.0.0-20260928053134956099
model_sha256:          55ec99ba451022fb75ff2b3076eb5c5e94e032d73812032a66cfb34b728def2b
feature_schema:        fs-2.0.0
feature_count:         55
training_dataset:      ds-1d-20260926202149-3f078494
oos_dataset:           ds-1d-20261001034811-ebdf74af
oos_start:             2025-01-01
oos_end:               2026-09-28
cost_model_version:    COST_MODEL_V2 (27.35bps equity, 7.26bps futures)
true_oos_ic:           -0.001284
true_oos_ic_pval:      0.6587
true_m1_win_rate:      0.470
true_m1_precision:     0.493
true_m1_ev:            -0.090%/trade
m1_portfolio_return:   +5.83% (18 months, 36 trades)
final_verdict:         RESEARCH_ONLY — NOT production ready
gates_passed:          9/21
gates_failed:          9/21
gates_partial:         3/21
```

---

## Section 6 — Recommended Immediate Actions

### Week 1
1. Demote model from SHADOW to BACKTEST stage
2. Stop forwarding paper-trading signals as evidence of production readiness
3. Discard all prior certification documents that claimed IC=0.4136 or ECE=0.000 as OOS metrics

### Month 1
4. Retrain on 7-day asymmetric labels (`src/labels/seven_day.py`)
5. Implement proper held-out OOS test period (e.g., 2024-01 onwards, never touched)
6. Fix `FeatureNormalizer.load_state()`
7. Run placebo tests on the new model

### Month 2
8. Run 20+ live trading days across 4 regimes (not just bear days)
9. If live IC > 0.005 and win rate > 56.9% (equity) → proceed to CHALLENGER
10. Otherwise: redesign feature set or model objective

---

*This certification supersedes all prior certification documents dated 2026-09-28 and 2026-09-29.*  
*Generated: 2026-10-01 | Output dir: reports/forensic_cert_2026_10_01/*
