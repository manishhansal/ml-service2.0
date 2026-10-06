# REPORT RECONCILIATION MATRIX
**Repository:** ml-service2.0  
**Audit Date:** 2026-10-01  
**Auditor:** Kiro Institutional Forensic System  
**Output Dir:** `reports/forensic_cert_2026_10_01/`

---

## Legend

| Status | Meaning |
|--------|---------|
| **VERIFIED** | Claim matches actual repository evidence |
| **PROXY** | Claim is true for label-proxy (P0) but NOT the actual model (M1) |
| **CONTRADICTED** | Claim is directly false per empirical evidence |
| **STALE** | Claim was true at some earlier date but no longer accurate |
| **UNVERIFIED** | Cannot confirm without additional evidence |
| **INVALID** | Claim contains methodological error |

---

## Critical Definitions Applied

- **MODE P0** = Label/hindsight proxy. NOT model performance.  
- **MODE M1** = Actual trained LightGBM model inference.  
- **MODE P2** = Portfolio execution of M1 signals.  
- **All IC / win-rate / accuracy claims from prior reports are P0 or in-sample unless explicitly marked M1.**

---

## Matrix

### ML_PIPELINE_FORENSIC_AUDIT.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | Model: expanded_lgbm, fs-4.0.0, 279 symbols | **STALE** | Actual model is `fs-2.0.0`, 55 features, trained on `ds-1d-20260926202149` |
| 2 | Label: triple_barrier, 5 bars, ±2%, next_open | **VERIFIED** | Confirmed in dataset metadata |
| 3 | Mean net return = −0.287%/trade | **VERIFIED** | Computed from dataset |
| 4 | Label balance 49.6/50.4% | **VERIFIED** | Confirmed |
| 5 | No look-ahead bias in feature code | **VERIFIED** | Static audit: 0 INVALID findings |
| 6 | IC = 0.4136 (claimed) | **CONTRADICTED** | True OOS IC (M1) = −0.001 (p=0.659). Claimed value is within-training CV, not true OOS. |
| 7 | Sep 28-29 80% SHORT win rate | **PROXY/STALE** | 2-day regime-specific result. Oct 1: 23.8%. Aggregate (Sep 29 + Sep 30 + Oct 1) = 46%. |
| 8 | Oct 1 forward paper −28.5% mean net | **UNVERIFIED** | Possible unit error; `resolve_forward_paper.py` not fully audited |
| 9 | ECE = 0.000 | **INVALID** | This is in-sample calibration fit on training data. True OOS calibration not measured. Brier score = 0.206. |
| 10 | No 7-day specific backtest exists | **VERIFIED** | `SevenDayBacktestEngine` added this cycle |

---

### ML_7_DAY_BACKTEST_REPORT.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | Win rate 64.2% | **PROXY** | This is label-proxy P0 mode (hindsight labels as scores). NOT M1 model. |
| 2 | Mean net P&L +1.66%/trade | **PROXY** | Same — label-proxy result, not model result. |
| 3 | Profitable opportunities exist (68.5% capture) | **VERIFIED** | Confirms opportunity set exists in data |
| 4 | Signal decay peaks T+3 | **PROXY** | Computed on proxy scores, not model scores |
| 5 | No .lgb model artifact found, fell back to proxy | **VERIFIED** | Root cause: model is `.pkl`, backtest searched for `.lgb` |
| 6 | "True model evaluation requires the actual model artifact" | **VERIFIED** | M1 inference now confirmed |

---

### MODEL_CALIBRATION_REPORT.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | ECE = 0.000 | **INVALID** | Isotonic regression fit on training data trivially achieves ECE ≈ 0 in-sample. True OOS calibration: severely compressed scores (std=0.0096), Brier = 0.206 |
| 2 | Calibration is "perfect" | **CONTRADICTED** | The calibrator DESTROYS score variance. OOS: raw std=0.0285 → calibrated std=0.0096. Calibrated IC (0.009) barely beats raw IC (−0.001). |
| 3 | Model output represents probabilities | **UNVERIFIED** | Scores range 0.28–0.62; distribution is not well-calibrated probability |

---

### LABEL_AUDIT.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | Mean net return = −0.287% at 10bps | **VERIFIED** | Confirmed from dataset |
| 2 | Break-even win rate = 56.9% | **VERIFIED** | Math confirmed |
| 3 | Label fails break-even test | **VERIFIED** | 48.94% positive rate < 56.9% |
| 4 | TIME_EXPIRY threshold wrong | **VERIFIED** | Fixed in `src/labels/seven_day.py` |
| 5 | 7-day labels recommended | **VERIFIED** | Implemented in `src/labels/seven_day.py` |

---

### LOOKAHEAD_BIAS_AUDIT.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | 0 INVALID shift(-N) in feature code | **VERIFIED** | Static audit confirms |
| 2 | No center=True patterns | **VERIFIED** | Confirmed |
| 3 | Intraday feature timestamps conditional | **UNVERIFIED** | Source timestamps not independently confirmed |
| 4 | News feature timestamps conditional | **UNVERIFIED** | SentinelPulse lineage not traced |
| 5 | Normalizer uses saved state in inference | **UNVERIFIED** | `FeatureNormalizer.load_state()` method missing; model pkl loads normalizer separately |

---

### FEATURE_ABLATION_REPORT.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | Returns/momentum are dominant features | **UNVERIFIED** | Based on training-time LightGBM feature importance, not OOS ablation |
| 2 | Top feature: ret_1 (importance 925) | **VERIFIED** | Confirmed from model artifact |
| 3 | Feature ablation is "static importance" | **VERIFIED** | Report correctly notes it's not OOS retraining |
| 4 | True OOS ablation not performed | **VERIFIED** | Documented as limitation |

---

### BASELINE_COMPARISON.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | ML beats random (64.2% vs 44.6%) | **PROXY** | Both numbers from P0 (proxy). With M1: true accuracy 52.3% vs random ~50% — gap nearly zero. |
| 2 | ML beats momentum | **PROXY** | Not verified for M1 |
| 3 | All baselines lose money after costs | **VERIFIED** | Confirmed |

---

### BEFORE_AFTER_MODEL_COMPARISON.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | New 7-day labels improve EV | **VERIFIED** | Math confirmed: break-even drops from 56.9% to ~35% with 2:1 R:R |
| 2 | Label EV old: −0.310%, new: −0.260% | **VERIFIED** | Computed on 10 sample symbols |
| 3 | Time periods covered are meaningful | **VERIFIED** | — |

---

### WALK_FORWARD_REPORT.md (existing, pre-audit)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | IC_rank = 0.4136 | **INVALID** | This is within-training walk-forward CV IC, not true OOS. True OOS IC = −0.001. Overfitting gap = 0.4149. |
| 2 | All 5 folds positive | **UNVERIFIED** | CV folds are within training period (2021–2026). No fold represents truly held-out future data. |
| 3 | PBO = 0.000 | **INVALID** | Computed as fraction of negative-Sharpe CV folds — not proper Combinatorial PBO. |
| 4 | ECE = 0.000 | **INVALID** | In-sample isotonic regression fit. |
| 5 | Live Sep 28: SHORT +0.655% net, 80% win rate | **STALE** | Regime-specific. Aggregate 3-day: 46% win rate. Oct 1 collapse invalidates. |

---

### FINAL_QUANT_CERTIFICATION.md (existing, 2026-09-29)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | "SHADOW PRODUCTION — 2 LIVE SESSIONS CONFIRMED" | **CONTRADICTED** | Oct 1 session shows model has no OOS predictive power. |
| 2 | G4: IC_continuous > 0.02 PASS (0.3757) | **CONTRADICTED** | True OOS IC = −0.001. The reported IC is within-training CV. |
| 3 | G5: CPCV PBO = 0.000 PASS | **INVALID** | Simplified PBO formula (fold-count), not proper CPCV. |
| 4 | G8: Calibration ECE < 0.10 PASS (0.000) | **INVALID** | In-sample isotonic. OOS calibration not measured. |
| 5 | "2-day avg +0.800% net, 80% win rate" | **STALE** | Based on 2 bear-market days. Not sustained. |
| 6 | G9 Net Sharpe > 0 PASS | **CONTRADICTED** | True M1 OOS EV = −0.090%/trade (negative) |
| 7 | G10/G11 PENDING → resolved Sep 30 | **CONTRADICTED** | Oct 1 forward paper: 23.8% win rate (catastrophic reversal). Gates should have FAILED. |

---

### PRODUCTION_READINESS_REPORT.md (existing, 2026-09-29)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | "AUTHORIZED FOR SHADOW" | **CONTRADICTED** | Based on false IC/performance claims. Model has no OOS alpha. |
| 2 | "Tests: 1,867 pass / 0 fail" | **VERIFIED** | Tests pass, but tests do not verify OOS model performance |
| 3 | "G7 DOUBLY CONFIRMED" | **CONTRADICTED** | Only 2 bear-market days. Not a valid regime test. |

---

### PORTFOLIO_EXECUTION_MODEL.md / PORTFOLIO_BACKTEST_REPORT.md (2026-10-01)

| # | Claim | Status | Evidence |
|---|-------|--------|---------|
| 1 | Win rate 77.1% (Mode B) | **PROXY** | Proxy-label mode (P0). M1 win rate < 50%. |
| 2 | Total return +61.3% | **PROXY** | P0 mode. NOT achievable by actual model. |
| 3 | Mode A/B distinction documented | **VERIFIED** | Documented in reports |
| 4 | Portfolio engine correctly implements execution | **VERIFIED** | 25 tests pass |

---

## Summary of Contradictions

| Report | Critical Contradictions |
|--------|------------------------|
| FINAL_QUANT_CERTIFICATION.md | IC claim, PBO claim, ECE claim, live performance claim |
| PRODUCTION_READINESS_REPORT.md | Authorization based on false metrics |
| WALK_FORWARD_REPORT.md | IC, PBO, ECE, live performance |
| ML_7_DAY_BACKTEST_REPORT.md | All performance numbers are P0 proxy, not M1 |
| PORTFOLIO_BACKTEST_REPORT.md | All returns are P0 proxy |
| BASELINE_COMPARISON.md | ML vs baselines based on P0 |

---

## Most Important Finding

**The claimed IC of 0.4136 and all downstream performance metrics (80% win rate, +0.8% live net, ECE=0.000) are based on within-training cross-validation, not a true held-out OOS period.**

The true OOS M1 metrics (2025 onwards) are:
- IC: −0.001 (not statistically significant)
- Binary accuracy: 52.3%
- EV: −0.090%/trade (negative)
- Direction: inverted (1−score outperforms score)

**The model demonstrates no genuine out-of-sample predictive power.**

---

*Generated: 2026-10-01 | Supersedes all previous certification claims.*
