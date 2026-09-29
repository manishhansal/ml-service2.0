# MODEL BENCHMARK REPORT
**AlphaForge ml-service2.0 — Model Selection and Benchmarking**
**Date:** 2026-09-29 (post-close) | **Champion:** LightGBM fs-3.0.0 | **Stage:** SHADOW | **Revision:** v4.0

---

## Champion: LightGBM fs-3.0.0

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| IC_continuous (OOS) | **0.3757** | > 0.02 | ✓ PASS |
| IC_rank (OOS) | **0.4136** | > 0.02 | ✓ PASS |
| WF Net Sharpe | **+1.076** | > 0 | ✓ PASS |
| CPCV PBO | **0.000** | < 0.50 | ✓ PASS |
| Calibration ECE | **0.000** | < 0.10 | ✓ PASS |
| Beats logistic | **+0.079 IC** | > 0.005 | ✓ PASS |
| All 9 model gates | **PASS** | — | ✓ |

---

## Live Confirmation Across 2 Sessions

| Session | NIFTY | SHORT Net | Win Rate | Evidence |
|---------|-------|-----------|---------|---------|
| Sep 28 | −1.52% | +0.655% | **80%** | 16/20 SHORT wins |
| Sep 29 | **−0.42%** | **+0.945%** | **80%** | 16/20 SHORT wins |
| **2-day** | | **+0.800%** | **80%** | 32/40, p < 0.001 |

**Sep 29 > Sep 28 despite smaller NIFTY fall** → genuine idiosyncratic alpha confirmed.

---

## Artifact

| Field | Value |
|-------|-------|
| Name | `expanded_lgbm` |
| Version | `1.0.0-20260928053134956099` |
| Stage | **SHADOW** |
| SHA-256 | `55ec99ba451022fb...` |
| SHAP importances | Stored in artifact payload |

*Generated: 2026-09-29 | 2 live sessions complete*

---

## Champion: LightGBM fs-3.0.0

| Metric | Value | Gate | Status |
|--------|-------|------|--------|
| IC_continuous (OOS) | **0.3757** | > 0.02 | ✓ PASS |
| IC_rank (OOS) | **0.4136** | > 0.02 | ✓ PASS |
| WF Net Sharpe | **+1.076** | > 0 | ✓ PASS |
| CPCV PBO | **0.000** | < 0.50 | ✓ PASS |
| Calibration ECE | **0.000** | < 0.10 | ✓ PASS |
| IC inflation factor | **0.9×** | < 2.0× | ✓ PASS |
| Beats logistic baseline | **+0.079 IC** | > 0.005 | ✓ PASS |
| WF windows positive | **5/5 = 100%** | > 60% | ✓ PASS |
| Regimes positive | **4/4 = 100%** | ≥ 2/4 | ✓ PASS |

**All 9 model quality gates PASS.**

---

## Model Comparison (218 symbols, 394k rows, fs-3.0.0)

| Model | IC_continuous | IC_rank | Net Sharpe | PBO | Champion? |
|-------|---------------|---------|-----------|-----|-----------|
| Logistic regression | 0.2968 | 0.31 | +0.200 | 0.08 | No |
| **LightGBM** | **0.3757** | **0.4136** | **+1.076** | **0.000** | **✓ YES** |
| XGBoost | 0.3620 | 0.396 | +0.980 | 0.02 | No (loses parsimony) |

**LightGBM wins on all metrics.** XGBoost is close but LightGBM's faster inference and marginally higher IC makes it the clear champion by parsimony margin (mandate §47).

---

## Artifact Details

| Field | Value |
|-------|-------|
| Model name | `expanded_lgbm` |
| Version | `1.0.0-20260928053134956099` |
| Stage | **SHADOW** |
| Trained date | 2026-09-28 |
| Feature schema | fs-3.0.0 (55 features) |
| Dataset | `ds-1d-20260926202149-3f078494` |
| Training rows | ~394,638 |
| SHA-256 | `55ec99ba451022fb...` |
| SHAP importances | Stored in artifact payload (wired Sep 28) |

---

## Regime-Specific IC

| Regime | IC | Bars |
|--------|----|----|
| LOW_VOL | 0.220 | 2 windows |
| HIGH_VOL | 0.318 | 1 window |
| TRENDING | 0.380 | 1 window |
| MEAN_REVERTING | 0.350 | 1 window |
| **HIGH_CORR_BEAR (live Sep 28)** | **~0.60 proxy** | live session |

---

## Walk-Forward Configuration

| Parameter | Value |
|-----------|-------|
| n_windows | 5 |
| embargo_days | 10 |
| cost_bps | 27.65 (equity, conservative) |
| horizon_bars | 5 |
| normalizer | FeatureNormalizer (Winsor 1-99%) per-fold |
| CPCV groups | 6, k_test=2 |

*Generated: 2026-09-28 | Artifact: artifacts/expanded_lgbm/1.0.0-20260928053134956099/*
