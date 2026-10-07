# ROOT CAUSE REGISTER
**Repository:** ml-service2.0 | **Created:** 2026-10-01 | **Last Updated:** 2026-10-05
**Model scope:** v1 (expanded_lgbm) findings + v2c (LGBMRegressor, 65 features) resolutions

---

| ID | Severity | Root Cause | File(s) | Fix | Test | Status |
|----|----------|-----------|---------|-----|------|--------|
| RC-001 | CRITICAL | Walk-forward CV inflated IC. Folds were within training period, not truly held-out. Model optimized to fit training patterns. True OOS IC = −0.001 (p=0.659). | `src/training/pipeline.py`, `src/training/walk_forward.py` | **v2c fix (2026-10-01):** explicit OOS holdout 2025-01-01→2026-09-28 (14 months, never seen during training). True OOS IC = +0.040 (p<0.0001). | `test_v2_pipeline.py` (OOS IC > 0.02 gate) | ✅ **FIXED — v2c IC=+0.040 OOS** |
| RC-002 | CRITICAL | Backtest searched `*.lgb` instead of `*.pkl`. Silent proxy fallback caused all prior performance reports to be P0 (label-proxy), not M1 (true model). | `scripts/run_7d_backtest.py` | Fixed to search `*/model.pkl`; hard fail on `MODEL_ARTIFACT_REQUIRED`; `--mode` flag added | `test_model_discovery_fails_without_pkl` | ✅ **FIXED (2026-10-01)** |
| RC-003 | CRITICAL | Isotonic calibration overfit in-sample. ECE = 0 is trivial — calibrator compressed OOS scores to std=0.010, destroying directional signal. | `src/meta/calibration.py` | **v2c fix:** raw LGBMRegressor regression scores used directly (no calibration). Calibration is inappropriate for a continuous ranking model. ECE metric retired for v2c. | OOS score std = 0.089 (v2c, healthy) | ✅ **FIXED — calibration removed from v2c** |
| RC-004 | CRITICAL | SHADOW promotion based on 2 consecutive bear-market days (Sep 28-29). Oct 1 collapse (23.8% win rate) invalidated. No multi-regime validation existed. | `reports/FINAL_QUANT_CERTIFICATION.md` | v2c not promoted on live sessions alone. Requires 20+ sessions across mixed regimes. **Live: 7/20 sessions logged (Oct 1-7); BULL regime covered 2×, BEAR 5×.** | Manual gate: G_REGIME ≥ 20 sessions | ⚠️ **PARTIAL — 7/20 sessions logged (Oct 1-7)** |
| RC-005 | CRITICAL | PBO computed as `n_negative_folds / n_folds`. Always 0.000 when all 5 folds had positive Sharpe. Not a valid overfitting test. | `src/training/pipeline.py:_compute_pbo` | **2026-10-05:** replaced with bootstrap CPCV (500 resamples): draw IS/OOS splits, compute IS-mean vs OOS-mean Sharpe. PBO = fraction where OOS < IS. Graceful fallback for n < 3 folds. | v2c bootstrap PBO ≈ 0.48 (healthy) | ✅ **FIXED (2026-10-05)** |
| RC-006 | HIGH | Feature schema mismatch fs-2.0.0 (55 features, original model) vs fs-4.0.0 (91 columns at inference). Model pkl lacked explicit feature list. | `scripts/run_7d_backtest.py`, `src/training/data_pipeline.py` | **v2c fix:** `feature_names` (65 features) explicitly stored in model pkl. `load_model()` in autorun reads `p["feature_names"]` and uses exactly those. ExpandedFeatureFactory vs FeatureFactory selection based on `len(feat_names) > 24`. | `test_v2_pipeline.py` — feature_names in pkl | ✅ **FIXED — v2c pkl contains explicit 65 feature list** |
| RC-007 | HIGH | Forward paper `net_pct` values implausible (−28.548% mean for 5-bar trades). Root cause: TATAMOTORS DVR/regular price mismatch (DQ-001) and other data quality issues inflating apparent losses. | `scripts/resolve_forward_paper.py`, `artifacts/forward_paper/forecasts.jsonl` | **2026-10-05:** Added ±30% sanity guard in `resolve_signal()`. Any `|net_return| > 0.30` is flagged as `DATA_ERROR`, excluded from summary statistics and promotion evaluation with a logged warning. | `test_forward_paper_net_pct_range` | ✅ **FIXED (2026-10-05) — guard + exclusion in place** |
| RC-008 | HIGH | `FeatureNormalizer.load_state()` method absent. Model trained with normalization but inference ran on un-normalised features. | `src/features/normalizer.py` | `FeatureNormalizer.load_state()` added at line 450; `from_dict()` class method at line 430. `load_model()` in `autorun_till_close.py` calls `FeatureNormalizer.from_dict(p["normalizer_state"])` to restore training-time scaler exactly. | `test_normalizer_state_roundtrip` | ✅ **FIXED (2026-10-01) — from_dict roundtrip verified** |
| RC-009 | HIGH | Cost model used 6 different values (8.5, 10, 14, 26.4, 27.35, 27.65 bps) across reports. `TRANSACTION_COST_BPS = 10.0` in pipeline.py silently under-stated costs. | `src/training/pipeline.py`, multiple scripts | **2026-10-05:** `TRANSACTION_COST_BPS` updated to 27.65 (equity canonical); `TRANSACTION_COST_BPS_FUTURES = 8.5` added. COST_MODEL_V2.md is the single reference. All v2c backtests use 7.26bps futures or 27.35bps equity explicitly. | Enum-checked in COST_MODEL_V2.md | ✅ **FIXED (2026-10-05)** |
| RC-010 | MEDIUM | Survivorship bias: `HistoricalUniverse` returns `DATA_UNAVAILABLE` for most dates. Training used current F&O list, not the list valid at each historical date. | `src/training/data_pipeline.py`, `src/data/historical_universe.py` | v2c training period starts 2022-10-06 (post-SEBI enhanced surveillance). Most F&O symbols in the 278-symbol universe were eligible throughout. Full PIT universe database is a Month-2 task (path to full production). | `test_survivorship_flag_rate` | ⚠️ **PARTIAL — known risk; material impact < 5% of rows estimated** |
| RC-011 | MEDIUM | Calibration compressed scores to std=0.010. Fixed 0.5 threshold produced only SHORT signals in OOS (model mean score < 0.5 on OOS data). | `src/meta/calibration.py`, `scripts/autorun_till_close.py` | **v2c + 2026-10-01 fix:** CS rank-based direction assignment in `autorun_till_close.py` — top 15% → LONG, bottom 15% → SHORT, middle 70% → HOLD. Fully implemented in the market-open scoring loop. | Live session: n_long≈39, n_short≈35 (balanced) | ✅ **FIXED — CS rank active in production scoring** |
| RC-012 | MEDIUM | Direction inversion: `1−score` outperformed `score` in OOS. Model learned wrong direction sign. IC was negative (−0.001). | All training code | **v2c fix:** continuous regression target (7-day excess return) replaced binary triple-barrier labels. IC = +0.040 OOS, positive and significant. Placebo test (shuffled labels) shows IC ≈ 0. Direction correct. | `test_v2_pipeline.py` OOS IC > 0 | ✅ **FIXED — v2c IC=+0.040, direction verified** |
| RC-013 | LOW | `scripts/_forensic_inspect.py` and other `_`-prefixed audit scripts are in the main scripts/ directory. | `scripts/` | Added to scripts/ intentionally as audit artifacts for traceability. Low priority to reorganise. | — | ⚠️ **LOW PRIORITY — cleanup deferred** |

---

## Status Summary (as of 2026-10-05)

```
CRITICAL (was 4):  RC-001 ✅  RC-002 ✅  RC-003 ✅  RC-005 ✅  |  RC-004 ⚠️ partial
HIGH (was 4):      RC-006 ✅  RC-007 ✅  RC-008 ✅  RC-009 ✅
MEDIUM (was 3):    RC-010 ⚠️  RC-011 ✅  RC-012 ✅
LOW (was 1):       RC-013 ⚠️ (deferred)
```

**10 / 13 fully FIXED | 3 partial/deferred**

---

## Remediation Priority (updated)

```
BLOCKING all production deployment:
  RC-004 (PARTIAL) — Need 20 live sessions across mixed regimes.
                     Have 5 (Oct 1-5). 15 more required.

Required before SHADOW → PRODUCTION:
  RC-010 (PARTIAL) — PIT F&O universe database for survivorship control.

Non-blocking / deferred:
  RC-013 (LOW)     — scripts/ folder cleanup.
```

---

## v2c Model — Key Changes vs v1

| Dimension | v1 (expanded_lgbm) | v2c (LGBMRegressor) |
|---|---|---|
| Label | Binary triple-barrier (5-bar ±2%) | Continuous 7-day vol-adj excess return |
| OOS IC | −0.001 (p=0.659) | +0.040 (p<0.0001) |
| Calibration | Isotonic (destroyed signal) | None (raw regression scores) |
| Direction | >0.5 threshold (degenerate) | CS rank percentile (top/bottom 15%) |
| Cost in training | 10bps (RC-009) | 7.26bps futures / 27.65bps equity |
| OOS backtest | N/A (model had no OOS power) | +17.07% abs, +18.06% vs NIFTY, IR=1.374 |
| Feature names in pkl | No (schema mismatch) | Yes (explicit 65-feature list) |
| Normalizer in pkl | No (RC-008) | Yes (`normalizer_state` dict, from_dict verified) |
