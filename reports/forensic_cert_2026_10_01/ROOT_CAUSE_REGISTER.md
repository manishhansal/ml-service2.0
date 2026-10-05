# ROOT CAUSE REGISTER
**Repository:** ml-service2.0 | **Date:** 2026-10-01

---

| ID | Severity | Root Cause | File(s) | Fix | Test | Status |
|----|----------|-----------|---------|-----|------|--------|
| RC-001 | CRITICAL | Walk-forward CV inflated IC. Folds were within training period, not truly held-out. Model optimized to fit training patterns. | `src/training/pipeline.py`, `src/training/walk_forward.py` | Separate final test period; evaluate IC on completely unseen future data | `test_true_oos_ic_below_threshold` | OPEN — requires retraining |
| RC-002 | CRITICAL | Backtest searched `*.lgb` instead of `*.pkl`. Silent proxy fallback. | `scripts/run_7d_backtest.py` | Fixed to search `*/model.pkl`; hard fail on `MODEL_ARTIFACT_REQUIRED` | `test_model_discovery_fails_without_pkl` | FIXED |
| RC-003 | CRITICAL | Isotonic calibration overfit in-sample. ECE = 0 is trivial. | `src/meta/calibration.py` | Use isotonic only on a dedicated calibration holdout; validate ECE on separate future period | `test_calibration_ece_on_holdout` | OPEN |
| RC-004 | CRITICAL | SHADOW promotion based on 2 bear-market days. No multi-regime validation. | `reports/FINAL_QUANT_CERTIFICATION.md` | Require min 20 trading days across BULL, BEAR, SIDEWAYS | `test_regime_validation_min_days` | OPEN |
| RC-005 | CRITICAL | PBO computed as fraction of negative-Sharpe folds. Not valid CPCV. | `src/training/pipeline.py:_compute_pbo` | Implement proper CPCV combinatorics | `test_pbo_uses_cpcv_not_fold_count` | OPEN |
| RC-006 | HIGH | Feature schema mismatch fs-2.0.0 vs fs-4.0.0. Inference possible but requires explicit subset selection. | `scripts/run_7d_backtest.py`, `src/training/data_pipeline.py` | Always record `feature_names` in model pkl (done) and use those at inference (partially done) | `test_inference_uses_model_feature_names` | PARTIAL |
| RC-007 | HIGH | Forward paper `net_pct` values implausible (−28.548% mean for 5-bar trades). Unit/sign error. | `scripts/resolve_forward_paper.py`, `artifacts/forward_paper/forecasts.jsonl` | Audit and reproduce; see FORWARD_PAPER_PNL_FORENSIC_AUDIT.md | `test_forward_paper_net_pct_range` | OPEN |
| RC-008 | HIGH | `FeatureNormalizer.load_state()` method absent. Model trained with normalization but inference runs without it. | `src/features/normalizer.py` | Add `load_state()` method; document normalization contract | `test_normalizer_state_roundtrip` | OPEN |
| RC-009 | HIGH | Cost model used 6 different values (8.5, 10, 14, 26.4, 27.35, 27.65 bps) across reports. | Multiple scripts and reports | Standardize to `NSECostModelV2` (see COST_MODEL_V2.md) | `test_all_backtests_use_same_cost_version` | OPEN |
| RC-010 | MEDIUM | Survivorship bias: HistoricalUniverse returns DATA_UNAVAILABLE for most dates. Training used current F&O list. | `src/training/data_pipeline.py`, `src/data/historical_universe.py` | Quantify affected rows; add survivorship flag | `test_survivorship_flag_rate` | OPEN |
| RC-011 | MEDIUM | Calibration compresses scores to std=0.010. Direction threshold (>0.5) generates only SHORT signals in OOS. | `src/meta/calibration.py`, `scripts/autorun_till_close.py` | Use cross-sectional rank percentile for direction, not fixed threshold | `test_signal_direction_not_degenerate` | PARTIAL (7-day engine uses CS rank) |
| RC-012 | MEDIUM | Direction inversion: 1−score outperforms score in OOS. Model learned wrong direction sign or feature orientation changed. | All training code | Re-examine feature sign conventions; retrain with OOS validation of direction | `test_signal_direction_positive_ic` | OPEN |
| RC-013 | LOW | `scripts/_forensic_inspect.py`, `_run_label_comparison.py` etc. use `_` prefix and should be cleaned up. | `scripts/` | Move to a dedicated `scripts/audit/` subfolder | — | LOW PRIORITY |

---

## Remediation Priority

```
P0 (Block all production):  RC-001, RC-003, RC-004, RC-005
P1 (Required for shadow):   RC-002 (FIXED), RC-007, RC-008, RC-011
P2 (Required before live):  RC-006, RC-009, RC-010, RC-012
```
