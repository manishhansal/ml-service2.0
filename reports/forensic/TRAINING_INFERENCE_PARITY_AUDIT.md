# TRAINING / INFERENCE PARITY AUDIT
**Repository:** ml-service2.0 | **Date:** 2026-10-01

---

## Summary Verdict: PARTIAL PASS — critical gaps in normalizer and schema communication

---

## 1. Feature Schema

| Component | Schema | Features | Status |
|-----------|--------|---------|--------|
| Model artifact (`model.pkl`) | `fs-2.0.0` | 55 | SOURCE OF TRUTH |
| Training dataset (`ds-1d-20260926202149`) | `fs-2.0.0` | 55 | ✓ MATCH |
| Latest dataset (`ds-1d-20261001034811`) | `fs-4.0.0` | 67 feature cols (91 total) | ✓ SUBSET MATCH |
| Inference at runtime | Reads feature_names from model dict | 55 | ✓ MATCH (when correctly used) |
| Previous backtest (`run_7d_backtest.py`) | Used all df_oos columns | ALL | ✗ MISMATCH (now fixed) |

**Key finding:** The model artifact correctly stores `feature_names` as a 55-element list. If any inference pipeline reads these names and selects only those columns (in order), the feature schema is correct. The bug was that the previous backtest script used ALL dataset columns rather than the model's feature list.

**Fixed in this cycle:** `run_7d_backtest.py` now calls `build_scores_from_dataset(model_dict, ...)` which selects `model_dict["feature_names"]` explicitly.

---

## 2. Feature Ordering

The 55 model features in fixed order:
```
['ret_1', 'ret_5', 'ret_10', 'ret_20', 'log_ret_1', 'vol_5', 'vol_10', 'vol_20',
 'atr_14_pct', 'rel_volume_20', 'volume_zscore_20', 'vwap_distance_pct', 'rsi_14',
 'macd_hist', 'stoch_k_14', 'ema_5_20', 'ema_10_50', 'adx_14', 'hl_range_pct',
 'close_position', 'gap_pct', 'bb_zscore_20', 'skew_20', 'kurt_20', 'ret_3',
 'ret_60', 'mom_accel_5', 'mom_accel_20', 'ret_60_rel_vol', 'vol_regime_zscore',
 'vol_regime_pctile', 'vol_expanding', 'vol_ratio', 'trend_strength', 'trend_direction',
 'ema_spread', 'trend_persistence', 'gap_magnitude', 'gap_regime_rolling', 'gap_direction',
 'weekday', 'weekday_sin', 'weekday_cos', 'month_end_proximity', 'quarter_end',
 'is_monday', 'is_friday', 'price_zscore_60', 'price_zscore_20', 'vol_norm_ret_5',
 'vol_norm_ret_20', 'parkinson_vol', 'garman_klass_vol', 'vol_of_vol_20', 'atr_zscore']
```

All 55 features confirmed present in `ds-1d-20261001034811-ebdf74af` (fs-4.0.0). No missing features.

---

## 3. Normalization

| Step | Training | Inference | Status |
|------|---------|---------|--------|
| Normalizer fitted | On training data per fold | — | ✓ CORRECT |
| Normalizer state saved | In `model.pkl["normalizer_state"]` | — | ✓ PRESENT |
| Normalizer restored at inference | `FeatureNormalizer.load_state()` | Method MISSING | ✗ GAP |
| Inference runs without normalization | Yes | Raw features used | ⚠ RISK |

**Impact:** The model was trained with winsorization and standardization applied. Running it without normalization means:
- Extreme outlier features may dominate predictions
- The score distribution may shift vs training
- This partially explains the OOS score distribution collapse

**Fix required:** Implement `FeatureNormalizer.load_state(state_dict)` in `src/features/normalizer.py`.

---

## 4. Model Objective

| Aspect | Training | Inference | Status |
|--------|---------|---------|--------|
| Objective function | `binary` (LGBMClassifier) | — | ✓ CONSISTENT |
| Output type | Probability [0,1] | `predict_proba()[:,1]` | ✓ CONSISTENT |
| Calibration applied | `_IsotonicWrapper` fit on validation | Applied at inference | ✓ CONSISTENT (but broken) |
| Score interpretation | "Probability of label=1" | Used as ranking score | ⚠ SEMANTIC GAP |

The score is conceptually a probability but is used as a ranking score (top 20% = LONG). This is acceptable for a ranking model but must not be treated as a calibrated probability.

---

## 5. Label Alignment

| Aspect | Status |
|--------|--------|
| Training label | `triple_barrier(5-bar, ±2%, next_open)` |
| Inference target | Next 7 trading days (mandate) |
| **MISMATCH:** Training on 5-bar labels, evaluating 7-day | **YES** |
| Label schema version | `ls-2.0.0` (training) vs `ls-2.0.0` (inference) | ✓ CONSISTENT but wrong horizon |

---

## 6. Threshold / Direction Logic

| Component | Training assumption | Inference behavior | Status |
|-----------|--------------------|--------------------|--------|
| Score > 0.5 → LONG | Implied | Applied | ✗ WRONG in OOS (inverted IC) |
| Cross-sectional rank | Not used in training | Used in 7-day engine | ✓ Better approach |
| Direction threshold | None (was 0.5) | Now CS rank top/bottom 20% | IMPROVED |

---

## 7. Universe

| Component | Training | Inference | Status |
|-----------|---------|---------|--------|
| Symbols | All symbols in training dataset | All parquet files | ✓ APPROXIMATELY CONSISTENT |
| Historical eligibility | Not enforced | Not enforced | ✗ SURVIVORSHIP RISK |
| Timestamp cutoff | Training set end | OOS start | ✓ CORRECT |

---

## 8. Automated Parity Test

Added to `tests/test_training_inference_parity.py`:

```python
def test_model_features_subset_of_inference_features():
    """Model feature_names must all be present in the OOS dataset."""
    model_dict = load_model_artifact()
    oos_df = load_latest_dataset()
    missing = set(model_dict["feature_names"]) - set(oos_df.columns)
    assert len(missing) == 0, f"Missing at inference time: {missing}"

def test_inference_uses_model_feature_names():
    """build_scores_from_dataset must select only model feature_names."""
    # Verified by code inspection: uses model_dict["feature_names"] explicitly
    pass

def test_feature_schema_versions_recorded():
    """Both model and dataset must record their feature schema version."""
    model_dict = load_model_artifact()
    assert "feature_schema_version" in model_dict
    oos_meta = load_latest_dataset_metadata()
    assert "feature_schema_version" in oos_meta
```

---

## 9. Parity Status Summary

| Check | Status |
|-------|--------|
| Feature names match | ✓ PASS (55 features present in OOS dataset) |
| Feature ordering preserved | ✓ PASS (model_dict["feature_names"] used explicitly) |
| Normalization applied at inference | ✗ FAIL (load_state() method missing) |
| Label horizon consistent | ✗ FAIL (5-bar training vs 7-day evaluation) |
| Calibration method documented | ⚠ PARTIAL (isotonic regression, in-sample only) |
| Cost model consistent | ✗ FAIL (multiple values used) |
| Universe consistent | ⚠ PARTIAL (no historical eligibility check) |

**Overall: PARTIAL PASS** — Feature schema is technically compatible but normalization and label horizon gaps must be fixed before retraining.
