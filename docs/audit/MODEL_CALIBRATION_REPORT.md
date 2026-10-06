# MODEL CALIBRATION REPORT
**Repository:** ml-service2.0 | **Date:** 2026-10-01

## 1. Calibration Claim vs Evidence

The existing reports claim **ECE = 0.000** (perfect calibration). This is suspicious and requires scrutiny.

## 2. What ECE = 0.000 Means

Expected Calibration Error (ECE) measures whether predicted probabilities match observed frequencies:
- ECE = 0 means: when the model says "70% probability of success", exactly 70% of those signals succeed
- ECE = 0.000 is essentially impossible for any real ML model without deliberate calibration fitting

Possible explanations for ECE = 0.000:
1. ECE computed on training data (in-sample) — trivial with a good fit
2. ECE computed on a test set that was inadvertently used for threshold/calibration optimization
3. The calibration evaluation uses a binned estimate with a single bin (trivially zero)
4. The model output was post-processed with perfect isotonic regression
5. Measurement error or implementation bug

## 3. Calibration Assessment from Available Evidence

The model generates scores in [0, 100] (normalised rank scores). These are NOT probabilities — they are rank positions within the universe. The statement "ECE = 0.000" for a rank score is meaningless because rank scores are not probabilities.

If calibration was computed by: ECE = |fraction_above_threshold − predicted_probability|, and the threshold was set to 50 (median rank), then by construction of the normalisation:
- Fraction above 50: exactly 50%
- Predicted "probability" at score=50: 50%
- ECE = |50% − 50%| = 0 trivially

**Conclusion:** The ECE = 0.000 claim likely reflects the normalisation artifact, not genuine probability calibration.

## 4. Proper Calibration Measurement

To properly calibrate the model:

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Step 1: Compute calibration curve on VALIDATION data only
fraction_pos, mean_predicted = calibration_curve(
    y_val, model.predict(X_val), n_bins=10
)

# Step 2: Compute ECE
ece = sum(
    abs(fraction_pos[i] - mean_predicted[i]) * n_i / n_total
    for i, n_i in enumerate(bin_counts)
)

# Step 3: Apply calibration if ECE > 0.05
from sklearn.isotonic import IsotonicRegression
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(model.predict(X_val), y_val)
calibrated_score = calibrator.transform(model.predict(X_test))
```

This calibration must be fit ONLY on validation data, never on test data.

## 5. Confidence Bucket Analysis (Proxy)

From the proxy backtest (label-proxy scores), confidence-stratified win rates:

| Confidence | Win Rate | Above Break-Even (56.9%) |
|-----------|---------|------------------------|
| 0–20% | ~60% | ✓ Yes |
| 20–40% | ~62% | ✓ Yes |
| 40–60% | ~64% | ✓ Yes |
| 60–80% | ~66% | ✓ Yes |
| 80–100% | ~68% | ✓ Yes |

These are proxy results — actual model confidence calibration requires the trained artifact and validation data.

## 6. Recommendations

1. **Re-compute ECE** with proper binned calibration on validation data
2. **Generate reliability diagram** (predicted probability vs observed frequency)
3. **Apply isotonic calibration** after verifying ECE > 0.05 on validation data
4. **Never use ECE = 0.000 as a production gate** until independently verified
5. **Use Brier score** as an additional calibration metric (already partly implemented in `ForecastLedger`)
