# SURVIVORSHIP BIAS CERTIFICATION
**Repository:** ml-service2.0 | **Date:** 2026-10-01

---

## Verdict: FAIL — Historical F&O Universe Not Validated

The training universe uses today's NSE F&O eligible list applied backward to 2022. The `HistoricalUniverse` and `FnOStateStore` infrastructure is implemented but returns `DATA_UNAVAILABLE` for most historical dates.

---

## Evidence

### 7-Step PIT Validation Status

Step 3 (Historical Universe Membership) returns:
```
FO_ELIGIBILITY_UNKNOWN: Historical F&O eligibility for {symbol}
on {date} is DATA_UNAVAILABLE. Using current universe as approximation.
```

This warning fires for essentially ALL historical observations.

### Universe Size Comparison

| Check | Observation |
|-------|------------|
| Current F&O universe (2026) | ~285 symbols |
| 2022 F&O universe (estimated) | ~175-200 symbols |
| Symbols in training dataset | 279 |
| F&O symbols added since 2022 | ~80-100 (estimate) |
| Affected training rows (est.) | 30-40% of 263,709 |

### Known F&O Universe Changes (2022–2026)

Symbols that entered F&O after training data starts (incomplete list):
- JIOFIN: Added to F&O ~2023 (post-listing)
- ZOMATO: Added ~2023
- ADANIENSOL: Added ~2024
- ATGL: Added ~2024
- Several others

These symbols appear in the training data before they were F&O eligible, which means:
1. Their inclusion in the training universe is historically impossible
2. Any historical data for them pre-eligibility is not tradable
3. Model may have learned patterns from non-tradable history

---

## Quantification of Affected Data

Without the complete historical F&O eligibility database, we cannot precisely quantify affected rows. Conservative estimate based on known additions:

```
~15-20% of training symbols may have partial pre-eligibility observations
~5-10% of all training rows may represent non-tradable history
Impact on IC: Unknown (may inflate or deflate — depends on whether
              pre-eligibility stocks had distinctive return patterns)
```

---

## Remediation

1. **Build complete historical F&O eligibility dataset** from NSE circulars and SEBI records
2. **Implement effective HistoricalUniverse** with real DATA instead of DATA_UNAVAILABLE
3. **Filter training data** to include only rows where symbol was F&O eligible on that date
4. **Re-evaluate IC** after filtering — if IC changes materially, prior results are compromised

Until completed, all performance metrics carry a survivorship bias flag. The magnitude of impact is unknown but likely non-trivial for the 2022–2023 period.

---

## Gate Status

**G_SURVIVORSHIP: FAIL**  
Reason: Historical eligibility not validated for any observation in training data. Impact unquantified.
