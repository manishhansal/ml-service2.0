# PROBABILITY OF BACKTEST OVERFITTING (PBO) REPORT
**Repository:** ml-service2.0 | **Date:** 2026-10-01

---

## Old PBO Method — INVALID

The existing `_compute_pbo()` in `src/training/pipeline.py`:
```python
def _compute_pbo(fold_sharpes: list[float]) -> float:
    n_negative = sum(1 for s in fold_sharpes if s < 0)
    return float(n_negative / len(fold_sharpes))
```

This counts the **fraction of CV folds with negative Sharpe**, not the probability of backtest overfitting. With 0 of 5 folds negative, it returns 0.000 — which is NOT PBO in the López de Prado (2018) sense.

**The G5 gate certification ("PBO < 0.50 PASS: 0.000") is based on this invalid formula.**

---

## Proper PBO via Combinatorial Purged CV

PBO is defined as the fraction of simulated backtest paths where the selected strategy underperforms out-of-sample, calculated over all C(N,K) combinations of K test groups from N total groups.

### Implementation Results

```
Dataset:       ds-1d-20260926202149 (training period, 2021–2026)
Method:        C(6,2) = 15 paths (n_groups=6, k_test=2)
Strategy:      signal = (m1_score − 0.5) × realized_return (Spearman proxy)

IS  Sharpe: mean = 1.1443, std = 0.1912
OOS Sharpe: mean = 1.1371, std = 0.3825
PBO_within_training = 0.0000 (0/15 paths with OOS Sharpe < 0)
```

### Why PBO_within_training = 0 Does NOT Mean No Overfitting

The CPCV paths are computed **within the training dataset** (2021–2026). Even with test groups held out, the model was ultimately selected and trained on all of this data. The C(N,K) paths all share the same in-sample period.

The correct interpretation of CPCV is that it reduces (but cannot eliminate) overfitting from strategy selection. It does NOT test a true held-out future period.

```
CPCV within training:  PBO = 0.000  (0/15 paths negative Sharpe)
True OOS (2025 +):     IC  = −0.001 (statistically not significant)
True OOS win rate:     47.0% (below random)
```

The CPCV says "within the training period, the model has consistent positive Sharpe." The true OOS period says the model collapsed. This collapse is not visible to CPCV because it only looks backward.

---

## Deflated Sharpe Ratio

For a proper overfitting assessment, the Deflated Sharpe Ratio (DSR) should be reported:
```
Observed Sharpe (WF within-training):    +1.076
Annualized (√252 factor):                applied
Number of trials:                         ≥50 (Optuna)
Autocorrelation:                          unknown
DSR estimate:                             < 0 (the selection bias from 50+ HPO trials 
                                           inflates the nominal Sharpe beyond what chance expects)
```

A formal DSR calculation requires recording all trial Sharpe values from Optuna. This is not currently saved in the model artifacts. **Recommendation:** Log all trial Sharpe values to MLflow for future DSR computation.

---

## Summary

| Metric | Old Method | Proper Method | Assessment |
|--------|-----------|--------------|------------|
| PBO | 0.000 (fold-count) | 0.000 (CPCV within-training) | Both within training only |
| True test | Not performed | IC = −0.001 in 2025+ | OVERFITTING CONFIRMED |
| DSR | Not computed | Likely < 0 | OVERFITTING LIKELY |

**Gate G5 should be FAIL:** The 0.000 PBO from fold-counting is trivially achievable by any model that overfits the training period. Proper independent OOS testing is the only valid overfitting test, and it clearly shows model collapse.
