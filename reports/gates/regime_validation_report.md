# G_REGIME Validation Report
**Model:** v2c (LGBMRegressor, 65 features, 7-day CS rank label)
**OOS Period:** 2025-01-01 → 2026-09-28 (14 months)   *(ISSUE-07: corrected from Sep 21)*
**Generated:** 2026-10-05 15:34 UTC | **Last updated:** 2026-10-07

## Gate Verdict: ⚠️ PARTIAL   *(ISSUE-01: corrected from ✅ PASS)*

> **Note on IC values (ISSUE-02):** Two different IC metrics appear across documents:
> - **+0.0178** (this report) = *portfolio-level time-series IC* — correlation of daily strategy
>   return vs universe, computed per observation-row across the 14-month OOS window.
> - **+0.040** (PRODUCTION_GATES, REMEDIATION_BACKLOG) = *cross-sectional Spearman rank IC* —
>   per-day Spearman correlation between model scores and 7-day forward returns, averaged
>   across 116k symbol×date observations. Both are genuine OOS metrics of the same model;
>   they measure different things and are not directly comparable.

Overall OOS IC (portfolio time-series) = **+0.0178** (p=1.2073e-09, n=116,178 observations)

## Regime-Split IC

| Regime | N | IC | p-value | Verdict |
|--------|---|----|---------|----|
| BULL | 16,666 | -0.0174 | 2.4712e-02 | ❌ FAIL |
| BEAR | 19,723 | +0.0337 | 2.0000e-06 | ✅ PASS |
| SIDEWAYS | 79,789 | +0.0213 | 0.0000e+00 | ✅ PASS |

**BULL regime has statistically significant *negative* IC (−0.017, p=0.025).** The model is a
BEAR/SIDEWAYS detector. In pure bull markets, cross-sectional momentum dispersion collapses
and the model generates negative-alpha signals. **BULL suppressor added to
`strategy/feature_weights.json` on 2026-10-06** (ALL.multiplier=0.3, max_positions=4).

## Volatility-Split IC

| Vol Regime | N | IC | p-value | Significant? |
|---|---|---|---|---|
| HIGH_VOL | 29,045 | +0.0285 | 1.0000e-06 | ✅ Yes |
| MID_VOL | 58,088 | +0.0075 | 7.0063e-02 | ⚠️ No (p=0.07) |
| LOW_VOL | 29,045 | +0.0224 | 1.3900e-04 | ✅ Yes |

## Quarterly IC (Decay Check)

| Quarter | N | IC | p-value | Significant? |
|---|---|---|---|---|
| 2025-Q1 | 16,730 | +0.0150 | 5.2519e-02 | ⚠️ Borderline |
| 2025-Q2 | 16,531 | +0.0212 | 6.4580e-03 | ✅ Yes |
| 2025-Q3 | 17,343 | -0.0060 | 4.3071e-01 | ❌ No (p=0.43) |
| 2025-Q4 | 16,915 | +0.0178 | 2.0336e-02 | ✅ Yes |
| 2026-Q1 | 16,554 | +0.0465 | 0.0000e+00 | ✅ Yes |
| 2026-Q2 | 16,584 | +0.0042 | 5.8836e-01 | ❌ No (p=0.59) |
| 2026-Q3 | 15,521 | +0.0260 | 1.1960e-03 | ✅ Yes |

> **ISSUE-08 disclosure:** 2 of 7 quarters (2025-Q3 and 2026-Q2) have non-significant IC.
> Alpha consistency risk: the model is not uniformly predictive across all time periods.
> The 2026-Q3 result (+0.0260) is reassuring for current deployment, but 2025-Q3 (−0.006)
> and 2026-Q2 (+0.004) represent acknowledged gaps in alpha consistency.

## Interpretation   *(ISSUE-03: corrected BULL interpretation)*

- **BULL regime:** model has **negative IC (−0.017)** — cross-sectional dispersion collapses
  when everything rises together. BULL suppressor mitigates this; SHORT signals not authorized
  in BULL regime. *(Previous incorrect text: "model captures momentum continuation" — WRONG)*
- **BEAR regime:** model captures defensive rotation and cross-sectional short signals (IC=+0.034)
- **SIDEWAYS:** model relies on cross-sectional dispersion; primary operating regime (IC=+0.021)

## Gate Rationale: PARTIAL (not PASS)

The G_REGIME gate is **⚠️ PARTIAL** because:
1. BULL regime IC = −0.017 (statistically significant negative) — BULL suppressor added as mitigation
2. 2 of 7 quarters have non-significant IC (alpha consistency risk)
3. MID_VOL IC not statistically significant (p=0.07)
4. Live sessions: 7/20 accumulated (Oct 1-7); gate requires 20 mixed-regime sessions

The OOS backtest evidence is sufficient to authorize LIMITED SHADOW deployment with the BULL
suppressor active. Full G_REGIME PASS requires 20 live sessions across all regimes.

**This report remains PARTIAL until 20 live sessions are accumulated (~Nov 3).**
Continuing to accumulate live sessions for ongoing monitoring.
