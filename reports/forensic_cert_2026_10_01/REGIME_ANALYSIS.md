# REGIME ANALYSIS — TRUE MODEL (M1)
**Repository:** ml-service2.0 | **Date:** 2026-10-01

---

## True Model IC by Regime (OOS 2025–2026)

| Regime | N Rows | IC | p-value | Assessment |
|--------|--------|----|---------|------------|
| SIDEWAYS (20d NIFTY ±5%) | 118,266 | −0.001 | 0.659 | NO SIGNAL |
| 2025 (full year) | 67,766 | −0.001 | 0.890 | NO SIGNAL |
| 2026 (YTD) | 50,500 | −0.006 | 0.167 | NO SIGNAL |

The entire OOS period shows near-zero to negative IC. No regime shows positive, significant IC in OOS.

---

## By Symbol IC (Cross-Sectional Variation)

```
Symbols analyzed: 279
Mean IC:          −0.006 (negative)
Std IC:            0.056
Positive IC:       45.9% of symbols (below 50%!)
```

| Category | Symbols |
|----------|---------|
| Top 5 by IC | ATUL (+0.126), ASIANPAINT (+0.112), SUNPHARMA (+0.111), BERGEPAINT (+0.106), BOSCHLTD (+0.103) |
| Bottom 5 by IC | PETRONET (−0.157), SUNTV (−0.132), PATANJALI (−0.125), HDFCLIFE (−0.124), AARTIIND (−0.115) |

**Notable pattern:** High-IC symbols are defensive (FMCG, Pharma, Quality). Negative-IC symbols are high-beta (Energy, Media, Small-cap). This suggests the model learned to predict defensive sector outperformance in bull markets — which reversed in the 2025+ OOS sideways period.

---

## Regime Transition Risk (Oct 1 Analysis)

The Sep 28-29 (BEAR) → Oct 1 (RECOVERY) transition caused a dramatic performance drop:

| Date | NIFTY | SHORT Win Rate | Notes |
|------|-------|--------------|-------|
| Sep 28 | −1.52% | 80% | Bear market, SHORT naturally profitable |
| Sep 29 | −0.42% | 80% | Still mild bear |
| Oct 1 | Recovery | 23.8% | First non-bear day, model collapses |

**The Oct 1 23.8% win rate is NOT a random fluctuation.** It is a regime transition that exposed the model's fundamental bias toward SHORT signals in falling markets.

---

## Regime Recommendations

1. **Do NOT hard-code BEAR → SHORT ONLY**: This was mentioned in REGIME_ANALYSIS.md based on 2 days. Empirically invalid.
2. **Do NOT apply regime filters without OOS validation**: Any regime-conditional model must be validated across ALL regimes OOS.
3. **The model has no validated cross-regime alpha**: Until IC > 0.005 is demonstrated in at least 3 distinct regimes, no production deployment.

---

## What Is Needed for Production

1. Minimum 20 trading days × 4 regimes = 80 days of live/paper tracking across BULL, BEAR, SIDEWAYS, HIGH_VOL
2. Each regime must independently show win rate > break-even
3. No cherry-picking: include ALL signals, not just favorable ones
