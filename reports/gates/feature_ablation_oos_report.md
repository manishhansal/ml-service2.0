# G_ABLATION Feature Ablation Report
**Model:** v2c | **OOS Period:** 2025-01-01 → 2026-09-21
**Generated:** 2026-10-05 15:35 UTC

## Gate Verdict: ✅ PASS

Baseline OOS IC = **+0.0178** (p=1.2073e-09)

*Method: zero out each feature group, measure IC drop on held-out OOS period.
A group is "critical" if removing it reduces IC by >20%.*

## Results (sorted by impact)

| Group | N features | IC w/o group | IC drop | % drop | Importance |
|-------|-----------|-------------|---------|--------|-----------|
| D_vol_rsi | 13 | +0.0085 | +0.0093 | +52.2% | 🔑 CRITICAL |
| A_price_returns | 8 | +0.0104 | +0.0074 | +41.6% | 🔑 CRITICAL |
| B_ext_momentum | 13 | +0.0249 | -0.0071 | -39.8% | 🔑 CRITICAL |
| E_cross_sectional | 15 | +0.0208 | -0.0030 | -16.6% | 📌 IMPORTANT |
| C_time_context | 7 | +0.0198 | -0.0020 | -11.0% | 📌 IMPORTANT |
| F_regime | 3 | +0.0195 | -0.0017 | -9.3% | ➡ MODERATE |
| G_news_sentiment | 1 | +0.0176 | +0.0003 | +1.6% | ➡ MODERATE |

## Interpretation

The ablation shows which feature groups contribute genuine OOS IC:
- Removing a **CRITICAL** group causes >20% IC drop — the group carries irreplaceable signal.
- **MINIMAL** groups add noise or are redundant with other groups.

This report closes the **G_ABLATION gate** — all feature groups are evaluated on
genuinely held-out OOS data (2025-01-01 onwards, never seen during training).
