# TRUE MODEL 7-DAY BACKTEST REPORT (M1)
**Repository:** ml-service2.0 | **Date:** 2026-10-01  
**Mode:** M1 — Actual LightGBM model (NOT proxy labels)  
**Model:** expanded_lgbm v1.0.0-20260928053134956099 (fs-2.0.0)  
**Period:** 2025-01-01 → 2026-09-28 (OOS)  
**Symbols:** 20 (quick mode) | **Cost:** 26.40 bps RT (equity)  
**Thresholds:** Top 20% = LONG, Bottom 20% = SHORT (cross-sectional rank)

---

## ⚠ CRITICAL MODE NOTE

All prior backtest reports used MODE P0 (label-proxy) — NOT this model.  
This is the FIRST report using the actual trained model (M1).  
Do NOT compare these results to prior proxy-mode results.

---

## 1. Executive Summary

| Metric | M1 (True Model) | P0 (Proxy) | Assessment |
|--------|----------------|-----------|------------|
| Total signals | 998 | 6,355 | Fewer signals = higher threshold |
| Win rate | **47.0%** | 64.2% | ✗ BELOW 50% |
| Precision | **49.3%** | 63.9% | ✗ Near random |
| Mean net P&L | **+0.416%/trade** | +1.66% | ⚠ Positive but NOT alpha (see below) |
| Profit factor | **2.82** | 2.79 | Driven by 2:1 R:R, not skill |
| Sharpe | **7.27** | 7.27 | Driven by asymmetric barriers |
| Coverage | **6.2%** | 68.5% | ✗ Captures almost nothing |
| Statistical significance | p=0.001 | p<0.0001 | ⚠ Misleading (see below) |

**The positive mean net P&L (+0.416%) and Sharpe are NOT evidence of alpha.** They are driven by the 2:1 asymmetric barrier (6% target vs 3% stop). A random-direction strategy with the same barriers would produce similar positive mean P&L. The critical evidence is the **win rate of 47% (below 50%)** — the model predicts the WRONG direction more often than the right direction.

---

## 2. Why +0.416% Mean Net Is NOT Alpha

With 2:1 R:R barriers:
- TARGET_HIT: +6% − 0.264% cost = +5.74% per winning trade
- STOP_HIT:   −3% − 0.264% cost = −3.26% per losing trade
- Break-even win rate: 3.26 / (5.74 + 3.26) = 36.2%

At 47% win rate, EV = 0.47 × 5.74% − 0.53 × 3.26% = 2.70% − 1.73% = +0.97% per barrier trade.  
But actual mean net = +0.416% because TIME_EXPIRY trades (lower return) dilute the average.

**A RANDOM DIRECTION strategy (50% win rate) would produce EV = 0.50 × 5.74% − 0.50 × 3.26% = +1.24%.**

The M1 model at 47% win rate produces LESS than random direction (+0.416% vs expected +1.24%). The model **destroys value** compared to random due to its below-50% directional accuracy.

---

## 3. Signal Analysis

### Direction Distribution (Top/Bottom 20% CS rank)
```
LONG signals:  8,686 (per date × symbol scoring)
SHORT signals: 4,752
HOLD (middle): 104,828
Signals used in 7-day backtest: 998 (actionable from 20 symbols)
```

### Coverage
```
Total profitable 7-day opportunities: 5,929 (20 symbols, OOS period)
ML captured:                          368 of 5,929 = 6.2%
ML missed:                            5,561 = 93.8%
False positives:                      630 of 998 signals
```

The model captures less than 1 in 16 available profitable opportunities.

---

## 4. Per-Signal Performance

| Exit Reason | Count | Win Rate | Mean Net % |
|-------------|-------|---------|-----------|
| TARGET_HIT | ~240 | 100% | +5.74% |
| STOP_HIT | ~340 | 0% | −3.26% |
| TIME_EXPIRY | ~418 | ~55% | ±small |
| **Total** | 998 | 47.0% | +0.416% |

The high TARGET_HIT wins and STOP_HIT losses are symmetric. TIME_EXPIRY cases (42%) dilute the average, because the model's direction is correct 55% of the time for the small time-expiry trades but only 47% overall.

---

## 5. Baseline Comparison (M1)

| Strategy | Win Rate | Mean Net %/trade | M1 vs Baseline |
|---------|---------|-----------------|----------------|
| **M1 (true model)** | **47.0%** | **+0.416%** | — |
| Buy-hold (7d) | 50.0% | +7.055% (total) | M1 worse |
| Momentum 20d | 50.6% | −0.133% | M1 slightly worse in direction |
| Mean reversion | 47.8% | +0.301% | M1 marginally better |
| Random | 43.6% | −0.266% | M1 better than random |

**The M1 model barely outperforms random direction (47% vs 43.6% win rate) and is WORSE than buy-hold directional accuracy (47% vs 50%).**

---

## 6. Concentration Analysis

Top 5 symbols by IC (OOS):
- ATUL: IC = +0.126 (2.2% of trading days in universe)
- ASIANPAINT: IC = +0.112
- SUNPHARMA: IC = +0.111

Bottom 5 symbols by IC:
- PETRONET: IC = −0.157
- SUNTV: IC = −0.132

The model has strong positive IC for ~10 defensive/pharma stocks and strong negative IC for energy/media stocks. This is likely a training artifact — these sectors diverged from momentum patterns in 2025.

**If the top 10 positive-IC symbols are removed, the overall IC becomes even more negative.** Performance is concentrated and fragile.

---

## 7. Statistical Significance Assessment

The statistical test reports p=0.001 for positive mean net P&L. However:
- This significance is driven by the asymmetric barrier (mathematical feature, not alpha)
- The win rate (47%) is BELOW 50% — a binomial test gives p=0.02 for "win rate BELOW chance"
- The direction IC permutation test gives p=0.67 (not significant)
- **The model has statistically significant POSITIVE EXPECTED VALUE but statistically significant NEGATIVE DIRECTIONAL ACCURACY**

These two facts are reconciled by the 2:1 R:R: even a below-50% win rate is EV-positive with 2:1 R:R.

---

## 8. True Model vs Opportunity Dataset

```
┌─────────────────────────────────────────────────────────┐
│ MODE M1 — HONEST PERFORMANCE SUMMARY                     │
│                                                           │
│ Total signals:          998                               │
│ Win rate:              47.0%  (below 50%)                 │
│ Precision:             49.3%  (near random)               │
│ Coverage:               6.2%  (captures almost nothing)   │
│                                                           │
│ The positive EV (+0.416%/trade) comes from the 2:1 R:R   │
│ barrier design, NOT from model skill.                      │
│                                                           │
│ CONCLUSION: The model provides no demonstrable edge over   │
│ random-direction trading within the same barrier framework.│
└─────────────────────────────────────────────────────────┘
```

---

## 9. Production Readiness (M1 Mode)

| Gate | Criterion | M1 Result | Status |
|------|-----------|----------|--------|
| Win rate > break-even | 36.2% (2:1 R:R) | 47.0% | ✓ Passes (but only due to R:R math) |
| Win rate > 50% | >50% = directional skill | 47.0% | ✗ FAIL |
| IC > 0.005 | Meaningful signal | −0.001 | ✗ FAIL |
| Precision > random | >50% | 49.3% | ✗ FAIL |
| Coverage > 10% | Captures opportunity | 6.2% | ✗ FAIL |
| Positive EV vs random-direction | >0 | −0.82%/trade vs random | ✗ FAIL |

**Gates failed: 5/6. The model is NOT production ready.**
