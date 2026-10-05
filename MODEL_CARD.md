# MODEL CARD
**Model:** expanded_lgbm  
**Version:** 1.0.0-20260928053134956099  
**Stage:** SHADOW (demoted to CHALLENGER pending G10/G11 resolution)  
**Date:** 2026-10-01  
**Author:** AlphaForge ML Team  
**Reviewer:** Kiro Forensic Audit System

---

## 1. Model Overview

| Field | Value |
|-------|-------|
| Purpose | Cross-sectional NSE F&O stock ranking for 5–7 day alpha |
| Type | LightGBM gradient boosted trees (ranking/classification) |
| Input | 67-feature vector per stock per trading day |
| Output | Score in [0, 1]; >0.5 → LONG signal, <0.5 → SHORT signal |
| Universe | 279 NSE F&O symbols |
| Primary horizon | 5 bars (~1 week) — **should be changed to 7 trading days** |
| Feature schema | fs-4.0.0 |
| Label schema | ls-2.0.0 (triple_barrier, horizon=5, ±2%) |
| Serving | FastAPI endpoint (POST /v2/predict/rankings) |

---

## 2. Training Data

| Parameter | Value |
|-----------|-------|
| Data source | NSE Bhavcopy via Angel One / Upstox / yfinance fallback |
| Date range | 2022-10-06 → 2026-09-28 |
| Total observations | 263,709 (279 symbols × ~944 days average) |
| Features | 67 (price, returns, volatility, volume, momentum, derivatives, news, macro) |
| Label type | Triple-barrier (TARGET_HIT / STOP_HIT / TIME_EXPIRY) |
| Label horizon | 5 bars (approximately 5 trading days) |
| Label barriers | ±2% fixed symmetric |
| Label execution | next_open (entry at open[T+1]) |
| Label cost | 27.65 bps round-trip |
| Class balance | 49.6% positive / 50.4% negative |
| Temporal coverage | 2022-10 → 2026-09 (47 months) |
| Survivorship | Partially controlled (some historical F&O eligibility DATA_UNAVAILABLE) |

---

## 3. Features

### Group A — Price/Returns (5)
`ret_1, ret_3, ret_5, ret_10, ret_20, ret_60, log_ret_1`

### Group B — Volatility (8)
`vol_5, vol_10, vol_20, atr_14_pct, parkinson_vol, garman_klass_vol, vol_of_vol_20, atr_zscore`

### Group C — Volume (6)
`rel_volume_20, volume_zscore_20, vwap_distance_pct, vol_ratio, vol_expanding, vol_spike_ratio`

### Group D — Momentum/Trend (10)
`rsi_14, rsi_7, macd_hist, stoch_k_14, ema_5_20, ema_10_50, adx_14, trend_strength, trend_direction, mom_accel_5`

### Group E — Price Structure (8)
`hl_range_pct, close_position, gap_pct, bb_zscore_20, bb_pct_b, price_zscore_60, price_from_20d_high_pct, price_from_20d_low_pct`

### Group F — Statistical (6)
`skew_20, kurt_20, vol_regime_zscore, vol_regime_pctile, vol_norm_ret_5, vol_norm_ret_20`

### Group G — Time (7)
`weekday, weekday_sin, weekday_cos, month_end_proximity, quarter_end, is_monday, is_friday`

### Group H — Intraday (8)
`intraday_morning_ret, intraday_close_vs_vwap, intraday_vol_profile, intraday_realized_vol, intraday_range_pct, intraday_last_hour_ret, intraday_open_to_high_pct, intraday_open_to_low_pct`

### Group I — News/Sentiment (6)
`news_market_sentiment, news_macro_sentiment, news_risk_sentiment, news_regime_score, news_asset_sentiment, news_event_importance`

### Group J — Index Alignment (3)
`nifty_ret_1d, nifty_ret_5d, nifty_ret_20d`

PIT status: All features use only data ≤ T (verified by static audit and mutation tests).

---

## 4. Target (Label)

The primary training target is a **binary classification label** from the triple-barrier method:

```
For bar T (close of day T):
  entry_price = open[T+1]          (next-day open, executable)
  target      = entry_price × 1.02  (+2% barrier)
  stop        = entry_price × 0.98  (-2% barrier)
  horizon     = 5 trading days

  label = 1  if target hit before stop, or TIME_EXPIRY with close[T+5] > open[T+1]
  label = 0  if stop hit first, or TIME_EXPIRY with close[T+5] < open[T+1]
```

**Known issue (CRIT-002):** This labeling produces a near-50/50 class balance with negative expected value at realistic transaction costs (break-even requires >56.9% win rate). The improved label in `src/labels/seven_day.py` addresses this.

---

## 5. Model Architecture

```
Algorithm:     LightGBM Gradient Boosted Decision Trees
Objective:     binary (cross-entropy)
Training:      PurgedKFoldSplitter (5 folds, 10-day embargo)
HPO:           Optuna (≥50 trials, inner CV on training folds only)
Normalisation: StandardScaler fitted on training data only, applied to test
```

---

## 6. Validation Method

Walk-forward OOS validation with 5 temporal folds:

| Fold | Train | Test | IC | Net Sharpe |
|------|-------|------|----|-----------|
| 1 | 2021-09 → 2022-09 | 2022-09 → 2023-01 | 0.367 | +0.82 |
| 2 | 2021-09 → 2023-01 | 2023-01 → 2023-06 | 0.390 | +0.91 |
| 3 | 2021-09 → 2023-06 | 2023-06 → 2024-01 | 0.431 | +1.21 |
| 4 | 2021-09 → 2024-01 | 2024-01 → 2024-08 | 0.418 | +1.08 |
| 5 | 2021-09 → 2024-08 | 2024-08 → 2026-09 | 0.377 | +0.93 |

Note: Sharpe computed at 10 bps cost (not equity 27.35bps). Recalculated at realistic costs: Sharpe ≈ +0.35 (estimated).

CPCV PBO: Claimed 0.000 using simplified formula (fraction of negative-Sharpe folds). Proper López de Prado CPCV not implemented.

---

## 7. Limitations

1. **Label horizon mismatch:** Trained on 5-bar labels, evaluated on 7 trading days
2. **Negative label EV:** At equity costs, the training labels have negative expected value
3. **Regime sensitivity:** Demonstrated alpha only in 2-day bear market (Sep 28-29, 2026)
4. **IC suspiciously high:** IC = 0.4136 is 5–20× typical equity factor IC; requires independent verification
5. **Calibration unverified:** ECE = 0.000 claim not independently reproduced
6. **F&O universe approximation:** Historical eligibility DATA_UNAVAILABLE for some dates
7. **Intraday features:** Source timestamps not independently verified for PIT compliance
8. **News features:** SentinelPulse timestamp alignment not independently verified
9. **Cost inconsistency:** WF Sharpe computed at 10bps; live uses 27.65bps

---

## 8. Known Failure Cases

1. **Market recoveries after bear periods:** Oct 1, 2026 showed 23.8% win rate immediately following 2-day bear period
2. **LONG signals in bear markets:** Sep 28-29 showed LONG book with −2.5% average net
3. **Defensive sector SHORT signals:** DRREDDY, ASIANPAINT, AXISBANK were consistent SHORT misses
4. **Low-volume periods:** Model performance not evaluated on low-volume/pre-holiday sessions
5. **Earnings announcements:** No explicit earnings filter; model may be confused during earnings season

---

## 9. Performance Metrics

### Classification (WF OOS, 5-bar labels)
```
IC_continuous:    0.3757
IC_rank:          0.4136
Calibration ECE:  0.000  (claimed, unverified)
PBO:              0.000  (simplified estimate)
```

### Trading (WF OOS, estimated at 26.4bps equity)
```
Net Sharpe (equity):  ~0.35 (estimated; claimed +1.076 at 10bps)
Net Sharpe (futures): ~1.06 (claimed at 12.75bps)
Win rate (live):      80% (Sep 28-29, SHORT only, 40 obs)
Win rate (Oct 1):     23.8% (ALL signals, 84 resolved)
```

### 7-Day Backtest (proxy evaluation, proxy upper bound)
```
Win rate (proxy):     64.2%
Mean net/trade:       +1.66%
Profit factor:        2.79
Sharpe (proxy):       7.27
```

---

## 10. Market Assumptions

- **NSE F&O market hours:** 09:15–15:30 IST
- **Daily settlement:** Cash equity (T+2); F&O (T+1)
- **Execution:** Market-on-open order at T+1 open
- **Short selling:** Requires F&O position (futures/options) — not equity short sell
- **Liquidity:** All 279 F&O symbols assumed liquid; not verified for smallest caps
- **Gap events:** No filter for overnight gaps or earnings surprises

---

## 11. Intended Use

The model is intended for:
- **Paper trading and signal research** during SHADOW stage
- Generating relative rankings of F&O stocks for cross-sectional alpha
- Identifying the top/bottom quartile of stocks for directional bias

**Not intended for:**
- Standalone trade execution without human review
- Stocks outside the NSE F&O universe
- Intraday trading (model uses EOD data)
- Position sizing determination
- Risk management (no explicit risk model)

---

## 12. Prohibited Interpretations

| Claim | Status |
|-------|--------|
| "This model generates 90% accurate signals" | FALSE — no evidence for this claim |
| "80% win rate is stable and regime-independent" | FALSE — based on 2 bear market days only |
| "The model is production-ready" | FALSE — G7, G10, G11 fail |
| "IC = 0.4136 means 70%+ directional accuracy" | MISLEADING — IC does not map directly to accuracy |
| "ECE = 0.000 means perfect calibration" | UNVERIFIED — calculation method not documented |
| "PBO = 0.000 means zero overfitting risk" | MISLEADING — simplified formula, not proper CPCV |

---

## 13. Risk Statement

This model is a research prototype. It has demonstrated:
- Genuine alpha structure in historical cross-sectional data (IC > 0 across 5 folds)
- Consistent SHORT alpha during 2 consecutive bear market days

It has NOT demonstrated:
- Stable performance across mixed market regimes
- Statistical significance with adequate sample size (>100 forward observations)
- Production-level reliability in forward paper trading
- Positive expected value at realistic equity transaction costs

**Recommended deployment stage:** Continue SHADOW monitoring for minimum 20 mixed-regime trading days before considering PRODUCTION promotion.

---

*Kiro Forensic Audit System — 2026-10-01*
