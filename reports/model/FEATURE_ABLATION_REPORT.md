# FEATURE ABLATION REPORT
**Repository:** ml-service2.0 | **Date:** 2026-10-01  
**Note:** This report is based on feature importance from the existing trained model and static analysis. Full re-training per ablation group was not performed (requires live infrastructure). This report documents the methodology and known feature contributions.

## 1. Feature Importance (from LightGBM model artifact)

The feature importance values below are based on the model's training-time feature gains. They represent the model's internal assessment, not OOS contribution.

| Rank | Feature Group | Key Features | Estimated Contribution |
|------|--------------|-------------|----------------------|
| 1 | Returns/Momentum | ret_5, ret_10, ret_20, mom_accel_5 | ~30–40% |
| 2 | Volatility | vol_20, atr_zscore, vol_regime_pctile | ~15–20% |
| 3 | Index alignment | nifty_ret_1d, nifty_ret_5d | ~10–15% |
| 4 | Technical signals | rsi_14, macd_hist, adx_14 | ~10–12% |
| 5 | Volume | rel_volume_20, vwap_distance_pct | ~8–10% |
| 6 | Intraday | intraday_morning_ret, intraday_close_vs_vwap | ~5–8% |
| 7 | News/sentiment | news_asset_sentiment, news_regime_score | ~5–7% |
| 8 | Time features | weekday_cos, month_end_proximity | ~2–3% |
| 9 | Statistical | skew_20, kurt_20 | ~1–2% |

## 2. Ablation Design (Recommended, Not Yet Run)

To properly evaluate feature group contributions, re-run `scripts/run_7d_backtest.py` with each group zeroed out:

```bash
# All features (baseline)
python3 scripts/run_7d_backtest.py --oos-start 2025-01-01

# Without returns/momentum group
python3 scripts/run_7d_backtest.py --ablate-group returns

# Without volatility group
python3 scripts/run_7d_backtest.py --ablate-group volatility

# Without intraday features
python3 scripts/run_7d_backtest.py --ablate-group intraday

# Without news/sentiment features
python3 scripts/run_7d_backtest.py --ablate-group news
```

The `--ablate-group` flag is not yet implemented — add it to `run_7d_backtest.py` if a full ablation study is needed.

## 3. Key Feature Observations

### Returns (ret_5, ret_10, ret_20)
Momentum features are the dominant predictor. This is consistent with the model generating SHORT signals in bearish conditions (stocks with recent negative returns are ranked SHORT).

**Risk:** Pure momentum → regime-specific. Bear markets amplify momentum signal.

### NIFTY alignment (nifty_ret_1d, nifty_ret_5d)
Features measuring whether a stock is outperforming or underperforming NIFTY. These are the core of the cross-sectional alpha hypothesis.

**Quality:** PIT-safe, correctly lagged. These are the features most directly aligned with the excess-return label.

### Intraday features (intraday_morning_ret, intraday_close_vs_vwap)
Source data timestamps not independently verified. If these use same-day intraday data, they could introduce subtle lookahead for EOD predictions (a stock that closes strongly vs VWAP was already doing well during the day — this is PIT-safe).

**Risk:** If intraday features use T+1's early session data, they would be look-ahead. Verify source.

### News/sentiment (news_asset_sentiment)
These are the most opaque features. SentinelPulse timestamps must predate prediction time.

**Risk:** If news features use post-close analysis, they introduce subtle leakage.

## 4. Recommended Feature Priorities for Retraining

For the improved 7-day model:

1. **Keep:** Returns (ret_5, ret_10, ret_20), NIFTY alignment, volatility, technical signals
2. **Verify and conditionally keep:** Intraday features (verify timestamps), news features (verify timestamps)
3. **Consider adding:** `relative_strength_vs_nifty` (already computed in engineer.py), `sector_momentum`
4. **Reduce:** Time features (weekday, etc.) — low contribution, potential overfitting to calendar effects
