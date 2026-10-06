# Feature Specification
**ml-service2.0 — Institutional Feature Engineering Framework**

*Date: 2026-09-24*
*Status: TARGET specification — most groups not yet implemented*

---

## 1. Feature Engineering Principles

1. Every feature has a single source of truth (DataService or SentinelPulse)
2. Every feature carries an explicit `feature_as_of` timestamp
3. Missing values use domain-appropriate handling — never zero-fill derivatives data
4. Features are grouped by signal type and documented with expected ranges
5. Feature importance is tracked per model per regime over time
6. Highly correlated features (|r| > 0.9) should be pruned to one representative
7. Feature stability is tested across time periods — unstable features are flagged

---

## 2. Feature Groups

### Group A: Price Action
*Source: DataService OHLCV bars*

| Feature | Formula | Lookback | Expected Range | Notes |
|---|---|---|---|---|
| `ret_1` | log(close_t / close_{t-1}) | 1 bar | [-0.1, 0.1] | 1-bar log return |
| `ret_5` | log(close_t / close_{t-5}) | 5 bars | [-0.2, 0.2] | 5-bar log return |
| `ret_10` | log(close_t / close_{t-10}) | 10 bars | [-0.3, 0.3] | 10-bar log return |
| `ret_20` | log(close_t / close_{t-20}) | 20 bars | [-0.4, 0.4] | 20-bar log return |
| `momentum_5` | sign(ret_5) × |ret_5|^0.5 | 5 bars | [-0.5, 0.5] | Square-root dampened momentum |
| `momentum_10` | sign(ret_10) × |ret_10|^0.5 | 10 bars | [-0.6, 0.6] | |
| `acceleration` | ret_1 - ret_1_{t-1} | 2 bars | [-0.05, 0.05] | Return acceleration |
| `atr_5` | mean(TR) over 5 bars | 5 bars | [0, 0.05] | Average True Range normalized by close |
| `atr_14` | mean(TR) over 14 bars | 14 bars | [0, 0.05] | ATR (14) normalized |
| `atr_expansion` | atr_5 / atr_20 | 20 bars | [0.5, 3.0] | Volatility expansion ratio |
| `range_pct` | (high - low) / close | 1 bar | [0, 0.05] | Intraday range |
| `gap_pct` | (open - prev_close) / prev_close | 1 bar | [-0.05, 0.05] | Gap percentage |
| `hl_position` | (close - low) / (high - low) | 1 bar | [0, 1] | Close position in day range |
| `trend_5` | OLS slope of close over 5 bars | 5 bars | [-1, 1] | Normalized trend slope |
| `realized_vol_5` | std(log_returns) × sqrt(bars_per_day) | 5 bars | [0, 0.5] | Annualized realized vol |
| `realized_vol_20` | std(log_returns) × sqrt(bars_per_day) | 20 bars | [0, 0.5] | |

### Group B: Volume
*Source: DataService OHLCV bars*

| Feature | Formula | Lookback | Expected Range | Notes |
|---|---|---|---|---|
| `vol_ratio_5` | volume_t / mean(volume, 5) | 5 bars | [0, 10] | Relative volume |
| `vol_ratio_20` | volume_t / mean(volume, 20) | 20 bars | [0, 10] | Relative volume (longer) |
| `vol_acceleration` | vol_ratio_5 / vol_ratio_20 | 20 bars | [0, 5] | Volume trend |
| `vwap_distance` | (close - vwap) / vwap | Intraday | [-0.03, 0.03] | Distance from VWAP |
| `vol_imbalance` | (buy_vol - sell_vol) / total_vol | 1 bar | [-1, 1] | If tick data available |
| `delivery_pct` | delivery_volume / total_volume | 1 day | [0, 1] | NSE delivery quality signal |

### Group C: Market Microstructure
*Source: DataService option chain / order book*

| Feature | Formula | Notes |
|---|---|---|
| `bid_ask_spread` | (ask - bid) / mid | Null if market closed |
| `effective_spread` | 2 × |trade_price - mid| / mid | Requires tick data |
| `order_imbalance` | (bid_qty - ask_qty) / (bid_qty + ask_qty) | [-1, 1] |
| `trade_intensity` | trades_per_minute / baseline | Relative activity |
| `vpin` | see VPIN engine | Volume-synchronised PIN |

**Note:** Microstructure features require tick/L2 data. DataService availability determines feasibility. Fall back gracefully — never substitute fake values.

### Group D: Derivatives
*Source: DataService option chain*

| Feature | Formula | Notes |
|---|---|---|
| `atm_iv` | IV of nearest ATM call | Null if IV unavailable |
| `iv_rank` | (iv_today - iv_52w_low) / (iv_52w_high - iv_52w_low) | [0, 1] |
| `iv_percentile` | percentile(iv_today, 252 days) | [0, 100] |
| `iv_change_1d` | (atm_iv - atm_iv_prev) / atm_iv_prev | Daily IV change |
| `skew_25d` | IV(25D put) - IV(25D call) | Risk reversal proxy |
| `term_structure` | IV(far_expiry) - IV(near_expiry) | Term structure slope |
| `pcr_volume` | put_volume / call_volume | PCR by volume |
| `pcr_oi` | put_oi / call_oi | PCR by OI |
| `max_pain` | strike minimizing total payout | Computed from chain |
| `max_pain_distance` | (spot - max_pain) / spot | Gravity from max pain |
| `futures_basis` | (futures_price - spot) / spot | Futures premium/discount |
| `futures_oi_change` | (oi_today - oi_yesterday) / oi_yesterday | OI build/unwind |
| `rollover_pct` | near_oi_change vs. far_oi_change | Monthly rollover signal |
| `gex_total` | Σ dealer GEX across chain | Net gamma exposure |
| `gamma_flip` | Strike where cumulative GEX = 0 | Support/resistance level |
| `gex_distance` | (spot - gamma_flip) / spot | Distance from gamma flip |

**Critical null handling:** If `oi=None` or `iv=None` for a strike, exclude that strike from computation. Never impute.

### Group E: Cross-Sectional
*Source: DataService (requires universe-level computation)*

| Feature | Formula | Notes |
|---|---|---|
| `relative_strength_nifty` | ret_20 - nifty_ret_20 | Excess return vs. NIFTY |
| `relative_strength_sector` | ret_20 - sector_ret_20 | Excess return vs. sector |
| `rank_in_universe` | percentile_rank(ret_5, universe) | [0, 100] |
| `sector_rank` | rank within own sector | [0, 100] |
| `beta_nifty` | rolling_beta(symbol, nifty, 20) | Market beta |
| `residual_return` | ret_5 - beta_nifty × nifty_ret_5 | Alpha component |
| `dispersion_score` | std(universe_returns) | Market breadth |
| `correlation_rank` | avg_pairwise_corr position | |

**Note:** Cross-sectional features require simultaneous data for all F&O stocks (~200). This needs a batch feature computation pipeline not currently implemented.

### Group F: Regime
*Source: DataService (indices) + ml-service2.0 RegimeClassifier*

| Feature | Formula | Notes |
|---|---|---|
| `nifty_trend_5` | OLS slope of NIFTY over 5 bars | Normalized |
| `nifty_trend_20` | OLS slope of NIFTY over 20 bars | |
| `vix_level` | India VIX level | From DataService |
| `vix_change` | (vix_today - vix_5d_avg) / vix_5d_avg | |
| `vix_percentile` | percentile(vix, 252 days) | |
| `market_breadth` | advance_count / (advance + decline) | A/D ratio |
| `new_highs_lows` | (52w_highs - 52w_lows) / universe | |
| `advance_decline_ratio` | advancing / declining | |
| `sector_dispersion` | std(sector_returns_today) | Rotation signal |
| `risk_on_off` | signal based on VIX + breadth + trend | [-1, 1] composite |
| `is_expiry` | 1 if current week is options expiry | Binary |
| `is_monthly_expiry` | 1 if current week is monthly expiry | Binary |
| `session_phase` | 0=pre_open, 1=morning, 2=afternoon, 3=post | Encoded |
| `day_of_week` | 0=Mon,...,4=Fri | Encoded |

### Group G: News (from SentinelPulse)
*Source: SentinelPulse*

| Feature | Formula | Notes |
|---|---|---|
| `news_impact_score` | Direct from SentinelPulse | [0, 1] |
| `impact_direction_encoded` | BULLISH=1, NEUTRAL=0, BEARISH=-1 | |
| `impact_confidence` | Direct from SentinelPulse | [0, 1] |
| `sentiment_overall` | Direct from SentinelPulse | [-1, 1] |
| `sentiment_market` | Direct from SentinelPulse | [-1, 1] |
| `sentiment_company` | Direct from SentinelPulse | [-1, 1] |
| `sentiment_macro` | Direct from SentinelPulse | [-1, 1] |
| `sentiment_risk` | Direct from SentinelPulse | [-1, 1] |
| `news_velocity` | article_rate vs. baseline | [0, 5] |
| `novelty_score` | how novel is this event | [0, 1] |
| `source_quality` | weighted tier score | [0, 1] |
| `event_importance` | from NewsEvent.importance | [0, 1] |
| `surprise_score` | from NewsEvent.surpriseScore | null if no expectation |
| `news_age_minutes` | now - news_as_of | [0, 1440] |

**PIT enforcement:** All news features use `SentinelNewsContext.as_of` as the PIT boundary. Features derived from news published after `prediction_timestamp` are forbidden.

### Group H: Time/Session
*Derived from prediction_timestamp*

| Feature | Formula | Notes |
|---|---|---|
| `minutes_since_open` | minutes(now - 09:15 IST) | [0, 375] NSE session |
| `minutes_to_close` | minutes(15:30 IST - now) | [0, 375] |
| `session_quartile` | 0-3 based on session position | |
| `day_of_week` | 0=Mon,...,4=Fri | |
| `week_of_month` | 1-5 | |
| `is_expiry_week` | weekly options expiry | Binary |
| `is_monthly_expiry_week` | monthly options expiry | Binary |
| `days_to_next_expiry` | calendar days | [0, 7] |
| `is_result_week` | earnings season overlap | Binary |

### Group I: Market Context
*Source: DataService (indices)*

| Feature | Formula | Notes |
|---|---|---|
| `nifty_return_1d` | NIFTY daily return | Index context |
| `banknifty_return_1d` | BANKNIFTY daily return | Banking sector |
| `nifty_change_pct` | change from previous close | |
| `banknifty_change_pct` | change from previous close | |
| `usdinr_change` | USD/INR rate change | FX context |
| `crude_change` | Crude oil change (if available) | |
| `dow_overnight` | Dow Jones overnight change | Global context |
| `sgx_nifty_change` | SGX Nifty overnight premium | Pre-market signal |

---

## 3. Feature Validation Requirements

Every feature must pass:
1. **Range test:** `assert feature_value ∈ expected_range` for 99.9% of live observations
2. **Leakage test:** `|Pearson(feature, future_return)| < 0.05` for all look-ahead windows
3. **Stability test:** IC does not drop below zero for > 3 consecutive months
4. **Null handling test:** null input produces null output (never zero)
5. **PIT test:** all input data timestamps < `feature_as_of`

---

## 4. Current Implementation Status

| Group | Target Features | Currently Implemented | Gap |
|---|---|---|---|
| A: Price Action | 16 | ~12 (via QlibFeatureEngine) | 4 missing |
| B: Volume | 6 | 3 | 3 missing (delivery_pct, vol_imbalance, tick-based) |
| C: Microstructure | 5 | 1 (VPIN) | 4 missing |
| D: Derivatives | 17 | ~8 (Greeks, GEX, PCR) | 9 missing |
| E: Cross-sectional | 8 | 0 | **8 missing** — no batch pipeline |
| F: Regime | 16 | ~6 (in RegimeClassifier inputs) | 10 missing |
| G: News | 14 | ~8 (via SentinelPulse) | 6 missing |
| H: Time/Session | 9 | ~4 | 5 missing |
| I: Market Context | 8 | ~5 (in RegimeClassifier inputs) | 3 missing |
| **Total** | **99** | **~47** | **~52 missing** |

---

## 5. Feature Selection Policy

Before any feature group is added to production models:

1. **Correlation pruning:** Remove one of any pair with |r| > 0.9
2. **Mutual information:** Retain features with MI > 0.01 vs. target
3. **Permutation importance:** Feature importance > 0 in at least 3 OOS folds
4. **Temporal stability:** IC does not flip sign in more than 2 consecutive quarters
5. **Regime stability:** IC is positive in at least 4 of 6 regimes tested
6. **SHAP stability:** SHAP direction consistent with economic intuition
7. **Leakage re-test:** Confirm after addition to feature set (interaction effects)

---

*End of Feature Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by `src/features/factory.py` (FeatureFactory: 24 explicit PIT-safe features + FeatureAvailability metadata) and `src/features/qlib_engine.py`. Missing values are NaN (never zero-filled). PIT-safety proven by a truncation-invariant test. Note (mandate 13): features are NOT padded to ~99; each has a definition, source, and test.

Status: IMPLEMENTED and tested. Authoritative certification: `reports/ML_SERVICE_FINAL_CERTIFICATION.md`.
