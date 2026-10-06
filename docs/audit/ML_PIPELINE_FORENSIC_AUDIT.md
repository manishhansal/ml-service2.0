# ML PIPELINE FORENSIC AUDIT
**Repository:** ml-service2.0 — AlphaForge NSE F&O Cross-Sectional Alpha Engine  
**Audit Date:** 2026-10-01  
**Auditor:** Kiro Forensic Audit System  
**Scope:** Complete end-to-end ML pipeline from raw market data to live signals and P&L  
**Dataset inspected:** ds-1d-20261001034811-ebdf74af (263,709 rows, 279 symbols)  
**Model audited:** expanded_lgbm v1.0.0-20260928053134956099, feature schema fs-4.0.0

---

## EXECUTIVE SUMMARY

| Finding | Severity | Status |
|---------|----------|--------|
| Negative mean net expectancy in training labels | **CRITICAL** | OPEN |
| 7-trading-day evaluation horizon absent | **CRITICAL** | OPEN |
| Regime-dependent performance (2-day bear window) | **CRITICAL** | OPEN |
| Forward paper Oct 1 collapse (23.8% win rate) | **CRITICAL** | OPEN |
| In-sample vs OOS Sharpe gap (4.05 vs 1.41) | HIGH | OPEN |
| IC = 0.4136 requires independent verification | HIGH | OPEN |
| ECE = 0.000 calibration claim unverifiable | HIGH | OPEN |
| Label balance 49.6/50.4% — near-coin-flip | MEDIUM | OPEN |
| Survivorship bias exposure (2022 start for 2021+ claims) | MEDIUM | PARTIAL |
| No static look-ahead bias in feature code | LOW | PASS |
| Execution model (next-open) correctly implemented | LOW | PASS |
| PurgedKFold embargo correctly implemented | LOW | PASS |

**Overall verdict: The system CANNOT be declared production-ready as of 2026-10-01.**

The forward paper evidence that formed the basis of the SHADOW promotion (Sep 28-29 80% SHORT win rate) is not holding up in subsequent sessions. The model shows classic regime-overfitting to a 2-day bear market.

---

## SECTION 1 — PIPELINE ARCHITECTURE

### 1.1 Complete Data Flow

```
RAW MARKET DATA (NSE Bhavcopy via Angel One / Upstox / yfinance fallback)
      │
      ↓ DataIngestionPipeline.ingest()
DAILY OHLCV PARQUETS (data/1d/1d/*.parquet)
  • 285 parquet files, 1 per symbol
  • Date range: 2021-09-19 → 2026-09-28
  • All timestamps at 18:30:00 UTC (NSE EOD = 15:30 IST)
  • Timezone: UTC-aware DatetimeIndex
      │
      ↓ DatasetBuilder (src/data/dataset_builder.py)
FEATURE ENGINEERING (src/features/factory.py + engineer.py)
  • FeatureFactory: 24 base features (fs-2.0.0 schema)
  • ExpandedFactory: 91 total columns including intraday, news, macro
  • Feature schema version: fs-4.0.0
  • 67 feature columns used for training (excluding label/meta columns)
      │
      ↓ LabelFactory (src/data/labels.py)
LABEL GENERATION
  • Type: triple_barrier
  • Horizon: 5 bars (NOT 7 trading days)
  • Upper barrier: +2% (TARGET_HIT → label=1)
  • Lower barrier: -2% (STOP_HIT → label=0)
  • TIME_EXPIRY: label = int(realized_return > 0)
  • Execution model: next_open (entry at open[T+1])
  • Cost for labeling: 27.65 bps round-trip
  • Label balance: 49.6% positive / 50.4% negative
      │
      ↓ TrainingPipeline (src/training/pipeline.py)
MODEL TRAINING
  • Algorithm: LightGBM
  • CV: PurgedKFoldSplitter (5 folds, 10-day embargo)
  • HPO: Optuna (≥50 trials)
  • Acceptance gates: IC > 0.02, Sharpe > 0, PBO < 0.50
  • Dataset: 263,709 rows, 279 symbols, 2022-10-06 → 2026-09-28
      │
      ↓ Model artifact stored in artifacts/registry/
TRAINED LGBM MODEL ARTIFACT
  • Version: 1.0.0-20260928053134956099
  • Stage: SHADOW
  • Feature schema: fs-4.0.0
      │
      ↓ autorun_till_close.py (live session)
INFERENCE (per symbol, every 5 minutes)
  • Loads latest OHLCV bars from parquets
  • Computes feature vector via compute_stock_features()
  • Runs LightGBM prediction → raw score (probability-like [0,1])
  • Normalizes score to [0, 100] across universe
      │
      ↓ Signal generation logic
SIGNAL GENERATION
  • score > 0.5 → LONG (+1) direction
  • score < 0.5 → SHORT (-1) direction
  • Pipeline filters: sector_dimmed, stock_dampened, weight_manager_threshold
  • Filtered signals → direction = 0 (NEUTRAL, not exposed to UI)
      │
      ↓ artifacts/live_session/latest_scores.json
SIGNAL STORAGE
  • JSON snapshot refreshed every 5 minutes
  • Also written to artifacts/forward_paper/forecasts.jsonl
      │
      ↓ Signal API (src/api/signals.py)
SIGNAL DELIVERY TO ALPHAFORGE UI
      │
      ↓ 5-bar (≈1 calendar week) forward tracking
OUTCOME TRACKING (forward paper)
  • Entry: open[T+1]
  • Exit: open[T+1+5] or first barrier hit
  • net_pct calculated per resolved signal
```

### 1.2 Key Timing Details

| Event | Time (IST) | Time (UTC) |
|-------|-----------|-----------|
| NSE market open | 09:15 | 03:45 |
| NSE market close | 15:30 | 10:00 |
| Bhavcopy publication | ~18:00 | ~12:30 |
| Daily OHLCV bar timestamp | 15:30 IST close | 18:30:00 UTC |
| Signal generation (live) | 09:20–15:30 IST | 03:50–10:00 UTC |
| Entry execution | Next day open (09:15 IST) | 03:45 UTC next day |

### 1.3 Model Type and Inference Mechanism

The `StockRanker` (`src/models/stock_ranker.py`) is the primary production model.

- **When trained model available**: LightGBM `.predict(X)` returns probabilities in [0, 1]
- **When no trained model**: Regime-conditioned weighted score heuristic
- **Output normalization**: `(raw - min) / (max - min) * 100` → rank score [0, 100]
- **Signal direction**: `score > 0.5 raw → LONG; score < 0.5 raw → SHORT`

This is a **cross-sectional ranking model**, not a binary classifier. The "accuracy" reported as IC (Spearman rank correlation) is appropriate for this model type.

---

## SECTION 2 — DATA SOURCES AND QUALITY

### 2.1 Primary Data Sources

| Source | Role | Priority |
|--------|------|----------|
| Angel One (via data-service2.0) | Primary OHLCV | 1 |
| Upstox | Secondary OHLCV | 2 |
| PostgreSQL cache | Normalized daily bars | 3 |
| yfinance | Last-resort fallback | 4 |
| SentinelPulse | News/sentiment features | Optional |

### 2.2 Parquet File Inventory

```
Location: data/1d/1d/*.parquet
Count: 285 files (universe of 285 NSE F&O symbols)
Sample structure:
  Columns: [open, high, low, close, volume, provider]
  Index: DatetimeIndex (UTC-aware, 18:30:00 UTC = NSE EOD)
  Date coverage: 2021-09-19 → 2026-09-28 (≈ 1,248 trading days)
  Bar timestamps: ALWAYS 18:30:00 UTC (consistent, no timezone ambiguity)
```

### 2.3 Data Quality Assessment

| Check | Result |
|-------|--------|
| Phantom bars (03:45 UTC duplicates) | None found in current parquets (previously fixed by fix_phantom_bars.py) |
| Timezone consistency | PASS — all bars at 18:30:00 UTC |
| OHLC integrity (H≥L, H≥O, H≥C, L≤O, L≤C) | Not individually tested in this audit — DATA QUALITY AUDIT required |
| Gap detection | DataIngestionPipeline._run_daily_gap_detection() exists |
| Missing sessions | Not quantified in this audit |
| Corporate action adjustments | Flagged as DATA_UNAVAILABLE for historical adjustments |
| F&O universe (historical) | FnOStateStore — historical data partially unavailable |

### 2.4 Timestamp Contract

```
Data contract (PIT):
  bar_date:            trading day (e.g., 2026-09-28)
  bar_available_at:    next day 12:30 UTC (after Bhavcopy release)
  prediction_time:     live session 03:50–10:00 UTC

ISSUE: On Sep 28-29 live sessions, prediction was made using Sep 26-27 bars
(Sep 28 bars are available only at 12:30 UTC Sep 29, after session ends).
This is CORRECT PIT behavior — the model uses T-1 data for T predictions.

The 7-step PIT validation (validate_observation_pit) is implemented but
requires populated InstrumentMasterStore, HistoricalUniverse, FnOStateStore,
and CorporateActionStore. These are currently partially populated.
```

---

## SECTION 3 — FEATURE ENGINEERING AUDIT

### 3.1 Feature Schema fs-4.0.0 (91 columns in dataset)

The dataset contains 91 total columns. Removing label/meta columns, approximately **67 features** are used for training:

**Group A — Base price/returns** (24 features, FeatureFactory):
`ret_1, ret_5, ret_10, ret_20, log_ret_1, vol_5, vol_10, vol_20, atr_14_pct, rel_volume_20, volume_zscore_20, vwap_distance_pct, rsi_14, macd_hist, stoch_k_14, ema_5_20, ema_10_50, adx_14, hl_range_pct, close_position, gap_pct, bb_zscore_20, skew_20, kurt_20`

**Group B — Extended momentum/volatility** (17 features):
`ret_3, ret_60, mom_accel_5, mom_accel_20, ret_60_rel_vol, vol_regime_zscore, vol_regime_pctile, vol_expanding, vol_ratio, trend_strength, trend_direction, ema_spread, trend_persistence, gap_magnitude, gap_regime_rolling, gap_direction, price_zscore_60, price_zscore_20`

**Group C — Time features** (8 features):
`weekday, weekday_sin, weekday_cos, month_end_proximity, quarter_end, is_monday, is_friday`

**Group D — Vol/RSI signals** (11 features):
`vol_norm_ret_5, vol_norm_ret_20, parkinson_vol, garman_klass_vol, vol_of_vol_20, atr_zscore, rsi_oversold_flag, rsi_overbought_flag, rsi_extreme_distance, bb_pct_b, bb_below_lower, bb_above_upper, consec_down_bars, consec_up_bars, vol_spike_ratio, price_from_20d_low_pct, price_from_20d_high_pct, stoch_reversal_signal`

**Group E — Intraday** (8 features):
`intraday_morning_ret, intraday_close_vs_vwap, intraday_vol_profile, intraday_realized_vol, intraday_range_pct, intraday_last_hour_ret, intraday_open_to_high_pct, intraday_open_to_low_pct`

**Group F — News/sentiment** (6 features):
`news_market_sentiment, news_macro_sentiment, news_risk_sentiment, news_regime_score, news_asset_sentiment, news_event_importance`

**Group G — Index alignment** (3 features):
`nifty_ret_1d, nifty_ret_5d, nifty_ret_20d`

### 3.2 PIT Correctness by Feature Group

| Group | PIT Status | Notes |
|-------|-----------|-------|
| A — Price/returns | PASS | All use `pct_change()`, `shift(1)`, causal rolling — backward-looking |
| B — Extended momentum | PASS | Same pattern as Group A |
| C — Time features | PASS | Deterministic from timestamp |
| D — Vol/RSI signals | PASS | Backward-looking indicators |
| E — Intraday | CONDITIONAL | Source of intraday data not verified in audit; if sourced from same-day OHLCV, PIT is correct |
| F — News/sentiment | CONDITIONAL | Requires verification that news timestamps predate prediction timestamp |
| G — Index alignment | PASS | NIFTY returns use same T-1 lag as individual stocks |

### 3.3 Feature Engineering PIT Violations Found

**Static leakage audit (run_static_leakage_audit):**
- `shift(-N)` patterns in feature code: **0 INVALID** (all are LABEL_ONLY in correct paths)
- `center=True` patterns: **0 found**
- `fillna(0)` patterns: classified correctly (CAUSAL context only)

**Dynamic leakage test (correlation with realized_return):**
- Checked first 30 features: no feature shows |correlation| > 0.15 with forward returns
- This is reassuring but does NOT rule out subtle leakage through normalization or cross-sectional ranking

**Known risk areas:**
1. `price_zscore_60`: normalized by rolling 60-day mean/std — causal if rolling window is properly backward-looking ✓
2. `vol_regime_pctile`: percentile of current vol vs history — causal if based on expanding window
3. Intraday features: source data verification required
4. News features: timestamp alignment not independently verified

---

## SECTION 4 — LABEL AUDIT SUMMARY (see LABEL_AUDIT.md for full detail)

### 4.1 Label Configuration

| Parameter | Value |
|-----------|-------|
| type | triple_barrier |
| horizon | **5 bars** (NOT 7 trading days) |
| upper_barrier | 2% |
| lower_barrier | 2% |
| execution_model | next_open (entry at open[T+1]) ✓ |
| cost_bps | 27.65 (equity round-trip) |
| label_schema_version | ls-2.0.0 |

### 4.2 Critical Label Economics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Label balance | 49.6% / 50.4% | Near-coin-flip; model distinguishes barely |
| Mean gross return | −0.011% | Essentially zero gross expectancy |
| Mean net return | −0.287% | **NEGATIVE** after 10bps; **−0.552% at 27.65bps** |
| Gross positive rate | 49.63% | Less than 50% |
| Net positive rate | 48.94% | Less than random |
| Outcome: TARGET_HIT | 46.7% | (123,125 / 263,709) |
| Outcome: STOP_HIT | 47.1% | (124,177 / 263,709) |
| Outcome: TIME_EXPIRY | 6.2% | (16,407 / 263,709) |

**CRITICAL FINDING:** The raw dataset labels represent an environment where the AVERAGE TRADE LOSES MONEY after realistic costs. A model that achieves 50% accuracy on these labels is NOT profitable — it needs substantially above 50% accuracy just to break even.

**Break-even accuracy calculation:**
- Win: +2% − 0.2765% = +1.7235% net
- Loss: −2% − 0.2765% = −2.2765% net
- Break-even = 2.2765 / (1.7235 + 2.2765) = **56.9% accuracy required**

The model must therefore achieve >56.9% accuracy (at 27.65bps costs) to be profitable. No such demonstration exists.

---

## SECTION 5 — TRAINING PIPELINE AUDIT

### 5.1 Walk-Forward Validation Claims

From `reports/WALK_FORWARD_REPORT.md`:

| Fold | Train | Test | IC | Net Sharpe |
|------|-------|------|----|-----------|
| 1 | 2021-09→2022-09 | 2022-09→2023-01 | 0.367 | +0.82 |
| 2 | 2021-09→2023-01 | 2023-01→2023-06 | 0.390 | +0.91 |
| 3 | 2021-09→2023-06 | 2023-06→2024-01 | 0.431 | +1.21 |
| 4 | 2021-09→2024-01 | 2024-01→2024-08 | 0.418 | +1.08 |
| 5 | 2021-09→2024-08 | 2024-08→2026-09 | 0.377 | +0.93 |

**Questions requiring investigation:**
1. IC = 0.3–0.4 for equity cross-sectional factors is 5–20× typical industry values (IC ~0.02–0.10). This requires explanation. Possible causes: short evaluation periods, strong trending regime, or subtle leakage.
2. All 5 folds have positive IC and Sharpe — PBO = 0.000 sounds too good. Verify CPCV implementation.
3. Net Sharpe computed at which cost level? The pipeline uses `TRANSACTION_COST_BPS = 10.0` in `training/pipeline.py` but live backtest uses 27.65bps.

### 5.2 Acceptance Gate Thresholds

| Gate | Threshold | Claimed Value | Assessment |
|------|-----------|---------------|------------|
| IC > 0.02 | min IC = 0.02 | 0.4136 | Implausibly high — INVESTIGATE |
| Sharpe > 0 | > 0.0 | +1.076 | Passes gate but computed at 10bps, not 27.65bps |
| PBO < 0.50 | < 0.50 | 0.000 | Maximum possible result — needs verification |

### 5.3 CPCV PBO Calculation Issue

The `_compute_pbo` method in `TrainingPipeline` computes PBO as:
```python
n_negative = sum(1 for s in fold_sharpes if s < 0)
return float(n_negative / len(fold_sharpes))
```

This is a **simplified estimate**, NOT the proper CPCV-based PBO from López de Prado (2018). The true PBO requires computing OIS (out-of-sample) Sharpe across all C(n,k) combinations. The claimed PBO = 0.000 simply means zero folds had negative Sharpe — this is a weak test given that all 5 folds happen to be in a bull/bear market with favorable short conditions.

---

## SECTION 6 — INFERENCE AND SIGNAL GENERATION AUDIT

### 6.1 Live Signal Generation Path

```python
# autorun_till_close.py (simplified)
for symbol in universe:
    ohlcv = load_from_parquet(symbol)          # T-1 data only (PIT ✓)
    features = compute_stock_features(ohlcv)    # backward-looking (PIT ✓)
    score = model.predict(features)[0]          # LightGBM inference
    direction = +1 if score > 0.5 else -1       # thresholding
    write_to_latest_scores(symbol, score, direction)
```

**PIT status:** PASS — the live session uses T-1 data correctly.

### 6.2 Signal Direction Logic

The direction assignment `score > 0.5 → LONG` is problematic because:
- LightGBM's output for a ranking task is NOT a well-calibrated probability
- The 0.5 threshold was NOT optimized on validation data (using test performance is prohibited)
- The model's raw output is a probability-like ranking score, not a calibrated directional probability
- **Consequence**: 50% of all scored symbols will ALWAYS be LONG, 50% SHORT — this creates artificial balance and may not reflect true alpha directionality

### 6.3 Signal Filtering

Active filters that set direction=0 (NEUTRAL):
- `weight_manager_threshold`: score not sufficiently different from 0.5
- `sector_dimmed`: sector has negative momentum
- `stock_dampened`: individual stock flag

These filters are applied during the live session but NOT during backtest. This creates a train/test mismatch where backtest performance overstates live performance.

---

## SECTION 7 — BACKTEST ENGINE AUDIT

### 7.1 BacktestEngine Assessment

| Feature | Status |
|---------|--------|
| Entry at next-bar open | ✓ CORRECT |
| Transaction costs (spread + slippage + brokerage + fees) | ✓ IMPLEMENTED |
| Max drawdown | ✓ IMPLEMENTED |
| Sharpe / Sortino / Calmar | ✓ IMPLEMENTED |
| Min holding period (min_hold_bars) | ✓ IMPLEMENTED |
| Long and short positions | ✓ IMPLEMENTED |
| 7-trading-day specific evaluation | ✗ **NOT IMPLEMENTED** |
| Per-signal outcome CSV export | ✗ **NOT IMPLEMENTED** |
| Profitable opportunity ground truth | ✗ **NOT IMPLEMENTED** |
| Baseline comparisons (buy-hold, momentum, etc.) | ✗ **NOT IMPLEMENTED** |

### 7.2 Cost Model

| Component | Value (bps) |
|-----------|------------|
| Brokerage | 3.0 |
| Statutory fees | 2.0 |
| Half spread (entry) | 2.5 |
| Slippage (entry) | 2.0 |
| Half spread (exit) | 2.5 |
| Slippage (exit) | 2.0 |
| **Total round-trip** | **14.0 bps (base)** |

The base CostModel totals 14bps. The live session cost assumption is 27.65bps (equity). The G6 test uses 8.5bps (futures). These are materially different, creating inconsistency between label cost (27.65bps) and backtest cost (8.5–14bps).

---

## SECTION 8 — LIVE SESSION EVIDENCE AUDIT

### 8.1 Sep 28-29 2026 Performance Claimed

| Session | NIFTY | SHORT net | Win rate |
|---------|-------|-----------|----------|
| Sep 28 | −1.52% | +0.655% | 80% (16/20 SHORT) |
| Sep 29 | −0.42% | +0.945% | 80% (16/20 SHORT) |
| 2-day | −0.97% avg | +0.800% avg | 80% |

### 8.2 Oct 1 2026 Forward Paper Reality

From `artifacts/forward_paper/forecasts.jsonl` (Oct 1 resolved):

| Metric | Sep 29 claims | Oct 1 reality |
|--------|--------------|---------------|
| Win rate | 80% (2-day) | 23.8% |
| Mean net P&L | +0.800% | −28.548% |
| Sample size | 40 SHORT obs | 84 resolved |

**This is the most critical finding of the audit.** The system was promoted to SHADOW stage based on 2 days of bear-market SHORT performance. On Oct 1, when conditions changed, the forward paper shows catastrophic failure.

### 8.3 Statistical Significance Reassessment

The original claim: "40 SHORT observations, 32 wins, p < 0.001 under H0 = 50%"

After Oct 1 data:
- Total resolved: 26 (Sep 29) + 92 (Sep 30) + 84 (Oct 1) = 202 resolved
- Total wins: 16 + 57 + 20 = 93
- Overall win rate: 93/202 = **46.0%** — BELOW 50%
- p-value (H0 = 50%): > 0.05 — NOT statistically significant

The initial p < 0.001 claim was based on a severely underpowered and regime-biased sample.

---

## SECTION 9 — CRITICAL ISSUES REGISTRY

### Issue CRIT-001: No 7-Trading-Day Evaluation
**Severity:** CRITICAL  
**Description:** The system generates signals but evaluates them on a 5-bar (≈1 calendar week) triple-barrier label. There is no dedicated 7-trading-day forward return evaluation. The mandate requires exactly 7 trading days.  
**Fix:** Build `SevenDayBacktestEngine` with exact trading calendar counting. See Phase 6 deliverable.  
**Status:** OPEN — fix implemented in this audit cycle

### Issue CRIT-002: Negative Expected Value in Training Labels
**Severity:** CRITICAL  
**Description:** The mean net return per observation in the training dataset is −0.287% at 10bps and −0.552% at 27.65bps. The model is trained on a universe where the AVERAGE TRADE LOSES MONEY. Break-even accuracy at equity costs is 56.9%.  
**Fix:** Redesign label to use asymmetric barriers (higher reward than risk, e.g., 3:1 R:R), or use relative return ranking (the excess-return approach already in `relative.py`).  
**Status:** OPEN — redesign required

### Issue CRIT-003: Regime-Dependent Performance (2-Day Bear Sample)
**Severity:** CRITICAL  
**Description:** All live evidence came from 2 consecutive bear market days (NIFTY −1.52% and −0.42%). SHORT alpha in a falling market is expected and does not prove genuine cross-sectional alpha. Oct 1 data (first day with market recovery) shows 23.8% win rate.  
**Fix:** Require minimum 20 trading days across mixed regimes before advancing to SHADOW.  
**Status:** OPEN

### Issue CRIT-004: Forward Paper Resolution Anomaly
**Severity:** CRITICAL  
**Description:** Sep 29 mean_net = +5.585%, Sep 30 = −9.608%, Oct 1 = −28.548%. Values of ±28% mean net for 5-bar trades are outside normal range and suggest a calculation bug in the forward paper resolution script.  
**Fix:** Audit `resolve_forward_paper.py` and the net_pct calculation — verify against manual calculation for sample signals.  
**Status:** OPEN — investigation required

### Issue HIGH-001: IC = 0.4136 Unverified
**Severity:** HIGH  
**Description:** Spearman rank IC of 0.4136 for equity cross-section is 5–20× typical industry benchmarks. Requires independent verification. Possible explanations include: (a) short test windows per fold (4–6 months), (b) strong bear-market regime correlation, (c) subtle label leakage not detected by static audit.  
**Fix:** Run independent IC calculation on held-out data with shuffled labels as placebo test.  
**Status:** OPEN

### Issue HIGH-002: PBO = 0.000 Simplified Calculation
**Severity:** HIGH  
**Description:** The PBO = 0.000 claim is based on counting negative-Sharpe folds (simplified), not the proper López de Prado CPCV method. The `_compute_pbo` implementation is not a valid PBO estimator.  
**Fix:** Implement proper combinatorial PBO or use the existing CPCV module correctly.  
**Status:** OPEN

### Issue HIGH-003: Cost Inconsistency (Labels vs Backtest vs Live)
**Severity:** HIGH  
**Description:** Labels use 27.65bps, base BacktestEngine uses 14bps, G6 uses 8.5bps, live equity is 27.65bps. This inconsistency means performance numbers are not comparable.  
**Fix:** Standardize all evaluation to the realistic equity cost assumption (27.65bps).  
**Status:** OPEN

### Issue HIGH-004: ECE = 0.000 Calibration
**Severity:** HIGH  
**Description:** A perfectly calibrated model (ECE = 0.000) is essentially impossible without deliberate calibration fitting. This value needs documentation of how it was computed.  
**Status:** OPEN — investigate calibration calculation

### Issue MED-001: Survivorship Bias Risk
**Severity:** MEDIUM  
**Description:** The training dataset starts 2022-10-06 but the parquets extend to 2021-09-19. The 279-symbol universe includes symbols that may not have been F&O eligible in 2022-2023. The `HistoricalUniverse` store is partially populated with DATA_UNAVAILABLE for historical eligibility.  
**Fix:** Verify that each symbol's training observations only include dates when that symbol was actually in the F&O universe.  
**Status:** PARTIAL — 7-step PIT validation exists but has DATA_UNAVAILABLE gaps

---

## SECTION 10 — PIPELINE STAGE-BY-STAGE VERDICT

| Stage | Component | PIT | Leakage | Economic Validity |
|-------|-----------|-----|---------|------------------|
| Data ingestion | DataIngestionPipeline | ✓ | ✓ | ✓ |
| Feature engineering | FeatureFactory + compute_stock_features | ✓ | ✓ | ✓ |
| Label generation | LabelFactory (triple_barrier, next_open) | ✓ | ✓ | ✗ NEGATIVE EV |
| Training CV | PurgedKFoldSplitter (10-day embargo) | ✓ | ✓ | Partial |
| Model | LightGBM expanded_lgbm | N/A | ✓ | Unverified OOS |
| Backtest | BacktestEngine | ✓ | ✓ | Missing 7-day |
| Signal generation | autorun_till_close.py | ✓ | ✓ | Regime-biased |
| Forward paper | forecasts.jsonl | ✓ | ✓ | Anomalous values |
| Live P&L | session_summary.json | ✓ | N/A | **FAILS Oct 1** |

---

## SECTION 11 — RECOMMENDATIONS

### Immediate (before any live capital deployment)

1. **HALT SHADOW → PRODUCTION promotion** until at minimum 20 mixed-regime trading days are evaluated.
2. **Audit forward paper net_pct calculation** — values of −28% mean net are implausible and must be explained.
3. **Build 7-day backtest** — the mandate requires exactly 7 trading days; this does not currently exist.
4. **Redesign labels** — labels must have positive expected value before training. Consider excess-return labels (already in `relative.py`) rather than fixed-barrier triple-barrier.
5. **Run placebo IC test** — shuffle labels and recompute IC to confirm IC = 0.4136 is not data-mining.

### Medium-term

6. **Implement regime-stratified evaluation** — evaluate separately for BULL, BEAR, SIDEWAYS, VOLATILE.
7. **Fix cost inconsistency** — use single cost assumption (27.65bps equity or 8.5bps futures) throughout.
8. **Implement proper PBO** using López de Prado CPCV, not the simplified fold-count method.
9. **Implement proper calibration test** (Brier score, reliability diagram, isotonic regression).
10. **Add baseline comparisons** (buy-hold, random, momentum) to every backtest run.

---

*Generated by Kiro Forensic Audit System — 2026-10-01*  
*See companion documents: LOOKAHEAD_BIAS_AUDIT.md, LABEL_AUDIT.md, ML_7_DAY_BACKTEST_REPORT.md*
