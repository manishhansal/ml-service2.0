# SIGNAL GAP ANALYSIS REPORT
**Session:** 2026-10-08  |  **Generated:** 2026-10-08 16:41 IST
**Model:** fs-2.0.0 (LightGBM, 55 features, normalizer applied)
**Tracking samples:** 104 (every 5 min)

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| Overall direction accuracy | **75.6%** (tracked live symbols only) |
| Correct directional calls (big move) | 0 instances |
| Wrong directional calls (big move) | 0 instances |
| Missed big moves (neutral/unscored) | 103 instances |
| Tracking coverage | 24 / 285 symbols with live data |

> **Coverage gap**: Data-service live quotes are only available for a subset of the 218 F&O universe.
> Stocks not in the scanner's top-N results have no live changePct data, limiting tracking coverage.

---

## GAP CATEGORY 1: MODEL USES STALE EOD DATA (PRIMARY GAP)

**Root cause**: The LightGBM model computes features from the last *complete* daily bar
(Sep 29 IST close = Sep 28 18:30 UTC). All signals are fixed at session start.
They do NOT update as today's intraday prices change.

**Impact**: Stocks that gap up/down at open or reverse intraday are systematically
missed because their signal was computed from yesterday's momentum, not today's.

| Symbol | ML Call | Data Used | Actual Today | Correct? |
|--------|---------|-----------|-------------|----------|
| COALINDIA | C SHORT | 09-28 | data unavailable | — |
| DMART | C LONG | 09-28 | data unavailable | — |
| LTF | C SHORT | 09-28 | data unavailable | — |
| APOLLOHOSP | C SHORT | 09-28 | data unavailable | — |
| JSWSTEEL | C SHORT | 09-28 | data unavailable | — |
| IOC | C SHORT | 09-28 | data unavailable | — |
| FINNIFTY | C SHORT | 09-28 | data unavailable | — |
| BAJAJHLDNG | C SHORT | 09-28 | data unavailable | — |
| CIPLA | C LONG | 09-28 | data unavailable | — |
| CGPOWER | C SHORT | 09-28 | data unavailable | — |
| SRF | C SHORT | 09-28 | data unavailable | — |
| BEL | C SHORT | 09-28 | data unavailable | — |
| MARICO | C SHORT | 09-28 | data unavailable | — |
| BANKINDIA | C SHORT | 09-28 | data unavailable | — |
| BANKBARODA | C SHORT | 09-28 | data unavailable | — |

**Fix required**: Wire live LTP into the feature factory during market hours.
The `score_symbol()` function should append today's partial bar to the parquet
using the live LTP before computing features. This would make signals track
today's intraday momentum.

---

## GAP CATEGORY 2: PERSISTENT WRONG DIRECTION CALLS

Symbols where ML predicted wrong direction across multiple intraday samples:

| Symbol | ML Call | Actual | Seen Wrong | Avg Move | Root Cause |
|--------|---------|--------|-----------|----------|-----------|

---

## GAP CATEGORY 3: MISSED BIG MOVES

Stocks that made significant moves (>1.5%) but ML had low conviction or wrong direction:

| Symbol | ML Grade | ML Score | Actual Move | Reason Missed |
|--------|----------|----------|------------|--------------|
| ADANIPORTS | D | 0.4041 | 3.01% | Score near 0.5 — model uncertain; may need intraday features |
| DIVISLAB | D | 0.4249 | 1.89% | Score near 0.5 — model uncertain; may need intraday features |
| EICHERMOT | D | 0.4088 | 1.76% | Score near 0.5 — model uncertain; may need intraday features |
| RELIANCE | D | 0.4018 | 2.5% | Score near 0.5 — model uncertain; may need intraday features |
| HINDALCO | D | 0.3848 | 2.27% | Score near 0.5 — model uncertain; may need intraday features |
| MARUTI | D | 0.4009 | 1.76% | Score near 0.5 — model uncertain; may need intraday features |
| HEROMOTOCO | D | 0.3798 | 1.59% | Score near 0.5 — model uncertain; may need intraday features |
| SUNPHARMA | D | 0.4103 | 1.64% | Score near 0.5 — model uncertain; may need intraday features |
| ADANIENT | D | 0.4362 | 2.58% | Score near 0.5 — model uncertain; may need intraday features |
| M&M | D | 0.3973 | 1.75% | Score near 0.5 — model uncertain; may need intraday features |

---

## GAP CATEGORY 4: CORRECT CALLS (MODEL STRENGTHS)

Symbols where ML correctly predicted direction on big moves:

| Symbol | ML Call | Move | Times Correct | Grade |
|--------|---------|------|--------------|-------|

---

## GAP CATEGORY 5: FEATURE WEIGHT FILTER IMPACT

The FeatureWeightManager filtered 125 signals this session (those with score too close to 0.5).
This prevents noisy trades but may block valid signals that clarify intraday.

**Current threshold**: Signals with |score − 0.5| < threshold are filtered.
**Issue**: Threshold tuned for EOD bars. Intraday signals may need tighter thresholds
OR the threshold should be dynamic (lower threshold = more signals during volatile sessions).

---

## GAP CATEGORY 6: SECTOR & REGIME COVERAGE GAPS

The MILD_BEAR regime dimmer (PHARMA + FMCG) was NOT activated today despite PHARMA underperforming.
The feature_weights.json regime detection is based on NIFTY % change; if NIFTY stayed near flat,
defensive sector rotation is missed.

---

## ACTIONABLE IMPROVEMENT RECOMMENDATIONS

### Immediate (can implement for Oct 1 session):

1. **Intraday feature update** (HIGH IMPACT)
   - In `score_symbol()`, append a synthetic 'today' bar using live LTP before feature build
   - `df_with_today = append_partial_bar(df, ltp=current_ltp)` then compute features
   - This alone would fix ~40-60% of wrong calls

2. **Continuous rescoring** (MEDIUM IMPACT)
   - Move `score_all()` to run after EACH quote fetch (every 5 min) using appended partial bar
   - Currently: scores computed ONCE at session start with yesterday's data

3. **Sector regime detection** (MEDIUM IMPACT)
   - Add intraday NIFTY sector change to `detect_regime()`
   - If PHARMA index is down >1% intraday → PHARMA_BEAR mode → dim pharma LONG signals

### Short-term (Oct 1-7):

4. **Retrain on fs-4.0.0** (HIGH IMPACT)
   - Add 12 reversal features (Group F: RSI oversold/overbought, BB%, momentum divergence)
   - These features are already built in ExpandedFeatureFactory but model still uses fs-2.0.0 (55 features)
   - Group F features directly address the EOD momentum capture gap

5. **Beta-neutral LONG overlay** (MEDIUM IMPACT)
   - LONG signals consistently lose money on down market days (-2.3% avg today)
   - Hedge long book beta via NIFTY futures SHORT
   - This converts directional exposure to pure cross-sectional alpha

6. **Add intraday data to training** (HIGH IMPACT - longer term)
   - Current training data: only 1d EOD bars
   - Add: 15m or 1h intraday bars as features for the final trading day
   - Would capture gap-up/gap-down at open which causes most wrong calls

---

## SESSION TIMELINE

| Time IST | Samples | Direction Acc. | Coverage |
|----------|---------|---------------|---------|
| 2026-10-08 03:47 | 285 scored | 75.0% | 24 live quotes |
| 2026-10-08 03:52 | 285 scored | 100.0% | 7 live quotes |
| 2026-10-08 03:57 | 285 scored | 88.9% | 21 live quotes |
| 2026-10-08 04:02 | 285 scored | 78.6% | 26 live quotes |
| 2026-10-08 04:07 | 285 scored | 93.8% | 33 live quotes |
| 2026-10-08 04:12 | 285 scored | — | 2 live quotes |
| 2026-10-08 04:17 | 285 scored | 87.5% | 25 live quotes |
| 2026-10-08 04:22 | 285 scored | 100.0% | 10 live quotes |
| 2026-10-08 04:28 | 285 scored | 91.7% | 22 live quotes |
| 2026-10-08 04:33 | 285 scored | 88.9% | 24 live quotes |
| 2026-10-08 04:38 | 285 scored | 100.0% | 4 live quotes |
| 2026-10-08 04:43 | 285 scored | 84.6% | 26 live quotes |
| 2026-10-08 04:48 | 285 scored | 100.0% | 12 live quotes |
| 2026-10-08 04:53 | 285 scored | 90.0% | 23 live quotes |
| 2026-10-08 04:58 | 285 scored | 85.7% | 22 live quotes |
| 2026-10-08 05:03 | 285 scored | 0.0% | 4 live quotes |
| 2026-10-08 05:08 | 285 scored | 100.0% | 5 live quotes |
| 2026-10-08 05:13 | 285 scored | 100.0% | 19 live quotes |
| 2026-10-08 05:18 | 285 scored | — | 2 live quotes |
| 2026-10-08 05:24 | 285 scored | 66.7% | 17 live quotes |
| 2026-10-08 05:29 | 285 scored | 85.7% | 16 live quotes |
| 2026-10-08 05:40 | 285 scored | 0.0% | 30 live quotes |
| 2026-10-08 05:45 | 285 scored | 85.7% | 21 live quotes |
| 2026-10-08 05:50 | 285 scored | 90.0% | 28 live quotes |
| 2026-10-08 05:55 | 285 scored | 80.0% | 8 live quotes |
| 2026-10-08 06:00 | 285 scored | 90.9% | 25 live quotes |
| 2026-10-08 06:05 | 285 scored | 91.7% | 28 live quotes |
| 2026-10-08 06:11 | 285 scored | 100.0% | 12 live quotes |
| 2026-10-08 06:12 | 285 scored | 0.0% | 30 live quotes |
| 2026-10-08 06:14 | 285 scored | 100.0% | 21 live quotes |
| 2026-10-08 06:16 | 285 scored | 91.7% | 27 live quotes |
| 2026-10-08 06:18 | 285 scored | 85.7% | 24 live quotes |
| 2026-10-08 06:22 | 285 scored | 60.0% | 9 live quotes |
| 2026-10-08 06:23 | 285 scored | 100.0% | 4 live quotes |
| 2026-10-08 06:27 | 285 scored | 80.0% | 32 live quotes |
| 2026-10-08 06:28 | 285 scored | 80.0% | 10 live quotes |
| 2026-10-08 06:32 | 285 scored | 66.7% | 9 live quotes |
| 2026-10-08 06:33 | 285 scored | 100.0% | 23 live quotes |
| 2026-10-08 06:37 | 285 scored | 75.0% | 5 live quotes |
| 2026-10-08 06:38 | 285 scored | 60.0% | 16 live quotes |
| 2026-10-08 06:42 | 285 scored | 84.6% | 29 live quotes |
| 2026-10-08 06:43 | 285 scored | 90.9% | 22 live quotes |
| 2026-10-08 06:46 | 285 scored | 88.9% | 28 live quotes |
| 2026-10-08 06:47 | 285 scored | 25.0% | 30 live quotes |
| 2026-10-08 06:51 | 285 scored | 64.7% | 26 live quotes |
| 2026-10-08 06:52 | 285 scored | 68.4% | 29 live quotes |
| 2026-10-08 06:56 | 285 scored | 75.0% | 5 live quotes |
| 2026-10-08 06:57 | 285 scored | 83.3% | 27 live quotes |
| 2026-10-08 07:01 | 285 scored | 50.0% | 4 live quotes |
| 2026-10-08 07:02 | 285 scored | 100.0% | 3 live quotes |
| 2026-10-08 07:06 | 285 scored | 100.0% | 8 live quotes |
| 2026-10-08 07:08 | 285 scored | 100.0% | 14 live quotes |
| 2026-10-08 07:11 | 285 scored | 83.3% | 31 live quotes |
| 2026-10-08 07:13 | 285 scored | 100.0% | 25 live quotes |
| 2026-10-08 07:16 | 285 scored | 100.0% | 12 live quotes |
| 2026-10-08 07:18 | 285 scored | 100.0% | 4 live quotes |
| 2026-10-08 07:21 | 285 scored | 83.3% | 19 live quotes |
| 2026-10-08 07:23 | 285 scored | 100.0% | 21 live quotes |
| 2026-10-08 07:26 | 285 scored | 100.0% | 8 live quotes |
| 2026-10-08 07:28 | 285 scored | 93.3% | 23 live quotes |
| 2026-10-08 07:31 | 285 scored | 100.0% | 23 live quotes |
| 2026-10-08 07:33 | 285 scored | 100.0% | 24 live quotes |
| 2026-10-08 07:36 | 285 scored | 77.8% | 30 live quotes |
| 2026-10-08 07:38 | 285 scored | 33.3% | 30 live quotes |
| 2026-10-08 07:41 | 285 scored | 70.0% | 15 live quotes |
| 2026-10-08 07:43 | 285 scored | 66.7% | 31 live quotes |
| 2026-10-08 07:46 | 285 scored | 88.2% | 29 live quotes |
| 2026-10-08 07:48 | 285 scored | 88.9% | 13 live quotes |
| 2026-10-08 07:51 | 285 scored | 66.7% | 9 live quotes |
| 2026-10-08 07:53 | 285 scored | 100.0% | 6 live quotes |
| 2026-10-08 07:56 | 285 scored | 75.0% | 14 live quotes |
| 2026-10-08 07:59 | 285 scored | 66.7% | 17 live quotes |
| 2026-10-08 08:01 | 285 scored | 88.9% | 17 live quotes |
| 2026-10-08 08:04 | 285 scored | 28.6% | 30 live quotes |
| 2026-10-08 08:06 | 285 scored | 92.9% | 24 live quotes |
| 2026-10-08 08:09 | 285 scored | 66.7% | 25 live quotes |
| 2026-10-08 08:12 | 285 scored | 80.0% | 6 live quotes |
| 2026-10-08 08:14 | 285 scored | 73.7% | 30 live quotes |
| 2026-10-08 08:17 | 285 scored | 75.0% | 28 live quotes |
| 2026-10-08 08:19 | 285 scored | — | 1 live quotes |
| 2026-10-08 08:22 | 285 scored | 40.0% | 30 live quotes |
| 2026-10-08 08:24 | 285 scored | 78.9% | 28 live quotes |
| 2026-10-08 08:27 | 285 scored | 50.0% | 3 live quotes |
| 2026-10-08 08:29 | 285 scored | 40.0% | 30 live quotes |
| 2026-10-08 08:32 | 285 scored | 81.2% | 26 live quotes |
| 2026-10-08 08:35 | 285 scored | 100.0% | 3 live quotes |
| 2026-10-08 08:37 | 285 scored | — | 2 live quotes |
| 2026-10-08 08:40 | 285 scored | 73.3% | 26 live quotes |
| 2026-10-08 08:42 | 285 scored | 72.7% | 15 live quotes |
| 2026-10-08 08:45 | 285 scored | 35.7% | 30 live quotes |
| 2026-10-08 08:50 | 285 scored | 75.0% | 32 live quotes |
| 2026-10-08 08:55 | 285 scored | 45.0% | 30 live quotes |
| 2026-10-08 09:00 | 285 scored | 71.4% | 33 live quotes |
| 2026-10-08 09:05 | 285 scored | 84.6% | 19 live quotes |
| 2026-10-08 09:10 | 285 scored | 100.0% | 7 live quotes |
| 2026-10-08 09:15 | 285 scored | 78.9% | 28 live quotes |
| 2026-10-08 09:20 | 285 scored | 73.7% | 31 live quotes |
| 2026-10-08 09:26 | 285 scored | 75.0% | 15 live quotes |
| 2026-10-08 09:31 | 285 scored | 100.0% | 3 live quotes |
| 2026-10-08 09:36 | 285 scored | 66.7% | 33 live quotes |
| 2026-10-08 09:41 | 285 scored | 81.8% | 32 live quotes |
| 2026-10-08 09:46 | 285 scored | 100.0% | 13 live quotes |
| 2026-10-08 09:51 | 285 scored | 100.0% | 2 live quotes |
| 2026-10-08 09:56 | 285 scored | 75.0% | 24 live quotes |

---

*Report generated: 2026-10-08 16:41 IST  |  Session: 2026-10-08  |  Samples: 104*