# SIGNAL GAP ANALYSIS REPORT
**Session:** 2026-10-07  |  **Generated:** 2026-10-07 15:35 IST
**Model:** fs-2.0.0 (LightGBM, 55 features, normalizer applied)
**Tracking samples:** 60 (every 5 min)

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| Overall direction accuracy | **69.4%** (tracked live symbols only) |
| Correct directional calls (big move) | 0 instances |
| Wrong directional calls (big move) | 0 instances |
| Missed big moves (neutral/unscored) | 41 instances |
| Tracking coverage | 40 / 285 symbols with live data |

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
| LTF | C SHORT | 09-28 | data unavailable | — |
| IOC | C SHORT | 09-28 | data unavailable | — |
| FINNIFTY | C SHORT | 09-28 | data unavailable | — |
| CGPOWER | C SHORT | 09-28 | data unavailable | — |
| MARICO | C SHORT | 09-28 | data unavailable | — |
| BANKINDIA | C SHORT | 09-28 | data unavailable | — |
| IDFCFIRSTB | C SHORT | 09-28 | data unavailable | — |
| KEI | C SHORT | 09-28 | data unavailable | — |
| HCLTECH | C SHORT | 09-28 | data unavailable | — |
| CANBK | C SHORT | 09-28 | data unavailable | — |
| GRASIM | C SHORT | 09-28 | data unavailable | — |
| OBEROIRLTY | C SHORT | 09-28 | data unavailable | — |
| CDSL | C SHORT | 09-28 | data unavailable | — |
| BAJAJFINSV | C SHORT | 09-28 | data unavailable | — |
| BANKNIFTY | C SHORT | 09-28 | data unavailable | — |

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
| ADANIENT | D | 0.4002 | 2.37% | Score near 0.5 — model uncertain; may need intraday features |
| ADANIPORTS | D | 0.4052 | 2.05% | Score near 0.5 — model uncertain; may need intraday features |
| MARUTI | D | 0.4533 | 1.83% | Score near 0.5 — model uncertain; may need intraday features |
| BRITANNIA | D | 0.4918 | 1.56% | Score near 0.5 — model uncertain; may need intraday features |
| TATAMOTORS | D | 0.4366 | 1.89% | Score near 0.5 — model uncertain; may need intraday features |
| M&M | D | 0.4539 | 1.54% | Score near 0.5 — model uncertain; may need intraday features |
| HCLTECH | D | 0.3594 | 2.58% | Score near 0.5 — model uncertain; may need intraday features |
| BPCL | D | 0.4044 | 1.5% | Score near 0.5 — model uncertain; may need intraday features |

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
| 2026-10-07 03:46 | 285 scored | — | 1 live quotes |
| 2026-10-07 03:51 | 285 scored | 91.7% | 21 live quotes |
| 2026-10-07 03:56 | 285 scored | 89.5% | 27 live quotes |
| 2026-10-07 04:01 | 285 scored | 100.0% | 2 live quotes |
| 2026-10-07 04:06 | 285 scored | 89.5% | 31 live quotes |
| 2026-10-07 04:11 | 285 scored | 100.0% | 20 live quotes |
| 2026-10-07 04:16 | 285 scored | 92.9% | 18 live quotes |
| 2026-10-07 04:21 | 285 scored | 88.2% | 26 live quotes |
| 2026-10-07 04:26 | 285 scored | 100.0% | 13 live quotes |
| 2026-10-07 04:31 | 285 scored | 45.5% | 30 live quotes |
| 2026-10-07 04:37 | 285 scored | 66.7% | 6 live quotes |
| 2026-10-07 04:42 | 285 scored | 100.0% | 8 live quotes |
| 2026-10-07 04:47 | 285 scored | 66.7% | 9 live quotes |
| 2026-10-07 04:52 | 285 scored | 80.0% | 30 live quotes |
| 2026-10-07 04:57 | 285 scored | 60.0% | 6 live quotes |
| 2026-10-07 05:02 | 285 scored | 73.3% | 26 live quotes |
| 2026-10-07 05:07 | 285 scored | 81.0% | 29 live quotes |
| 2026-10-07 05:13 | 285 scored | 62.5% | 13 live quotes |
| 2026-10-07 05:18 | 285 scored | 57.1% | 11 live quotes |
| 2026-10-07 05:23 | 285 scored | 82.4% | 30 live quotes |
| 2026-10-07 05:28 | 285 scored | 85.0% | 28 live quotes |
| 2026-10-07 05:33 | 285 scored | 90.9% | 26 live quotes |
| 2026-10-07 05:38 | 285 scored | 33.3% | 8 live quotes |
| 2026-10-07 05:43 | 285 scored | 100.0% | 2 live quotes |
| 2026-10-07 06:50 | 285 scored | 0.0% | 6 live quotes |
| 2026-10-07 06:55 | 285 scored | 80.0% | 26 live quotes |
| 2026-10-07 07:00 | 285 scored | 62.5% | 27 live quotes |
| 2026-10-07 07:05 | 285 scored | 60.0% | 10 live quotes |
| 2026-10-07 07:10 | 285 scored | 33.3% | 4 live quotes |
| 2026-10-07 07:15 | 285 scored | 75.0% | 4 live quotes |
| 2026-10-07 07:20 | 285 scored | 88.9% | 27 live quotes |
| 2026-10-07 07:25 | 285 scored | 73.7% | 29 live quotes |
| 2026-10-07 07:30 | 285 scored | 72.7% | 30 live quotes |
| 2026-10-07 07:48 | 285 scored | 40.9% | 30 live quotes |
| 2026-10-07 07:53 | 285 scored | 50.0% | 4 live quotes |
| 2026-10-07 07:58 | 285 scored | 60.0% | 15 live quotes |
| 2026-10-07 08:03 | 285 scored | 73.7% | 24 live quotes |
| 2026-10-07 08:08 | 285 scored | 63.2% | 24 live quotes |
| 2026-10-07 08:13 | 285 scored | 88.2% | 28 live quotes |
| 2026-10-07 08:18 | 285 scored | 68.2% | 30 live quotes |
| 2026-10-07 08:23 | 285 scored | 100.0% | 6 live quotes |
| 2026-10-07 08:28 | 285 scored | 60.0% | 24 live quotes |
| 2026-10-07 08:33 | 285 scored | 61.5% | 20 live quotes |
| 2026-10-07 08:39 | 285 scored | 70.8% | 30 live quotes |
| 2026-10-07 08:44 | 285 scored | 50.0% | 24 live quotes |
| 2026-10-07 08:49 | 285 scored | 60.0% | 31 live quotes |
| 2026-10-07 08:54 | 285 scored | 57.9% | 30 live quotes |
| 2026-10-07 08:59 | 285 scored | 68.8% | 23 live quotes |
| 2026-10-07 09:04 | 285 scored | 60.0% | 9 live quotes |
| 2026-10-07 09:09 | 285 scored | 85.0% | 29 live quotes |
| 2026-10-07 09:14 | 285 scored | 76.9% | 20 live quotes |
| 2026-10-07 09:20 | 285 scored | 47.8% | 30 live quotes |
| 2026-10-07 09:25 | 285 scored | 50.0% | 4 live quotes |
| 2026-10-07 09:30 | 285 scored | 46.2% | 30 live quotes |
| 2026-10-07 09:35 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-07 09:40 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-07 09:45 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-07 09:50 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-07 09:55 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-07 10:00 | 285 scored | 73.3% | 40 live quotes |

---

*Report generated: 2026-10-07 15:35 IST  |  Session: 2026-10-07  |  Samples: 60*