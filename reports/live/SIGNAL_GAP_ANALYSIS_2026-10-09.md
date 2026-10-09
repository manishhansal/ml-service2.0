# SIGNAL GAP ANALYSIS REPORT
**Session:** 2026-10-09  |  **Generated:** 2026-10-09 15:39 IST
**Model:** fs-2.0.0 (LightGBM, 55 features, normalizer applied)
**Tracking samples:** 67 (every 5 min)

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| Overall direction accuracy | **73.4%** (tracked live symbols only) |
| Correct directional calls (big move) | 0 instances |
| Wrong directional calls (big move) | 0 instances |
| Missed big moves (neutral/unscored) | 158 instances |
| Tracking coverage | 30 / 285 symbols with live data |

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
| APOLLOHOSP | C SHORT | 09-28 | data unavailable | — |
| LTF | C SHORT | 09-28 | data unavailable | — |
| BANKBARODA | C SHORT | 09-28 | data unavailable | — |
| DMART | C SHORT | 09-28 | data unavailable | — |
| BANKNIFTY | C SHORT | 09-28 | data unavailable | — |
| BHARTIARTL | C SHORT | 09-28 | data unavailable | — |
| BEL | C SHORT | 09-28 | data unavailable | — |
| BANKINDIA | C SHORT | 09-28 | data unavailable | — |
| HINDALCO | C SHORT | 09-28 | data unavailable | — |
| TATASTEEL | C SHORT | 09-28 | data unavailable | — |
| AUBANK | C SHORT | 09-28 | data unavailable | — |
| HINDZINC | C SHORT | 09-28 | data unavailable | — |
| COALINDIA | C SHORT | 09-28 | data unavailable | — |
| GAIL | C SHORT | 09-28 | data unavailable | — |
| JSWSTEEL | C SHORT | 09-28 | data unavailable | — |

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
| TCS | D | 0.3829 | 4.64% | Score near 0.5 — model uncertain; may need intraday features |
| INFY | D | 0.3905 | 2.43% | Score near 0.5 — model uncertain; may need intraday features |
| M&M | D | 0.4539 | 1.85% | Score near 0.5 — model uncertain; may need intraday features |
| BRITANNIA | D | 0.4575 | 1.74% | Score near 0.5 — model uncertain; may need intraday features |
| MARUTI | D | 0.4011 | 1.68% | Score near 0.5 — model uncertain; may need intraday features |
| ADANIENT | D | 0.395 | 1.98% | Score near 0.5 — model uncertain; may need intraday features |
| TATAMOTORS | D | 0.4596 | 1.92% | Score near 0.5 — model uncertain; may need intraday features |
| SBIN | D | 0.4173 | 1.66% | Score near 0.5 — model uncertain; may need intraday features |
| NESTLEIND | D | 0.4231 | 1.73% | Score near 0.5 — model uncertain; may need intraday features |
| HEROMOTOCO | D | 0.4173 | 1.58% | Score near 0.5 — model uncertain; may need intraday features |

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
| 2026-10-09 03:45 | 285 scored | 25.0% | 15 live quotes |
| 2026-10-09 03:50 | 285 scored | 71.4% | 27 live quotes |
| 2026-10-09 03:55 | 285 scored | 100.0% | 4 live quotes |
| 2026-10-09 04:00 | 285 scored | 28.6% | 30 live quotes |
| 2026-10-09 04:06 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-09 04:11 | 285 scored | 40.0% | 27 live quotes |
| 2026-10-09 04:16 | 285 scored | 71.4% | 16 live quotes |
| 2026-10-09 04:38 | 285 scored | 73.3% | 28 live quotes |
| 2026-10-09 04:43 | 285 scored | 72.7% | 27 live quotes |
| 2026-10-09 04:48 | 285 scored | 57.1% | 19 live quotes |
| 2026-10-09 04:53 | 285 scored | 35.7% | 30 live quotes |
| 2026-10-09 04:58 | 285 scored | 92.9% | 24 live quotes |
| 2026-10-09 05:03 | 285 scored | 62.5% | 27 live quotes |
| 2026-10-09 05:08 | 285 scored | 88.2% | 28 live quotes |
| 2026-10-09 05:13 | 285 scored | 57.1% | 10 live quotes |
| 2026-10-09 05:18 | 285 scored | 42.1% | 30 live quotes |
| 2026-10-09 05:23 | 285 scored | 84.2% | 30 live quotes |
| 2026-10-09 05:30 | 285 scored | 88.2% | 24 live quotes |
| 2026-10-09 05:35 | 285 scored | 84.2% | 27 live quotes |
| 2026-10-09 05:52 | 285 scored | — | 1 live quotes |
| 2026-10-09 05:57 | 285 scored | 90.9% | 17 live quotes |
| 2026-10-09 06:02 | 285 scored | 83.3% | 27 live quotes |
| 2026-10-09 06:09 | 285 scored | 84.2% | 27 live quotes |
| 2026-10-09 06:14 | 285 scored | 75.0% | 26 live quotes |
| 2026-10-09 06:19 | 285 scored | 80.0% | 22 live quotes |
| 2026-10-09 06:24 | 285 scored | 70.6% | 26 live quotes |
| 2026-10-09 06:29 | 285 scored | 86.7% | 20 live quotes |
| 2026-10-09 06:34 | 285 scored | 100.0% | 8 live quotes |
| 2026-10-09 06:40 | 285 scored | 90.0% | 30 live quotes |
| 2026-10-09 06:45 | 285 scored | 90.9% | 22 live quotes |
| 2026-10-09 06:50 | 285 scored | 100.0% | 3 live quotes |
| 2026-10-09 06:55 | 285 scored | 100.0% | 4 live quotes |
| 2026-10-09 07:00 | 285 scored | 76.2% | 30 live quotes |
| 2026-10-09 07:05 | 285 scored | 78.3% | 31 live quotes |
| 2026-10-09 07:10 | 285 scored | 77.8% | 10 live quotes |
| 2026-10-09 07:15 | 285 scored | 100.0% | 3 live quotes |
| 2026-10-09 07:25 | 285 scored | 42.1% | 30 live quotes |
| 2026-10-09 07:30 | 285 scored | 93.3% | 23 live quotes |
| 2026-10-09 07:35 | 285 scored | — | 1 live quotes |
| 2026-10-09 07:40 | 285 scored | 70.6% | 27 live quotes |
| 2026-10-09 07:46 | 285 scored | 71.4% | 13 live quotes |
| 2026-10-09 07:51 | 285 scored | 100.0% | 5 live quotes |
| 2026-10-09 07:56 | 285 scored | 90.0% | 29 live quotes |
| 2026-10-09 08:01 | 285 scored | 100.0% | 15 live quotes |
| 2026-10-09 08:06 | 285 scored | 80.0% | 9 live quotes |
| 2026-10-09 08:11 | 285 scored | 94.7% | 28 live quotes |
| 2026-10-09 08:16 | 285 scored | 90.0% | 29 live quotes |
| 2026-10-09 08:21 | 285 scored | 94.4% | 25 live quotes |
| 2026-10-09 08:27 | 285 scored | 38.1% | 30 live quotes |
| 2026-10-09 08:32 | 285 scored | 80.0% | 30 live quotes |
| 2026-10-09 08:37 | 285 scored | 78.3% | 28 live quotes |
| 2026-10-09 08:42 | 285 scored | 100.0% | 9 live quotes |
| 2026-10-09 08:47 | 285 scored | 83.3% | 8 live quotes |
| 2026-10-09 08:52 | 285 scored | 71.4% | 29 live quotes |
| 2026-10-09 08:57 | 285 scored | 88.9% | 13 live quotes |
| 2026-10-09 09:02 | 285 scored | 38.9% | 30 live quotes |
| 2026-10-09 09:07 | 285 scored | 91.7% | 31 live quotes |
| 2026-10-09 09:12 | 285 scored | 91.7% | 15 live quotes |
| 2026-10-09 09:18 | 285 scored | 89.5% | 27 live quotes |
| 2026-10-09 09:23 | 285 scored | 88.2% | 27 live quotes |
| 2026-10-09 09:28 | 285 scored | 100.0% | 12 live quotes |
| 2026-10-09 09:33 | 285 scored | 100.0% | 5 live quotes |
| 2026-10-09 09:38 | 285 scored | 47.4% | 30 live quotes |
| 2026-10-09 09:43 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-09 09:48 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-09 09:53 | 285 scored | 30.0% | 30 live quotes |
| 2026-10-09 09:58 | 285 scored | 30.0% | 30 live quotes |

---

*Report generated: 2026-10-09 15:39 IST  |  Session: 2026-10-09  |  Samples: 67*