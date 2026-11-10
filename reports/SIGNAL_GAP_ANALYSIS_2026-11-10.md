# SIGNAL GAP ANALYSIS REPORT
**Session:** 2026-11-10  |  **Generated:** 2026-11-10 15:48 IST
**Model:** fs-2.0.0 (LightGBM, 55 features, normalizer applied)
**Tracking samples:** 40 (every 5 min)

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| Overall direction accuracy | **61.6%** (tracked live symbols only) |
| Correct directional calls (big move) | 16 instances |
| Wrong directional calls (big move) | 15 instances |
| Missed big moves (neutral/unscored) | 0 instances |
| Tracking coverage | 10 / 346 symbols with live data |

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
| HFCL | S SHORT | 09-28 | data unavailable | — |
| LTM | S SHORT | 09-28 | data unavailable | — |
| RECLTD | S SHORT | 09-28 | data unavailable | — |
| KOTAKBANK | S SHORT | 09-28 | data unavailable | — |
| TIINDIA | S SHORT | 09-28 | data unavailable | — |
| 3BBLACKBIO | S SHORT | 09-28 | data unavailable | — |
| ADVANCE | S SHORT | 09-28 | data unavailable | — |
| ABANSENT | S SHORT | 09-28 | data unavailable | — |
| ABINFRA | S SHORT | 09-28 | data unavailable | — |
| ADL | S SHORT | 09-28 | data unavailable | — |
| ADFFOODS | S SHORT | 09-28 | data unavailable | — |
| 5PAISA | S SHORT | 09-28 | data unavailable | — |
| AEGISLOG | S LONG | 09-28 | data unavailable | — |
| AARTIPHARM | S SHORT | 09-28 | data unavailable | — |
| CYIENT | S SHORT | 09-28 | data unavailable | — |

**Fix required**: Wire live LTP into the feature factory during market hours.
The `score_symbol()` function should append today's partial bar to the parquet
using the live LTP before computing features. This would make signals track
today's intraday momentum.

---

## GAP CATEGORY 2: PERSISTENT WRONG DIRECTION CALLS

Symbols where ML predicted wrong direction across multiple intraday samples:

| Symbol | ML Call | Actual | Seen Wrong | Avg Move | Root Cause |
|--------|---------|--------|-----------|----------|-----------|
| TATASTEEL | A SHORT | UP | 6x | 2.04% | Stale features (EOD only) |
| DIVISLAB | S SHORT | UP | 5x | 4.17% | Stale features (EOD only) |
| INFY | B LONG | DOWN | 2x | 1.77% | Stale features (EOD only) |
| HDFCBANK | B SHORT | UP | 1x | 1.75% | Stale features (EOD only) |
| ICICIBANK | A SHORT | UP | 1x | 1.53% | Stale features (EOD only) |

---

## GAP CATEGORY 3: MISSED BIG MOVES

Stocks that made significant moves (>1.5%) but ML had low conviction or wrong direction:

| Symbol | ML Grade | ML Score | Actual Move | Reason Missed |
|--------|----------|----------|------------|--------------|

---

## GAP CATEGORY 4: CORRECT CALLS (MODEL STRENGTHS)

Symbols where ML correctly predicted direction on big moves:

| Symbol | ML Call | Move | Times Correct | Grade |
|--------|---------|------|--------------|-------|
| HINDALCO | LONG | 1.64% | 9x | A |
| INFY | SHORT | 1.77% | 2x | S |
| ITC | SHORT | 1.64% | 2x | S |
| ICICIBANK | LONG | 1.53% | 2x | B |
| HDFCBANK | LONG | 1.75% | 1x | B |

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
| 2026-11-10 05:09 | 346 scored | 50.0% | 11 live quotes |
| 2026-11-10 05:19 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 05:55 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 06:38 | 346 scored | 0.0% | 3 live quotes |
| 2026-11-10 06:43 | 346 scored | 42.9% | 30 live quotes |
| 2026-11-10 06:48 | 346 scored | 66.7% | 30 live quotes |
| 2026-11-10 06:54 | 346 scored | 66.7% | 30 live quotes |
| 2026-11-10 06:59 | 346 scored | 66.7% | 30 live quotes |
| 2026-11-10 07:04 | 346 scored | 63.6% | 30 live quotes |
| 2026-11-10 07:09 | 346 scored | 63.6% | 30 live quotes |
| 2026-11-10 07:14 | 346 scored | 63.6% | 30 live quotes |
| 2026-11-10 07:19 | 346 scored | 63.6% | 30 live quotes |
| 2026-11-10 07:24 | 346 scored | 100.0% | 11 live quotes |
| 2026-11-10 07:29 | 346 scored | 100.0% | 10 live quotes |
| 2026-11-10 07:45 | 346 scored | 71.4% | 30 live quotes |
| 2026-11-10 07:51 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 07:56 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 08:01 | 346 scored | 80.0% | 12 live quotes |
| 2026-11-10 08:06 | 346 scored | 85.7% | 30 live quotes |
| 2026-11-10 08:11 | 346 scored | 71.4% | 30 live quotes |
| 2026-11-10 08:16 | 346 scored | 33.3% | 8 live quotes |
| 2026-11-10 08:21 | 346 scored | 33.3% | 12 live quotes |
| 2026-11-10 08:26 | 346 scored | 57.1% | 30 live quotes |
| 2026-11-10 08:31 | 346 scored | 33.3% | 30 live quotes |
| 2026-11-10 08:37 | 346 scored | 83.3% | 30 live quotes |
| 2026-11-10 08:42 | 346 scored | 37.5% | 30 live quotes |
| 2026-11-10 08:47 | 346 scored | 66.7% | 30 live quotes |
| 2026-11-10 08:52 | 346 scored | 66.7% | 30 live quotes |
| 2026-11-10 08:57 | 346 scored | 33.3% | 7 live quotes |
| 2026-11-10 09:02 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 09:07 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 09:13 | 346 scored | 0.0% | 9 live quotes |
| 2026-11-10 09:18 | 346 scored | 100.0% | 6 live quotes |
| 2026-11-10 09:23 | 346 scored | 71.4% | 30 live quotes |
| 2026-11-10 09:28 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 09:33 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 09:38 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 09:43 | 346 scored | 62.5% | 30 live quotes |
| 2026-11-10 09:48 | 346 scored | 66.7% | 12 live quotes |
| 2026-11-10 09:53 | 346 scored | 0.0% | 10 live quotes |

---

*Report generated: 2026-11-10 15:48 IST  |  Session: 2026-11-10  |  Samples: 40*