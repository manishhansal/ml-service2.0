# SIGNAL GAP ANALYSIS REPORT
**Session:** 2026-11-03  |  **Generated:** 2026-11-03 15:36 IST
**Model:** fs-2.0.0 (LightGBM, 55 features, normalizer applied)
**Tracking samples:** 67 (every 5 min)

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| Overall direction accuracy | **98.5%** (tracked live symbols only) |
| Correct directional calls (big move) | 42 instances |
| Wrong directional calls (big move) | 1 instances |
| Missed big moves (neutral/unscored) | 0 instances |
| Tracking coverage | 30 / 346 symbols with live data |

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
| 3BBLACKBIO | S SHORT | 09-28 | data unavailable | — |
| ADVANCE | S SHORT | 09-28 | data unavailable | — |
| ABANSENT | S SHORT | 09-28 | data unavailable | — |
| SOLARINDS | S SHORT | 09-28 | data unavailable | — |
| ANGELONE | S SHORT | 09-28 | data unavailable | — |
| RAIN | S SHORT | 09-28 | data unavailable | — |
| ABINFRA | S SHORT | 09-28 | data unavailable | — |
| BHEL | S SHORT | 09-28 | data unavailable | — |
| ADL | S SHORT | 09-28 | data unavailable | — |
| ADFFOODS | S SHORT | 09-28 | data unavailable | — |
| 5PAISA | S SHORT | 09-28 | data unavailable | — |
| LICHSGFIN | S SHORT | 09-28 | data unavailable | — |
| POWERINDIA | S SHORT | 09-28 | data unavailable | — |
| MUTHOOTFIN | S SHORT | 09-28 | data unavailable | — |
| AEGISLOG | S SHORT | 09-28 | data unavailable | — |

**Fix required**: Wire live LTP into the feature factory during market hours.
The `score_symbol()` function should append today's partial bar to the parquet
using the live LTP before computing features. This would make signals track
today's intraday momentum.

---

## GAP CATEGORY 2: PERSISTENT WRONG DIRECTION CALLS

Symbols where ML predicted wrong direction across multiple intraday samples:

| Symbol | ML Call | Actual | Seen Wrong | Avg Move | Root Cause |
|--------|---------|--------|-----------|----------|-----------|
| TATASTEEL | S SHORT | UP | 1x | 2.24% | Stale features (EOD only) |

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
| HINDALCO | LONG | 1.61% | 42x | B |

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
| 2026-11-03 03:58 | 346 scored | 0.0% | 30 live quotes |
| 2026-11-03 04:03 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:09 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:14 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:19 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:24 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:29 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:34 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:39 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:44 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:49 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 04:54 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:00 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:05 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:10 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:15 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:20 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:25 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:30 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:35 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:40 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:46 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:51 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 05:56 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:01 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:06 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:11 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:16 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:22 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:27 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:32 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:37 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:42 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:47 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:52 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 06:57 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:03 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:08 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:13 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:18 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:23 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:28 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:33 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:38 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:43 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:49 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:54 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 07:59 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 08:04 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 08:09 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 08:14 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 08:19 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 08:24 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 08:54 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 08:59 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:05 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:10 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:15 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:20 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:25 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:30 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:35 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:40 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:45 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:51 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 09:56 | 346 scored | 100.0% | 30 live quotes |
| 2026-11-03 10:01 | 346 scored | 100.0% | 30 live quotes |

---

*Report generated: 2026-11-03 15:36 IST  |  Session: 2026-11-03  |  Samples: 67*