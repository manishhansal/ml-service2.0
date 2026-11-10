# SIGNAL GAP ANALYSIS REPORT
**Session:** 2026-11-09  |  **Generated:** 2026-11-09 15:36 IST
**Model:** fs-2.0.0 (LightGBM, 55 features, normalizer applied)
**Tracking samples:** 59 (every 5 min)

---

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| Overall direction accuracy | **50.0%** (tracked live symbols only) |
| Correct directional calls (big move) | 42 instances |
| Wrong directional calls (big move) | 34 instances |
| Missed big moves (neutral/unscored) | 2 instances |
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
| ABINFRA | S SHORT | 09-28 | data unavailable | — |
| ADL | S SHORT | 09-28 | data unavailable | — |
| ADFFOODS | S SHORT | 09-28 | data unavailable | — |
| 5PAISA | S SHORT | 09-28 | data unavailable | — |
| LICHSGFIN | S SHORT | 09-28 | data unavailable | — |
| MUTHOOTFIN | S SHORT | 09-28 | data unavailable | — |
| AEGISLOG | S LONG | 09-28 | data unavailable | — |
| AARTIPHARM | S SHORT | 09-28 | data unavailable | — |
| JIOFIN | S SHORT | 09-28 | data unavailable | — |
| PAYTM | S SHORT | 09-28 | data unavailable | — |
| AAREYDRUGS | S SHORT | 09-28 | data unavailable | — |
| IPCALAB | S SHORT | 09-28 | data unavailable | — |

**Fix required**: Wire live LTP into the feature factory during market hours.
The `score_symbol()` function should append today's partial bar to the parquet
using the live LTP before computing features. This would make signals track
today's intraday momentum.

---

## GAP CATEGORY 2: PERSISTENT WRONG DIRECTION CALLS

Symbols where ML predicted wrong direction across multiple intraday samples:

| Symbol | ML Call | Actual | Seen Wrong | Avg Move | Root Cause |
|--------|---------|--------|-----------|----------|-----------|
| APOLLOHOSP | B LONG | DOWN | 12x | 2.66% | Stale features (EOD only) |
| INFY | S SHORT | UP | 5x | 2.69% | Stale features (EOD only) |
| HDFCLIFE | S SHORT | UP | 4x | 2.2% | Stale features (EOD only) |
| ICICIBANK | S SHORT | UP | 3x | 2.93% | Stale features (EOD only) |
| HDFCBANK | S SHORT | UP | 2x | 1.96% | Stale features (EOD only) |
| ONGC | B LONG | DOWN | 2x | 1.89% | Stale features (EOD only) |
| HEROMOTOCO | A SHORT | UP | 2x | 1.7% | Stale features (EOD only) |
| SBIN | S SHORT | UP | 1x | 2.79% | Stale features (EOD only) |

---

## GAP CATEGORY 3: MISSED BIG MOVES

Stocks that made significant moves (>1.5%) but ML had low conviction or wrong direction:

| Symbol | ML Grade | ML Score | Actual Move | Reason Missed |
|--------|----------|----------|------------|--------------|
| M&M | D | 0.0667 | 3.6% | Score near 0.5 — model uncertain; may need intraday features |

---

## GAP CATEGORY 4: CORRECT CALLS (MODEL STRENGTHS)

Symbols where ML correctly predicted direction on big moves:

| Symbol | ML Call | Move | Times Correct | Grade |
|--------|---------|------|--------------|-------|
| TATASTEEL | LONG | 2.24% | 7x | B |
| INFY | SHORT | 2.46% | 6x | S |
| SUNPHARMA | LONG | 3.02% | 6x | B |
| ADANIENT | LONG | 2.35% | 5x | A |
| HDFCLIFE | LONG | 2.2% | 4x | S |
| TCS | LONG | 2.29% | 3x | S |
| AXISBANK | LONG | 3.09% | 3x | S |
| ONGC | SHORT | 1.89% | 3x | A |
| HDFCBANK | LONG | 1.96% | 2x | B |
| WIPRO | LONG | 1.69% | 1x | S |

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
| 2026-11-09 05:00 | 346 scored | 75.0% | 30 live quotes |
| 2026-11-09 05:05 | 346 scored | 50.0% | 30 live quotes |
| 2026-11-09 05:10 | 346 scored | 40.0% | 12 live quotes |
| 2026-11-09 05:15 | 346 scored | 50.0% | 10 live quotes |
| 2026-11-09 05:20 | 346 scored | 33.3% | 21 live quotes |
| 2026-11-09 05:26 | 346 scored | 0.0% | 7 live quotes |
| 2026-11-09 05:31 | 346 scored | 71.4% | 13 live quotes |
| 2026-11-09 05:36 | 346 scored | 50.0% | 20 live quotes |
| 2026-11-09 05:41 | 346 scored | 100.0% | 4 live quotes |
| 2026-11-09 05:46 | 346 scored | 28.6% | 20 live quotes |
| 2026-11-09 05:51 | 346 scored | 0.0% | 5 live quotes |
| 2026-11-09 05:56 | 346 scored | 66.7% | 24 live quotes |
| 2026-11-09 06:06 | 346 scored | 0.0% | 8 live quotes |
| 2026-11-09 06:11 | 346 scored | 100.0% | 3 live quotes |
| 2026-11-09 06:16 | 346 scored | 47.1% | 30 live quotes |
| 2026-11-09 06:21 | 346 scored | 50.0% | 18 live quotes |
| 2026-11-09 06:26 | 346 scored | 55.6% | 16 live quotes |
| 2026-11-09 06:31 | 346 scored | 47.1% | 30 live quotes |
| 2026-11-09 06:36 | 346 scored | 60.0% | 30 live quotes |
| 2026-11-09 06:41 | 346 scored | 50.0% | 3 live quotes |
| 2026-11-09 06:46 | 346 scored | 50.0% | 17 live quotes |
| 2026-11-09 06:51 | 346 scored | 53.8% | 30 live quotes |
| 2026-11-09 06:56 | 346 scored | 50.0% | 7 live quotes |
| 2026-11-09 07:02 | 346 scored | 75.0% | 30 live quotes |
| 2026-11-09 07:07 | 346 scored | 83.3% | 7 live quotes |
| 2026-11-09 07:12 | 346 scored | 64.7% | 30 live quotes |
| 2026-11-09 07:17 | 346 scored | 42.9% | 18 live quotes |
| 2026-11-09 07:23 | 346 scored | 25.0% | 12 live quotes |
| 2026-11-09 07:28 | 346 scored | 0.0% | 12 live quotes |
| 2026-11-09 07:33 | 346 scored | 10.0% | 20 live quotes |
| 2026-11-09 07:38 | 346 scored | 28.6% | 23 live quotes |
| 2026-11-09 07:43 | 346 scored | 25.0% | 19 live quotes |
| 2026-11-09 07:48 | 346 scored | 80.0% | 30 live quotes |
| 2026-11-09 07:53 | 346 scored | 40.0% | 14 live quotes |
| 2026-11-09 07:58 | 346 scored | 0.0% | 7 live quotes |
| 2026-11-09 08:03 | 346 scored | 66.7% | 30 live quotes |
| 2026-11-09 08:08 | 346 scored | 57.1% | 18 live quotes |
| 2026-11-09 08:13 | 346 scored | 50.0% | 22 live quotes |
| 2026-11-09 08:19 | 346 scored | 25.0% | 12 live quotes |
| 2026-11-09 08:24 | 346 scored | 60.0% | 30 live quotes |
| 2026-11-09 08:29 | 346 scored | 100.0% | 2 live quotes |
| 2026-11-09 08:34 | 346 scored | 0.0% | 1 live quotes |
| 2026-11-09 08:39 | 346 scored | 42.9% | 20 live quotes |
| 2026-11-09 08:44 | 346 scored | — | 2 live quotes |
| 2026-11-09 08:49 | 346 scored | 50.0% | 5 live quotes |
| 2026-11-09 08:54 | 346 scored | 100.0% | 4 live quotes |
| 2026-11-09 08:59 | 346 scored | 50.0% | 12 live quotes |
| 2026-11-09 09:04 | 346 scored | — | 1 live quotes |
| 2026-11-09 09:10 | 346 scored | 60.0% | 30 live quotes |
| 2026-11-09 09:15 | 346 scored | 60.0% | 30 live quotes |
| 2026-11-09 09:20 | 346 scored | 57.1% | 30 live quotes |
| 2026-11-09 09:25 | 346 scored | — | 2 live quotes |
| 2026-11-09 09:30 | 346 scored | 57.1% | 30 live quotes |
| 2026-11-09 09:35 | 346 scored | 40.0% | 20 live quotes |
| 2026-11-09 09:40 | 346 scored | — | 6 live quotes |
| 2026-11-09 09:45 | 346 scored | 0.0% | 3 live quotes |
| 2026-11-09 09:50 | 346 scored | 14.3% | 17 live quotes |
| 2026-11-09 09:55 | 346 scored | 50.0% | 30 live quotes |
| 2026-11-09 10:01 | 346 scored | 57.1% | 30 live quotes |

---

*Report generated: 2026-11-09 15:36 IST  |  Session: 2026-11-09  |  Samples: 59*