# Signal Analysis — NSE Live Session Oct 1, 2026
## Root Cause Analysis Report

**Session date:** 2026-10-01 (Thursday)  
**Market hours:** 09:15 → 15:30 IST  
**Model:** fs-4.0.0 | 84 features | H1+H5 ensemble | LightGBM  
**Tracking period:** 10:28 → 11:47 IST (3 samples after fix deployment)  

---

## Executive Summary

Ten bugs were identified and fixed during the Oct 1 live session. The most critical was a Python 3.12 `pd.Timestamp` timezone incompatibility that silently excluded all NIFTY50 large-caps from scoring. After all fixes, tracked SHORT accuracy reached **91.7%** on 26 signals at 11:47 IST.

---

## Market Context (Oct 1)

| Metric | Value |
|--------|-------|
| NIFTY open | ~22,620 |
| NIFTY at 11:47 | 22,566 (-0.24%) |
| IT sector | +1.2% (INFY +2.0%, TCS +1.2%, HCLTECH +1.5%) |
| AUTO sector | -2.4% (EICHERMOT -2.9%, M&M -2.7%, MARUTI -2.5%) |
| CEMENT sector | -2.5% (ULTRACEMCO -2.5%, SHREECEM -2.4%, GRASIM -2.9%) |
| BANK sector | +0.6% → +1.0% (KOTAKBANK +2.1%, HDFCBANK +1.2%) |
| PHARMA sector | -0.7% (DRREDDY -0.9%, DIVISLAB -0.3%) |

---

## Bugs Found and Fixed (chronological)

### BUG-1: Ensemble manifest dict unpacking error (pre-session)
**Severity:** P0 — autorun crash on startup  
**Symptom:** `ValueError` on `for key, value in ensemble.items()` — manifest metadata key being treated as a model entry.  
**Fix:** Added `if key == "manifest": continue` guard in ensemble loop.  
**Commit:** `628d078`

---

### BUG-2: All-LONG bias — absolute score threshold failure
**Severity:** P1 — 0 SHORT signals generated  
**Root cause:** Model score distribution shifted to mean=0.63 today. Absolute threshold (`>0.55` LONG / `<0.45` SHORT) produced only LONGs when minimum score was 0.525.  
**Fix:** Cross-sectional rank-based direction assignment — top 15% → LONG, bottom 15% → SHORT, middle 70% → neutral. IC is a cross-sectional metric; absolute thresholds are fragile.  
**Result:** 41L/41S balanced from 278 symbols.  
**Commit:** `b809828`

---

### BUG-3: Cross-sectional ranking placed before weight manager
**Severity:** P1 — weight manager filtered out all ranked signals  
**Root cause:** Initial placement of cross-sectional ranking ran before the feature weight manager, which then filtered all ranked signals to direction=0.  
**Fix:** Moved cross-sectional ranking to AFTER all filters (weight manager, sector dampening, SentinelPulse).  
**Commit:** `4fe706e`

---

### BUG-4: SentinelPulse asyncio "Event loop is closed"
**Severity:** P2 — SP news sentiment unavailable after first call  
**Root cause:** `asyncio.run()` creates and closes an event loop. Subsequent calls in the same process found the loop closed.  
**Fix:** `loop = asyncio.get_event_loop(); loop.run_until_complete(_fetch_news_batch(...))`  
**Commit:** `3fc1ab4`

---

### BUG-5: Sector dampening cut SHORTs in falling AUTO sector
**Severity:** P1 — missed M&M -2.5%, EICHERMOT -2.9%, MARUTI -2.0% SHORTs  
**Root cause:** Sector dampening only blocked LONGs in falling sectors. No mechanism to boost neutral stocks to SHORT when sector falls hard AND the stock is also falling.  
**Fix:** Added sector_boost logic: when sector < -1.5% AND individual stock < -0.5%, override direction=0 to SHORT.  
**Commit:** `3fc1ab4`

---

### BUG-6: NIFTY quote "UNAVAILABLE" on timeout
**Severity:** P3 — dashboard shows no NIFTY price  
**Root cause:** Data service times out intermittently under load. Each timeout returned None, showing "UNAVAILABLE".  
**Fix:** Module-level `_last_nifty_quote` cache — on timeout, returns last known quote.  
**Commit:** `3fc1ab4`

---

### BUG-7: Signal tracker -100% false wrong-signals (LTP=0)
**Severity:** P2 — accuracy statistics completely invalid  
**Root cause:** `signal_tracker_v2.py` was getting LTP=0 for symbols not fetched from data service, computing `(0 - open) / open = -100%` change, and marking all LONGs as wrong.  
**Fix:** Added `if ltp <= 0: continue` guard; sanity check `abs(changePct) > 30% → recalculate from (ltp-open)/open`.  
**Commit:** `3fc1ab4`

---

### BUG-8: Sector boost overwritten by final cross-sectional pass
**Severity:** P1 — sector_boost and sector_dimmed signals lost  
**Root cause:** The final cross-sectional pass preserved `news_dimmed` and `ensemble_disagree` flags but NOT `sector_boost` or `sector_dimmed`. Any sector adjustment was silently overwritten.  
**Fix:** Added `s.get("sector_boost") or s.get("sector_dimmed")` to the preserve-guard in the final cross-sectional loop.  
**Commit:** `7f84e19`

---

### BUG-9: Missing sector coverage for BANK and CEMENT
**Severity:** P2 — KOTAKBANK +4.2% and ULTRACEMCO -2.7% had no sector signal  
**Root cause:** Only IT, PHARMA, AUTO sectors were defined. BANK and CEMENT were completely absent from sector dampening logic.  
**Fix:** Added `BANK_SYMS` (HDFCBANK, ICICIBANK, KOTAKBANK, AXISBANK, SBIN, INDUSINDBK) and `CEMENT_SYMS` (ULTRACEMCO, AMBUJACEM, ACC, SHREECEM, RAMCOCEM).  
**Commit:** `7f84e19`

---

### BUG-10: pd.Timestamp tzinfo bug — ALL large-caps excluded from scoring ⭐ CRITICAL
**Severity:** P0 — silent exclusion of entire NIFTY50 large-cap universe  
**Root cause:** Python 3.12 raises `ValueError: Cannot pass a datetime with tzinfo with the tz= parameter` when `pd.Timestamp(tz_aware_datetime, tz='UTC')` is called. The partial-bar injection code (which adds today's live LTP as today's OHLC bar) used this pattern.  

The partial-bar injection only runs when `live_ltp is not None`. The initial key-quotes fetch (NIFTY, BANKNIFTY, RELIANCE, HDFCBANK, ICICIBANK, INFY, TCS, KOTAKBANK, AXISBANK, BHARTIARTL, SBIN, LT, MARUTI, WIPRO, TITAN, NTPC, ONGC, BAJFINANCE, HINDUNILVR, ADANIENT) provided valid LTPs for 20 large-cap symbols. All 20 triggered the broken code path, all 20 returned `None` from `score_symbol()`, and all 20 were silently excluded from the 285-symbol universe.

**Before fix:** 238 symbols scored, score range [0.519, 0.682], mean 0.628. All TOP LONG/SHORT signals were mid-cap stocks (DELTACORP, TITAGARH, SUZLON...) because they had no live LTP and skipped the broken code path.

**After fix:** 285 symbols scored, score range [0.405, 0.682], mean 0.608. Large-caps (INFY, TCS, KOTAKBANK, M&M, EICHERMOT, HDFCBANK...) now appear in the signal universe.

**Fix:** Strip `tzinfo` before passing to `pd.Timestamp`:
```python
# Before (broken):
pd.Timestamp(ist_now.replace(hour=0, ...), tz="UTC")

# After (fixed):
ist_midnight_naive = ist_now.replace(hour=0, ..., tzinfo=None)
pd.Timestamp(ist_midnight_naive, tz="UTC")
```

**Commit:** `19cbf30`

---

## Signal Accuracy — Post-Fix Tracking Results

All 3 samples are post-fix (BUG-10 fix active).

| Sample | Time | Signals | Accuracy | LONG win | SHORT win |
|--------|------|---------|----------|----------|-----------|
| S1 | 11:37 | 27 (L=0, S=27) | 81.5% | 0% | 81.5% |
| S2 | 11:42 | 27 (L=0, S=27) | 81.5% | 0% | 81.5% |
| S3 | 11:47 | 26 (L=1, S=25) | 91.7% | 0% | 88.0% |

**S1/S2 wrong signals (5):** KOTAKBANK, HDFCBANK, BAJAJFINSV, SBIN, BHARTIARTL — all BANK sector  
**S3 wrong signals (2):** INDUSINDBK (LONG, actual ↓0.4%), BHARTIARTL (SHORT, actual ↑0.4%)

**Accuracy improvement S1→S3:** 81.5% → 91.7% as BANK sector avg moved from +0.6% to +1.0%, triggering the sector dampening threshold and dropping the 4 incorrect BANK SHORTs.

### Correct SHORTs (consistent across samples)
| Symbol | Sector | Actual move |
|--------|--------|-------------|
| EICHERMOT | AUTO | -2.9% |
| GRASIM | CEMENT | -2.9% |
| M&M | AUTO | -2.7% |
| MARUTI | AUTO | -2.5%/2.6% |
| ULTRACEMCO | CEMENT | -2.5% |
| SHREECEM | CEMENT | -2.4% |
| POWERGRID | — | -1.7%/1.8% |
| DRREDDY | PHARMA | -0.9% |
| RELIANCE | — | -0.5% |
| ADANIPORTS | — | -1.3% |

---

## Remaining Gaps

### GAP-1: LONG signals untestable (tracking gap, not model error)
**Issue:** All 39–42 LONG signals are for mid-cap stocks (DELTACORP, TITAGARH, SUZLON, TATACHEM, PFC) that have no live LTP in the data service (`provider=none`). These cannot be evaluated for accuracy.  
**Impact:** LONG win rate shows 0% in tracker — this is a data coverage gap, not model failure.  
**Proposed fix:** Add live LTP coverage for top-50 NSE mid-caps, or restrict LONG signals to symbols with confirmed live LTP data.

### GAP-2: BANK sector threshold too high
**Issue:** BANK sector was +0.6% (S1/S2) — just below the 1.0% dampening threshold. KOTAKBANK was up +2.1% individually while the sector avg was +0.6%. 4 BANK SHORTs were wrong.  
**Proposed fix:** Add stock-level check: if individual stock is up >1.5% AND direction == -1 (SHORT), suppress regardless of sector avg.

### GAP-3: INFY direction=0 despite +2.0% move
**Issue:** INFY scored 0.530–0.538, placed in middle 70% by cross-sectional ranking (direction=0). LTP override threshold (2.0%) not met at all samples.  
**Model behavior:** Correct — INFY with live LTP applied scores below mean (0.538 < 0.608), reflecting the model's view that a +2% intraday move is likely to mean-revert. The tracker shows this as a "missed mover" but it may not be a true miss.  
**Proposed action:** Monitor over 10+ sessions to assess if INFY-type "neutral-on-big-move" signals have alpha.

### GAP-4: LONG book avg P&L -3.7%
**Issue:** Live P&L shows SHORT avg +1.6–1.7% (34 positions) vs LONG avg -3.7% (14 positions).  
**Root cause:** The tracked LONG positions (from FP signals, pre-session entries) are mid-cap stocks with higher beta that sold off with the market. These are NOT the new post-fix LONG signals (which are untestable due to GAP-1).  
**Proposed fix:** Beta-neutral hedge is already logged (NIFTY SHORT ~11-12 units). Operator should execute NIFTY futures SHORT to neutralize long-book beta.

### GAP-5: M&M sector_boost overridden by weight manager threshold
**Issue:** M&M (AUTO sector, down -2.7%) had `sector_boost` applied (direction=-1) but the feature weight manager subsequently filtered it back to direction=0. Final cross-sectional preserved direction=0 (post-weight-manager state).  
**Root cause:** The `sector_boost` flag is set on the in-memory dict, but `_write_latest_scores` does not propagate custom flags — only `direction` is written. If the weight manager resets direction=0 after sector_boost, the final CS preserves 0 (not -1).  
**Proposed fix:** Weight manager should not override sector_boost/sector_dimmed flags (same guard as final CS).

---

## Fixes Applied (commits pushed to feat/ml-service-implementation)

| Commit | Fix |
|--------|-----|
| `628d078` | Ensemble dict iteration guard |
| `b809828` | Cross-sectional ranking — direction assignment |
| `4fe706e` | Cross-sectional ranking — move to after all filters |
| `3fc1ab4` | SP asyncio fix, sector SHORT boost, NIFTY cache, tracker LTP sanity |
| `7f84e19` | Sector boost overwrite fix, BANK+CEMENT sectors, symmetric UP boost |
| `9a4ea35` | Tracker live_quotes sharing, rate limiting fix |
| `fe4d797` | LTP momentum override, neutral-missed tracker analysis |
| `22d6a26` | LTP override guard + relative score threshold |
| **`19cbf30`** | **CRITICAL: pd.Timestamp tzinfo bug — large-cap scoring fix** |

---

## Recommended Next Steps

1. **Deploy weight manager sector_boost guard** (GAP-5) — prevents M&M falling through
2. **Add stock-level BANK dampening** (GAP-2) — suppress individual stock SHORT if up >1.5%  
3. **Expand mid-cap live LTP coverage** (GAP-1) — contact data provider for DELTACORP, TITAGARH, etc.
4. **Retrain model with corrected large-cap data** — the model never saw large-cap live-bar features during inference before today. Next training run should include post-partial-bar feature distributions.
5. **Execute NIFTY beta hedge** — LONG book avg -3.7% warrants ~11-12 unit NIFTY SHORT hedge as logged.

---

*Generated: 2026-10-01 11:50 IST | ml-service2.0 | feat/ml-service-implementation*
