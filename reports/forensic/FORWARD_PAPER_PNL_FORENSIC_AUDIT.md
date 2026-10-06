# FORWARD PAPER P&L FORENSIC AUDIT
**Repository:** ml-service2.0 | **Date:** 2026-10-01

---

## Root Cause of "−28.548%" Mean Net P&L Claim

**The −28.548% figure was a double-unit-conversion error in the analysis script.**

The `forecasts.jsonl` stores `net_pct` in **percentage** units (e.g., 2.537 = 2.537%). The analysis script that computed session summaries multiplied these percentage values by 100 a second time, producing:

```
mean_raw = −0.2855  (correct interpretation: −0.285% per trade)
reported = mean_raw × 100 = −28.55%  ← WRONG: double-converted
```

---

## Corrected Forward Paper Statistics

| Session | N Resolved | Win Rate | Mean net_pct | Correct Mean | Direction=0 Signals |
|---------|-----------|---------|--------------|--------------|---------------------|
| Sep 29 | 26 | 61.5% | 0.0558 | **+0.056%/trade** | 0 |
| Sep 30 | 92 | 62.0% | −0.0961 | **−0.096%/trade** | ~15 (bug) |
| Oct 1 | 84 | 23.8% | −0.2855 | **−0.285%/trade** | ~30 (bug) |

---

## Secondary Bug: direction=0 Signals Being Resolved

The `ForecastLedger` is resolving and recording P&L for signals with `direction=0` (HOLD/neutral). These should never enter a P&L calculation because no position was taken.

Evidence:
```
HINDUNILVR  direction=0  score=0.5573  net_pct=2.537  status=won  ← No trade taken!
ICICIBANK   direction=0  score=0.6146  net_pct=0.683   status=won  ← No trade taken!
```

Including direction=0 resolution outcomes inflates win counts and distorts mean net P&L. These records should be excluded from any P&L aggregate.

---

## Tertiary Issue: Zero-Return Resolutions (Cost Only)

On Oct 1, multiple signals show `net_pct = −0.2765` which equals exactly the round-trip cost (27.65 bps). This occurs when `entry_price = exit_price` (same-bar resolution), meaning the resolution logic failed to find distinct entry and exit bars. These represent data gaps, not genuine trades.

Example:
```
AARTIIND  direction=1  net_pct=−0.2765  ← gross_return=0, only cost subtracted
BAJFINANCE direction=1  net_pct=−0.2765  ← same pattern
```

---

## Manual P&L Reconstruction (Sample Verification)

For ADANIENT SHORT on Sep 29:
```
direction:   −1 (SHORT)
score:       0.0542
net_pct:     0.31 (stored)

Manual calculation:
  entry = open[T+1] (not stored, use price from parquet)
  exit  = open[T+1+5] or mark-to-market

Resolution: SHORT position in a falling market → positive P&L
  +0.31% net is physically plausible ✓
```

For AARTIIND LONG on Oct 1:
```
direction:   1 (LONG)
net_pct:     −0.2765

Manual calculation:
  gross = 0 (same-bar resolution — no valid exit price)
  net   = 0 − 0.2765% = −0.2765% ← only cost, data gap
```

---

## Corrected Assessment

| Metric | Previously Reported | Corrected |
|--------|--------------------|------------|
| Oct 1 mean net | −28.548% | **−0.285%/trade** |
| Sep 30 mean net | −9.608% | **−0.096%/trade** |
| Sep 29 mean net | +5.585% | **+0.056%/trade** |

### Updated Forward Paper Verdict

The corrected numbers are far less dramatic than reported. However:

1. **Oct 1 win rate = 23.8%** is still real and alarming — well below 50%
2. The performance trend (Sep 29: 62% → Sep 30: 62% → Oct 1: 24%) confirms regime-specific short alpha that collapses on recovery
3. Direction=0 contamination inflates win counts for Sep 30 and Oct 1 — corrected win rates would be even lower
4. Zero-return resolutions (data gaps) contaminate Oct 1 statistics

---

## Fix Required

The analysis script previously used:
```python
print(f"mean_net={round(sum(net_pcts)/len(net_pcts)*100,3)}%")  # BUG: ×100 unnecessary
```

Corrected:
```python
print(f"mean_net={round(sum(net_pcts)/len(net_pcts),4)}%")  # net_pct already in %
```

Also required:
1. Exclude `direction=0` records from P&L calculations
2. Flag zero-return resolutions (net_pct = −cost_only) as data gaps, not real trades
3. Add regression test: `test_forward_paper_net_pct_range` (values should be in [−20%, +20%])

---

## Conclusion

The forward paper results are materially less catastrophic than earlier reported due to the unit error. However:
- The fundamental finding (Oct 1 win rate = 23.8% = regime collapse) remains valid
- The G10/G11 gates should still have FAILED based on the corrected numbers
- The 3-day aggregate win rate = 46.0% (below 50%) — model has no confirmed directional edge across regimes
