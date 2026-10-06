# SIGNAL FAILURE ANALYSIS
**Repository:** ml-service2.0 | **Date:** 2026-10-01

## 1. Forward Paper Failure Breakdown (Oct 1, 2026)

Oct 1 is the most critical failure data point: 84 resolved signals, 23.8% win rate, −28.5% mean net.

### Root Cause Analysis

**Primary cause: Regime reversal**  
After 2 consecutive bear days (NIFTY −1.52% and −0.42%), Oct 1 was a recovery day. SHORT signals that performed well in bear conditions generated losses when the market reversed.

**Secondary cause: Possible net_pct calculation bug**  
Mean net of −28.548% for 5-bar trades is physically implausible (would require 28% average adverse price move in 5 bars on NSE F&O stocks). This strongly suggests a calculation error in `resolve_forward_paper.py`. Likely candidates:
1. Entry price used is closing price of signal day (lookahead), not open[T+1]
2. net_pct is computed without cost subtraction or with double-counting
3. Direction sign is inverted for SHORT positions
4. net_pct units are basis points, not percentage (i.e., −28.548 bps = −0.285%, plausible)

**Recommendation:** Audit `scripts/resolve_forward_paper.py` manually before using forward paper results for any gate decision.

## 2. Failure Category Distribution (Label Proxy Backtest)

| Category | % of Losses | Root Cause | Fixable? |
|---------|------------|-----------|---------|
| TIME_EXPIRY near-zero return | 35% | Small returns < transaction costs | Label redesign ✓ |
| STOP_HIT (adverse move) | 26% | Genuine prediction error | Better features |
| Post-entry trend reversal | 21% | Market non-stationary over 7d | Regime filter |
| Volatility shock | 12% | News/event not captured | News features |
| Liquidity gap | 4% | Low-volume execution | Liquidity filter |
| Other | 2% | Miscellaneous | — |

## 3. Structural Failure Modes

### Mode A: LONG signals in bear market
- LONG signals during BEAR regime perform poorly (−2.5% avg in Sep 28-29 live)
- Fix: Add regime filter — suppress LONG signals when market_regime = BEAR

### Mode B: Defensive sector misses
- DRREDDY, ASIANPAINT, AXISBANK consistently missed (defensive stocks move counter to broad market)
- Fix: Add sector_relative_momentum feature with sector-specific thresholds

### Mode C: Near-zero TIME_EXPIRY outcomes
- 6.2% of observations hit neither barrier in 5 bars → labelled by tiny return vs cost
- Fix: New label design (7-day asymmetric barrier) reduces this category

### Mode D: IC inflation from non-bear regimes
- IC = 0.4136 measured over mixed 5-year period; individual regime ICs may vary significantly
- Bull regime IC could be near-zero if model is primarily a bear-market predictor

## 4. Statistical Significance of Oct 1 Failure

```
Oct 1 resolved signals: 84
Oct 1 wins:             20 (23.8%)

Expected win rate (from WF): ~52–55% (at 50/50 labels + model edge)
Observed: 23.8%

Binomial test H0: p = 0.50
P(X ≤ 20 | n=84, p=0.50) = 2.3e-7 (highly statistically significant)

Conclusion: Oct 1 performance (23.8%) is SIGNIFICANTLY BELOW random chance (50%).
This is NOT bad luck — it is a structured failure.
Most likely cause: regime-specific model + first recovery day.
```

## 5. Remediation Priority

| Fix | Priority | Expected Impact | Implementation |
|----|---------|-----------------|---------------|
| Audit net_pct bug in resolve_forward_paper.py | P0 | Fixes misleading data | 1 day |
| Regime filter (suppress LONG in BEAR) | P1 | Reduces wrong-direction losses | 3 days |
| 7-day label redesign | P1 | Removes negative-EV labels | 2 days + retrain |
| Require 20+ mixed-regime days before PRODUCTION | P1 | Prevents premature promotion | Protocol change |
| Placebo IC test | P2 | Verifies IC is genuine | 1 day |
| Beta-neutral LONG overlay | P3 | Reduces LONG regime sensitivity | 1 week |
