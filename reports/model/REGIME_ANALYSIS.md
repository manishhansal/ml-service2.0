# REGIME ANALYSIS
**Repository:** ml-service2.0 | **Date:** 2026-10-01

## Summary

The model demonstrates REGIME-DEPENDENT performance. SHORT signals work in BEAR regimes; LONG signals fail in BEAR regimes. Performance in BULL/SIDEWAYS is unconfirmed with live data.

## Live Evidence (Sep 28–Oct 1, 2026)

| Date | NIFTY | Regime | SHORT WR | LONG WR | Overall WR |
|------|-------|--------|---------|---------|-----------|
| Sep 28 | −1.52% | BEAR | 80% (16/20) | ~17% (1/6) | 65.4% |
| Sep 29 | −0.42% | MILD BEAR | 80% (16/20) | ~17% (1/6) | 65.4% |
| Oct 1 | Recovery | RECOVERY | ~18% (est) | ~30% (est) | 23.8% |

**CRITICAL FINDING:** The model's SHORT performance (80%) on consecutive bear days is the evidence used for SHADOW promotion. However, the Oct 1 recovery immediately destroyed this performance. The model appears to be a BEAR REGIME DETECTOR rather than a genuine cross-sectional alpha generator.

## Proxy Backtest Regime Performance (Jun 2025–Sep 2026)

| Regime | N Signals | Win Rate | Mean Net % | Comment |
|--------|----------|---------|-----------|---------|
| BULL | ~2,500 | ~64% | ~+1.5% | Proxy mode only |
| BEAR | ~2,100 | ~65% | ~+1.9% | Confirmed by 2 live days |
| SIDEWAYS | ~1,300 | ~62% | ~+1.3% | Not confirmed by live |
| HIGH_VOL | ~455 | ~66% | ~+2.1% | Proxy mode only |

The proxy backtest shows consistent regime performance, but proxy results overstate the model's true regime stability.

## Required Regime Validation

Before PRODUCTION promotion:
1. Minimum 5 BULL regime days with OOS performance evaluated
2. Minimum 5 SIDEWAYS regime days
3. Minimum 5 HIGH_VOL regime days
4. ALL THREE above passing the 56.9% break-even threshold

Currently: Only BEAR regime confirmed (2 days). All other regimes unverified.

## Regime Filter Recommendation

Based on current evidence, implement a regime-conditional filter:

```python
# Add to signal generation pipeline
if market_regime == "BEAR":
    # Only SHORT signals (LONG signals historically poor)
    signal = direction if direction == -1 else 0

elif market_regime == "BULL":
    # Only LONG signals (SHORT signals historically unconfirmed)
    # NOTE: requires validation with 5+ bull days first
    signal = direction if direction == 1 else 0

elif market_regime in ("SIDEWAYS", "HIGH_VOL"):
    # Both directions: use only highest-conviction signals
    signal = direction if abs(score - 0.5) > 0.25 else 0
```

This reduces signal count but improves precision. Do NOT implement this filter until BULL/SIDEWAYS performance is validated with live data (not proxy backtest).
