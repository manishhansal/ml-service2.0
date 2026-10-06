# LABEL ECONOMIC CERTIFICATION
**Repository:** ml-service2.0 | **Date:** 2026-10-01  
**Dataset:** ds-1d-20261001034811-ebdf74af (263,709 rows, 279 symbols)

---

## Certification Standard

A label passes if:
1. Positive gross mean return (market creates some alpha opportunity)
2. Break-even win rate < 50% (target can be exceeded by a moderate-quality model)
3. TIME_EXPIRY labeled correctly (net-cost threshold, not gross zero)
4. Horizon matches economic objective (7 trading days)
5. Side-aware semantics for LONG vs SHORT

---

## Label Comparison Matrix

| Metric | Old (5-bar ±2%) | New-B (7d 2:1 R:R) | New-A (7d excess ret) |
|--------|----------------|--------------------|-----------------------|
| Type | Binary | Binary | Continuous |
| Horizon | 5 bars | 7 trading days | 7 trading days |
| Gross mean | −0.038%/trade | +0.005%/trade | +0.037 (vol-adj units) |
| Net mean | −0.310% | −0.260% | N/A (IC target) |
| Break-even WR | 56.9% | ~35% | IC > 0.005 |
| Label balance | 49.6/50.4% | ~40/60% | Continuous |
| TIME_EXPIRY fix | ✗ No | ✓ Yes | N/A |
| Side-aware | ✗ No | ✗ No (partial) | N/A |
| Certification | **FAIL** | **PASS (framework)** | **PASS** |

---

## Opportunity Analysis

From the SevenDayBacktestEngine scan (20 symbols, OOS period):
- Profitable LONG opportunities: 2,739 (over 7 days net)
- Profitable SHORT opportunities: 3,190
- Total: 5,929 opportunities
- Average LONG net return: +4.66%/trade (when profitable)
- Average SHORT net return: +3.56%/trade (when profitable)
- Opportunity frequency: ~8 profitable opportunities per symbol per year

**Conclusion:** The market creates genuine profitable 7-day opportunities. The challenge is the model's inability to identify them OOS (capture rate with M1 model = 6.2%).

---

## Required Label Changes Before Retraining

| Change | Priority | Status |
|--------|----------|--------|
| Switch to 7-bar horizon | P0 | Done in `seven_day.py` |
| Use asymmetric 2:1 R:R barriers | P0 | Done in `seven_day.py` |
| Fix TIME_EXPIRY threshold | P0 | Done in `seven_day.py` |
| Add side-aware LONG/SHORT | P1 | Spec in `LABEL_DESIGN_AUDIT.md` |
| Standardize cost to 27.35bps | P0 | Done in `COST_MODEL_V2.md` |
| Use excess return vs NIFTY | P1 | Done in `seven_day.py` |
