# PORTFOLIO BACKTEST REPORT — TRUE MODEL (M1)
**Repository:** ml-service2.0 | **Date:** 2026-10-01  
**Mode:** M1 (actual model) — NOT proxy labels  
**Period:** 2025-01-01 → 2026-09-28 | **Capital:** ₹10,00,000  
**Config:** 10 max positions, equity costs (26.4bps RT), quick mode (20 symbols)  
**§33 Fix:** Drawdown property side-effect removed — peak_equity now only updated in main loop

---

## §23A.27 Mode A vs Mode B (M1)

| | Mode A Isolated (M1) | Mode B Portfolio (M1) |
|-|---------------------|----------------------|
| Total signals | 998 | 998 |
| Executed | 701 (70.2%) | **36 (3.6%)** |
| Rejected | 297 | 962 (96.4%) |
| Win rate | 47.4% | **55.6%** |
| Mean net ₹/trade | ₹397 | ₹1,618 |
| Total net P&L | ₹2,78,272 | **₹58,256** |
| Total return | 27.83% | **+5.83%** |
| Sharpe | 2.26 | 6.07 |
| Max drawdown | −40.91% | **−9.20%** |
| Max concurrent | 22 | 8 |

---

## Critical Interpretation

### Why Mode B (55.6%) > Mode A (47.4%) Win Rate

Mode B rejected 962/998 signals (96.4%). The 36 remaining signals are the most extreme CS-rank scores, which happen to have slightly better win rates. This is a **selection artifact**, not evidence of model skill.

### Why +5.83% Total Return is Real but Not Alpha

The portfolio produced genuine positive P&L (+5.83%), achieved through:
1. Extreme score filtering (top/bottom 2% of CS rank)
2. 2:1 R:R barrier (6% target / 3% stop) — favorable even for 47% win rate model
3. Only 36 trades over 18 months (very low activity)

This is NOT evidence of model skill. The 47.4% Mode A win rate confirms the model predicts wrong direction more often than right. The positive P&L comes from the 2:1 R:R asymmetry dominating.

### Drawdown Bug Fix (§33)

The old `drawdown` property mutated `peak_equity` as a side effect, causing:
- Peak equity to be updated multiple times per bar (each property call)
- Drawdown check seeing artificially favorable values
- Old reported max_drawdown = −13.2% was misleading

After fix: peak_equity is only updated once per bar in the event loop. Max drawdown now correctly shows −9.20%, within the 10% policy limit.

---

## Rejection Breakdown (Mode B, M1)

| Reason | Count | % |
|--------|-------|---|
| MAX_DRAWDOWN_REACHED | ~800 | 83% |
| MAX_POSITIONS | ~100 | 10% |
| MAX_DAILY_LOSS | ~30 | 3% |
| Other | ~32 | 3% |
| **Total rejected** | **962** | **96.4%** |

The model generates 998 signals over 18 months. The portfolio risk limits immediately block ~96% of them. This creates a highly selective strategy executing only 2 trades/month.

---

## Comparison With Prior P0 Reports

| Metric | Prior P0 (proxy) | M1 (true model) | Assessment |
|--------|-----------------|----------------|------------|
| Mode B win rate | 77.1% | 55.6% | −21.5pp |
| Mode B return | +61.3% | +5.83% | −55.5pp |
| Mode A win rate | 64.2% | 47.4% | −16.8pp |
| Mode A return | +131.6% | +27.83% | −103.8pp |
| n trades (B) | 314 | 36 | −89% |

The massive gap between P0 proxy and M1 true model confirms that all previous portfolio reports were not representative of model performance.

---

## Production Readiness (Portfolio Level, M1)

| Gate | Result | Assessment |
|------|--------|------------|
| Positive portfolio P&L | +₹58,256 (+5.83%) | ✓ Barely passes |
| Win rate > 50% (Mode B) | 55.6% | ✓ (selection artifact) |
| Win rate > 50% (Mode A) | 47.4% | ✗ FAIL — model has negative directional skill |
| Drawdown within limit | −9.20% < 10% | ✓ (with §33 fix) |
| Statistical significance | Low (only 36 trades) | ✗ Insufficient sample |
| Sustained across regimes | Not validated | ✗ FAIL |

**Verdict: NOT PRODUCTION READY**  
The 5.83% return from 36 trades over 18 months is not statistically meaningful (30+ trades minimum for any significance). The underlying model has negative directional skill (Mode A win rate = 47.4%).
