# PRODUCTION READINESS REPORT — FORENSIC AUDIT
**Repository:** ml-service2.0  
**Date:** 2026-10-01  
**Audit:** Kiro Forensic Audit System  
**Supersedes:** PRODUCTION_READINESS_REPORT.md (Sep 29 version, pre-Oct-1 data)

---

## Final Verdict: NOT PRODUCTION READY

```
╔═════════════════════════════════════════════════════════════════╗
║  FORENSIC VERDICT: SHADOW MONITORING REQUIRED                   ║
║  DO NOT PROMOTE TO PRODUCTION UNTIL ALL MANDATORY GATES PASS   ║
╚═════════════════════════════════════════════════════════════════╝
```

The Oct 1 forward paper evidence (23.8% win rate, −28.5% mean net) overrides the Sep 28-29 live session performance (80% win rate) that justified the SHADOW promotion. The strategy exhibits strong regime-dependence that was not apparent from a 2-day bear market observation window.

---

## Gate Status

### Mandatory Pass/Fail

| Gate | Criterion | Status | Evidence |
|------|-----------|--------|---------|
| NO LOOK-AHEAD BIAS | Zero INVALID shift(-N) in features | ✓ **PASS** | Static audit, mutation tests |
| NO DATA LEAKAGE | Features uncorrelated with future returns | ✓ **PASS** | |r| < 0.15, structural check |
| VALID PIT DATA | Entry at open[T+1], exit at T+7 | ✓ **PASS** | BacktestEngine, 19 tests |
| REALISTIC EXECUTION | NSE equity 26.4bps or futures 7.3bps | ✓ **PASS** | NSECostModel verified |
| REPRODUCIBLE BACKTEST | Same inputs → same outputs | ✓ **PASS** | Deterministic pipeline |
| NO TEST-SET OPTIMIZATION | Thresholds not tuned on test data | ⚠ **PARTIAL** | G6 stress config post-hoc |
| POSITIVE NET EXPECTANCY | Mean net P&L > 0 after costs | ✗ **FAIL** | Oct 1: −28.5% (anomalous); Label mean: −0.26% |
| POSITIVE NET P&L | Cumulative net P&L > 0 | ✗ **UNVERIFIED** | Only 2 days of live P&L |
| SURVIVORSHIP-FREE UNIVERSE | Historical F&O membership validated | ⚠ **PARTIAL** | DATA_UNAVAILABLE for historical dates |
| NO CHERRY-PICKING | Evaluation includes all signals, all regimes | ✗ **FAIL** | Only bear market days evaluated |

### Target Performance Criteria

| Criterion | Status | Actual | Required |
|-----------|--------|--------|---------|
| >90% actionable signal precision | ✗ FAIL | 63.9% (proxy) | > 90% |
| Directional accuracy >90% | ✗ FAIL | 64.2% (proxy), 23.8% (live Oct 1) | > 90% |
| Positive net expectancy | ✗ FAIL (live) / ✓ PASS (proxy) | Live: negative; Proxy: +1.66% | > 0 |
| Statistical significance | ✓ PASS (proxy) / ✗ UNCERTAIN (live) | p<0.0001 (proxy); Oct 1 contradicts | p < 0.05 |
| Stable across regimes | ✗ FAIL | Only BEAR confirmed | All 4 regimes |
| Win rate > break-even | ✗ FAIL (live Oct 1) | 23.8% << 56.9% | > 56.9% (equity) |
| Profit factor > 1.0 | ✗ UNVERIFIED | Oct 1 negative; proxy 2.79 | > 1.0 |

---

## Section-by-Section Assessment

### Data — PASS (with caveats)
Parquets are clean, timestamps consistent (18:30:00 UTC), no phantom bars. Caveats: historical universe eligibility, corporate actions, lot sizes partially unavailable.

### Feature Generation — PASS
No static leakage, no dynamic leakage, PIT-safe. Conditional risks in intraday and news features require timestamp verification.

### Model — UNVERIFIED
IC = 0.4136 is unusually high and requires independent verification with a placebo test. PBO = 0.000 uses a simplified (not rigorous) formula.

### Signal Generation — PASS (structural) / FAIL (regime)
Signal generation is structurally correct. Performance fails in non-bear market conditions.

### Backtest (7-day) — NEW, PASS (proxy)
New `SevenDayBacktestEngine` correctly evaluates 7-trading-day outcomes with NSE calendar and realistic costs. 19 regression tests pass. Proxy evaluation shows viable economics.

### Leakage — PASS
Complete lookahead bias audit passes.

### Execution Realism — PASS
Next-bar-open execution enforced. BACKTEST_EXECUTION_CONTRACT.md documents all rules.

### Profitability — FAIL
Live Oct 1 data shows 23.8% win rate (significantly below 50% by binomial test). Two bear-day sessions insufficient to declare profitability.

### Statistical Significance — FAIL (live) / PASS (proxy)
Live evidence: 202 cumulative resolved signals (Sep 29 + Sep 30 + Oct 1), 93 wins = 46.0% win rate. P-value vs H0=50%: > 0.05. NOT STATISTICALLY SIGNIFICANT.

### Reproducibility — PASS
Pipeline is deterministic. Code, dataset version, model version, configuration all documented.

---

## What Must Change Before Production

### P0 — Blocking
1. **Audit `resolve_forward_paper.py`** for net_pct calculation bug (Oct 1 −28.5% anomaly)
2. **Accumulate 20+ trading days** across mixed regimes (BULL, BEAR, SIDEWAYS)
3. **Win rate > 56.9%** sustained over 20+ days at equity costs

### P1 — Required Before Production Promotion
4. **Retrain on 7-day asymmetric barrier labels** (`src/labels/seven_day.py`)
5. **Run WF validation at 26.4bps** (not 10bps) — verify Sharpe remains > 0
6. **Placebo IC test** — shuffle labels, confirm IC drops to ~0
7. **Independent ECE verification** — reproduce ECE claim or recalibrate
8. **Regime filter implementation** — suppress wrong-direction signals in confirmed regimes

### P2 — Required for Full Production Certification
9. Proper CPCV PBO implementation
10. Beta-neutral LONG overlay for bear market conditions
11. Live performance > 10pp above random baseline over 50+ resolved signals

---

## Maximum Honest Performance Assessment

```
Label EV (old 5-bar ±2%):
  NEGATIVE at equity costs (−0.31%/trade)
  Trains model on near-random labels

Label EV (new 7-day asymmetric):
  MARGINALLY NEGATIVE gross, POSITIVE framework (40.2% > 35% break-even)
  Addresses CRIT-002

True OOS performance (estimated from IC=0.4136):
  At equity (26.4bps): Sharpe ~0.35, win rate ~52–55%
  At futures (7.3bps): Sharpe ~1.0–1.5, win rate ~55–60%

Maximum leakage-free out-of-sample performance achieved:
  WIN RATE = ~55% (estimated from walk-forward IC and live data)
  This DOES NOT achieve the >90% target without significant model improvements

Why >90% is currently unachievable without compromising validity:
  1. Labels are near-50/50 binary — max theoretical accuracy ~65–70% at IC=0.4
  2. Transaction costs eliminate marginal signals
  3. Market is partially efficient over 7-day horizons
  4. Regime-dependent performance limits all-weather accuracy
```

---

*Supersedes all previous PRODUCTION_READINESS_REPORT.md versions.*  
*This is the authoritative production readiness assessment as of 2026-10-01.*
