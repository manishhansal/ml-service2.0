# COST MODEL v2 — STANDARDIZED
**Repository:** ml-service2.0 | **Date:** 2026-10-01 | **Version:** COST_MODEL_V2

---

## Problem

Six different round-trip cost assumptions existed in the codebase:

| Value (bps) | Where Used |
|-------------|-----------|
| 8.5 bps | G6 stress test, "NSE Futures" claim |
| 10 bps | `TrainingPipeline.TRANSACTION_COST_BPS` |
| 14 bps | `BacktestEngine` base `CostModel()` |
| 26.4 bps | `NSECostModel.equity()` (computed) |
| 27.35 bps | `COST_MODEL_V2.md` (first attempt) |
| 27.65 bps | `LabelFactory`, live session, reports |

**This is unacceptable.** Different cost assumptions produce incomparable results and can make an unprofitable strategy appear profitable.

---

## Canonical Cost Model v2

All evaluations must use ONE of these two models, recorded as `cost_model_version = COST_MODEL_V2`.

### EQUITY_DELIVERY (NSE cash segment)

| Component | Rate | Notes |
|-----------|------|-------|
| Brokerage (entry + exit) | 6.00 bps | 3 bps per side, flat-rate broker |
| STT (sell-side only) | 10.00 bps | 0.10% of sell turnover |
| Exchange charges (RT) | 0.07 bps | NSE + clearing |
| GST on brokerage | 1.08 bps | 18% of 6 bps |
| DP charges (sell-side) | 1.70 bps | NSDL/CDSL delivery charge |
| Stamp duty (buy-side) | 1.50 bps | 0.015% of buy turnover |
| Slippage (both sides) | 7.00 bps | 3.5 bps each way (typical NSE F&O) |
| **Total round-trip** | **27.35 bps** | |

### FUTURES (NSE F&O segment)

| Component | Rate | Notes |
|-----------|------|-------|
| Brokerage (RT) | 0.08 bps | ₹20/order flat at ₹25L notional |
| STT (sell-side futures) | 0.13 bps | 0.0125% of sell turnover |
| Exchange charges (RT) | 0.04 bps | NSE F&O + clearing |
| GST on brokerage | 0.01 bps | 18% of 0.08 bps |
| Slippage (both sides) | 7.00 bps | 3.5 bps each way |
| **Total round-trip** | **7.26 bps** | |

---

## Break-Even Win Rates (COST_MODEL_V2)

For symmetric barriers `±X%`:

| Barrier | Equity (27.35bps) | Futures (7.26bps) |
|---------|------------------|------------------|
| ±1% | 73.5% | 53.8% |
| ±2% | 56.9% | 51.9% |
| ±3% | 51.5% | 50.6% |

For asymmetric barriers `target T, stop S`:

| T / S | Equity BE | Futures BE |
|-------|-----------|-----------|
| 6% / 3% | 36.2% | 33.9% |
| 4% / 2% | 36.6% | 34.2% |
| 3% / 1.5% | 36.9% | 34.5% |

---

## Implementation

`COST_MODEL_V2` is the canonical cost constant. Use `NSECostModelV2` from `src/backtest/cost_model_v2.py` (to be created):

```python
from src.backtest.cost_model_v2 import NSECostModelV2

equity = NSECostModelV2.equity()    # 27.35 bps
futures = NSECostModelV2.futures()  # 7.26 bps
```

Every training experiment and backtest report must record:
```
cost_model_version: COST_MODEL_V2
cost_bps_equity: 27.35
cost_bps_futures: 7.26
```

---

## Historical Inconsistencies to Fix

| File | Current value | Correct value | Action |
|------|--------------|--------------|--------|
| `src/training/pipeline.py:TRANSACTION_COST_BPS` | 10.0 | 27.35 (equity) or 7.26 (futures) | Fix + retrain |
| `src/backtest/engine.py:CostModel()` | 14 bps | 27.35 | Fix |
| `src/data/labels.py:LabelConfig.cost_bps` | 27.65 | 27.35 | Fix |
| G6 test config | 8.5 bps | 7.26 bps (futures) | Fix |
| WF validation cost | 10 bps | 27.35 bps (equity) | Fix + retrain |

**Critical impact of training pipeline fix:** When `TRANSACTION_COST_BPS = 10` is changed to `27.35`, the reported net Sharpe will decrease. The current WF Sharpe of +1.076 was computed at 10bps. At 27.35bps, the estimated WF Sharpe ≈ +0.35 (based on OOS IC degradation).
