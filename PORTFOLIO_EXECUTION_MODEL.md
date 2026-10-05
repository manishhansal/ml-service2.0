# PORTFOLIO EXECUTION MODEL
**Repository:** ml-service2.0 | **Date:** 2026-10-01 | **Version:** v1.0  
**Implementation:** `src/backtest/portfolio_engine.py`

---

## 1. Overview

The portfolio execution model simulates how signals from the ML service would actually behave when executed within a real portfolio subject to capital constraints, risk limits, and position management rules. It distinguishes between two fundamentally different performance numbers:

| Mode | Purpose | Interpretation |
|------|---------|---------------|
| **Mode A (isolated)** | Theoretical signal quality | Sum of all individual signal returns assuming unlimited capital |
| **Mode B (portfolio)** | Actual tradable performance | P&L achievable with real capital constraints |

**Critical rule (§23A.27):** Only Mode B performance may be cited as the strategy's actual profitability. Mode A is the upper-bound signal quality diagnostic.

---

## 2. Event Processing Order (§23A.25)

At every trading bar, events are processed in this exact chronological order:

```
1. Receive market bar (OHLCV data for all symbols)
2. Mark-to-market all open positions at today's close
3. Fill pending orders at today's OPEN (+ slippage)
4. Evaluate stop-loss/target conditions using today's HIGH/LOW
5. Execute time exits for positions at T+7 bars
6. Process new signals arriving at this timestamp
7. Apply portfolio constraints → eligible/rejected decisions
8. Generate orders for eligible signals
9. Record equity curve point
```

**PIT guarantee:** Signals at bar T use only OHLCV data ≤ T. Orders fill at T+1 open (never at T close).

---

## 3. Portfolio State

Maintained at every bar:

| Field | Description |
|-------|-------------|
| `cash` | Available cash (reduces on position open, increases on close) |
| `positions` | Dict[symbol → List[Position]] — all open positions |
| `pending_orders` | Orders waiting to fill at next bar's open |
| `realized_pnl` | Cumulative realized P&L |
| `unrealized_pnl` | MTM unrealized P&L across all open positions |
| `total_equity` | `cash + Σ(committed_notional) + unrealized_pnl` |
| `gross_exposure` | `Σ(notional) / total_equity` |
| `net_exposure` | `Σ(direction × notional) / total_equity` |
| `peak_equity` | Running maximum equity (for drawdown) |
| `drawdown` | `(equity - peak_equity) / peak_equity` |
| `consecutive_losses` | Reset to 0 on any win |
| `daily_pnl_today` | Reset at each calendar day boundary |

**Equity formula:**
```
total_equity = cash 
             + Σ(position.quantity × position.entry_price)          [committed]
             + Σ(direction × (current_price - entry_price) × qty)   [unrealized]

This correctly handles both LONG and SHORT positions.
```

---

## 4. Signal Evaluation (Mode B)

Every signal passes through these checks in order. Any failure causes immediate rejection with a documented reason:

| Check | Rejection Code | Description |
|-------|---------------|-------------|
| 1. Deduplication | `DUPLICATE_SIGNAL` | Same signal_id seen before |
| 2. Daily loss halt | `MAX_DAILY_LOSS_REACHED` | `daily_pnl_today ≤ -(max_daily_loss × equity)` |
| 3. Drawdown halt | `MAX_DRAWDOWN_REACHED` | `|drawdown| ≥ max_total_drawdown` |
| 4. Consecutive losses | `MAX_CONSECUTIVE_LOSSES` | `consecutive_losses ≥ max_consecutive_losses` |
| 5. Open risk | `MAX_OPEN_RISK` | `open_risk ≥ max_open_risk` |
| 6. Position limit | `MAX_POSITIONS_REACHED` | `open + pending ≥ max_positions` |
| 7. Same-symbol policy | `SAME_SYMBOL_IGNORE_POLICY` | Duplicate direction (IGNORE mode) |
| 8. Opposite-signal | `OPPOSITE_SIGNAL_IGNORE_POLICY` | Opposite direction (IGNORE mode) |
| 9. Capital | `INSUFFICIENT_CAPITAL` | `available_capital < 1 share price` |
| 10. Gross exposure | `MAX_GROSS_EXPOSURE` | `estimated_committed_exp > max_gross_exposure` |
| 11. Net exposure | `MAX_NET_EXPOSURE` | `|net + new| > max_net_exposure` |
| 12. Sector exposure | `MAX_SECTOR_EXPOSURE` | `sector_exp ≥ max_sector_exposure` |
| 13. Expected value | `LOW_EXPECTED_VALUE` | `EV < min_expected_net_return` |

---

## 5. Position Lifecycle (§23A.12)

Every position goes through this exact lifecycle:

```
T+0: SIGNAL generated (model scores at close[T])
     ↓
T+0: ORDER generated (if signal passes all checks)
     ↓
T+1: ENTRY FILL at open[T+1] + slippage
     ↓ (7 trading days)
T+1..T+7: Daily mark-to-market
T+1..T+7: Stop-loss / take-profit monitoring (using intraday HIGH/LOW)
     ↓
T+7: TIME EXIT at close[T+7] if no earlier trigger
```

Early exit conditions (in priority order at each bar):
1. `high ≥ target_price` (LONG target) → TARGET_HIT at `target_price`
2. `low ≤ stop_price` (LONG stop) → STOP_HIT at `stop_price`
3. `bars_held ≥ 7` → TIME_EXPIRY at `close[T+7]`
4. Opposite signal received → OPPOSITE_SIGNAL (per policy)

---

## 6. Stop/Target Conflict Resolution (§23A.13)

**Rule:** If both HIGH ≥ target AND LOW ≤ stop in the same bar, the engine evaluates TARGET first, then STOP. This is the **optimistic assumption** for long positions.

For a more conservative assumption, STOP would be evaluated first. The current implementation uses TARGET_FIRST which is documented and applied consistently across the entire backtest.

Ambiguous cases are recorded in `portfolio_events.csv` as `CONFLICT_SAME_BAR` events.

---

## 7. Same-Symbol Policy

| Policy | Behavior |
|--------|---------|
| `IGNORE` (default) | Second signal on same symbol (same direction) is rejected |
| `ADD` | Adds to existing position if risk criteria met |
| `REPLACE` | Closes existing, opens new at new price |
| `AVERAGE` | Averages into existing position |

**Production default:** `IGNORE` — conservative, prevents pyramiding without validation evidence.

---

## 8. Opposite-Signal Policy

| Policy | Behavior |
|--------|---------|
| `CLOSE_ONLY` (default) | Closes existing position, does NOT open reverse |
| `CLOSE_AND_REVERSE` | Closes existing, opens opposite direction |
| `IGNORE` | Ignores opposite signal while position is open |
| `REDUCE` | Partially reduces position size |

**Production default:** `CLOSE_ONLY` — prevents unvalidated reverse positions.

---

## 9. Capital Allocation Methods

| Method | Formula | Use Case |
|--------|---------|---------|
| `EQUAL` | `available / remaining_slots` | Default; simple, fair |
| `CONFIDENCE_WEIGHTED` | `max_per_pos × conviction` | Higher allocation to strong signals |
| `RISK_WEIGHTED` | `risk_budget / stop_distance` | Fixed risk per trade |
| `VOLATILITY_SCALED` | `target_vol / atr` | Equal vol contribution |
| `FIXED_RISK` | `equity × risk_per_trade / stop_pct` | Institutional standard |

---

## 10. Cost Model

```
Equity (NSE cash delivery):
  Brokerage (RT):         6.00 bps
  STT (sell):            10.00 bps
  Exchange + clearing:    0.07 bps
  GST on brokerage:       1.08 bps
  DP charges (sell):      1.70 bps
  Stamp duty (buy):       1.50 bps
  Slippage (both sides):  7.00 bps
  ─────────────────────────────────
  Total round-trip:      ~27.35 bps

Futures (NSE F&O):
  Brokerage (RT):         0.08 bps
  STT (sell-side):        0.13 bps
  Exchange charges:       0.04 bps
  Slippage:               7.00 bps
  ─────────────────────────────────
  Total round-trip:       ~7.26 bps
```

---

## 11. Signal Deduplication (§23A.17)

Every signal gets a deterministic unique ID:
```python
signal_id = SHA256(f"{symbol}|{timestamp[:16]}|{direction}|{model_version}")[:16]
```

Signals with the same ID are accepted only once. Subsequent occurrences are logged as `DUPLICATE_SIGNAL` rejections.

---

## 12. Partial Fill Limitation (§23A.10)

At daily bar granularity, partial fills cannot be simulated (we don't have intraday order book data). The engine assumes 100% fill probability at each bar's open price + slippage. This is a documented limitation.

For high-volume NSE F&O stocks (RELIANCE, HDFCBANK, etc.), the 100% fill assumption is reasonable for positions up to ~₹50M notional. For smaller/less-liquid symbols, actual fills may be partial at the assumed entry price.

---

## 13. Execution Contract Reference

See `BACKTEST_EXECUTION_CONTRACT.md` for the binding execution rules. All implementations in this engine comply with that contract.
