# Backtesting Specification
**ml-service2.0 — Backtest Engine Design**

*Date: 2026-09-24*

---

## 1. Backtest Principles

1. **No look-ahead** — entry at next bar's open after signal bar's close
2. **Realistic costs** — 10-15bps round-trip for NSE F&O
3. **Slippage model** — minimum 2bps per side (half-spread)
4. **Session rules** — no entries in last 15 minutes, no overnight for intraday
5. **Deterministic** — same inputs always produce same results
6. **Reproducible** — dataset version + random seed documented

---

## 2. Cost Model

```python
# NSE F&O round-trip cost estimate (approximate, varies by broker)
BROKERAGE = 0.0003          # 3bps per side (discount broker)
STT_INTRADAY = 0.0001       # 1bp STT (F&O intraday)
EXCHANGE_CHARGES = 0.00003  # 0.3bps
GST_ON_BROKERAGE = 0.18 × BROKERAGE
SEBI_FEE = 0.000001         # Negligible

TOTAL_ROUND_TRIP = 0.0010   # 10bps (conservative estimate)
SLIPPAGE_PER_SIDE = 0.0002  # 2bps per side (minimum)

EFFECTIVE_COST = TOTAL_ROUND_TRIP + 2 × SLIPPAGE_PER_SIDE  # ~14bps total
```

---

## 3. Execution Model

```python
# Signal at bar_t close:
entry_price = bar_{t+1}.open  # NOT bar_t.close

# Stop loss:
# If stop triggered within bar_{t+k}:
#   exit_price = max(stop_price, bar_k.low)  # assume worst fill
# If stop is gapped through:
#   exit_price = bar_k.open (gap open)

# Target:
# If target triggered within bar_{t+k}:
#   exit_price = min(target_price, bar_k.high)

# Horizon expiry:
# At max_bars, exit at bar_{t+max_bars}.close
```

---

## 4. Required Metrics

All backtests must report the full suite from `10_VALIDATION_FRAMEWORK.md`.

---

## 5. Backtest Anti-Patterns

| Anti-pattern | Correct approach |
|---|---|
| Fill at signal bar close | Fill at next bar open |
| Assume full fill at any price | Model partial fills or slippage |
| Ignore transaction costs | Deduct costs from every trade |
| Assume bid=ask | Use half-spread as minimum cost |
| Use adjusted prices with future adjustments | Use unadjusted prices at trade time |
| Ignore short-selling constraints | Model locate costs or disable shorts if not feasible |

---

*End of Backtesting Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/backtest/engine.py — BacktestEngine with next-bar-open fills (no lookahead), a configurable CostModel (brokerage + fees + half-spread + slippage), gross vs net separation, and cost_sensitivity_analysis() at 5/10/20 bps. Never assumes signal price equals execution price. Tested in tests/test_backtest_engine.py. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
