# BACKTEST EXECUTION CONTRACT
**Repository:** ml-service2.0  
**Date:** 2026-10-01  
**Version:** v1.0

## 1. Governing Principle

Every backtest trade must use ONLY prices that would have been observable and executable at the time of trade entry. No simulation may access a price that was not available to the trader at the moment of the decision.

## 2. Signal Generation Timing

```
T = signal_date (NSE trading day)

Signal fires at:
  close of bar T (15:30 IST = 10:00 UTC)

Signal is generated using:
  ONLY features computed from data available at T close
  i.e., all OHLCV, derivative, macro, news data with timestamp ≤ T

Entry is available at:
  EARLIEST: open of bar T+1 (09:15 IST next trading day)
  LATEST:   intraday price during T+1

Standard execution assumed:
  OPEN of bar T+1 (market-on-open order)
```

## 3. Entry Price Rules

| Rule | Value |
|------|-------|
| Entry price | `open[T+1]` — next trading day's open |
| Entry timestamp | T+1 09:15 IST |
| Price availability | NSE pre-open session publishes indicative opens; actual open available at 09:15 IST |
| Prohibited entries | `close[T]` (signal bar) — this is LOOK-AHEAD EXECUTION and is FORBIDDEN |
| Prohibited entries | Any intraday price before the open of T+1 |
| Permitted adjustment | Slippage model applied to open price to simulate fill |

## 4. Exit Price Rules

| Rule | Value |
|------|-------|
| Primary exit (7-day) | `close[T+7]` — close of the 7th TRADING day after T |
| Early exit (stop) | First bar where `low ≤ stop_level` (LONG) or `high ≥ stop_level` (SHORT) |
| Early exit (target) | First bar where `high ≥ target_level` (LONG) or `low ≤ target_level` (SHORT) |
| Stop/target fill price | Barrier level itself (not the bar's close) |
| Exit timestamp | T+7 15:30 IST for time exit |

## 5. Trading Calendar

```
NSE F&O trading days:
  Monday–Friday, excluding NSE holidays defined in strategy/nse-calendar.json

7-day horizon:
  Count exactly 7 NSE trading days after T.
  Do NOT count T itself.
  Do NOT count Saturday, Sunday, or NSE holidays.

Example:
  T = Thursday 2026-09-25
  T+1 = Monday 2026-09-29 (entry)
  T+7 = Tuesday 2026-10-07 (primary exit)
  (skips: Sat Sep 27, Sun Sep 28)
```

## 6. Cost Model (Mandatory in all economic evaluations)

### NSE Equity Delivery
```
Brokerage (round-trip):          6.00 bps
STT (sell side):                10.00 bps
Exchange + clearing (RT):        0.07 bps
GST on brokerage:                1.08 bps
DP charges (sell side):          1.70 bps
Stamp duty (buy):                1.50 bps
Slippage (both sides):           7.00 bps
─────────────────────────────────────────
Total:                         ~27.35 bps
```

### NSE Futures (F&O Segment)
```
Brokerage (flat rate):           0.08 bps
STT (sell side futures):         0.13 bps
Exchange charges (RT):           0.04 bps
GST on brokerage:                0.01 bps
Slippage (both sides):           7.00 bps
─────────────────────────────────────────
Total:                          ~7.26 bps
```

These totals are used in `NSECostModel.equity()` and `NSECostModel.futures()` respectively.

## 7. Position Sizing

Default assumption: 100% capital deployed per signal (fractional Kelly = 1/n concurrent positions). Position sizing does not affect per-trade return metrics, only portfolio-level metrics.

## 8. Concurrent Position Handling

The `SevenDayBacktestEngine` evaluates each signal independently. Portfolio-level backtests must implement position caps to avoid unrealistic capital allocation.

## 9. Prohibited Backtest Practices

| Practice | Reason | Status |
|---------|--------|--------|
| Entry at signal bar's close | Look-ahead execution | FORBIDDEN |
| Entry at intraday low/high | Impossible to guarantee fill | FORBIDDEN |
| Exit at intraday extreme | Not executable at optimal price | FORBIDDEN |
| Reusing test data for threshold optimization | Data snooping | FORBIDDEN |
| Running walk-forward on the same data used for HPO | Test contamination | FORBIDDEN |
| Survivorship-only universe | Inflated returns | FORBIDDEN |
| Post-hoc regime filtering | Cherry-picking | FORBIDDEN |

## 10. Verification Tests

All backtest implementations must pass:
- `test_entry_price_is_next_bar_open` — entry ≠ signal-bar close
- `test_exit_after_exactly_7_trading_days` — exits at T+7 NSE calendar
- `test_net_pnl_equals_gross_minus_cost` — cost correctly deducted
- `test_short_signal_inverts_return` — direction inversion correct
- `test_exit_uses_only_forward_data` — no historical data in outcome calculation

Implementation: `tests/test_seven_day_engine.py` (19 tests, all passing)

---
*This contract is binding for all economic certification reports.*
