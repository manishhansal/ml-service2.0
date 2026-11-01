---
inclusion: fileMatch
fileMatchPattern: "src/{scanners,analytics}/**"
---

# Scanners & Analytics — Karpathy Guidelines (Service-Specific)

> Applies on top of `karpathy-guidelines.md` when working in `src/scanners/` or `src/analytics/`.

## Think Before Coding

- There are 5 SMC scanner types: OrderBlock, FairValueGap, SupplyDemand, LiquiditySweep, MarketStructure. Before adding a new scanner variant, confirm it is truly distinct from the 5 existing types and not a parameter variation.
- Scanner results feed directly into the live signal registry (Redis). A change to zone/signal schema breaks the `/v2/scanners/registry` endpoint consumed by `alpha-forge`.
- `ScoreThresholdSweep`, `ForecastLedger`, `CounterfactualLedger`, and `SignalPromotionEngine` all run in the live session pipeline. Changes here affect running sessions — state the impact before proceeding.
- The BULL suppressor multiplier (0.3) and max LONG cap (4 positions) are live risk controls. Do not change the logic that reads `strategy/feature_weights.json` without explicit instruction.

## Simplicity First

- Each scanner follows a `config + scanner` pattern (e.g., `OrderBlockConfig`, `ObScanner`). New scanners must follow the same pattern — one config dataclass, one scanner class with a `.scan()` method.
- Analytics components (ledgers, trackers) should write to their designated output files/DB tables. Do not add cross-component writes as a side effect.
- The intraday pre-filter (`passes_intraday_volume`, `passes_intraday_volatility`) applies before all 5 scanners. Do not add per-scanner pre-filters that diverge from this shared gate — instead, update the shared functions.

## Surgical Changes

- `src/scanners/registry.py` manages Redis signal state (including `clear_for_new_day` and `close_all_active`). Do not add Redis writes outside this registry module.
- Scanner configs (`OrderBlockConfig`, `FVGConfig`, etc.) carry default parameters tuned against NSE F&O data. Do not change defaults as a side effect of feature work.
- The `_LATEST_PATH_WATCH` file-watcher in `src/main.py` depends on `artifacts/live_session/latest_scores.json`. Do not change the filename or path as a side effect of other work.

## Goal-Driven Execution

- For scanner changes: success = the scanner returns valid zone/signal objects for known fixture OHLCV data. Verify: `pytest tests/scanners/ -q`.
- For analytics changes (e.g., SignalPromotionEngine): success = promotion/rejection logic matches the documented 6-gate spec and produces the correct lifecycle state transitions.
- For registry changes: success = Redis keys are written/read in the correct format, verified against an integration test with a Redis fixture.
- Use `pytest.mark.unit` for fixture-based scanner tests and `pytest.mark.integration` for Redis-backed registry tests.
