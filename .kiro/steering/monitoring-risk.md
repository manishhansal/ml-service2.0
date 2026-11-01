---
inclusion: fileMatch
fileMatchPattern: "src/{monitoring,risk}/**"
---

# Monitoring & Risk — Karpathy Guidelines (Service-Specific)

> Applies on top of `karpathy-guidelines.md` when working in `src/monitoring/` or `src/risk/`.

## Think Before Coding

- `DrawdownManager` has 4 states: NORMAL → CAUTION → DEFENSIVE → HALTED. Before changing any threshold or transition logic, state which state transitions your change affects and what the live-session consequence is.
- `DriftDetectorV3` drives RETRAIN/ROLLBACK decisions for the live model. Threshold changes are not neutral — tightening a threshold can trigger a rollback of the v2c model mid-session.
- There are 16 custom Prometheus metrics (`src/monitoring/metrics.py`). Before adding a new metric, check whether it already exists under a different name.

## Simplicity First

- `evidently` and `nannyml` are already imported for drift. Do not add a third drift library.
- Prometheus counter/gauge/histogram registration is global — duplicate registration causes runtime errors. Check before creating a new metric object.
- Risk state machine logic should remain in `DrawdownManager`. Do not replicate state checks in routers or the meta engine.

## Surgical Changes

- The BULL suppressor parameters (`strategy/feature_weights.json`) are live configuration. Do not modify them as a side effect of monitoring code changes.
- `src/monitoring/drift.py` is excluded from coverage metrics (requires live data). Do not add mocked-only tests to inflate its coverage number.
- Prometheus metric names must follow the existing snake_case convention with the `ml_service_` prefix. Renaming a metric is a breaking change for any Grafana dashboard scraping it.

## Goal-Driven Execution

- For DrawdownManager changes: success = all 4 state transitions produce the correct resulting state, verified by `pytest tests/risk/ -q`.
- For drift changes: success = PSI and wasserstein distance computations return correct outputs on known-good fixture data.
- For Prometheus changes: success = `GET /metrics` response contains the new metric with the correct type annotation (# TYPE line).
- Alert thresholds must be verified against the historical OOS data before being changed in production configuration.
