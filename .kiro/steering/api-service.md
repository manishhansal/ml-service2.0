---
inclusion: fileMatch
fileMatchPattern: "src/api/**"
---

# API Service — Karpathy Guidelines (Service-Specific)

> Applies on top of `karpathy-guidelines.md` when working in `src/api/`.

## Think Before Coding

- All routers live under `/v2/` prefix. Before adding a new endpoint, check the existing API reference (`docs/operations/API_REFERENCE.md`) to confirm it doesn't already exist.
- API contracts are consumed by `alpha-forge` (port 3000) — breaking changes silently break the live consumer. State the contract change explicitly before making it.
- Authentication is enforced via `X-API-KEY` middleware in `src/main.py`. New endpoints inherit it automatically; do not add per-route auth unless there's a specific reason.

## Simplicity First

- Router handlers should be thin: validate input → call a service/engine → return a Pydantic model. Business logic belongs in the service layer, not in the router.
- Response models must be Pydantic v2 schemas defined in `src/schemas/` — do not define inline response dicts.
- Do not add new routers to `src/main.py` without checking that the prefix and tags are consistent with the existing pattern.

## Surgical Changes

- `src/api/predict.py`, `signals.py`, and `monitoring.py` are called by the live session pipeline. Changes to their request/response shapes break running sessions.
- Do not modify `_EXEMPT_PATHS` in `src/main.py` without explicit instruction — this controls authentication bypass.
- Do not touch the WebSocket endpoint (`/v2/stream/signals`) when working on REST routes.

## Goal-Driven Execution

- For every new endpoint, success = the endpoint returns the correct HTTP status with a valid response body.
- Verify with: `pytest tests/api/<module_test>.py -q` — check both happy-path and error-path cases.
- If an endpoint touches the audit log, verify the audit entry format matches the existing JSONL schema.
