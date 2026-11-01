---
inclusion: fileMatch
fileMatchPattern: "{src/{cache,clients,streaming,audit,registry,reconciliation,core},scripts,Dockerfile*,docker-compose*}/**"
---

# Infrastructure, Scripts & Clients — Karpathy Guidelines (Service-Specific)

> Applies on top of `karpathy-guidelines.md` when working in `src/cache/`, `src/clients/`, `src/streaming/`, `src/audit/`, `src/registry/`, `src/reconciliation/`, `src/core/`, `scripts/`, or Docker/compose files.

## Think Before Coding

- `src/audit/logger.py` is a tamper-detected, append-only system. The startup sequence in `src/main.py` **refuses to start** if audit integrity fails. Before touching audit code, state what property you are modifying and how you will verify tamper-detection still works.
- `src/clients/market_data_pb2.py` and `market_data_pb2_grpc.py` are **generated from protobuf definitions** in `protos/`. Never hand-edit them — regenerate via `hatch run generate-stubs`.
- `src/cache/redis_cache.py` is a shared singleton used across the entire request lifecycle. Changes to its interface affect every service that calls `app.state.cache`.
- Scripts in `scripts/` are live session operators (`autorun_till_close.py`, `daily_autorun_scheduler.py`, `session_watchdog.py`). Before modifying them, state whether they will be run in a currently active session.

## Simplicity First

- Redis key naming: all scanner registry keys follow a defined pattern in `src/scanners/registry.py`. Do not invent a new key namespace without adding it to the registry module.
- `src/streaming/streamer.py` (SignalStreamer) uses a simple ConnectionManager. Do not add pub/sub libraries (e.g., aioredis pub/sub) unless the scale demonstrably requires it — the current file-watcher + broadcast model handles the current use case.
- `docker-compose.yml`, `docker-compose.full-stack.yml` are production compose files. Do not add experimental services to them without explicit instruction — use `docker-compose.test.yml` for test-only infrastructure.
- The `Makefile` contains the canonical dev commands. Add new targets there rather than documenting one-off `python -c` snippets.

## Surgical Changes

- Do not change `src/core/exceptions.py` as a side effect of other work — exception classes are referenced across the entire codebase and renaming one is a broad change.
- The Redis connection lifecycle (`connect()` / `disconnect()`) is managed in the FastAPI lifespan in `src/main.py`. Do not add connection management inside individual service modules.
- `src/reconciliation/` handles forward paper reconciliation. Do not add writes to the `positions` or `forecasts` PostgreSQL tables from outside this module.
- `src/registry/` is the single authority for model stage transitions. Do not hard-code a stage string (e.g., `"shadow"`) anywhere else in the codebase — use the registry enum.

## Goal-Driven Execution

- For Redis/cache changes: success = the cache returns the correct value after write, verified with a unit test using `fakeredis`.
- For streaming changes: success = a WebSocket client receives the broadcast message within one poll cycle (2 s), verified by an integration test.
- For Docker/compose changes: success = `docker-compose config` validates without errors AND the service starts and passes `/health`.
- For script changes: success = the script runs to completion on a dry-run (if the script supports `--dry-run`) or produces correct log output on a single iteration.
- For audit changes: success = tamper-detection still raises `AuditLogViolation` on a modified log, verified by `pytest tests/audit/ -q`.
