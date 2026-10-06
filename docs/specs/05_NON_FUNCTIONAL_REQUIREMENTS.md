# Non-Functional Requirements
**ml-service2.0**

*Date: 2026-09-24*

---

## Performance

| Requirement | Target | Measurement |
|---|---|---|
| Inference latency p50 | < 50ms | Prometheus histogram |
| Inference latency p95 | < 100ms | Prometheus histogram |
| Inference latency p99 | < 200ms | Prometheus histogram |
| Feature pipeline latency p95 | < 80ms | Internal timer |
| LLM news reasoning timeout | 120ms hard limit | asyncio.wait_for |
| Cold start (service startup) | < 10 seconds | Lifespan timer |
| Model load time | < 5 seconds per model | Startup log |
| WebSocket signal latency | < 20ms from generation | Internal timer |

## Availability

| Requirement | Target |
|---|---|
| Service uptime | 99.9% during NSE market hours (09:00-16:00 IST) |
| Recovery from DataService outage | Automatic via circuit breaker |
| Recovery from SentinelPulse outage | Automatic (market-only fallback) |
| Recovery from Redis outage | Automatic (LRU fallback) |
| Model rollback time | < 30 seconds |

## Security

| Requirement | Notes |
|---|---|
| All endpoints require X-API-KEY | Except /health |
| WebSocket requires api_key query param | HTTP headers unavailable on WS |
| API keys never logged | Structlog masks sensitive fields |
| Audit log integrity | SHA256 hash chain, tamper detection at startup |
| Model artifacts integrity | SHA256 on load |
| No credential storage in model artifacts | |
| Input validation on all endpoints | Pydantic V2 strict mode |

## Scalability

| Requirement | Target |
|---|---|
| Concurrent inference requests | >= 10 simultaneous (uvicorn workers=4) |
| Throughput | >= 50 requests/second under normal load |
| Feature computation | Stateless — scales horizontally |

## Reliability

| Requirement | Notes |
|---|---|
| Deterministic inference | Same input → same output (no stochastic state mutations in decide()) |
| Idempotent training | Same dataset + seed → materially identical IC (within 0.001) |
| Immutable model artifacts | Never mutate; always create new version |
| Immutable audit log | Append-only JSONL; tamper detection |

## Maintainability

| Requirement | Target |
|---|---|
| Test coverage (real assertions) | >= 90% |
| Type hints | 100% (mypy strict) |
| Linting | ruff, zero violations |
| Dependency pinning | Exact versions in pyproject.toml |
| Code documentation | Docstrings on all public methods |

---

*End of Non-Functional Requirements*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

NFRs partially validated. Determinism/reproducibility: dataset hashing is deterministic (tested) and estimators use fixed seeds. Data-integrity: SHA-256 artifact integrity, immutable audit/feedback stores. NOT yet benchmarked on a live deployment: latency SLAs (p50/p95/p99) with trained models loaded, throughput, and Prometheus metric export — these remain follow-ups and are not blockers for the current RESEARCH_READY status. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
