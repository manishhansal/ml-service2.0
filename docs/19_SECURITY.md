# Security
**ml-service2.0 — Security Controls and Threat Model**

*Date: 2026-09-24*

---

## 1. Authentication

- All `/v2/*` endpoints require `X-API-KEY` header
- WebSocket requires `?api_key=` query parameter
- `/health` is exempt (no business data exposed)
- API key validated via constant-time comparison (prevent timing attacks)
- 401 returned for missing or invalid key

## 2. Credential Management

**NEVER log:**
- API keys
- Broker secrets
- Database credentials
- Redis passwords
- Bearer tokens

Structlog configuration masks sensitive environment variable names. Keys are referenced by name in logs, not value.

## 3. Input Validation

- All inputs validated by Pydantic V2 strict mode
- ValidationError → HTTP 422 with field-level detail (no stack trace)
- Numeric bounds enforced (confidence ∈ [0,1], etc.)
- Symbol names uppercased and validated
- Enum values strictly validated

## 4. Model Artifact Security

- SHA256 integrity check on artifact load
- `ArtifactIntegrityFailure` exception raised on mismatch (service refuses to use corrupt model)
- Do NOT use `pickle.load()` on untrusted artifacts (use joblib with explicit allow list)
- Audit log integrity verified at startup

## 5. No Data Exfiltration

- ML Service does NOT call external market data providers directly
- All outbound calls go to: data-service2.0 (internal), SentinelPulse (internal), MLflow (internal), Redis (internal)
- No telemetry, no external API calls, no model weight upload

## 6. Injection Prevention

- All SQL queries use parameterized form (no ORM in ml-service2.0 currently)
- HTTP headers from external requests not reflected verbatim
- Command injection: no shell=True subprocess calls

## 7. CORS

- Origins restricted to `settings.allowed_origins` list
- Default: allow localhost in dev; restrict to AlphaForge domain in production

## 8. Secrets in Configuration

`.env.example` provides template — never commit real values. Required secrets:
- `ML_SERVICE_API_KEY` — generate with `openssl rand -hex 32`
- `DATA_SERVICE_API_KEY` — provided by data-service2.0 administrator
- `SENTINEL_PULSE_API_KEY` — provided by SentinelPulse administrator

---

*End of Security*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Security posture upheld through the implementation phase: market data flows exclusively via data-service2.0 (no direct provider calls); X-API-KEY auth enforced; SHA-256 artifact integrity; immutable append-only audit and feedback stores; no credentials committed (verified .env / artifacts gitignored). Dependency CVE scanning and secret scanning in CI remain a documented follow-up. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
