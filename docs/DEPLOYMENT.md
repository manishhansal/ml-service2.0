# Deployment Guide — ml-service2.0

This guide covers every step from a fresh checkout to a production-ready deployment, including integration with the other services in the AlphaForge stack.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Variables Reference](#environment-variables-reference)
3. [Development Setup](#development-setup)
4. [Generating gRPC Stubs](#generating-grpc-stubs)
5. [Running the Service](#running-the-service)
6. [Production Deployment](#production-deployment)
   - [Docker Build](#docker-build)
   - [docker-compose (Full Stack)](#docker-compose-full-stack)
   - [docker-compose (Integration Tests)](#docker-compose-integration-tests)
7. [Service Integrations](#service-integrations)
   - [alpha-forge](#alpha-forge)
   - [data-service2.0](#data-service20)
   - [SentinelPulse](#sentinelpulse)
   - [Redis](#redis)
   - [MLflow](#mlflow)
8. [Health Checks and Readiness Probes](#health-checks-and-readiness-probes)
9. [Observability](#observability)
10. [Known Limitations](#known-limitations)

---

## Prerequisites

| Dependency | Minimum version | Notes |
|---|---|---|
| Python | 3.11+ | `python3.11 --version` |
| Redis | 7.0+ | Feature cache + signal cache |
| Docker | 24.0+ | Optional — for containerized runs |
| Docker Compose | v2 (`docker compose`) | Optional |
| MLflow Tracking Server | 2.x | For experiment logging; can be local |
| gRPC tools | `grpcio-tools` | For stub generation from `.proto` |

Redis and MLflow can be omitted in development if you set `FEATURE_CACHE_TTL=0` (disables caching) and point `MLFLOW_TRACKING_URI` at a local SQLite file. The service starts and serves heuristic predictions without either dependency.

---

## Environment Variables Reference

Copy `.env.example` to `.env` and fill in the values marked **required**. All variables are read by `src/config.py` using `pydantic-settings`.

### Server

| Variable | Default | Required | Description |
|---|---|---|---|
| `PORT` | `8100` | No | Port ml-service2.0 listens on |
| `LOG_LEVEL` | `INFO` | No | structlog minimum level: `DEBUG` / `INFO` / `WARNING` / `ERROR` / `CRITICAL` |
| `DEPLOYMENT_MODE` | `research` | No | Controls heuristic fallback: `research` / `paper` / `shadow` / `validated_ml_only` |
| `ALLOWED_ORIGINS` | `http://localhost:3000` | No | Comma-separated CORS allowed origins (alpha-forge URL in production) |
| `UVICORN_WORKERS` | `4` | No | Number of Uvicorn worker processes (minimum 1) |

### Authentication

| Variable | Default | Required | Description |
|---|---|---|---|
| `ML_SERVICE_API_KEY` | — | **Yes** | API key required in `X-API-KEY` header for all prediction endpoints |

### Upstream: data-service2.0

| Variable | Default | Required | Description |
|---|---|---|---|
| `DATA_SERVICE_2_URL` | — | **Yes** | Base URL for data-service2.0, e.g. `http://localhost:8200` |
| `DATA_SERVICE_API_KEY` | — | **Yes (production)** | API key or JWT for data-service2.0 authentication |

### Upstream: SentinelPulse

| Variable | Default | Required | Description |
|---|---|---|---|
| `SENTINEL_PULSE_URL` | — | **Yes** | Base URL for SentinelPulse, e.g. `http://localhost:3001` |
| `SENTINEL_PULSE_API_KEY` | `""` | No (dev) / **Yes (production)** | API key for SentinelPulse |

### Storage: Redis

| Variable | Default | Required | Description |
|---|---|---|---|
| `REDIS_URL` | `redis://localhost:6379` | No | Redis connection URL |

### Storage: MLflow

| Variable | Default | Required | Description |
|---|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | No | MLflow tracking server URI |

### Storage: Filesystem

| Variable | Default | Required | Description |
|---|---|---|---|
| `MODEL_ARTIFACTS_PATH` | `./artifacts` | No | Root path for model artifact storage. Use a persistent volume in production |
| `AUDIT_LOG_PATH` | `./audit.jsonl` | No | Path to the append-only audit log (JSON Lines format) |

### Feature Pipeline

| Variable | Default | Constraints | Description |
|---|---|---|---|
| `FEATURE_CACHE_TTL` | `60` | ≥ 10 | TTL in seconds for feature vectors cached in Redis |
| `NEWS_CONTEXT_CACHE_TTL` | `90` | ≥ 10 | TTL in seconds for SentinelPulse news context in LRU cache |
| `MIN_CONFIDENCE_SCORE` | `70` | 0–100 | Minimum `DataConfidenceScore` from data-service2.0. Responses below this return `INSUFFICIENT_EVIDENCE` |
| `ALPHA360_ENABLED` | `false` | — | Enable Qlib Alpha360 360-dimensional factor computation (slower) |

### Training Pipeline

| Variable | Default | Constraints | Description |
|---|---|---|---|
| `EMBARGO_PERIOD_DAYS` | `10` | ≥ 5 | Embargo period in trading days for purged K-fold CV |
| `SHADOW_TRADING_MIN_DAYS` | `20` | ≥ 1 | Minimum trading days in shadow stage before live promotion eligibility |
| `OPTUNA_N_TRIALS` | `50` | ≥ 50 | Number of Optuna HPO trials per model |
| `CPCV_N_PATHS` | `10` | ≥ 10 | Number of overlapping test paths for CPCV |
| `MODEL_ACCEPTANCE_MIN_IC` | `0.02` | — | Minimum mean IC for the model acceptance gate |
| `MODEL_ACCEPTANCE_MAX_PBO` | `0.5` | — | Maximum backtest overfitting probability |

### Online Learning

| Variable | Default | Description |
|---|---|---|
| `ONLINE_LEARNING_ENABLED` | `false` | Enable incremental model updates on IC degradation |
| `MAX_CONSECUTIVE_ONLINE_UPDATES` | `5` | Maximum consecutive incremental updates before full retraining is required |
| `ONLINE_LEARNING_IC_DEGRADATION_THRESHOLD` | `0.20` | IC degradation fraction from 90-day baseline that triggers online learning |

### gRPC

| Variable | Default | Description |
|---|---|---|
| `GRPC_CONNECTION_TIMEOUT` | `5.0` | gRPC connection establishment timeout in seconds |
| `GRPC_PER_CALL_DEADLINE` | `2.0` | gRPC per-call deadline in seconds |
| `GRPC_MAX_RECONNECT_ATTEMPTS` | `3` | Maximum reconnect attempts after stream interruption before returning UNAVAILABLE |

### LLM / Meta-Decision Engine

| Variable | Default | Description |
|---|---|---|
| `LLM_INFERENCE_TIMEOUT_MS` | `120` | Maximum LLM inference time in milliseconds before falling back to cache |
| `LLM_MODEL_NAME` | `ProsusAI/finbert` | HuggingFace model identifier. Default: FinBERT. Alternative: `THUDM/FinGPT-v3.3` |

### Promotion Gates

| Variable | Default | Constraints | Description |
|---|---|---|---|
| `PREDICTIVE_GATE_MARGIN` | `0.005` | 0.001–0.05 | Minimum IC improvement over champion required for the PREDICTIVE gate |
| `APPROVAL_TOKEN_EXPIRY_HOURS` | `24` | — | Hours before an approvalToken expires |

---

## Development Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url> ml-service2.0
cd ml-service2.0

python3.11 -m venv .venv
source .venv/bin/activate      # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
# Production dependencies only
pip install -e .

# Full development install (pytest, ruff, mypy, hypothesis, etc.)
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
# Open .env and set ML_SERVICE_API_KEY, DATA_SERVICE_2_URL, SENTINEL_PULSE_URL
```

For a pure local development run where data-service2.0 and SentinelPulse are not available, the service will start and serve **heuristic predictions** for all endpoints. No trained ML artifacts are required.

### 4. Start a local Redis instance

```bash
# Docker one-liner
docker run -d -p 6379:6379 redis:7-alpine

# Or brew (macOS)
brew install redis && brew services start redis
```

### 5. Start a local MLflow instance (optional)

```bash
mlflow server --host 0.0.0.0 --port 5000
```

If MLflow is unavailable, training runs will fail but inference endpoints will continue to operate on any previously loaded artifacts.

---

## Generating gRPC Stubs

The data-service2.0 gRPC streaming interface requires Python stubs generated from `protos/market_data.proto`.

```bash
bash protos/generate_stubs.sh
```

This runs `python -m grpc_tools.protoc` and writes the generated stubs to `src/clients/protos/`. Re-run whenever `market_data.proto` changes. The stubs are not committed to the repository and must be regenerated on each fresh checkout.

Verify the stubs were generated:

```bash
ls src/clients/protos/
# market_data_pb2.py  market_data_pb2_grpc.py
```

---

## Running the Service

### Development (auto-reload)

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8100 --reload
```

The `--reload` flag watches `src/` for changes and restarts automatically. Do not use in production.

### Production (multiple workers)

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8100 --workers 4
```

The worker count is also configurable via `UVICORN_WORKERS`. Default is `4`.

### Verify the service is running

```bash
curl http://localhost:8100/health
# {"status":"healthy","timestamp":1752569100,"version":"2.0.0"}
```

---

## Running Tests

```bash
# Full test suite with coverage
pytest tests/ --cov=src --cov-fail-under=90 --timeout=120

# Property-based tests only
pytest tests/ -m hypothesis

# Latency benchmarks
pytest tests/test_signal_generation_latency.py --benchmark-only

# Type checking
mypy --strict src/

# Linting
ruff check src/ tests/
```

The full test suite requires live mock servers for data-service2.0 and SentinelPulse. The `docker-compose.test.yml` file spins up WireMock stubs automatically. See the [Integration Tests](#docker-compose-integration-tests) section.

---

## Production Deployment

### Docker Build

The `Dockerfile` uses a multi-stage build with three targets:

| Target | Purpose |
|---|---|
| `base` | Shared base layer (Python 3.11-slim, working directory) |
| `dev` | Includes `.[dev]` extras for testing in CI |
| `production` | Minimal production image, no test dependencies |

Build the production image:

```bash
docker build --target production -t ml-service2:latest .

# Tag with git SHA for immutable versioning
docker build --target production -t ml-service2:$(git rev-parse --short HEAD) .
```

Run the container:

```bash
docker run -d \
  --name ml-service2 \
  -p 8100:8100 \
  -e ML_SERVICE_API_KEY=your-key \
  -e DATA_SERVICE_2_URL=http://data-service2:8200 \
  -e SENTINEL_PULSE_URL=http://sentinel-pulse:3001 \
  -e REDIS_URL=redis://redis:6379 \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  -v /mnt/ml-artifacts:/app/artifacts \
  -v /mnt/audit-logs:/app/audit.jsonl \
  ml-service2:latest
```

**Important:** Mount `MODEL_ARTIFACTS_PATH` and `AUDIT_LOG_PATH` to persistent volumes. Artifacts and audit logs must survive container restarts.

### docker-compose (Full Stack)

Starts ml-service2.0 together with Redis, MLflow, and WireMock stubs for data-service2.0 and SentinelPulse:

```bash
docker compose up
```

Services started:

| Service | Port | Description |
|---|---|---|
| `ml-service` | `8100` | ml-service2.0 (dev target, auto-reload) |
| `redis` | `6379` | Redis 7 |
| `mlflow` | `5000` | MLflow tracking server |
| `mock-data-service` | `8080` (internal) | WireMock stub for data-service2.0 |
| `mock-sentinel-pulse` | `8081` (internal) | WireMock stub for SentinelPulse |

### docker-compose (Integration Tests)

Runs the full integration test suite in CI:

```bash
docker compose -f docker-compose.test.yml up --abort-on-container-exit
```

Exits with the test container's exit code. Suitable for CI pipelines — returns non-zero when tests fail.

---

## Service Integrations

### alpha-forge

alpha-forge is the primary consumer of ml-service2.0. In alpha-forge's environment configuration:

```bash
# alpha-forge .env
ML_SERVICE_URL=http://localhost:8100        # local dev
# ML_SERVICE_URL=http://ml-service2:8100   # docker-compose / k8s

ML_SERVICE_API_KEY=your-ml-service-api-key-here
```

alpha-forge calls ml-service2.0 exclusively via:
- `POST /v2/predict/*` — prediction requests
- `POST /v2/meta/decide` — trading decisions
- `WS /v2/stream/signals` — real-time signal stream
- `GET /health` — liveness check

The CORS configuration in ml-service2.0 must include the alpha-forge origin:

```bash
# ml-service2.0 .env
ALLOWED_ORIGINS=http://localhost:3000
# Production:
# ALLOWED_ORIGINS=https://alpha-forge.yourfirm.com
```

### data-service2.0

data-service2.0 (port 8200) is the **sole authority** for all market data. ml-service2.0 never connects directly to NSE, Upstox, Angel One, Binance, or any market data provider.

```bash
# ml-service2.0 .env
DATA_SERVICE_2_URL=http://localhost:8200
DATA_SERVICE_API_KEY=your-data-service-api-key-here
```

All feature requests to data-service2.0 include:
- `X-API-KEY` (or `Authorization: Bearer <jwt>`) header
- `?pit_date=YYYY-MM-DD` parameter in backtest mode

**Timeouts and retries:**
- REST request timeout: 10 seconds per call
- gRPC connection timeout: `GRPC_CONNECTION_TIMEOUT` (default 5s)
- gRPC per-call deadline: `GRPC_PER_CALL_DEADLINE` (default 2s)
- gRPC stream reconnect: up to `GRPC_MAX_RECONNECT_ATTEMPTS` (default 3) with 1s delay
- Circuit breaker: 3 failures in 30 seconds → OPEN (60s recovery)

When data-service2.0 is unreachable, all prediction endpoints return `PredictionProvenance.UNAVAILABLE`. No alternative data source is ever used.

### SentinelPulse

SentinelPulse (port 3001) provides PIT-correct NLP features.

```bash
# ml-service2.0 .env
SENTINEL_PULSE_URL=http://localhost:3001
SENTINEL_PULSE_API_KEY=your-sentinel-pulse-api-key-here
```

**Endpoints consumed:**

| Purpose | SentinelPulse endpoint |
|---|---|
| Inference — per-symbol news context | `GET /api/v1/alphaforge/news-context/:instrument` |
| Training — PIT-correct asset features | `GET /api/v1/ml/features/asset/:assetId` |
| Training — labeled samples | `GET /api/v1/ml/training/samples` |
| Training — historical reactions | `GET /api/v1/ml/historical-reactions` |

**Degraded mode:** When SentinelPulse is unreachable or returns an incomplete response, the Feature Pipeline substitutes:
- All numeric news fields → `0.0`
- `impact_direction` → `"neutral"`
- `market_regime_nlp` → `"NEUTRAL"`
- The affected instrument and zeroed fields are recorded in the `FeatureQualityReport`

**News context cache:** Responses are cached in an in-process LRU cache with:
- Max entries: 500 per symbol
- TTL: `NEWS_CONTEXT_CACHE_TTL` (default 90 seconds)

**Training data validation:** Every training sample fetched from SentinelPulse must carry `look_ahead_validated: true`. Samples missing this flag are discarded. If more than 10% of a batch is discarded, training is halted pending operator acknowledgement.

### Redis

Redis is used for:

| Key pattern | TTL | Content |
|---|---|---|
| `feature:{symbol}:{bucket}` | `FEATURE_CACHE_TTL` (60s) | Serialized `FeatureVector` JSON |
| `news:{symbol}` | `NEWS_CONTEXT_CACHE_TTL` (90s) | SentinelPulse news context |
| `signal:{symbol}:latest` | 300s | Latest `MetaOutput` JSON |
| `llm_news:{symbol}` | 60s | Cached `NewsSignal` from FinBERT/FinGPT |
| `model:status` | — | Current model load status |
| `drift:latest` | 86400s | Latest drift report |

All Redis access uses `redis.asyncio` with async/await.

**Configuration tips:**
- Increase `maxmemory` to at least 512MB for 200-symbol workloads at 60-second TTL
- Use `maxmemory-policy allkeys-lru` so Redis evicts old cache entries under memory pressure rather than returning errors
- For production, enable Redis persistence (`appendonly yes`) so the signal cache survives Redis restarts

```bash
REDIS_URL=redis://username:password@redis-host:6379/0
```

### MLflow

MLflow is used for experiment tracking, model registry, and artifact storage.

```bash
MLFLOW_TRACKING_URI=http://mlflow:5000
```

One MLflow experiment per model family:

| Model | MLflow experiment name |
|---|---|
| Regime Classifier | `alphaforge-regime_classifier-hpo` |
| Stock Ranker | `alphaforge-stock_ranker-hpo` |
| Strategy Selector | `alphaforge-strategy_selector-hpo` |
| Risk Predictor | `alphaforge-risk_predictor-hpo` |
| RL Execution Agent | `alphaforge-rl_execution_agent-hpo` |

Each MLflow run logs: model name, version, date range, all hyperparameters, IC per fold, net Sharpe, max drawdown, PBO, and the SHA-256 hash of the training dataset.

**Artifact storage:** Model files are stored on the filesystem under `MODEL_ARTIFACTS_PATH` (not inside MLflow's artifact store), with SHA-256 checksum files alongside each artifact. MLflow stores metadata and hyperparameters; the filesystem holds the actual model files.

**Local SQLite MLflow (dev only):**

```bash
MLFLOW_TRACKING_URI=sqlite:///mlflow.db
mlflow server --backend-store-uri sqlite:///mlflow.db --host 0.0.0.0 --port 5000
```

---

## Health Checks and Readiness Probes

### Liveness probe

```
GET /health
```

Returns `200 OK` immediately if the service process is alive. Does not check external dependencies. Use for container liveness probes.

```yaml
# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8100
  initialDelaySeconds: 10
  periodSeconds: 15
```

### Readiness probe

Use `/v2/models/status` as a readiness probe. The service is ready when at least the regime classifier and risk predictor are loaded (either `trained_model` or `heuristic`).

```yaml
# Kubernetes readiness probe
readinessProbe:
  httpGet:
    path: /health
    port: 8100
  initialDelaySeconds: 5
  periodSeconds: 10
```

For a deeper readiness check (verifies Redis connectivity), you can use:

```bash
curl http://localhost:8100/v2/models/status \
  -H "X-API-KEY: your-key"
```

If all models show `"loaded": false` and `"provenance": "unavailable"`, the service is running but not ready to serve predictions.

### Docker HEALTHCHECK

```dockerfile
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -f http://localhost:8100/health || exit 1
```

---

## Observability

### Structured logging

All logs are emitted as JSON via `structlog`. Every log entry includes:

| Field | Description |
|---|---|
| `timestamp` | UTC ISO-8601 |
| `service` | `"ml-service2.0"` |
| `version` | Service version string |
| `request_id` | `X-Request-ID` value |
| `model_name` | Relevant model (when applicable) |
| `latency_ms` | Request latency |
| `provenance` | Prediction provenance |
| `deployment_mode` | Current deployment mode |

Set `LOG_LEVEL=DEBUG` to see feature vector assembly, cache hits/misses, and SHAP computation details.

### Audit log

The append-only audit log at `AUDIT_LOG_PATH` records every training run, promotion decision, online learning update, and approvalToken issuance in JSON Lines format. Each entry is written synchronously with `fsync`. This file must never be deleted or modified — the service raises `AuditLogViolation` if an attempt is detected.

### Metrics

The service does not ship a Prometheus exporter by default. Latency, cache hit rate, and model provenance distributions can be computed from the structured logs. An optional `prometheus-fastapi-instrumentator` integration can be enabled by adding it to `pyproject.toml` and calling `Instrumentator().instrument(app).expose(app)` in `src/main.py`.

---

## Known Limitations

1. **90%+ test coverage requires live mocks.** The property-based and integration test files (`test_api_contracts.py`, `test_finrl_execution_agent.py`) start WireMock-backed mock servers for data-service2.0 and SentinelPulse. Running the test suite with `pytest tests/` without those mocks will fail ~15% of tests. Use `docker compose -f docker-compose.test.yml up` to get the full coverage pass.

2. **mypy strict requires optional library stubs.** Libraries like `qlib`, `finrl`, `nannyml`, and `riskfolio-lib` do not ship `py.typed` markers or bundled stubs. A `py.typed`-compatible stub package or `# type: ignore` comments are required for those import sites. The CI workflow installs stubs via `mypy` `extra_requirements` where available and uses `ignore_missing_imports = true` in `pyproject.toml` as a fallback for the remaining libraries.

3. **gRPC stubs must be regenerated after proto changes.** The `protos/generate_stubs.sh` script is not run automatically. If `market_data.proto` is updated in data-service2.0, regenerate stubs and commit the diff before deploying.

4. **LLM model weights download on first run.** FinBERT (`ProsusAI/finbert`, ~450 MB) is downloaded from HuggingFace Hub on first startup. In air-gapped environments, pre-download the weights and set `TRANSFORMERS_OFFLINE=1` with the model path mapped into the container.

5. **Online learning is disabled by default.** Set `ONLINE_LEARNING_ENABLED=true` to activate incremental model updates. In `DEPLOYMENT_MODE=validated_ml_only`, online learning updates still require IC validation before activation.

6. **Artifact storage is not replicated.** `MODEL_ARTIFACTS_PATH` is a local filesystem path. In a multi-worker or multi-node deployment, this path must be on a shared persistent volume (NFS, EFS, or equivalent) visible to all workers. MLflow does not replicate the binary artifacts.

7. **The 3m timeframe is permanently forbidden for Indian market data.** Any request to `/v2/predict/*` that includes 3-minute bars will return a `422` error with the field-level message `"3m timeframe is not valid for Indian market data"`.

8. **Alpha360 computation is slow.** With `ALPHA360_ENABLED=true`, processing 500 symbols may exceed the 60-second SLA on hardware without AVX2/AVX-512 SIMD support. Keep Alpha360 disabled unless running on modern server hardware.
