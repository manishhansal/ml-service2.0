# ml-service2.0

Institutional-grade standalone ML microservice for the AlphaForge trading platform.

Runs on **port 8100**. Consumes market data exclusively from `data-service2.0` (port 8200)
and news intelligence from `SentinelPulse` (port 3001). Serves predictions to `alpha-forge`
(port 3000) over REST and WebSocket.

---

## Architecture

```
alpha-forge (port 3000)  ←── REST + WebSocket ──┐
                                                  │
                               ml-service2.0      │ (port 8100)
                                    │
            ┌───────────────────────┼───────────────────────┐
            ▼                       ▼                       ▼
  data-service2.0          SentinelPulse             Redis / MLflow
  (port 8200)              (port 3001)
  Market data sole         NLP/news sole
  authority                authority
```

---

## Requirements

- Python 3.11+
- Redis 7+
- MLflow 2.x (experiment tracking)
- Docker (optional, for full stack)

---

## Setup

### 1. Clone and create virtual environment

```bash
git clone <repo-url> ml-service2.0
cd ml-service2.0
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
# Production
pip install -e .

# Development (includes pytest, ruff, mypy, hypothesis, etc.)
pip install -e ".[dev]"
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual values — see comments in the file
```

### 4. Generate gRPC stubs

```bash
bash protos/generate_stubs.sh
```

### 5. Run the service

```bash
# Development (auto-reload)
uvicorn src.main:app --host 0.0.0.0 --port 8100 --reload

# Production (4 workers)
uvicorn src.main:app --host 0.0.0.0 --port 8100 --workers 4
```

---

## Docker

ml-service2.0 runs as its own Docker Compose stack (redis + mlflow + ml-service). All ML execution happens inside Docker — do not run training directly on the host.

### Preferred workflow — Makefile targets

```bash
# Start the full stack (ml-service + redis + mlflow)
make up

# Stop the stack
make down

# Rebuild the image and restart (after dependency or source changes)
make rebuild

# Run the full test suite inside Docker (authoritative)
make docker-test

# Run the training-readiness gate against live services
make readiness

# Tail ml-service logs
make logs-ml

# Show container status + health
make status-docker

# Open a shell in the running ml-service container
make shell
```

### Direct Docker Compose (advanced)

```bash
# Development stack (ml-service + Redis + MLflow)
docker compose up -d

# Run with hot-reload source bind-mount (default in docker-compose.yml)
docker compose up -d ml-service

# Integration test stack (WireMock stubs for data-service and SentinelPulse)
docker compose -f docker-compose.test.yml up --abort-on-container-exit
```

### Ports

| Service | Host port | Notes |
|---|---|---|
| ml-service | `8100` | HTTP + WebSocket |
| redis | internal only | No host binding — avoids conflict with alpha-forge-redis |
| mlflow | internal only | Access via `docker compose port mlflow 5000` |

---

## Running Tests

> **All ML execution must run inside Docker** (mandate §2). The commands below are for quick host-side feedback during development. The authoritative test run is `make docker-test`.

```bash
# Run tests inside Docker (authoritative)
make docker-test

# Host-side (fast feedback, no Docker overhead)
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

---

## API Overview

All prediction endpoints are under `/v2/` and require `X-API-KEY` header authentication.

| Method | Path | Description | p95 SLA |
|--------|------|-------------|---------|
| POST | `/v2/predict/regime` | Market regime classification | 50ms |
| POST | `/v2/predict/rankings` | Stock ranking (≤200 symbols) | 200ms |
| POST | `/v2/predict/strategy` | Trading strategy selection | 50ms |
| POST | `/v2/predict/risk` | Per-trade risk estimation | 50ms |
| POST | `/v2/predict/portfolio` | Portfolio optimisation (legacy) | 500ms |
| POST | `/v2/predict/portfolio-v2` | Portfolio optimisation (Riskfolio-Lib) | 500ms |
| POST | `/v2/predict/execution` | RL execution agent action | 50ms |
| POST | `/v2/predict/price-regime` | Price regime forecast | 50ms |
| POST | `/v2/predict/iv-regime` | IV regime classification | 50ms |
| POST | `/v2/meta/decide` | LLM meta-decision engine | 150ms |
| WS | `/v2/stream/signals` | Real-time signal streaming | — |
| GET | `/health` | Health check (no auth) | — |

See `/docs` (FastAPI Swagger UI) for full schema documentation when the service is running.

---

## Directory Structure

```
ml-service2.0/
├── src/
│   ├── api/            FastAPI routers (predictions, meta, analytics, training, monitoring)
│   ├── features/       FeaturePipeline, QlibFeatureEngine, LeakageValidator
│   ├── models/         RegimeClassifier, StockRanker, StrategySelector, RiskPredictor, ...
│   ├── training/       TrainingPipeline, PurgedKFoldSplitter, CPCV, OptunaHPO, OnlineLearner
│   ├── meta/           MetaDecisionEngine, CalibrationLayer, EnsembleWeighter, LLMNewsReasoner
│   ├── monitoring/     DriftMonitor (Evidently AI + NannyML), AlertSystem
│   ├── registry/       ModelRegistry, ModelPromotion (six-gate pipeline)
│   ├── explainability/ ModelExplainer (SHAP TreeExplainer / KernelExplainer)
│   ├── streaming/      SignalStreamer (WebSocket)
│   ├── audit/          AuditLogger (append-only JSON Lines)
│   ├── clients/        DataServiceClient (REST + gRPC), SentinelPulseClient
│   ├── cache/          RedisCache (async)
│   ├── schemas/        Pydantic V2 schemas (base enums, features, predictions, meta, ...)
│   ├── analytics/      Greeks, GEX, VPIN, Vol Surface
│   ├── config.py       pydantic-settings Settings
│   └── main.py         FastAPI app + lifespan
├── tests/              Test suite (property-based + integration + latency)
├── protos/             market_data.proto + stub generation script
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── docker-compose.test.yml
├── .env.example
└── README.md
```

---

## Key Design Principles

- **PIT Correctness** — No feature in training or inference contains future information.
- **Evidence-Gated Execution** — Every prediction carries `PredictionProvenance`; only `TRAINED_MODEL` is live-eligible.
- **Self-Learning** — Models adapt via online learning (max 5 consecutive updates before full retrain).
- **Six-Gate Promotion** — Challengers pass DATA → PREDICTIVE → CALIBRATION → EXECUTION → RISK → STABILITY gates before becoming champion.
- **Full Observability** — Every prediction, training run, and promotion is in the append-only audit log.
- **Explainability by Default** — SHAP attributions on every model response; LLM natural-language rationale in MetaOutput.
