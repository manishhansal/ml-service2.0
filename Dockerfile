# ────────────────────────────────────────────────────────────────────────────
# ml-service2.0 — Multi-stage Dockerfile
#
# Stages:
#   base        — Python 3.11-slim with system-level build dependencies
#   dev         — Adds all dev + prod dependencies; used for local development
#                 and CI test runs
#   builder     — Builds a wheel from the project for the production stage
#   production  — Minimal runtime image; no build tools, runs as nobody
# ────────────────────────────────────────────────────────────────────────────

# ── Stage 1: base ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base
WORKDIR /app

# Install only packages needed to compile C extensions (numpy, LightGBM, XGBoost…)
# libgomp1  — OpenMP runtime required by LightGBM / XGBoost
# curl      — used by the HEALTHCHECK in the production stage
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# ── Stage 2: dev ──────────────────────────────────────────────────────────────
# Full development image: production + dev extras (pytest, hypothesis, ruff …)
# Used by docker-compose.yml and docker-compose.test.yml
FROM base AS dev
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[dev]"
COPY . .

# ── Stage 3: builder ──────────────────────────────────────────────────────────
# Builds a relocatable wheel that the production stage installs without
# needing build-essential or the full source tree.
FROM base AS builder
COPY pyproject.toml ./
# Install hatchling (build backend declared in pyproject.toml)
RUN pip install --no-cache-dir hatchling
COPY src/ ./src/
# Build wheel only (no sdist needed); --no-isolation re-uses already-installed
# hatchling rather than pulling it again inside a venv.
RUN pip install --no-cache-dir build && python -m build --wheel --no-isolation

# ── Stage 4: production ───────────────────────────────────────────────────────
# Minimal runtime image — only libgomp1, the wheel, and nobody user.
FROM python:3.11-slim AS production
WORKDIR /app

# libgomp1 is required at runtime by LightGBM and XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Copy wheel from builder and install; then remove it to keep the layer lean
COPY --from=builder /app/dist/*.whl .
RUN pip install --no-cache-dir *.whl && rm *.whl

EXPOSE 8100

# Liveness check — data-service must be reachable before marking healthy
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8100/health || exit 1

# Drop root; nobody (uid 65534) has no write access to /app by default
USER nobody

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8100", "--workers", "4"]
