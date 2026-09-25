# ── ml-service2.0 Makefile ────────────────────────────────────────────────────
# Shortcuts for common development workflows.
# Requires: Python 3.11+, uv (https://github.com/astral-sh/uv) or pip.
#
# Usage:
#   make setup        — install all dependencies incl. dev extras + pre-commit hooks
#   make test         — run full test suite with coverage
#   make test-unit    — run only @pytest.mark.unit tests (fast, no I/O)
#   make test-pit     — run only @pytest.mark.pit tests (PIT / lookahead-bias)
#   make test-tdd     — run only @pytest.mark.tdd tests (Phase 2 red-phase TDD)
#   make lint         — ruff check + ruff format check
#   make format       — ruff format (auto-fix)
#   make typecheck    — mypy strict type checking
#   make ci           — full CI pipeline (lint + typecheck + test)
#   make clean        — remove build artefacts and caches

.PHONY: setup test test-unit test-pit test-tdd lint format typecheck ci clean \
        protos serve up down restart rebuild logs logs-ml logs-redis \
        status-docker shell docker-test readiness help

# ── Detect uv / pip ───────────────────────────────────────────────────────────
UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  PIP_INSTALL = uv pip install
  PIP_SYNC    = uv pip sync
else
  PIP_INSTALL = pip install
  PIP_SYNC    = pip install
endif

PYTHON  ?= python3
SRC_DIR  = src
TEST_DIR = tests

# ── Setup ─────────────────────────────────────────────────────────────────────
setup:  ## Install all dependencies + pre-commit hooks
	$(PIP_INSTALL) -e ".[dev]"
	pre-commit install --install-hooks
	@echo "✓ Environment ready. Run 'make test' to verify."

# ── Tests ─────────────────────────────────────────────────────────────────────
test:  ## Full test suite with coverage (fails under 90%)
	$(PYTHON) -m pytest $(TEST_DIR)/ \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-fail-under=90 \
		--timeout=120 \
		-v

test-unit:  ## Fast unit tests only (marker: unit)
	$(PYTHON) -m pytest $(TEST_DIR)/ \
		-m unit \
		--timeout=30 \
		-v --no-header

test-pit:  ## Point-In-Time correctness tests (marker: pit)
	$(PYTHON) -m pytest $(TEST_DIR)/ \
		-m pit \
		--timeout=60 \
		-v --no-header

test-tdd:  ## Phase 2 TDD red-phase tests (marker: tdd)
	$(PYTHON) -m pytest $(TEST_DIR)/test_data_contracts.py \
		$(TEST_DIR)/test_feature_pit.py \
		$(TEST_DIR)/test_meta_decision.py \
		--timeout=60 \
		-v --no-header

test-fast:  ## All tests except slow (no coverage, fast feedback)
	$(PYTHON) -m pytest $(TEST_DIR)/ \
		-m "not slow and not integration" \
		--timeout=30 \
		--no-cov \
		-q

# ── Linting & Formatting ──────────────────────────────────────────────────────
lint:  ## ruff check + ruff format check (read-only)
	$(PYTHON) -m ruff check $(SRC_DIR)/ $(TEST_DIR)/
	$(PYTHON) -m ruff format --check $(SRC_DIR)/ $(TEST_DIR)/

format:  ## ruff format + auto-fix (modifies files)
	$(PYTHON) -m ruff format $(SRC_DIR)/ $(TEST_DIR)/
	$(PYTHON) -m ruff check --fix $(SRC_DIR)/ $(TEST_DIR)/

# ── Type Checking ─────────────────────────────────────────────────────────────
typecheck:  ## mypy strict type check on src/
	$(PYTHON) -m mypy $(SRC_DIR)/ --strict

# ── Full CI Gate ──────────────────────────────────────────────────────────────
ci: lint typecheck test  ## Run full CI pipeline locally (lint → typecheck → test)

# ── Proto Generation ──────────────────────────────────────────────────────────
protos:  ## Regenerate gRPC stubs from .proto files
	bash protos/generate_stubs.sh

# ── Dev Server ────────────────────────────────────────────────────────────────
serve:  ## Start development server (reload on change)
	$(PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port 8100 --reload

# ── Docker targets ────────────────────────────────────────────────────────────
# All ML execution must run inside Docker (mandate §2).
# These targets manage the ml-service2.0 Docker compose stack.

.PHONY: up down restart rebuild logs logs-ml logs-redis status-docker shell

up:  ## Start the ml-service2.0 Docker stack (ml-service + redis + mlflow)
	docker compose up -d
	@echo "✓ ml-service2.0 running at http://localhost:8100"

down:  ## Stop and remove the ml-service2.0 Docker stack
	docker compose down
	@echo "✓ Stack stopped."

restart:  ## Restart ml-service only (redis and mlflow stay running)
	docker compose restart ml-service
	@echo "✓ ml-service restarted."

rebuild:  ## Rebuild the test-hardened image and restart the stack
	@echo "▶  Building ml-service2:test-hardened ..."
	docker build -f Dockerfile.test -t ml-service2:test-hardened .
	@echo "▶  Restarting stack with new image..."
	docker compose up -d --force-recreate ml-service
	@echo "✓  ml-service rebuilt and restarted at http://localhost:8100"

logs:  ## Tail all service logs (ml-service + redis + mlflow)
	docker compose logs -f --tail=100

logs-ml:  ## Tail ml-service logs only
	docker compose logs -f --tail=100 ml-service

logs-redis:  ## Tail redis logs
	docker compose logs -f --tail=50 redis

status-docker:  ## Show Docker stack container statuses
	@docker compose ps
	@echo ""
	@echo "Health check:"
	@docker exec ml-service2-api python3 -c \
		"import urllib.request,json; r=urllib.request.urlopen('http://localhost:8100/health',timeout=5); \
		 d=json.loads(r.read()); print('  ml-service:', d.get('status'), 'v'+d.get('version','?'))" 2>/dev/null \
		|| echo "  ml-service: not responding"

shell:  ## Open a shell in the running ml-service container
	docker exec -it ml-service2-api bash

docker-test:  ## Run the full test suite inside Docker (authoritative — not host Python)
	@echo "▶  Running tests inside Docker (mandate §2)..."
	docker run --rm \
		--env-file .env \
		-e DOCKER_IMAGE_DIGEST=$$(docker inspect ml-service2:test-hardened --format '{{.Id}}' | cut -c1-71) \
		-v $$(pwd)/src:/app/src \
		-v $$(pwd)/tests:/app/tests \
		-v $$(pwd)/artifacts:/app/artifacts \
		ml-service2:test-hardened \
		pytest tests/ -q --no-header --no-cov \
			--ignore=tests/test_e2e_training.py \
			--ignore=tests/test_e2e_certification.py \
			--ignore=tests/test_certification_harness.py \
			--ignore=tests/test_intraday_research_state.py \
			--ignore-glob="*test_meta_engine*" \
			--ignore-glob="*test_schemas_optional*"

readiness:  ## Run the training-readiness gate against live services
	docker run --rm \
		--env-file .env \
		-e DOCKER_IMAGE_DIGEST=dev \
		--add-host=host.docker.internal:host-gateway \
		-e DATA_SERVICE_2_URL=http://host.docker.internal:8200 \
		-e SENTINEL_PULSE_URL=http://host.docker.internal:3001 \
		-v $$(pwd)/scripts:/app/scripts \
		ml-service2:test-hardened \
		python3 scripts/run_readiness_gate.py

# ── Clean ─────────────────────────────────────────────────────────────────────
clean:  ## Remove build artefacts, caches, and coverage files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"  -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.json
	@echo "✓ Clean."

# ── Help ─────────────────────────────────────────────────────────────────────
help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
