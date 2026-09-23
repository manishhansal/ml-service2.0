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
        protos serve help

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
