# ── ml-service2.0 Makefile ────────────────────────────────────────────────────
# AlphaForge NSE F&O Cross-Sectional Alpha Engine
# Model: LightGBM fs-3.0.0 | Stage: SHADOW | Port: 8100
#
# Usage: make <target>
#   make help          — show all available targets
#   make setup         — install Python deps + pre-commit hooks
#   make test          — run full host test suite (1,867 tests)
#   make docker-test   — run tests inside Docker (authoritative)
#   make build         — build the test-hardened Docker image
#   make up            — start the full stack
#   make health        — health check across all 4 services
#   make logs          — tail all service logs
# ─────────────────────────────────────────────────────────────────────────────

# ── Auto-detect package manager ───────────────────────────────────────────────
UV := $(shell command -v uv 2>/dev/null)
ifdef UV
  PIP_INSTALL = uv pip install
else
  PIP_INSTALL = pip install
endif

PYTHON        ?= python3
SRC_DIR        = src
TEST_DIR       = tests
IMAGE          = ml-service2:test-hardened
CONTAINER      = ml-service2-api
COMPOSE_FILE   = docker-compose.yml

# Test ignore flags shared between host and Docker runs
TEST_IGNORES = \
	--ignore=$(TEST_DIR)/test_coverage_boost.py \
	--ignore=$(TEST_DIR)/test_coverage_boost2.py \
	--ignore=$(TEST_DIR)/test_coverage_boost3.py \
	--ignore=$(TEST_DIR)/test_coverage_boost_mocked.py \
	--ignore=$(TEST_DIR)/test_coverage_boost_mocked2.py \
	--ignore=$(TEST_DIR)/test_coverage_boost_mocked3.py \
	--ignore=$(TEST_DIR)/test_coverage_boost_mocked4.py \
	--ignore=$(TEST_DIR)/test_coverage_boost_mocked5.py \
	--ignore=$(TEST_DIR)/test_coverage_boost_mocked6.py \
	--ignore=$(TEST_DIR)/test_coverage_boost_mocked7.py \
	--ignore=$(TEST_DIR)/test_schemas_optional.py \
	--ignore=$(TEST_DIR)/test_meta_engine.py

.PHONY: \
	setup test test-unit test-pit test-fast lint format typecheck ci \
	build rebuild up down restart logs logs-ml logs-redis logs-mlflow logs-all \
	health shell docker-test docker-shell prune reset \
	ingest ingest-resume ingest-universe universe-coverage \
	forward-paper forward-paper-resolve signal-promote \
	promote-shadow g6-test watcher threshold-sweep session \
	clean clean-docker help

# ─────────────────────────────────────────────────────────────────────────────
# HOST DEVELOPMENT
# ─────────────────────────────────────────────────────────────────────────────

setup:  ## Install all Python dependencies + pre-commit hooks
	$(PIP_INSTALL) -e ".[dev]"
	pre-commit install --install-hooks
	@echo "✓ Environment ready. Run 'make test' to verify."

test:  ## Run full host test suite (fast, no coverage; ~110s)
	$(PYTHON) -m pytest $(TEST_DIR)/ \
		$(TEST_IGNORES) \
		--no-cov -q --tb=short -p no:warnings
	@echo ""
	@echo "✓ Host tests complete."

test-cov:  ## Run full test suite WITH coverage (slow; fails under 90%)
	$(PYTHON) -m pytest $(TEST_DIR)/ \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-fail-under=90 \
		--timeout=120 -q

test-unit:  ## Fast unit tests only (marker: unit)
	$(PYTHON) -m pytest $(TEST_DIR)/ -m unit --no-cov -q --tb=short --timeout=30

test-pit:  ## Point-In-Time correctness tests (marker: pit)
	$(PYTHON) -m pytest $(TEST_DIR)/ -m pit --no-cov -q --tb=short --timeout=60

test-fast:  ## All tests except slow/integration
	$(PYTHON) -m pytest $(TEST_DIR)/ \
		-m "not slow and not integration" \
		$(TEST_IGNORES) --no-cov -q --timeout=30

lint:  ## ruff check + format check (read-only)
	$(PYTHON) -m ruff check $(SRC_DIR)/ $(TEST_DIR)/
	$(PYTHON) -m ruff format --check $(SRC_DIR)/ $(TEST_DIR)/

format:  ## ruff format + auto-fix (modifies files)
	$(PYTHON) -m ruff format $(SRC_DIR)/ $(TEST_DIR)/
	$(PYTHON) -m ruff check --fix $(SRC_DIR)/ $(TEST_DIR)/

typecheck:  ## mypy strict type check on src/
	$(PYTHON) -m mypy $(SRC_DIR)/ --strict

ci: lint typecheck test  ## Full local CI pipeline (lint → typecheck → test)

serve:  ## Start development server with hot-reload (host Python)
	$(PYTHON) -m uvicorn src.main:app --host 0.0.0.0 --port 8100 --reload

# ─────────────────────────────────────────────────────────────────────────────
# DOCKER IMAGE
# ─────────────────────────────────────────────────────────────────────────────

build:  ## Build the test-hardened Docker image (uses layer cache)
	@echo "▶  Building $(IMAGE) ..."
	docker build -f Dockerfile.test -t $(IMAGE) .
	@echo "✓  Image built: $(IMAGE)"
	@docker image ls $(IMAGE) --format "   Size: {{.Size}}  Created: {{.CreatedAt}}"

rebuild:  ## Force-rebuild image (no cache) then restart the stack
	@echo "▶  Force-rebuilding $(IMAGE) (no cache — installs all packages fresh) ..."
	@echo "   ⚠  This takes ~12 min due to pip installs. Use 'make build' if you only changed code."
	docker build --no-cache -f Dockerfile.test -t $(IMAGE) .
	@echo "▶  Restarting stack with new image ..."
	docker compose -f $(COMPOSE_FILE) up -d --force-recreate ml-service
	@$(MAKE) health

deploy:  ## Build image + restart container (fastest code-change deploy)
	@echo "▶  Building and deploying $(IMAGE) ..."
	docker build -f Dockerfile.test -t $(IMAGE) .
	docker compose -f $(COMPOSE_FILE) up -d --force-recreate ml-service
	@sleep 20
	@$(MAKE) health

# ─────────────────────────────────────────────────────────────────────────────
# DOCKER STACK MANAGEMENT
# ─────────────────────────────────────────────────────────────────────────────

up:  ## Start full ml-service2.0 stack (ml-service + redis + mlflow)
	docker compose -f $(COMPOSE_FILE) up -d
	@sleep 20
	@echo "✓  Stack started"
	@$(MAKE) health

down:  ## Stop and remove the ml-service2.0 stack
	docker compose -f $(COMPOSE_FILE) down
	@echo "✓  Stack stopped."

restart:  ## Restart ml-service only (redis + mlflow stay running)
	docker compose -f $(COMPOSE_FILE) restart ml-service
	@sleep 15
	@$(MAKE) health

reset:  ## Tear down, rebuild image, and restart everything from scratch
	@echo "▶  Full reset: down → build → up ..."
	docker compose -f $(COMPOSE_FILE) down
	$(MAKE) build
	docker compose -f $(COMPOSE_FILE) up -d
	@sleep 25
	@$(MAKE) health

prune:  ## Remove stopped containers and dangling images (frees disk space)
	docker container prune -f
	docker image prune -f
	@echo "✓  Pruned stopped containers and dangling images."

# ─────────────────────────────────────────────────────────────────────────────
# DOCKER LOGS
# ─────────────────────────────────────────────────────────────────────────────

logs:  ## Tail ml-service logs (last 100 lines, then follow)
	docker compose -f $(COMPOSE_FILE) logs -f --tail=100 ml-service

logs-ml:  ## Tail ml-service logs only (alias for logs)
	docker logs -f --tail=100 $(CONTAINER)

logs-redis:  ## Tail Redis logs
	docker compose -f $(COMPOSE_FILE) logs -f --tail=50 redis

logs-mlflow:  ## Tail MLflow logs
	docker compose -f $(COMPOSE_FILE) logs -f --tail=50 mlflow

logs-all:  ## Tail ml-service logs + show data-service and sentinelpulse status
	@echo "▶  ml-service2.0 logs (Ctrl+C to stop) ..."
	@echo "   For data-service: cd ../data-service2.0 && docker compose logs -f"
	@echo "   For sentinelpulse: cd ../sentinelpulse && docker compose logs -f"
	docker compose -f $(COMPOSE_FILE) logs -f --tail=100

logs-errors:  ## Show only ERROR-level log lines from ml-service (last 200 lines)
	docker logs --tail=200 $(CONTAINER) 2>&1 | grep -i "error\|exception\|traceback\|FAIL" || echo "(no errors in last 200 lines)"

# ─────────────────────────────────────────────────────────────────────────────
# DOCKER HEALTH & STATUS
# ─────────────────────────────────────────────────────────────────────────────

health:  ## Health check across all 4 services
	@echo "─────────────────────────────────────"
	@echo " Service Health Check"
	@echo "─────────────────────────────────────"
	@printf "  ml-service2.0  (8100):  "
	@curl -sf http://localhost:8100/health | $(PYTHON) -c \
		"import sys,json; d=json.load(sys.stdin); \
		 print('✓ ' + d.get('status','?') + ' v' + str(d.get('version','?')))" \
		2>/dev/null || echo "✗  not responding"
	@printf "  data-service2.0 (8200):  "
	@curl -sf http://localhost:8200/health | $(PYTHON) -c \
		"import sys,json; d=json.load(sys.stdin); print('✓ ' + d.get('status','?'))" \
		2>/dev/null || echo "✗  not responding"
	@printf "  SentinelPulse   (3001):  "
	@curl -sf http://localhost:3001/health | $(PYTHON) -c \
		"import sys,json; d=json.load(sys.stdin); print('✓ ' + str(d.get('status',d))[:30])" \
		2>/dev/null || echo "✗  not responding"
	@printf "  alpha-forge     (3000):  "
	@curl -sf http://localhost:3000/health >/dev/null 2>&1 && echo "✓  responding" || echo "✗  not responding"
	@echo "─────────────────────────────────────"
	@printf "  DEPLOYMENT_MODE:         "
	@docker exec $(CONTAINER) printenv DEPLOYMENT_MODE 2>/dev/null || echo "unknown"

status:  ## Show running Docker containers (all services)
	@echo "Docker containers:"
	@docker ps --format "  {{.Names}}\t{{.Status}}\t{{.Ports}}" \
		| column -t
	@echo ""
	@echo "ml-service2 stack:"
	@docker compose -f $(COMPOSE_FILE) ps

shell:  ## Open an interactive bash shell inside the ml-service container
	docker exec -it $(CONTAINER) bash

docker-shell:  ## Run a one-off container with a shell (for debugging without running stack)
	docker run --rm -it \
		--env-file .env \
		--add-host=host.docker.internal:host-gateway \
		-v $$(pwd)/src:/app/src \
		-v $$(pwd)/tests:/app/tests \
		-v $$(pwd)/artifacts:/app/artifacts \
		-v $$(pwd)/strategy:/app/strategy \
		-v $$(pwd)/data:/app/data \
		$(IMAGE) bash

# ─────────────────────────────────────────────────────────────────────────────
# DOCKER TESTS
# ─────────────────────────────────────────────────────────────────────────────

docker-test:  ## Run full test suite inside Docker (authoritative — matches CI)
	@echo "▶  Running tests inside Docker (Python 3.11, $(IMAGE)) ..."
	docker run --rm \
		--env-file .env \
		-e DOCKER_IMAGE_DIGEST=$$(docker inspect $(IMAGE) --format '{{.Id}}' | cut -c1-71) \
		-v $$(pwd)/src:/app/src \
		-v $$(pwd)/tests:/app/tests \
		-v $$(pwd)/artifacts:/app/artifacts \
		-v $$(pwd)/strategy:/app/strategy \
		$(IMAGE) \
		pytest tests/ -q --no-header --no-cov -p no:warnings \
			$(TEST_IGNORES) \
			--ignore=tests/test_phase3c.py \
			--ignore=tests/test_phase3d.py \
			--ignore=tests/test_phase3k.py
	@echo "✓  Docker tests complete."

docker-test-cov:  ## Run Docker tests WITH coverage report
	docker run --rm \
		--env-file .env \
		-v $$(pwd)/src:/app/src \
		-v $$(pwd)/tests:/app/tests \
		$(IMAGE) \
		pytest tests/ --cov=src --cov-report=term-missing -q $(TEST_IGNORES)

docker-test-fast:  ## Run fast subset of Docker tests (unit + pit markers only)
	docker exec $(CONTAINER) python3 -m pytest tests/ \
		-m "unit or pit" --no-cov -q --tb=short -p no:warnings

# ─────────────────────────────────────────────────────────────────────────────
# DATA INGESTION
# ─────────────────────────────────────────────────────────────────────────────

check-token:  ## Check Upstox token status + data freshness across all 218 symbols
	@python3 scripts/check_upstox_token.py

refresh-data:  ## Full universe data refresh: backfill via data-service + ingest into parquets
	@echo "▶  Refreshing data for all 218 F&O symbols ..."
	@echo "   Step 1: Backfilling via data-service (Angel One + Upstox)"
	@echo "   Step 2: Ingesting fresh bars into parquets"
	@echo "   Step 3: Phantom bar cleanup"
	PYTHONPATH=. $(PYTHON) scripts/refresh_all_universe_data.py
	@echo "▶  Ingesting latest bars ..."
	PYTHONPATH=. $(PYTHON) scripts/fast_ingest.py
	@echo "✓  Ingestion complete."

ingest-universe:  ## Full 218-symbol F&O universe ingestion (resumable, ~10 min)
	@echo "▶  Running broad-universe ingestion (rate: 2 req/s, 218 symbols) ..."
	@echo "   Checkpoint: data/1d/_checkpoint.json (already-done symbols skipped)"
	docker run --rm \
		--env-file .env \
		--add-host=host.docker.internal:host-gateway \
		-e DATA_SERVICE_2_URL=http://host.docker.internal:8200 \
		-v $$(pwd)/scripts:/app/scripts \
		-v $$(pwd)/src:/app/src \
		-v $$(pwd)/data:/app/data \
		$(IMAGE) \
		python3 scripts/ingest_broad_universe.py --interval 1d --max-symbols 220

ingest-resume:  ## Resume an interrupted ingestion from checkpoint
	@echo "▶  Resuming ingestion from checkpoint ..."
	@if [ -f data/1d/_checkpoint.json ]; then \
		echo "   Checkpoint found."; \
		$(PYTHON) -c "import json; cp=json.load(open('data/1d/_checkpoint.json')); \
		  done=len([k for k in cp.get('completed',{}) if k.startswith('1d:')]); \
		  print('   Already ingested:', done, '/ 220 symbols')"; \
	fi
	$(MAKE) ingest-universe

universe-coverage:  ## Report how many of 220 F&O symbols have up-to-date parquets
	@echo "▶  Checking universe parquet coverage ..."
	PYTHONPATH=. $(PYTHON) -c "\
from pathlib import Path; import pandas as pd; \
pqs = sorted(Path('data/1d/1d').glob('*.parquet')); \
dates = {p.stem: str(pd.read_parquet(p).index[-1])[:10] for p in pqs}; \
fresh = [s for s,d in dates.items() if d >= '2026-09-23']; \
stale = [s for s,d in dates.items() if d < '2026-09-23']; \
print(f'Total: {len(pqs)}/220'); \
print(f'Fresh (>=Sep 23): {len(fresh)}'); \
print(f'Stale (<Sep 23): {len(stale)}'); \
[print(f'  STALE: {s} ({d})') for s,d in dates.items() if d < '2026-09-23']"

# ─────────────────────────────────────────────────────────────────────────────
# LIVE SESSION & FORWARD PAPER
# ─────────────────────────────────────────────────────────────────────────────

session:  ## Start live scoring session until 15:30 IST (autorun_till_close.py)
	@echo "▶  Starting live session — scores 218 symbols every 5 min until 15:30 IST ..."
	@echo "   Press Ctrl+C to stop early. Logs: artifacts/live_session/autorun_log.jsonl"
	PYTHONPATH=. $(PYTHON) -W ignore scripts/autorun_till_close.py

fix-phantoms:  ## Remove phantom 03:45 UTC duplicate bars from all 218 parquets (run once)
	@echo "▶  Fixing phantom duplicate bars (201,774 fake rows in 214/218 parquets) ..."
	PYTHONPATH=. $(PYTHON) scripts/fix_phantom_bars.py
	@echo "✓  Parquets cleaned. Run 'make ingest' to refresh with latest bars."

fix-phantoms-dry:  ## Dry run: show how many phantom bars would be removed
	PYTHONPATH=. $(PYTHON) scripts/fix_phantom_bars.py --dry-run
	@echo "▶  Running forward paper session (218 symbols) ..."
	PYTHONPATH=. $(PYTHON) scripts/run_forward_paper_session_v2.py
	@echo "✓  Signals written to artifacts/forward_paper/signals_v2.jsonl"

forward-paper-resolve:  ## Resolve forward paper signals (requires Sep 30 open prices)
	@echo "▶  Resolving forward paper signals ..."
	PYTHONPATH=. $(PYTHON) scripts/resolve_forward_paper.py
	@echo "✓  Results: artifacts/forward_paper/outcomes.jsonl"

forward-paper-audit:  ## Show forward paper status (n open, n resolved, win rate)
	PYTHONPATH=. $(PYTHON) scripts/run_forward_paper_audit.py

signal-promote:  ## Run SignalPromotionEngine on resolved outcomes (G11 gate)
	@echo "▶  Running SignalPromotionEngine ..."
	PYTHONPATH=. $(PYTHON) scripts/run_signal_promotion.py
	@echo "✓  Report: reports/signal_promotion_preliminary.json"

# ─────────────────────────────────────────────────────────────────────────────
# SHADOW / PRODUCTION PROMOTION
# ─────────────────────────────────────────────────────────────────────────────

promote-shadow:  ## Promote CHALLENGER → SHADOW (requires --approver and --note args)
	@echo "▶  Promoting to SHADOW stage ..."
	@echo "   Usage: make promote-shadow APPROVER='name' NOTE='reason'"
	@echo "   Example: make promote-shadow APPROVER='head_of_quant' NOTE='G10+G11 pass'"
	PYTHONPATH=. $(PYTHON) scripts/promote_to_shadow.py \
		--approver "$(APPROVER)" \
		--note "$(NOTE)"

# ─────────────────────────────────────────────────────────────────────────────
# RESEARCH & ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

g6-test:  ## Run G6 cost robustness test (confirms strategy viable at 1.5× primary cost)
	@echo "▶  Running G6 cost robustness test (LightGBM, 218 symbols) ..."
	PYTHONPATH=. $(PYTHON) scripts/run_g6_robustness_test.py
	@echo "✓  Report: reports/g6_cost_robustness_test.json"

reversal-scan:  ## Run real-time reversal scan — detect oversold bounces and overbought drops
	@echo "▶  Scanning 218 symbols for reversal setups ..."
	PYTHONPATH=. $(PYTHON) -c "\
import sys; sys.path.insert(0,'.'); \
from src.analytics.reversal_detector import ReversalDetector; \
d = ReversalDetector(override_threshold=0.60); \
r = d.scan(); \
s = d.top_setups(r, n=10); \
print('OVERSOLD BOUNCE candidates (LONG reversal):'); \
[print(f'  {x[\"symbol\"]:15} score={x[\"score\"]:.2f}  {x[\"evidence\"]}') for x in s['oversold_bounce']]; \
print(); print('OVERBOUGHT DROP candidates (SHORT reversal):'); \
[print(f'  {x[\"symbol\"]:15} score={x[\"score\"]:.2f}  {x[\"evidence\"]}') for x in s['overbought_drop']]; \
print(f'\nTotal: {len(r)} reversal setups | Strong (>=0.60): {sum(1 for x in r if x.reversal_score>=0.6)}')"
	@echo "▶  NSE event watcher check ..."
	PYTHONPATH=. $(PYTHON) scripts/nse_event_watcher.py check

watcher-status:  ## Show NSE watcher fire budget + cooldown state
	PYTHONPATH=. $(PYTHON) scripts/nse_event_watcher.py status

watcher-add-event:  ## Add an NSE calendar event (usage: make watcher-add-event LABEL=RBI_OCT FIRE_AT=2026-10-01T10:00:00+05:30)
	PYTHONPATH=. $(PYTHON) scripts/nse_event_watcher.py add-event \
		--label "$(LABEL)" \
		--fire-at "$(FIRE_AT)" \
		--window 60

threshold-sweep:  ## Run score threshold sweep to find optimal min-conviction floor
	@echo "▶  Running score threshold sweep ..."
	@$(PYTHON) -c "\
import sys; sys.path.insert(0,'.'); \
from src.analytics.score_threshold_sweep import ScoreThresholdSweep; \
from pathlib import Path; import json; \
p = Path('reports/g6_cost_robustness_test.json'); \
print('Sweep requires resolved outcomes. Run make forward-paper-resolve first.' \
  if not p.exists() else 'Use ScoreThresholdSweep.run() from Python.');"

brier-report:  ## Print brier delta report from ForecastLedger (measures signal calibration)
	PYTHONPATH=. $(PYTHON) scripts/run_brier_report.py 2>/dev/null || \
		$(PYTHON) -c "import sys; sys.path.insert(0,'.'); \
from src.analytics.forecast_ledger import ForecastLedger; \
ledger = ForecastLedger(); s = ledger.status(); \
print('ForecastLedger status: total=%d  open=%d  settled=%d' % (s['total_records'],s['open'],s['settled'])); \
r = ledger.brier_report(); \
print('No settled forecasts yet — run make forward-paper-resolve first.' if r.get('n',0)==0 \
  else ('brier_delta=%+.6f  win_rate=%.1f%%' % (r['brier_delta'], r['win_rate']*100)))"

# ─────────────────────────────────────────────────────────────────────────────
# CLEAN
# ─────────────────────────────────────────────────────────────────────────────

clean:  ## Remove Python build artefacts and caches (host)
	find . -type d -name "__pycache__"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info"   -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache"  -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache"  -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage coverage.json
	@echo "✓  Clean."

clean-docker:  ## Remove the test-hardened image (forces full rebuild next time)
	docker rmi $(IMAGE) 2>/dev/null || echo "Image not found."
	@echo "✓  Docker image removed. Run 'make build' to rebuild."

# ─────────────────────────────────────────────────────────────────────────────
# HELP
# ─────────────────────────────────────────────────────────────────────────────

help:  ## Show this help message
	@echo ""
	@echo "ml-service2.0 — NSE F&O Alpha Engine (SHADOW stage)"
	@echo "Model: LightGBM fs-3.0.0 | Port: 8100 | Tests: 1,867"
	@echo ""
	@echo "  \033[1mHOST DEVELOPMENT\033[0m"
	@grep -E '^(setup|test|test-cov|test-unit|test-pit|test-fast|lint|format|typecheck|ci|serve):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mDOCKER IMAGE\033[0m"
	@grep -E '^(build|rebuild|deploy):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mDOCKER STACK\033[0m"
	@grep -E '^(up|down|restart|reset|prune):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mDOCKER LOGS\033[0m"
	@grep -E '^(logs|logs-ml|logs-redis|logs-mlflow|logs-all|logs-errors):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mHEALTH & STATUS\033[0m"
	@grep -E '^(health|status|shell|docker-shell):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mDOCKER TESTS\033[0m"
	@grep -E '^(docker-test|docker-test-cov|docker-test-fast):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mDATA\033[0m"
	@grep -E '^(ingest|ingest-universe|ingest-resume|universe-coverage):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mLIVE SESSION & FORWARD PAPER\033[0m"
	@grep -E '^(session|forward-paper|forward-paper-resolve|forward-paper-audit|signal-promote):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mSHADOW / PRODUCTION\033[0m"
	@grep -E '^(promote-shadow):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mRESEARCH & ANALYSIS\033[0m"
	@grep -E '^(g6-test|watcher|watcher-status|threshold-sweep|brier-report):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  \033[1mCLEAN\033[0m"
	@grep -E '^(clean|clean-docker):.*?## .*$$' \
		$(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-24s\033[0m %s\n", $$1, $$2}'
	@echo ""

.DEFAULT_GOAL := help
