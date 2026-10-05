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
	refresh-data fast-backfill fast-backfill-today rescore \
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

fast-backfill:  ## Fast parallel backfill (5d lookback, 8 workers) — use after token refresh
	@echo "▶  Fast parallel backfill for recent bars (last 5 days) ..."
	@echo "   Tip: run 'make check-token' first to ensure Upstox token is valid"
	PYTHONPATH=. $(PYTHON) scripts/fast_backfill_recent.py --days 5 --workers 8
	@echo "✓  Fast backfill complete."

fast-backfill-today:  ## Backfill only today's bars (2d lookback, 10 workers) — urgent recovery
	@echo "▶  Targeted backfill for today's bars ..."
	PYTHONPATH=. $(PYTHON) scripts/fast_backfill_recent.py --days 2 --workers 10
	@echo "✓  Done."

rescore:  ## Re-score all 218 symbols from latest parquets (use outside live session)
	@echo "▶  Rescoring all 218 symbols from latest parquet data ..."
	PYTHONPATH=. $(PYTHON) scripts/rescore_now.py
	@echo "✓  Fresh scores written to artifacts/live_session/latest_scores.json"

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

# ─────────────────────────────────────────────────────────────────────────────
# FORENSIC AUDIT & 7-DAY BACKTEST (added 2026-10-01)
# ─────────────────────────────────────────────────────────────────────────────

audit:  ## Full forensic audit — run all leakage, label, and data quality checks
	@echo "▶  Running forensic audit pipeline ..."
	@echo "   Step 1: Static leakage audit (shift(-N) and center=True scan)"
	PYTHONPATH=. $(PYTHON) -c "\
from src.features.leakage_validator import run_static_leakage_audit; \
from pathlib import Path; \
findings = run_static_leakage_audit([Path('src')]); \
invalid = [f for f in findings if f.classification == 'INVALID']; \
print(f'Static leakage audit: {len(invalid)} INVALID findings (expected: 0)'); \
[print(f'  INVALID: {f.file}:{f.line} — {f.code_snippet[:60]}') for f in invalid]; \
print('PASS' if not invalid else 'FAIL')"
	@echo "   Step 2: Dataset label quality check"
	PYTHONPATH=. $(PYTHON) -c "\
import pandas as pd, json; \
from pathlib import Path; \
ds = sorted(Path('artifacts/datasets').glob('ds-1d-*/data.parquet')); \
if ds: \
    df = pd.read_parquet(str(ds[-1])); \
    lbl = df['label']; \
    rr = df['realized_return_net'].dropna() if 'realized_return_net' in df.columns else df['realized_return'].dropna(); \
    print(f'Dataset: {ds[-1].parent.name}'); \
    print(f'Label balance: {(lbl==1).mean()*100:.1f}% positive / {(lbl==0).mean()*100:.1f}% negative'); \
    print(f'Mean net return: {rr.mean()*100:.4f}%'); \
    print(f'Positive rate: {(rr>0).mean()*100:.1f}%'); \
    print('FAIL: Negative expected value' if rr.mean() < 0 else 'PASS') \
else: print('No dataset found')"
	@echo "   Step 3: PIT timestamp check on parquets"
	PYTHONPATH=. $(PYTHON) -c "\
from pathlib import Path; import pandas as pd; \
pqs = list(Path('data/1d/1d').glob('*.parquet')); \
naive = [p.stem for p in pqs if pd.read_parquet(str(p)).index.tz is None]; \
future = [p.stem for p in pqs if len(pd.read_parquet(str(p))) > 0 \
  and pd.read_parquet(str(p)).index[-1] > pd.Timestamp.now(tz='UTC')]; \
print(f'Total parquets: {len(pqs)}'); \
print(f'Naive timezone (error): {len(naive)}'); \
print(f'Future timestamps (error): {len(future)}'); \
print('PASS' if not naive and not future else 'FAIL')" 2>/dev/null
	@echo "✓  Audit complete. See ML_PIPELINE_FORENSIC_AUDIT.md for full results."

validate-data:  ## Validate parquet data quality (gaps, OHLC integrity, timezone)
	@echo "▶  Validating data quality across all parquets ..."
	PYTHONPATH=. $(PYTHON) -c "\
from pathlib import Path; import pandas as pd, numpy as np; \
pqs = sorted(Path('data/1d/1d').glob('*.parquet')); \
issues = 0; \
for p in pqs: \
    df = pd.read_parquet(str(p)); \
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns: \
        bad_ohlc = (df['high'] < df['low']).sum() + (df['high'] < df['close']).sum() + (df['low'] > df['close']).sum(); \
        if bad_ohlc > 0: print(f'OHLC error: {p.stem} ({bad_ohlc} rows)'); issues += bad_ohlc; \
    dup = df.index.duplicated().sum(); \
    if dup > 0: print(f'Duplicate index: {p.stem} ({dup} rows)'); issues += dup; \
print(f'Total issues: {issues}'); print('PASS' if issues == 0 else 'FAIL')"
	@echo "✓  Data validation complete."

backtest-7d:  ## Run 7-trading-day backtest (full universe, equity costs)
	@echo "▶  Running 7-day backtest (full universe, equity costs) ..."
	@echo "   This may take 5–15 minutes for the full 285-symbol universe."
	PYTHONPATH=. $(PYTHON) scripts/run_7d_backtest.py \
		--oos-start 2025-01-01 --cost equity
	@echo "✓  Backtest complete. Results: artifacts/backtest_7d/"

backtest-7d-quick:  ## Run 7-day backtest quick mode (20 symbols, ~60 seconds)
	@echo "▶  Running 7-day backtest (quick mode, 20 symbols) ..."
	PYTHONPATH=. $(PYTHON) scripts/run_7d_backtest.py \
		--oos-start 2025-06-01 --quick --cost equity
	@echo "✓  Quick backtest complete. Results: artifacts/backtest_7d/"

backtest-7d-futures:  ## Run 7-day backtest with futures cost model (8.5bps)
	@echo "▶  Running 7-day backtest with futures costs ..."
	PYTHONPATH=. $(PYTHON) scripts/run_7d_backtest.py \
		--oos-start 2025-01-01 --cost futures
	@echo "✓  Futures backtest complete."

portfolio-backtest:  ## Run portfolio-level overlapping-signal backtest (Mode A + B, §23A)
	@echo "▶  Running portfolio backtest (§23A) ..."
	@echo "   Mode A: isolated signal quality | Mode B: portfolio-constrained"
	PYTHONPATH=. $(PYTHON) scripts/run_portfolio_backtest.py \
		--oos-start 2025-01-01 --capital 1000000
	@echo "✓  Portfolio backtest complete. Results: artifacts/portfolio_backtest/"

portfolio-backtest-quick:  ## Portfolio backtest quick mode (20 symbols, ~30s)
	@echo "▶  Running portfolio backtest quick mode ..."
	PYTHONPATH=. $(PYTHON) scripts/run_portfolio_backtest.py \
		--oos-start 2025-06-01 --quick
	@echo "✓  Quick portfolio backtest complete."

portfolio-backtest-stress:  ## Portfolio backtest with execution stress scenarios
	@echo "▶  Running portfolio backtest with stress scenarios ..."
	PYTHONPATH=. $(PYTHON) scripts/run_portfolio_backtest.py \
		--oos-start 2025-01-01 --stress --capacity
	@echo "✓  Stress test complete."

# ── V2 MODEL PIPELINE ─────────────────────────────────────────────────────────

v2-build:  ## Build v2c dataset (7-day CS rank labels + CS/regime features)
	@echo "▶  Building v2 labeled dataset (step 1: 7-day labels)..."
	PYTHONPATH=. $(PYTHON) scripts/_build_v2_dataset.py
	@echo "▶  Building v2b ranked dataset (step 2: CS rank labels)..."
	PYTHONPATH=. $(PYTHON) scripts/_build_v2b_dataset.py
	@echo "▶  Building v2c CS+regime dataset (step 3: CS + regime features)..."
	PYTHONPATH=. $(PYTHON) scripts/_build_v2c_dataset.py
	@echo "✓  V2 dataset ready: artifacts/datasets/v2c_cs_regime/"

v2-train:  ## Train v2c model with all forensic fixes (requires v2-build first)
	@echo "▶  Training v2c model (LGBMRegressor, CS rank label, 65 features)..."
	PYTHONPATH=. $(PYTHON) scripts/train_v2.py \
		--dataset artifacts/datasets/v2c_cs_regime/data.parquet \
		--estimators 500
	@echo "✓  V2 model trained. See artifacts/v2_model/"

v2-backtest:  ## Run v2c model 7-day backtest (M1 mode, no proxy)
	@echo "▶  Running v2c 7-day backtest (M1 mode) ..."
	PYTHONPATH=. $(PYTHON) scripts/run_7d_backtest.py \
		--mode m1 --oos-start 2025-01-01 --quick
	@echo "✓  V2 7-day backtest complete. See artifacts/backtest_7d/"

v2-cs-backtest:  ## Run cross-sectional long-short portfolio backtest with v2c model
	@echo "▶  Running v2c CS L/S portfolio backtest ..."
	PYTHONPATH=. $(PYTHON) scripts/run_cs_portfolio_backtest.py --rebalance 7
	@echo "✓  CS backtest complete. See artifacts/cs_portfolio/"

v2-long-only:  ## Run long-only strategy backtest with v2c model (RECOMMENDED)
	@echo "▶  Running long-only strategy backtest (top 10%, weekly, futures) ..."
	PYTHONPATH=. $(PYTHON) scripts/_run_long_only_backtest.py
	@echo "✓  Long-only backtest complete."

v2-full:  ## Complete v2 pipeline: build + train + backtest (takes 15–30 min)
	@echo "▶  Running complete v2 pipeline ..."
	$(MAKE) v2-build
	$(MAKE) v2-train
	$(MAKE) v2-cs-backtest
	$(MAKE) v2-long-only
	@echo "✓  Complete v2 pipeline finished."

v2-alpha-measure:  ## Measure true alpha (IC on actual executable returns)
	@echo "▶  Measuring true alpha (IC on actual open→close returns) ..."
	PYTHONPATH=. $(PYTHON) scripts/_measure_true_alpha.py
	@echo "✓  Alpha measurement complete."

v2-max-alpha:  ## Exhaustive grid search for maximum alpha configuration (takes 10-20 min)
	@echo "▶  Running maximum alpha grid search (270 configurations) ..."
	PYTHONPATH=. $(PYTHON) scripts/_max_alpha_backtest.py
	@echo "✓  Grid search complete. See artifacts/max_alpha/best_config.json"

v2-ensemble:  ## Train parsimonious ensemble model (A+B+C+D) for higher IC
	@echo "▶  Training parsimonious ensemble ..."
	PYTHONPATH=. $(PYTHON) scripts/_train_parsimonious_ensemble.py
	@echo "✓  Ensemble complete. See artifacts/v2_ensemble/"

leakage-test:  ## Run static + dynamic leakage audit across all src/ modules
	@echo "▶  Running comprehensive leakage audit ..."
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_static_leakage_audit.py \
		tests/test_feature_pipeline_pit_correctness.py \
		tests/test_feature_pit.py \
		--no-cov -q --tb=short
	@echo "✓  Leakage audit complete."

test-7d-engine:  ## Run 7-day backtest engine regression tests (19 tests)
	@echo "▶  Running SevenDayBacktestEngine regression tests ..."
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_seven_day_engine.py \
		tests/test_sprint3_alpha_signal.py \
		-v --no-cov --tb=short
	@echo "✓  Engine tests complete."

compare-labels:  ## Compare old vs new label designs (EV analysis)
	@echo "▶  Comparing 5-bar symmetric vs 7-day asymmetric label economics ..."
	PYTHONPATH=. $(PYTHON) scripts/_run_label_comparison.py 2>/dev/null
	@echo "✓  Label comparison complete."

audit-forward-paper:  ## Audit forward paper signals and resolution quality
	@echo "▶  Auditing forward paper signals ..."
	PYTHONPATH=. $(PYTHON) -c "\
import json; from pathlib import Path; from collections import defaultdict; import numpy as np; \
fp = Path('artifacts/forward_paper/forecasts.jsonl'); \
if not fp.exists(): print('No forecasts.jsonl found'); exit(); \
records = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]; \
by_date = defaultdict(list); \
[by_date[r.get('session_date','?')].append(r) for r in records]; \
print(f'Sessions found: {sorted(by_date.keys())}'); \
for d, recs in sorted(by_date.items()): \
    resolved = [r for r in recs if r.get('net_pct') is not None]; \
    if not resolved: continue; \
    net = [r['net_pct'] for r in resolved]; \
    wins = sum(1 for p in net if p > 0); \
    print(f'  {d}: {len(resolved)} resolved | win={wins/len(net)*100:.1f}% | mean_net={np.mean(net)*100:.3f}%'); \
    if abs(np.mean(net)) > 0.05: print(f'    WARNING: |mean_net| > 5% — possible calculation error')"
	@echo "✓  Forward paper audit complete."

generate-report:  ## Generate all audit reports summary (pipe-friendly)
	@echo "▶  Generating audit report summary ..."
	@echo ""
	@echo "=== ML-SERVICE 2.0 INITIAL AUDIT DOCUMENTS ==="
	@for f in ML_PIPELINE_FORENSIC_AUDIT.md LOOKAHEAD_BIAS_AUDIT.md LABEL_AUDIT.md \
		BACKTEST_EXECUTION_CONTRACT.md ML_7_DAY_BACKTEST_REPORT.md \
		SIGNAL_FAILURE_ANALYSIS.md SIGNAL_DECAY_ANALYSIS.md \
		REGIME_ANALYSIS.md BASELINE_COMPARISON.md FEATURE_ABLATION_REPORT.md \
		MODEL_CALIBRATION_REPORT.md BEFORE_AFTER_MODEL_COMPARISON.md \
		MODEL_CARD.md PRODUCTION_READINESS_REPORT_FORENSIC.md \
		PORTFOLIO_EXECUTION_MODEL.md OVERLAPPING_SIGNAL_ANALYSIS.md \
		PORTFOLIO_BACKTEST_REPORT.md; do \
		if [ -f "$$f" ]; then echo "  ✓ $$f"; else echo "  ✗ $$f"; fi; \
	done
	@echo ""
	@echo "=== FORENSIC CERTIFICATION (reports/forensic_cert_2026_10_01/) ==="
	@for f in REPORT_RECONCILIATION_MATRIX.md MASTER_FORENSIC_AUDIT.md \
		ROOT_CAUSE_REGISTER.md TRAINING_INFERENCE_PARITY_AUDIT.md \
		COST_MODEL_V2.md FORWARD_PAPER_PNL_FORENSIC_AUDIT.md \
		LABEL_DESIGN_AUDIT.md LABEL_ECONOMIC_CERTIFICATION.md \
		SURVIVORSHIP_BIAS_CERTIFICATION.md PLACEBO_TEST_REPORT.md \
		TRUE_MODEL_7_DAY_BACKTEST_REPORT.md PBO_REPORT.md \
		REGIME_ANALYSIS.md PORTFOLIO_BACKTEST_REPORT_M1.md \
		PRODUCTION_GATES.md FINAL_QUANT_CERTIFICATION.md \
		IMPROVEMENT_SPRINT_REPORT.md BEFORE_AFTER_V2_COMPARISON.md \
		FINAL_PRODUCTION_RECOMMENDATION.md; do \
		if [ -f "reports/forensic_cert_2026_10_01/$$f" ]; then \
			echo "  ✓ reports/forensic_cert_2026_10_01/$$f"; \
		else \
			echo "  ✗ reports/forensic_cert_2026_10_01/$$f (MISSING)"; \
		fi; \
	done
	@echo ""
	@echo "=== MACHINE-READABLE OUTPUTS ==="
	@for f in artifacts/backtest_7d/backtest_signals.csv \
		artifacts/backtest_7d/profitable_opportunities.csv \
		artifacts/backtest_7d/performance_by_symbol.csv \
		artifacts/backtest_7d/performance_by_regime.csv \
		artifacts/backtest_7d/performance_by_confidence.csv \
		artifacts/backtest_7d/backtest_report.json \
		artifacts/backtest_7d/baseline_comparison.json \
		artifacts/backtest_7d/signal_decay.json \
		artifacts/portfolio_backtest/portfolio_equity_curve.csv \
		artifacts/portfolio_backtest/executed_orders.csv \
		artifacts/portfolio_backtest/rejected_signals.csv \
		artifacts/portfolio_backtest/portfolio_events.csv \
		artifacts/portfolio_backtest/signal_execution_mapping.csv \
		artifacts/portfolio_backtest/open_positions.csv; do \
		if [ -f "$$f" ]; then \
			echo "  ✓ $$f"; \
		else \
			echo "  ✗ $$f (run make backtest-7d-quick or portfolio-backtest-quick first)"; \
		fi; \
	done
	@echo ""

certify:  ## Run full institutional forensic certification (M1 mode)
	@echo "▶  Running full forensic certification pipeline ..."
	@echo "   Step 1: True model inference (M1)"
	PYTHONPATH=. $(PYTHON) scripts/_run_true_model_inference.py 2>/dev/null
	@echo "   Step 2: Placebo tests"
	PYTHONPATH=. $(PYTHON) scripts/_run_placebo_tests.py 2>/dev/null || true
	@echo "   Step 3: 7-day backtest (M1 mode)"
	PYTHONPATH=. $(PYTHON) scripts/run_7d_backtest.py --mode m1 --oos-start 2025-01-01 --quick
	@echo "   Step 4: Portfolio backtest (M1 mode)"
	PYTHONPATH=. $(PYTHON) scripts/run_portfolio_backtest.py --quick --oos-start 2025-01-01
	@echo "   Step 5: Report inventory"
	$(MAKE) generate-report
	@echo "✓  Certification complete. See reports/forensic_cert_2026_10_01/FINAL_QUANT_CERTIFICATION.md"

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
