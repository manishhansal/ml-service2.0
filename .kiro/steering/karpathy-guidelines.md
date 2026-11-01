# Karpathy-Inspired Coding Guidelines

> Derived from Andrej Karpathy's observations on LLM coding pitfalls.
> Source: https://github.com/multica-ai/andrej-karpathy-skills
>
> **Tradeoff:** These guidelines bias toward caution over speed.
> For trivial tasks (a typo fix, an obvious one-liner) use judgment — not every change needs the full rigor.

---

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing anything:
- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them — don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

**In this project specifically:**
- This is a production-grade, LIMITED SHADOW quant ML service. Wrong assumptions about PIT correctness, feature leakage, or label construction can silently corrupt live alpha. Say what you assume about data timing before touching any feature, label, or training code.
- Gate implications: if a change could affect any of the 21 production gates (G_PIT, G_LEAK, G_PARITY, etc.) flag it explicitly before proceeding.
- Do not assume `settings.*` values — read `src/config.py` first.

---

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: *Would a senior quant engineer say this is overcomplicated?* If yes, simplify.

**In this project specifically:**
- The existing codebase already has deep abstraction layers (MetaDecisionEngine, SignalPromotionEngine, DrawdownManager, etc.). Do not stack another layer unless it clearly eliminates duplication.
- Prefer plain functions over new classes for logic that is only used once.
- Do not add new dependencies to `pyproject.toml` without explicit user approval.
- All new code must pass `ruff` and `mypy --strict` — write types from the start, not as an afterthought.

---

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style (structlog, pydantic v2, async/await, `from __future__ import annotations`).
- If you notice unrelated dead code, mention it — don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: *Every changed line should trace directly to the user's request.*

**In this project specifically:**
- The audit log (`src/audit/logger.py`) is append-only and tamper-detected — do not touch it unless the task is explicitly about the audit system.
- `strategy/feature_weights.json` is the live agent-editable config — do not modify it as a side effect of unrelated work.
- `artifacts/` and `reports/` are output directories — never edit them directly unless the user asks.
- Do not change the 65-feature schema in `fs-2.0.0` (ExpandedFeatureFactory) as a side effect. Any feature addition must go through the leakage gate.
- `src/clients/market_data_pb2.py` and `market_data_pb2_grpc.py` are generated — never hand-edit them.

---

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform imperative tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

**In this project specifically:**
- Run the relevant test slice before and after any change: `pytest tests/<module> -q --no-cov`
- The coverage gate is **90%** — do not reduce coverage. If a change would drop coverage, add tests or note the gap explicitly.
- IC/model metric changes must be verified against the OOS backtest output, not just "it runs."
- For any change to a live-session script (`scripts/autorun_till_close.py`, `scripts/daily_autorun_scheduler.py`), the success criterion must include a dry-run execution, not just a lint pass.
- The project uses `pytest.mark.unit`, `.integration`, `.slow`, `.pit`, `.tdd` markers — use the appropriate marker when writing new tests.

---

## Project Architecture Snapshot

| Layer | Path | Notes |
|-------|------|-------|
| API | `src/api/` | FastAPI routers under `/v2/` prefix |
| ML Models | `src/models/` | LGBMRegressor v2c (65 features) |
| Training | `src/training/` | WalkForwardValidator, CPCV, SelfLearningLoop |
| Features | `src/features/` | 65 PIT-certified features (fs-2.0.0) |
| Analytics | `src/analytics/` | ScoreThresholdSweep, SignalPromotionEngine, ledgers |
| Monitoring | `src/monitoring/` | DriftDetectorV3, 16 Prometheus metrics |
| Risk | `src/risk/` | DrawdownManager (4 states) |
| Scanners | `src/scanners/` | SMC: OB, FVG, SD, Sweep, MarketStructure |
| Streaming | `src/streaming/` | WebSocket fan-out (WS /v2/stream/signals) |
| Registry | `src/registry/` | CHALLENGER→SHADOW→PRODUCTION lifecycle |
| Data | `src/data/` | SignalDB (PostgreSQL/TimescaleDB port 5445) |
| Audit | `src/audit/` | Tamper-detected append-only audit log |
| Cache | `src/cache/` | Redis (hiredis) |

**Active model:** v2c LGBMRegressor | **Stage:** LIMITED SHADOW | **Port:** 8100 | **Tests:** 1,891 ✓
