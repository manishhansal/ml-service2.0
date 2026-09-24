# Software Development Lifecycle
**ml-service2.0**

*Date: 2026-09-24*

---

## Development Workflow

```
DISCOVERY → FORENSIC AUDIT → REQUIREMENTS → ARCHITECTURE → DESIGN →
TEST PLAN → IMPLEMENTATION → UNIT TEST → INTEGRATION TEST → BACKTEST →
WALK-FORWARD → SHADOW → CERTIFICATION → RELEASE
```

No shortcut is acceptable. A stage cannot be skipped even under time pressure.

---

## Git Discipline

**Commit message format:**
```
feat(ml): add triple-barrier label factory
feat(training): add walk-forward validator
feat(validation): add CPCV path enumeration
feat(decision): add expected value engine
feat(learning): add feedback loop endpoint
test(leakage): add future-data detection tests
test(calibration): add ECE property tests
docs(architecture): update decision engine spec
fix(pit): thread data_quality through to abstention
refactor(ensemble): consolidate duplicate drift detectors
```

**Branch strategy:**
- `main` — production-stable
- `feature/[name]` — individual features
- One PR per focused change
- All CI gates must pass before merge

---

## Code Quality Gates (CI)

Every PR must pass:
1. `ruff check src/ tests/` — zero lint violations
2. `mypy src/` — zero type errors
3. `pytest tests/ -m "unit"` — all unit tests pass
4. `pytest tests/ -m "pit"` — all PIT tests pass
5. `pytest tests/ -m "integration"` — integration tests pass
6. Coverage >= 90% (real assertions, not mocks)

---

## Review Requirements

- P0 changes: 2 reviewers, including financial logic review
- P1 changes: 1 reviewer
- Test-only changes: 1 reviewer
- Documentation: 1 reviewer

---

*End of SDLC*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

The model SDLC (HYPOTHESIS -> BACKTEST -> CHALLENGER -> SHADOW -> APPROVED -> PRODUCTION) is enforced by ModelLifecycleStage, TrainingPipeline forward-only transitions, ModelPromotion (six gates), and ChampionChallengerManager. Development followed TDD (test-first) for the new modules. On the live-data run the pipeline advanced a model to CHALLENGER/SHADOW but correctly did NOT promote to PRODUCTION (no cost-surviving edge). Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
