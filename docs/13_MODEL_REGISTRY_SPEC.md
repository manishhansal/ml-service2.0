# Model Registry Specification
**ml-service2.0 — Model Lifecycle and Governance**

*Date: 2026-09-24*

---

## 1. Lifecycle Stages

```
HYPOTHESIS → BACKTEST → CHALLENGER → SHADOW → APPROVED → PRODUCTION
                                                              │
                                                         DEPRECATED → ARCHIVED
```

All transitions are forward-only. Rollback creates a new PRODUCTION version from an archived artifact — it never mutates the failed model.

---

## 2. Promotion Gates

### Gate 1: DATA
- `final_oos_used_for_selection` must be False
- Evidence level must be LEVEL_A or LEVEL_B
- `look_ahead_validated` must be True
- OOS observation count >= 60

### Gate 2: PREDICTIVE
- Mean IC >= champion IC + `ic_margin_vs_champion` (default 0.005)
- If no champion: IC > 0.0

### Gate 3: CALIBRATION
- Brier score <= champion brier_score + 0.01
- If no champion: Brier score < 0.25

### Gate 4: EXECUTION
- Net return >= 0 on validation partition
- Turnover <= champion turnover × 1.20

### Gate 5: RISK
- Max drawdown <= champion drawdown + 2 percentage points
- If no champion: max drawdown <= 0.20

### Gate 6: STABILITY
- IC decay status NOT in {FAILED, SIGNIFICANT_DECAY}
- Feature drift severity NOT in {HIGH, CRITICAL}

### Required for PROMOTE:
- All 6 gates PASS
- Human ApprovalToken (time-limited)
- Audit log entry written

---

## 3. Model Artifact Schema

```python
class ModelArtifact(BaseSchema):
    model_id: str              # "{name}-{version}", e.g. "regime_classifier-1.2.3"
    model_name: str
    version: str               # semantic version
    lifecycle_stage: ModelLifecycleStage
    artifact_path: str         # relative path in artifacts/
    sha256: str                # integrity hash
    training_period: tuple[date, date] | None  # (start, end)
    dataset_hash: str | None
    feature_schema_version: str | None  # hash of feature column names
    hyperparameters: dict
    metrics: dict[str, float]  # ic_mean, sharpe_net, brier_score, etc.
    created_at: datetime
    promoted_at: datetime | None
    promoted_by: str | None    # human approval identifier
```

---

## 4. Registry Storage

Current: file-based in `artifacts/`  
Future consideration: MLflow Model Registry (already in dependencies)

File structure:
```
artifacts/
├── registry.json              # master registry manifest
├── regime_classifier/
│   ├── v1.0.0/
│   │   ├── model.pkl
│   │   ├── calibrator.pkl
│   │   └── metadata.json
│   └── v1.1.0/
│       ├── model.pkl
│       └── ...
└── stock_ranker/
    └── ...
```

---

*End of Model Registry Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/registry/registry.py (immutable artifacts with SHA-256 integrity, versioning, lifecycle states), src/registry/promotion.py (six-gate promotion), and src/registry/lifecycle.py (ChampionChallengerManager: challenger -> shadow -> gated champion, with tested rollback). Shadow models cannot allocate capital (enforced). Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
