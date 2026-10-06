# Online Learning Specification
**ml-service2.0 — Controlled Self-Learning System**

*Date: 2026-09-24*

---

## 1. Online Learning Safety Rules

Online learning must NEVER degrade the production champion. The architecture must be:

```
Champion (PRODUCTION) ──────────────────────────── immutable
    │
    └── Shadow copy (SHADOW) ← online updates applied here
            │
            ├── If performance improves → promote shadow to champion (via gates)
            └── If performance degrades → rollback shadow to champion copy
```

---

## 2. Safety Limits (Implemented in OnlineLearner)

| Limit | Default | Description |
|---|---|---|
| `max_consecutive_updates` | 10 | Force full retrain after this many sequential updates |
| `min_samples_per_update` | 30 | Reject updates with fewer samples |
| `max_drift_threshold` | 0.25 | Block updates when feature PSI > threshold |
| `min_performance_threshold` | 0.0 | Rollback if rolling IC drops below this |
| `checkpoint_interval` | 5 | Save checkpoint every N updates |

---

## 3. Update Flow

```
AlphaForge trade outcome received
    │
    ├── Validate: outcome has signal_id, symbol, return_net, resolved_at
    ├── Fetch original signal from ML audit log (by signal_id)
    ├── Construct (X, y) pair for online update
    │
    ├── OnlineLearner.update(X_new, y_new, timestamps_new)
    │       ├── Check min_samples (reject if < 30)
    │       ├── Check consecutive_updates (force retrain if > max)
    │       ├── Check drift (block if PSI > threshold)
    │       ├── Apply incremental update to SHADOW model only
    │       ├── Compute rolling IC on recent predictions
    │       ├── If IC < min_performance_threshold → rollback
    │       └── Checkpoint if interval reached
    │
    └── Shadow performance evaluated periodically
            → If shadow IC > champion IC for 20+ periods → promote
```

---

## 4. Implementation Status

| Component | Implemented | Connected |
|---|---|---|
| OnlineLearner safety limits | ✓ | ✗ |
| Feedback endpoint `/train/feedback` | ✗ | N/A |
| AlphaForge outcome sender | ✗ | N/A |
| Shadow/champion separation | ✗ | N/A |
| Periodic evaluation | ✗ | N/A |

**P1 action required:** Build feedback loop before enabling online learning.

---

*End of Online Learning Specification*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/training/online_learner.py and src/training/self_learning.py (SelfLearningLoop). The champion is NEVER auto-mutated: the loop goes feedback -> retrain decision -> challenger -> shadow only, and promotion to champion still requires the 6-gate pass plus an approval token. Feedback substrate: src/data/feedback.py (immutable FeedbackStore + OutcomeResolver) and POST /train/feedback. Proven by test that the champion is never auto-promoted. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
