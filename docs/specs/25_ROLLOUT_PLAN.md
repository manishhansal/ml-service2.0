# Rollout Plan
**ml-service2.0 — Production Deployment Plan**

*Date: 2026-09-24*

---

## Rollout Stages

### Stage 1: Heuristic (Current State)
- `deployment_mode=research`
- `ML_MODE=fallback` in AlphaForge
- ML contributes 15% to opportunity score as heuristic enhancer
- No capital at risk from ML decisions specifically

### Stage 2: First Trained Model (After Phase 5)
- `deployment_mode=paper`
- First CHAMPION model (RegimeClassifier or RiskPredictor)
- Shadow mode: compare ML predictions vs. actual outcomes for 20 days
- No capital allocation change

### Stage 3: Multi-Model Ensemble (After Phase 6)
- `deployment_mode=shadow`
- All 4+ models trained and validated
- Shadow: run MetaDecision alongside rule engine, compare outcomes
- Begin calibrating thresholds from shadow outcomes

### Stage 4: Paper Trading Validation (After Phase 13)
- All 20 acceptance criteria met
- 20+ trading days of shadow showing positive incremental return
- `ML_MODE=fallback` → `ML_MODE=required` in AlphaForge staging
- Paper trade with ML signals for 1 month

### Stage 5: Production (After Phase 15)
- `deployment_mode=validated_ml_only`
- `ML_MODE=required` (or `fallback` with monitoring)
- Continuous monitoring active
- Rollback procedure tested and documented

---

## Rollback at Each Stage

Each stage has an explicit rollback:
- Stage 1→2: Remove trained model artifacts; reverts to heuristic
- Stage 2→3: Remove additional models; revert IC registry
- Stage 3→4: Change deployment_mode back to paper
- Stage 4→5: Set ML_MODE=fallback in AlphaForge

---

*End of Rollout Plan*

---

## IMPLEMENTATION STATUS ADDENDUM (2026-09-24, post-execution)

Rollout state: **NOT STARTED beyond RESEARCH/PAPER.** The full pipeline and the
champion/challenger/shadow rollout machinery (ChampionChallengerManager, 6-gate
promotion, tested rollback) are implemented, but no model has been promoted past
CHALLENGER because none passed the cost/edge gates on real data-service2.0 data
(OOS IC=0.0, PBO=0.6, net Sharpe negative). Shadow and production rollout stages
remain gated on discovering a cost-surviving edge. This is an intentional,
honest outcome — the rollout controls are ready and correctly refused to
advance a model with no verified edge.
