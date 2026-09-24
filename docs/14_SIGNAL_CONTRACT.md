# Signal Contract
**ml-service2.0 — Canonical Signal Schema and Decision Semantics**

*Date: 2026-09-24*

---

## 1. Signal Philosophy

A signal is not just a prediction. It is a complete, auditable, time-bounded record of a decision and its justification. Every signal must be able to answer:

1. Why was this trade considered?
2. What information was available at that exact time?
3. What model produced the prediction?
4. How calibrated is the probability?
5. What is the expected net edge?
6. What is the downside risk?
7. What regime are we in?
8. What did the news engine contribute?
9. Why was this trade accepted instead of rejected?
10. What would have caused NO_TRADE?
11. What happened afterward?
12. Was the prediction correct?
13. Did the signal make money after costs?

If a signal cannot answer all 13 questions from persisted records, it is not production-ready.

---

## 2. Signal Actions

| Action | Meaning | Capital Allocation |
|---|---|---|
| `LONG` | Enter long position | Yes, if risk gate passes |
| `SHORT` | Enter short position | Yes, if risk gate passes |
| `EXIT` | Exit existing position | Yes — always honor |
| `HOLD` | Maintain existing position | Yes — no change |
| `NO_TRADE` | Do not enter / abstain | No — first-class decision |

**`NO_TRADE` is a first-class decision.** It is the most important output when evidence is insufficient. The system should become increasingly selective when historical NO_TRADE accuracy shows it was correct to abstain.

---

## 3. Signal Quality Tiers

| Tier | Criteria | Interpretation |
|---|---|---|
| `REJECTED` | PIT violation, data gate failure, model unavailable | Signal blocked before scoring |
| `ABSTAIN` | AbstentionPolicy triggered | Evidence insufficient — do not trade |
| `WATCH` | Signal valid but confidence < WATCH_THRESHOLD | Track but do not allocate capital |
| `VALID` | confidence >= VALID_THRESHOLD, positive EV | Trade-eligible signal |
| `HIGH_CONVICTION` | confidence >= HC_THRESHOLD, EV > EV_HC, strong agreement | Maximum allocation eligible |

**Thresholds must be learned from historical outcomes, not hardcoded arbitrary values.** Initial values:
- `VALID_THRESHOLD`: 0.55 confidence + agreement_ratio >= 0.6 + EV > 0
- `HIGH_CONVICTION_THRESHOLD`: 0.70 confidence + agreement_ratio >= 0.75 + EV > 2 × cost

These must be recalibrated after 500+ paper trades.

---

## 4. Current Signal Schema (Implemented in MetaOutput)

```python
# src/schemas/meta.py — MetaOutput (current)
class MetaOutput(BaseSchema):
    action: str                    # BUY | SELL | WAIT | NO_TRADE
    confidence: float              # [0, 1]
    uncertainty: float             # [0, 1]
    agreement: float               # [0, 1]
    agreement_ratio: float         # [0, 1]
    ensemble_score: float | None
    reason_codes: list[str]
    contributing_models: list[str]
    abstention: bool
    provenance: PredictionProvenance
    news_sentiment_signal: NewsSignal | None
    decomposition: ConfidenceDecomposition | None
    explainability: MetaOutputExplainability | None
    symbol: str
    
# MISSING FIELDS — must add:
#   prediction_timestamp: datetime   ← P0 gap
#   feature_as_of: datetime          ← P0 gap
#   data_as_of: datetime             ← P0 gap
#   expires_at: datetime             ← P1 gap
#   signal_id: str                   ← P0 gap (for audit)
```

---

## 5. Target Signal Schema (Full Production)

```python
class Signal(BaseSchema):
    # ── Identity ──────────────────────────────────────────────────────────────
    signal_id: str                     # UUID v4 — unique per signal
    timestamp: datetime                # UTC — signal generation time (= now)
    prediction_timestamp: datetime     # UTC — when prediction was made
    feature_as_of: datetime            # UTC — latest data timestamp used
    data_as_of: datetime               # UTC — DataService data freshness
    news_as_of: datetime | None        # UTC — SentinelPulse data freshness
    expires_at: datetime               # UTC — signal expiry (reject if past this)
    
    # ── Instrument ────────────────────────────────────────────────────────────
    symbol: str                        # NSE symbol uppercase (e.g. "RELIANCE")
    market: str                        # NSE | NFO | CDS | BFO | MCX
    timeframe: str                     # 5m | 15m | 1h | 1d
    
    # ── Decision ──────────────────────────────────────────────────────────────
    action: str                        # LONG | SHORT | EXIT | HOLD | NO_TRADE
    quality_tier: str                  # REJECTED | ABSTAIN | WATCH | VALID | HIGH_CONVICTION
    abstention: bool
    abstention_reason: str | None      # Human-readable reason code string
    decision_reason: str               # ≤ 200 chars — why this decision was made
    
    # ── Model Provenance ──────────────────────────────────────────────────────
    model_version: str                 # e.g. "regime_classifier/v1.2.3"
    feature_version: str               # Feature schema version hash
    data_version: str | None           # Dataset version hash if from batch
    provenance: PredictionProvenance   # weakest-link across all models
    contributing_models: list[str]     # list of model IDs that contributed
    
    # ── Calibrated Probabilities ──────────────────────────────────────────────
    direction_probability: float       # P(correct direction) ∈ [0, 1] — calibrated
    target_probability: float          # P(target hit before stop) ∈ [0, 1]
    stop_probability: float            # P(stop hit before target) ∈ [0, 1]
    # Invariant: target_probability + stop_probability <= 1.0
    
    # ── Expected Value ────────────────────────────────────────────────────────
    expected_return_gross: float       # Gross expected % return
    expected_return_net: float         # Net expected % return after costs
    expected_value: float              # Risk-adjusted EV
    # EV = P(target) × target_pct − P(stop) × stop_pct − cost_pct − uncertainty_penalty
    uncertainty: float                 # Epistemic uncertainty ∈ [0, 1]
    
    # ── Trade Structure ───────────────────────────────────────────────────────
    entry: float | None                # Suggested entry price
    stop: float | None                 # Stop loss price
    targets: list[float]               # Target prices (primary, secondary, etc.)
    risk_reward: float | None          # (target - entry) / (entry - stop) gross R:R
    holding_horizon_bars: int | None   # Expected max holding period
    
    # ── Risk and Position Sizing ──────────────────────────────────────────────
    position_size_pct: float           # Fraction of risk budget [0, 1]
    expected_drawdown_pct: float | None # Expected adverse excursion
    expected_adverse_excursion: float | None  # MAE estimate
    expected_favorable_excursion: float | None  # MFE estimate
    
    # ── Execution ─────────────────────────────────────────────────────────────
    expected_slippage_pct: float | None  # Estimated execution slippage
    expected_cost_pct: float | None      # Total transaction costs
    execution_quality: str | None        # IMMEDIATE | LIMIT | AVOID_CLOSE
    
    # ── Regime ────────────────────────────────────────────────────────────────
    regime: str                        # MarketRegime value
    regime_probability: float          # P(current regime is correct) ∈ [0, 1]
    
    # ── Model Confidence ─────────────────────────────────────────────────────
    confidence: float                  # [0, 1] calibrated ensemble confidence
    agreement_ratio: float             # [0, 1] model agreement fraction
    calibration_score: float           # ECE ∈ [0, 1] (lower = better calibration)
    data_confidence: float             # DataConfidenceScore / 100 ∈ [0, 1]
    
    # ── News Intelligence ────────────────────────────────────────────────────
    news_impact: float | None          # [0, 1] news impact score
    news_confidence: float | None      # [0, 1] news signal confidence
    news_direction: int | None         # -1 | 0 | 1 news sentiment direction
    news_contribution: float | None    # Incremental contribution of news to confidence
    
    # ── Explainability ────────────────────────────────────────────────────────
    top_features: list[str]            # Top-3 contributing feature names
    feature_contributions: dict[str, float]  # feature_name → SHAP value (when available)
    dominant_model: str | None         # Which model had highest ensemble weight
    reason_codes: list[str]            # Machine-readable decision codes
    
    # ── Outcome (populated post-trade) ───────────────────────────────────────
    outcome_resolved: bool             # False until trade closes
    outcome_return_net: float | None   # Actual net return (populated after close)
    outcome_hit_stop: bool | None
    outcome_hit_target: bool | None
    outcome_holding_bars: int | None
    outcome_resolved_at: datetime | None
```

---

## 6. Signal Lifecycle

```
GENERATED
    │
    ├── quality_tier == REJECTED → discard, log
    ├── quality_tier == ABSTAIN → log, do not send to broker
    ├── expires_at < now → discard, log
    │
    ▼
VALID or HIGH_CONVICTION
    │
    ├── AlphaForge risk gate (portfolio constraints, daily loss limit)
    ├── AlphaForge execution gate (is market open? is instrument liquid?)
    │
    ▼
SENT_TO_BROKER
    │
    ├── Fill confirmation → UPDATE signal with fill_price, fill_timestamp
    │
    ▼
OPEN_POSITION
    │
    ├── Stop hit → outcome_hit_stop=True, outcome resolved
    ├── Target hit → outcome_hit_target=True, outcome resolved
    ├── Horizon elapsed → outcome resolved
    ├── Manual exit → outcome resolved
    │
    ▼
CLOSED
    │
    └── Feedback to ML Service → OnlineLearner.update()
```

---

## 7. Signal Persistence Requirements

Every signal must be persisted with sufficient information for post-trade analysis. The AlphaForge database must store:

| Table | Fields |
|---|---|
| `ml_signals` | signal_id, timestamp, symbol, action, quality_tier, confidence, regime, provenance, feature_as_of, prediction_timestamp, expires_at |
| `ml_signal_models` | signal_id, model_id, model_action, model_confidence, model_direction, model_provenance, weight |
| `ml_signal_features` | signal_id, feature_name, feature_value, feature_as_of |
| `ml_signal_outcomes` | signal_id, outcome_return_net, outcome_hit_stop, outcome_hit_target, resolved_at |

Without this persistence, post-trade analysis and online learning are impossible.

---

## 8. Signal Expiry Policy

| Timeframe | Default expiry | Notes |
|---|---|---|
| 1m | 5 minutes | Stale within one bar |
| 5m | 15 minutes | 3 bars |
| 15m | 45 minutes | 3 bars |
| 1h | 2 hours | Half session |
| 1d | Market close (15:30 IST) | End of trading day |

Signals past `expires_at` must be silently discarded — never executed.

---

*End of Signal Contract*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

Implemented by src/schemas/meta.py (MetaOutput now carries the full PIT chain — prediction_timestamp, feature_as_of, data_as_of, news_as_of, expires_at — plus signal_id, model_versions, dataset_version, calibration_version, expected_net_edge, prob_target/stop) and src/meta/signal.py (is_executable, attach_expiry, validate_signal_contract). Strict expiry boundary and the PIT invariant (feature_as_of/news_as_of <= prediction_timestamp) are enforced and tested. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
