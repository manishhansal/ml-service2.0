"""
src.validation.protocol — pre-registered CONFIRMATION_PROTOCOL (mandate §36).

This protocol MUST be written and hashed BEFORE running any confirmation
experiments.  It defines:
  - exact hypotheses
  - primary/secondary metrics
  - acceptance gates
  - stopping conditions

Modification after seeing results is FORBIDDEN.  Any post-hoc experiment is
classified EXPLORATORY_POST_HOC in the research trial ledger.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_PATH = Path("artifacts/confirmation/CONFIRMATION_PROTOCOL.json")
PROTOCOL_HASH_PATH = Path("artifacts/confirmation/CONFIRMATION_PROTOCOL.sha256")

PROTOCOL: dict[str, Any] = {
    "protocol_id": "CP-V1-20260925",
    "baseline_id": "CONFIRMATION_BASELINE_V1",
    "created_before_results": True,
    "note": (
        "This protocol was registered before any confirmation experiments ran. "
        "All acceptance/rejection decisions are governed by these pre-registered "
        "criteria. Post-hoc changes are forbidden."
    ),

    # ── Hypotheses ──────────────────────────────────────────────────────────
    "hypotheses": {
        "H0": (
            "The lightgbm model trained on 65 F&O symbols with next-open "
            "triple-barrier labels has zero predictive information content "
            "for future next-open returns (IC = 0)."
        ),
        "H1": (
            "The model has statistically significant and economically viable "
            "predictive information for next-open returns after realistic costs."
        ),
        "H2_ic_artifact": (
            "The reported IC=0.45 is entirely explained by the triple-barrier "
            "clamping artifact (returns forced to ±2%), and the honest "
            "continuous-return IC is not meaningfully positive."
        ),
        "H3_cross_sectional": (
            "The per-symbol IC result is consistent with the cross-sectional "
            "rank-IC result (both measure the same signal)."
        ),
    },

    # ── Primary metric ───────────────────────────────────────────────────────
    "primary_metric": {
        "name": "oos_rank_ic_continuous",
        "description": (
            "Out-of-sample Spearman rank IC between model predictions and "
            "realized continuous (unclamped) next-open returns. This is the "
            "honest IC that is not inflated by barrier clamping. "
            "Measured across all 5 walk-forward OOS windows."
        ),
        "acceptance_gate": 0.02,
        "direction": "greater_than",
    },

    # ── Secondary metrics ────────────────────────────────────────────────────
    "secondary_metrics": [
        {
            "name": "net_sharpe_10bps",
            "description": "Annualized net Sharpe ratio after 10 bps round-trip cost",
            "acceptance_gate": 0.0,
            "direction": "greater_than",
        },
        {
            "name": "cpcv_pbo",
            "description": "Probability of backtest overfitting (CPCV)",
            "acceptance_gate": 0.5,
            "direction": "less_than",
        },
        {
            "name": "cluster_robust_pvalue",
            "description": (
                "Two-sided cluster-robust (date-clustered) p-value for H0: IC=0. "
                "Clustering unit: trading date."
            ),
            "acceptance_gate": 0.05,
            "direction": "less_than",
        },
        {
            "name": "null_test_ic_percentile",
            "description": (
                "Percentile of observed IC in the label-permutation null distribution. "
                "Must be >= 95th percentile to reject H0."
            ),
            "acceptance_gate": 95.0,
            "direction": "greater_than",
        },
        {
            "name": "positive_oos_window_fraction",
            "description": "Fraction of walk-forward OOS windows with positive IC",
            "acceptance_gate": 0.6,
            "direction": "greater_than",
        },
        {
            "name": "effective_sample_size_ratio",
            "description": (
                "N_effective / N_raw after date-clustering. "
                "If < 0.01, the sample is severely dependent."
            ),
            "acceptance_gate": 0.01,
            "direction": "greater_than",
            "note": "Informational — does not gate acceptance alone, but modifies interpretation.",
        },
        {
            "name": "sector_neutral_ic",
            "description": (
                "OOS IC after removing sector-mean return from labels. "
                "Tests whether the signal survives sector-neutralization."
            ),
            "acceptance_gate": 0.01,
            "direction": "greater_than",
        },
        {
            "name": "beta_neutral_ic",
            "description": (
                "OOS IC against NIFTY-beta-neutral residual returns. "
                "Tests whether the signal is individual-stock alpha vs NIFTY timing."
            ),
            "acceptance_gate": 0.0,
            "direction": "greater_than",
        },
    ],

    # ── Required analyses (all mandatory regardless of primary metric result) ──
    "required_analyses": [
        "cross_sectional_dependence",
        "effective_sample_size",
        "cluster_robust_standard_errors_date",
        "cluster_robust_standard_errors_symbol",
        "cluster_robust_standard_errors_twoway",
        "block_bootstrap_ci",
        "stationary_bootstrap_ic",
        "label_permutation_null_100",
        "time_permutation_null_100",
        "symbol_permutation_null_100",
        "block_permutation_null_100",
        "beta_neutral_validation",
        "sector_neutral_validation",
        "symbol_robustness_leave_one_out",
        "regime_robustness_by_year",
        "walk_forward_per_window_details",
        "cpcv_full",
        "deflated_sharpe_ratio",
        "cost_sensitivity_5_10_15_20_30_bps",
        "ic_vs_continuous_returns",  # mandatory — resolves the barrier artifact question
        "ic_vs_barrier_clamped_returns",  # for comparison
    ],

    # ── Universe ─────────────────────────────────────────────────────────────
    "universe": "65 symbols from CONFIRMATION_BASELINE_V1 — fixed, not modified",
    "survivorship_classification": "CURRENT_UNIVERSE_ONLY",
    "survivorship_note": (
        "Survivorship bias is explicitly accepted as a known limitation. "
        "All results carry SURVIVORSHIP_LIMITED qualifier."
    ),

    # ── Time periods ─────────────────────────────────────────────────────────
    "data_period": "2021-10-08 to 2026-09-23",
    "oos_definition": "Walk-forward: 5 non-overlapping OOS windows, embargo=10 days",
    "regime_periods_predefined": ["2021", "2022", "2023", "2024", "2025", "2026_YTD"],

    # ── Models evaluated ─────────────────────────────────────────────────────
    "models_in_scope": ["logistic", "lightgbm", "xgboost"],
    "primary_model": "lightgbm (CONFIRMATION_BASELINE_V1 champion)",
    "no_new_models": True,
    "no_hyperparameter_tuning": True,

    # ── Cost assumptions (fixed) ─────────────────────────────────────────────
    "primary_cost_bps": 10.0,
    "cost_sensitivity_levels": [5, 10, 15, 20, 30],
    "cost_components": [
        "brokerage", "exchange_charges", "STT", "GST", "stamp_duty", "slippage"
    ],
    "cost_note": (
        "10 bps round-trip is the primary economic test. "
        "No switching to a lower cost level if 10 bps fails. "
        "All 5 cost levels are reported."
    ),

    # ── Execution convention ─────────────────────────────────────────────────
    "execution_convention": "next_open",
    "entry": "open[T+1]",
    "exit": "open[T+1+horizon]",
    "horizon_bars": 5,
    "execution_note": "Fixed — cannot be changed after seeing confirmation results.",

    # ── Stopping rules ────────────────────────────────────────────────────────
    "stopping_rules": [
        "STOP if model artifact SHA256 does not match CONFIRMATION_BASELINE_V1",
        "STOP if dataset hash does not match CONFIRMATION_BASELINE_V1",
        "STOP if PIT violation detected in confirmation dataset",
        "STOP if leakage validator fails on confirmation dataset",
        "STOP if IC against continuous returns is indistinguishable from zero "
        "AND null test fails to reject",
        "STOP if result depends entirely on one sector (SECTOR_DEPENDENT)",
        "STOP if top 5 symbols contribute > 90% of total IC (SYMBOL_CONCENTRATED)",
        "REPORT_AND_CONTINUE (not stop) if net Sharpe is negative — report as "
        "ECONOMICALLY_UNVIABLE, document, do not tune",
    ],

    # ── What MUST NOT happen ──────────────────────────────────────────────────
    "forbidden_after_results": [
        "tuning hyperparameters based on confirmation results",
        "selecting best label after seeing results",
        "removing poor-performing symbols",
        "removing poor-performing periods",
        "modifying features after observing OOS results",
        "modifying transaction costs to obtain acceptance",
        "modifying holding periods after observing results",
        "switching execution convention after observing results",
        "cherry-picking profitable windows/symbols/sectors",
        "converting historical replay to forward-paper evidence",
    ],

    # ── Acceptance criteria ────────────────────────────────────────────────────
    "acceptance_for_research_signal_confirmed": (
        "Primary metric IC_continuous >= 0.02 AND "
        "cluster-robust p-value < 0.05 AND "
        "null test percentile >= 95th AND "
        "positive OOS window fraction >= 0.6. "
        "Net Sharpe may be negative — this gives RESEARCH_SIGNAL_CONFIRMED only, "
        "not PAPER_ELIGIBLE."
    ),
    "acceptance_for_paper_eligible": (
        "All RESEARCH_SIGNAL_CONFIRMED criteria plus "
        "net Sharpe (10 bps) >= 0 AND "
        "PBO < 0.5 AND "
        "signal is not sector-concentrated AND "
        "signal is not symbol-concentrated."
    ),
    "acceptance_for_no_verified_edge": (
        "Primary metric IC_continuous < 0.02, OR "
        "cluster-robust p-value >= 0.05, OR "
        "null test fails to reject H0 at 95th percentile."
    ),

    # ── Multiple testing adjustment ────────────────────────────────────────────
    "multiple_testing_note": (
        "All 32 experiments from the prior cross-sectional research run are "
        "included in the DSR calculation. The confirmation analysis is "
        "pre-registered (1 primary hypothesis) but must be interpreted in "
        "the context of the full research history."
    ),
    "dsr_n_trials_minimum": 32,  # from cross_sectional_research.json
}


def _hash(d: dict[str, Any]) -> str:
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write() -> Path:
    """Write the confirmation protocol and its hash.  Must be called before results."""
    PROTOCOL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PROTOCOL_PATH.exists():
        existing = json.loads(PROTOCOL_PATH.read_text())
        if existing == PROTOCOL:
            return PROTOCOL_PATH  # already written — no-op
        raise RuntimeError(
            "PROTOCOL DRIFT: CONFIRMATION_PROTOCOL.json already exists and differs "
            "from the current protocol definition. This is forbidden."
        )
    PROTOCOL_PATH.write_text(json.dumps(PROTOCOL, indent=2, default=str))
    h = _hash(PROTOCOL)
    PROTOCOL_HASH_PATH.write_text(h + "\n")
    print(f"CONFIRMATION_PROTOCOL written: {PROTOCOL_PATH}")
    print(f"Protocol SHA256: {h}")
    return PROTOCOL_PATH


def load() -> dict[str, Any]:
    if not PROTOCOL_PATH.exists():
        raise RuntimeError("CONFIRMATION_PROTOCOL not found — run write() first.")
    return json.loads(PROTOCOL_PATH.read_text())


if __name__ == "__main__":
    write()
    p = load()
    print(f"Protocol ID: {p['protocol_id']}")
    print(f"Primary metric: {p['primary_metric']['name']} >= {p['primary_metric']['acceptance_gate']}")
    print(f"Required analyses: {len(p['required_analyses'])}")
