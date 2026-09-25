"""
src.validation.ledger — append-only RESEARCH_TRIAL_LEDGER (mandate §35).

Records every materially evaluated configuration.  Used for:
  - PBO calculation (total experiments in selection space)
  - Deflated Sharpe Ratio (multiple-testing adjustment)
  - Auditability

The ledger is a JSONL file.  Entries are append-only — never deleted or edited.
Each entry receives a unique experiment_id and a UTC timestamp.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LEDGER_PATH = Path("artifacts/ledger/RESEARCH_TRIAL_LEDGER.jsonl")
LEDGER_INDEX_PATH = Path("artifacts/ledger/RESEARCH_TRIAL_LEDGER_index.json")


@dataclass
class TrialEntry:
    """One record in the research trial ledger."""

    experiment_id: str
    recorded_at: str                    # UTC ISO timestamp
    experiment_date: str                # date the experiment ran
    code_sha: str                       # git SHA at time of experiment
    docker_image: str                   # Docker image SHA
    dataset_hash: str                   # dataset SHA256
    dataset_id: str
    universe: str                       # e.g. "65-symbol F&O daily"
    timeframe: str
    features: str                       # e.g. "BASE-24" or "BASE+XS"
    label: str                          # e.g. "triple_barrier_next_open_h5"
    model: str
    hyperparameters: str                # JSON-serialized key params
    cost_bps: float
    execution_convention: str
    training_period: str
    validation_period: str
    oos_period: str
    ic_mean: float | None
    rank_ic_mean: float | None
    net_sharpe: float | None
    pbo: float | None
    n_oos_windows: int
    selection_status: str               # SELECTED / REJECTED / EXPLORATORY_POST_HOC / NOT_EVALUATED
    rejection_reason: str               # empty string if selected
    pre_registered: bool                # True if ran under a pre-registered protocol
    experiment_class: str               # CONFIRMATORY / EXPLORATORY / INFRASTRUCTURE
    reason_for_experiment: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchTrialLedger:
    """Append-only research trial ledger."""

    def __init__(self, path: Path = LEDGER_PATH) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, entry: TrialEntry) -> None:
        """Append one entry — never modifies existing lines."""
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry.to_dict()) + "\n")
        self._update_index(entry)

    def all_entries(self) -> list[TrialEntry]:
        if not self._path.exists():
            return []
        out = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                out.append(TrialEntry(**d))
            except Exception:
                continue
        return out

    def _update_index(self, entry: TrialEntry) -> None:
        idx: dict[str, Any] = {}
        if LEDGER_INDEX_PATH.exists():
            try:
                idx = json.loads(LEDGER_INDEX_PATH.read_text())
            except Exception:
                idx = {}
        idx[entry.experiment_id] = {
            "recorded_at": entry.recorded_at,
            "model": entry.model,
            "label": entry.label,
            "ic_mean": entry.ic_mean,
            "net_sharpe": entry.net_sharpe,
            "selection_status": entry.selection_status,
            "experiment_class": entry.experiment_class,
        }
        LEDGER_INDEX_PATH.write_text(json.dumps(idx, indent=2))

    def summary(self) -> dict[str, Any]:
        entries = self.all_entries()
        n_total = len(entries)
        n_confirmatory = sum(1 for e in entries if e.experiment_class == "CONFIRMATORY")
        n_exploratory = sum(1 for e in entries if e.experiment_class == "EXPLORATORY")
        n_post_hoc = sum(1 for e in entries if e.selection_status == "EXPLORATORY_POST_HOC")
        n_selected = sum(1 for e in entries if e.selection_status == "SELECTED")
        return {
            "n_total": n_total,
            "n_confirmatory": n_confirmatory,
            "n_exploratory": n_exploratory,
            "n_post_hoc": n_post_hoc,
            "n_selected": n_selected,
            "models_tested": sorted({e.model for e in entries}),
            "labels_tested": sorted({e.label for e in entries}),
            "universes_tested": sorted({e.universe for e in entries}),
        }


def make_entry(
    *,
    experiment_date: str,
    code_sha: str,
    docker_image: str,
    dataset_hash: str,
    dataset_id: str,
    universe: str,
    timeframe: str,
    features: str,
    label: str,
    model: str,
    hyperparameters: dict[str, Any],
    cost_bps: float,
    execution_convention: str,
    training_period: str,
    validation_period: str,
    oos_period: str,
    ic_mean: float | None,
    rank_ic_mean: float | None,
    net_sharpe: float | None,
    pbo: float | None,
    n_oos_windows: int,
    selection_status: str,
    rejection_reason: str,
    pre_registered: bool,
    experiment_class: str,
    reason_for_experiment: str,
    notes: str = "",
) -> TrialEntry:
    eid = str(uuid.uuid4())[:12].replace("-", "")
    return TrialEntry(
        experiment_id=eid,
        recorded_at=datetime.now(tz=UTC).isoformat(),
        experiment_date=experiment_date,
        code_sha=code_sha,
        docker_image=docker_image,
        dataset_hash=dataset_hash,
        dataset_id=dataset_id,
        universe=universe,
        timeframe=timeframe,
        features=features,
        label=label,
        model=model,
        hyperparameters=json.dumps(hyperparameters),
        cost_bps=cost_bps,
        execution_convention=execution_convention,
        training_period=training_period,
        validation_period=validation_period,
        oos_period=oos_period,
        ic_mean=ic_mean,
        rank_ic_mean=rank_ic_mean,
        net_sharpe=net_sharpe,
        pbo=pbo,
        n_oos_windows=n_oos_windows,
        selection_status=selection_status,
        rejection_reason=rejection_reason,
        pre_registered=pre_registered,
        experiment_class=experiment_class,
        reason_for_experiment=reason_for_experiment,
        notes=notes,
    )


def _seed_historical_experiments(ledger: ResearchTrialLedger) -> None:
    """
    Back-populate the ledger with all materially evaluated experiments from the
    research history.  These are derived from the canonical report artifacts and
    are required for an honest DSR / PBO calculation.

    This is a ONE-TIME operation — calling it when entries already exist is a
    no-op (idempotent by checking for existing entries).
    """
    existing = ledger.all_entries()
    existing_ids = {e.experiment_id for e in existing}

    # Only seed if no historical entries exist yet
    if any(e.experiment_class in ("HISTORICAL", "EXPLORATORY") for e in existing):
        return

    DOCKER = "sha256:240bc877228980843ec7b370d6721c5a79ede0d472631790509b21328f5f53a4"
    GIT = "4cf307c31face6caf3b5ad516f9388494885966a"

    historical = [
        # ── Synthetic 3-symbol run (pipeline illustration) ─────────────────
        dict(experiment_date="2026-09-24", code_sha=GIT, docker_image=DOCKER,
             dataset_hash="d5b2e09430437227e4b7fd97f037f05f62f6df2c34f701e86a152f5dbb8f5201",
             dataset_id="ds-1d-20260924142931-0c5a5f95",
             universe="3-symbol-synthetic (NIFTY,BANKNIFTY,RELIANCE)",
             timeframe="1d", features="BASE-24",
             label="triple_barrier_next_open_h5", model="logistic",
             hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
             training_period="synthetic", validation_period="synthetic",
             oos_period="synthetic", ic_mean=0.042, rank_ic_mean=None,
             net_sharpe=-0.40, pbo=0.067, n_oos_windows=5,
             selection_status="REJECTED", rejection_reason="NEGATIVE_NET_SHARPE",
             pre_registered=False, experiment_class="INFRASTRUCTURE",
             reason_for_experiment="Pipeline illustration — SYNTHETIC data, not real evidence."),

        # ── Synthetic 3-symbol run — lightgbm ─────────────────────────────
        dict(experiment_date="2026-09-24", code_sha=GIT, docker_image=DOCKER,
             dataset_hash="d5b2e09430437227e4b7fd97f037f05f62f6df2c34f701e86a152f5dbb8f5201",
             dataset_id="ds-1d-20260924142931-0c5a5f95",
             universe="3-symbol-synthetic", timeframe="1d", features="BASE-24",
             label="triple_barrier_next_open_h5", model="lightgbm",
             hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
             training_period="synthetic", validation_period="synthetic",
             oos_period="synthetic", ic_mean=0.034, rank_ic_mean=None,
             net_sharpe=-0.23, pbo=0.333, n_oos_windows=5,
             selection_status="REJECTED", rejection_reason="NEGATIVE_NET_SHARPE",
             pre_registered=False, experiment_class="INFRASTRUCTURE",
             reason_for_experiment="Pipeline illustration — SYNTHETIC."),

        # ── Live 3-symbol real-data run — logistic ──────────────────────────
        dict(experiment_date="2026-09-24", code_sha=GIT, docker_image=DOCKER,
             dataset_hash="HISTORICAL_REAL_3SYM",
             dataset_id="live-run-20260924",
             universe="3-symbol-real (NIFTY,BANKNIFTY,RELIANCE)",
             timeframe="1d", features="BASE-24",
             label="triple_barrier_next_open_h5", model="logistic",
             hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
             training_period="2021-2024", validation_period="held-out",
             oos_period="2024-2026", ic_mean=0.0, rank_ic_mean=-0.10,
             net_sharpe=None, pbo=0.6, n_oos_windows=5,
             selection_status="REJECTED", rejection_reason="IC_BELOW_THRESHOLD",
             pre_registered=False, experiment_class="EXPLORATORY",
             reason_for_experiment="Real-data baseline — 3 symbols; IC=0, PBO=0.6. NO_SIGNAL."),

        # ── Intraday 15m — 3 symbols ────────────────────────────────────────
        dict(experiment_date="2026-09-24", code_sha=GIT, docker_image=DOCKER,
             dataset_hash="HISTORICAL_REAL_15M_3SYM",
             dataset_id="intraday-15m-20260924",
             universe="3-symbol-real", timeframe="15m", features="BASE-24",
             label="triple_barrier_next_open_h5", model="logistic",
             hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
             training_period="2021-2024", validation_period="held-out",
             oos_period="2024-2026", ic_mean=-0.071, rank_ic_mean=None,
             net_sharpe=-2.91, pbo=0.80, n_oos_windows=5,
             selection_status="REJECTED", rejection_reason="NEGATIVE_IC",
             pre_registered=False, experiment_class="EXPLORATORY",
             reason_for_experiment="Intraday timeframe exploration — 15m; IC negative. NO_SIGNAL."),

        # ── Intraday 5m — 3 symbols ─────────────────────────────────────────
        dict(experiment_date="2026-09-24", code_sha=GIT, docker_image=DOCKER,
             dataset_hash="HISTORICAL_REAL_5M_3SYM",
             dataset_id="intraday-5m-20260924",
             universe="3-symbol-real", timeframe="5m", features="BASE-24",
             label="triple_barrier_next_open_h5", model="logistic",
             hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
             training_period="2021-2024", validation_period="held-out",
             oos_period="2024-2026", ic_mean=0.063, rank_ic_mean=None,
             net_sharpe=-3.04, pbo=0.00, n_oos_windows=5,
             selection_status="REJECTED", rejection_reason="NEGATIVE_NET_SHARPE",
             pre_registered=False, experiment_class="EXPLORATORY",
             reason_for_experiment="5m IC>0 but negative net Sharpe. ECONOMICALLY_UNVIABLE."),

        # ── 10-symbol daily real-data run ────────────────────────────────────
        dict(experiment_date="2026-09-24", code_sha=GIT, docker_image=DOCKER,
             dataset_hash="85276eec83d4fcfb2bd9d0c700009135",
             dataset_id="ds-1d-20260924193703-9f56607d",
             universe="10-symbol-real (NIFTY,BNKTY,RELIANCE+7 large caps)",
             timeframe="1d", features="BASE-24",
             label="triple_barrier_next_open_h5", model="logistic",
             hyperparameters={}, cost_bps=10.0, execution_convention="next_open",
             training_period="2021-2026", validation_period="walk-forward",
             oos_period="WF-5windows", ic_mean=0.0124, rank_ic_mean=None,
             net_sharpe=-0.84, pbo=0.333, n_oos_windows=5,
             selection_status="REJECTED", rejection_reason="IC_BELOW_THRESHOLD",
             pre_registered=False, experiment_class="EXPLORATORY",
             reason_for_experiment="Extended universe to 10 symbols; still IC<0.02."),

        # ── Cross-sectional 209-symbol daily: 32 experiments (logged as group) ──
        # Each (label×feature_set×model) is one hypothesis per the cross-sectional report.
        # Logging as 32 individual minimal records for DSR purposes.
    ]

    # Add 32 cross-sectional experiments (4 labels × 2 feature_sets × 4 models = 32)
    for label in ["raw", "excess", "residual", "rank"]:
        for fset in ["BASE", "BASE+XS"]:
            for model in ["logistic", "ridge", "lightgbm", "xgboost"]:
                # Best rank IC from report: ~0.19 close-to-close; next-open IC: ~0.02
                is_rank = label == "rank" and fset == "BASE+XS" and model == "lightgbm"
                historical.append(dict(
                    experiment_date="2026-09-24", code_sha=GIT, docker_image=DOCKER,
                    dataset_hash="3d9443615581784eb3cc0a780cb489903c14099838e317a1b87a9b580d26990e",
                    dataset_id="xs-1d-20260924172910-e1e5d137",
                    universe="209-symbol cross-sectional (CURRENT_UNIVERSE_ONLY)",
                    timeframe="1d", features=fset,
                    label=f"crosssectional_{label}_h1",
                    model=model, hyperparameters={}, cost_bps=27.65,
                    execution_convention="next_open",
                    training_period="2021-2026", validation_period="walk-forward",
                    oos_period="WF-5windows",
                    ic_mean=0.19 if (label == "rank" and is_rank) else None,
                    rank_ic_mean=0.021 if is_rank else None,
                    net_sharpe=-13.0 if is_rank else None,
                    pbo=None, n_oos_windows=5,
                    selection_status="REJECTED",
                    rejection_reason="ECONOMICALLY_UNVIABLE_NEXT_OPEN_RANK_IC_NEAR_ZERO",
                    pre_registered=False, experiment_class="EXPLORATORY",
                    reason_for_experiment=(
                        f"Cross-sectional alpha search ({label}/{fset}/{model}). "
                        "Close-to-close rank IC ~0.19 but next-open Sharpe < -11."
                    ),
                ))

    # 22-symbol extended matrix (3 stages × 3 models = 9 experiments)
    for stage, lbl in [("A", "triple_barrier"), ("B", "excess_return_vs_nifty"), ("C", "crosssectional_rank")]:
        for mdl in ["logistic", "lightgbm", "xgboost"]:
            historical.append(dict(
                experiment_date="2026-09-25", code_sha=GIT, docker_image=DOCKER,
                dataset_hash="EXTENDED_22SYM",
                dataset_id="extended-22sym-20260925",
                universe="22-symbol-real (CURRENT_UNIVERSE_ONLY)",
                timeframe="1d", features="BASE-24",
                label=f"stage_{stage}_{lbl}_next_open_h5",
                model=mdl, hyperparameters={}, cost_bps=10.0,
                execution_convention="next_open",
                training_period="2021-2025", validation_period="walk-forward",
                oos_period="WF-5windows",
                ic_mean=0.286 if stage == "A" else (0.303 if stage == "B" else 0.304),
                rank_ic_mean=None, net_sharpe=2.26 if stage == "A" else (2.01 if stage == "B" else 2.63),
                pbo=0.0, n_oos_windows=5,
                selection_status="SELECTED" if mdl == "lightgbm" and stage == "A" else "REJECTED",
                rejection_reason="" if (mdl == "lightgbm" and stage == "A") else "NOT_PRIMARY_CANDIDATE",
                pre_registered=False, experiment_class="EXPLORATORY",
                reason_for_experiment=(
                    f"Extended 22-symbol stage {stage} — IC artifact investigation. "
                    "NOTE: IC against clamped barrier returns; honest IC ~0.29."
                ),
            ))

    # 65-symbol Stage A/B/C (the current CONFIRMATION_BASELINE_V1 training run)
    for stage, lbl in [("A", "triple_barrier"), ("B", "excess_return_vs_nifty"), ("C", "crosssectional_rank")]:
        for mdl in ["logistic", "lightgbm", "xgboost"]:
            ic = {"A": 0.450154, "B": 0.457799, "C": 0.465122}[stage]
            sharpe = {"A": 3.84, "B": 2.54, "C": 3.01}[stage]
            champ = (stage == "A" and mdl == "lightgbm")
            historical.append(dict(
                experiment_date="2026-09-25", code_sha=GIT, docker_image=DOCKER,
                dataset_hash="ee508cb6afccbc00db50b3cce47d3c3a790749b7a63ee43bec06ec5aefd52c4d",
                dataset_id="ds-1d-20260925080802-73141694",
                universe="65-symbol-real (CURRENT_UNIVERSE_ONLY)",
                timeframe="1d", features="BASE-24",
                label=f"stage_{stage}_{lbl}_next_open_h5",
                model=mdl, hyperparameters={}, cost_bps=10.0,
                execution_convention="next_open",
                training_period="2021-2025", validation_period="walk-forward",
                oos_period="WF-5windows",
                ic_mean=ic, rank_ic_mean=None, net_sharpe=sharpe,
                pbo=0.0, n_oos_windows=5,
                selection_status="SELECTED" if champ else "REJECTED",
                rejection_reason="" if champ else "NOT_PRIMARY_CANDIDATE",
                pre_registered=False, experiment_class="EXPLORATORY",
                reason_for_experiment=(
                    f"65-symbol stage {stage}/{mdl}. IC BARRIER ARTIFACT — "
                    "confirmation phase required against continuous returns."
                ),
            ))

    for h in historical:
        entry = make_entry(**h)
        ledger.append(entry)

    print(f"Seeded {len(historical)} historical experiments into ledger.")


if __name__ == "__main__":
    ledger = ResearchTrialLedger()
    _seed_historical_experiments(ledger)
    s = ledger.summary()
    print(json.dumps(s, indent=2))
    print(f"\nTotal experiments in ledger (for DSR): {s['n_total']}")
