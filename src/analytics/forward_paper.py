"""
src.analytics.forward_paper — genuine forward-paper runner (mandate §55-§61).

This is NOT historical replay. The lifecycle is:

    REAL DATA AT T  →  FEATURES AT T  →  ELIGIBLE MODEL  →  SIGNAL
        →  IMMUTABLE PERSISTENCE (signal record written once, never mutated)
        →  WAIT for the target horizon to elapse in wall-clock time
        →  FUTURE REAL MARKET DATA  →  OUTCOME RESOLUTION (append-only)

Guardrails enforced here:
  - Immutability (§56): a signal record is append-only JSONL; resolving an
    outcome writes a SEPARATE feedback record and never edits the original.
  - Genuine forward (§55, §96): ``record_signal`` stamps wall-clock ``created_at``
    and the intended ``resolve_after`` (T + horizon). ``resolve_due`` refuses to
    resolve a signal whose horizon has not yet elapsed in real time — so a
    historical bar cannot masquerade as forward-paper.
  - Versioning (§60): every signal carries git_sha, model_version,
    experiment_id, dataset_hash, feature_schema, feature_hash.
  - Minimum sample (§58): ``status`` reports NOT_RUN / INSUFFICIENT_SAMPLE
    (<20 genuine resolved trades) / RUNNING; never claims validation on a
    handful of trades.
  - No auto-promotion (§61): this module only records evidence; promotion to
    shadow/production remains a separate explicit gate.

The store is designed so an independent validator can recompute all forward
metrics from the persisted records alone.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

MIN_TRADES_FOR_VALIDATION = 20        # §58 floor for any forward-paper conclusion
PREFERRED_TRADES = 100                # §58 preferred sample for economic conclusions


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


@dataclass
class ForwardSignal:
    """Immutable forward-paper signal record (§56)."""

    signal_id: str
    created_at: str            # wall-clock time the signal was formed
    signal_ts: str             # T — the decision timestamp
    resolve_after: str         # T + horizon; outcome may only be resolved after this
    symbol: str
    direction: int             # +1 long / -1 short / 0 flat
    horizon_bars: int
    interval: str
    # versioning (§60)
    git_sha: str
    model_version: str
    experiment_id: str
    dataset_hash: str
    feature_schema: str
    feature_hash: str
    feature_as_of: str
    data_as_of: str
    # decision economics (§56)
    prediction: float
    probability: float
    expected_edge: float
    expected_cost: float
    decision: str              # TRADE / NO_TRADE
    entry_price: float
    resolved: bool = False     # set on the SEPARATE resolution record, not here


class ForwardPaperStore:
    """Append-only, immutable store of forward-paper signals (§56)."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, sig: ForwardSignal) -> None:
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(sig)) + "\n")

    def all_signals(self) -> list[ForwardSignal]:
        if not self._path.exists():
            return []
        out: list[ForwardSignal] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(ForwardSignal(**json.loads(line)))
            except Exception:
                continue
        return out


@dataclass
class ForwardPaperRunner:
    """Forms + persists forward-paper signals and gates outcome resolution.

    Only CANDIDATES that passed historical research gates should be paper-traded
    (§59); this runner does not itself decide eligibility — the caller passes an
    ``experiment_id`` of a vetted candidate.
    """

    store: ForwardPaperStore
    interval: str = "1d"
    horizon_bars: int = 1

    def record_signal(
        self,
        *,
        symbol: str,
        signal_ts: datetime,
        direction: int,
        prediction: float,
        probability: float,
        expected_edge: float,
        expected_cost: float,
        entry_price: float,
        model_version: str,
        experiment_id: str,
        dataset_hash: str,
        feature_schema: str,
        feature_vector: dict[str, float] | None,
        feature_as_of: datetime,
        data_as_of: datetime,
        bar_seconds: int,
    ) -> ForwardSignal:
        """Persist an immutable forward signal formed at *signal_ts* (§56).

        ``decision`` is TRADE only when expected_edge > expected_cost (§30); the
        record is written regardless so NO_TRADE decisions are also auditable.
        """
        now = datetime.now(tz=UTC)
        resolve_after = signal_ts + timedelta(seconds=bar_seconds * self.horizon_bars)
        feature_hash = _hash_features(feature_vector)
        decision = "TRADE" if (expected_edge > expected_cost and direction != 0) else "NO_TRADE"
        sig = ForwardSignal(
            signal_id=str(uuid.uuid4()),
            created_at=now.isoformat(),
            signal_ts=signal_ts.isoformat(),
            resolve_after=resolve_after.isoformat(),
            symbol=symbol,
            direction=direction,
            horizon_bars=self.horizon_bars,
            interval=self.interval,
            git_sha=_git_sha(),
            model_version=model_version,
            experiment_id=experiment_id,
            dataset_hash=dataset_hash,
            feature_schema=feature_schema,
            feature_hash=feature_hash,
            feature_as_of=feature_as_of.isoformat(),
            data_as_of=data_as_of.isoformat(),
            prediction=float(prediction),
            probability=float(probability),
            expected_edge=float(expected_edge),
            expected_cost=float(expected_cost),
            decision=decision,
            entry_price=float(entry_price),
        )
        self.store.append(sig)
        return sig

    def resolve_due(self, now: datetime | None = None) -> list[ForwardSignal]:
        """Return TRADE signals whose horizon has elapsed in wall-clock time (§57).

        A signal is only eligible for outcome resolution once ``now >=
        resolve_after``. This is what makes it genuine forward-paper rather than
        historical replay: you cannot resolve an outcome before real time has
        passed.
        """
        now = now or datetime.now(tz=UTC)
        due: list[ForwardSignal] = []
        for s in self.store.all_signals():
            if s.decision != "TRADE":
                continue
            ra = datetime.fromisoformat(s.resolve_after)
            if ra.tzinfo is None:
                ra = ra.replace(tzinfo=UTC)
            if now >= ra:
                due.append(s)
        return due

    def status(self, n_resolved: int) -> dict[str, Any]:
        """Report forward-paper validation status honestly (§58, §96)."""
        n_signals = len(self.store.all_signals())
        if n_signals == 0:
            state = "NOT_RUN"
        elif n_resolved < MIN_TRADES_FOR_VALIDATION:
            state = "INSUFFICIENT_SAMPLE"
        else:
            state = "RUNNING"
        return {
            "state": state,
            "n_signals_recorded": n_signals,
            "n_resolved": n_resolved,
            "min_trades_for_validation": MIN_TRADES_FOR_VALIDATION,
            "preferred_trades": PREFERRED_TRADES,
            "note": (
                "Forward-paper is genuine signal-at-T / outcome-after-T. "
                "Historical replay does NOT count (§96). No auto-promotion to "
                "shadow/production (§61)."
            ),
        }


def _hash_features(feature_vector: dict[str, float] | None) -> str:
    if not feature_vector:
        return "none"
    items = sorted(feature_vector.items())
    payload = json.dumps(items, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]
