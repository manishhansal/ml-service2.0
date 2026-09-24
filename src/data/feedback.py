"""
src.data.feedback — outcome resolution + immutable feedback store (Phase Q).

OutcomeResolver deterministically classifies how a trade concluded from its
realized price path — it never marks a trade resolved merely because a timer
expired. The exit_reason records the actual cause (target/stop/time/manual/
forced/missing) and MAE/MFE/holding period are computed from the path.

FeedbackStore is an APPEND-ONLY JSONL store: feedback records are immutable
once written. This is the substrate of the self-learning loop — every paper /
live trade outcome enters here and feeds challenger retraining decisions.

Requirements: Phase Q, Phase 47, Phase 48, Phase 67, 12_ONLINE_LEARNING_SPEC.
"""
from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.logging_config import get_logger
from src.schemas.meta import FeedbackRecord, OutcomeResolution

logger = get_logger(__name__)


class OutcomeResolver:
    """
    Resolves a trade outcome from its realized price path.

    Usage::

        resolver = OutcomeResolver(cost_bps=10.0)
        resolution = resolver.resolve(
            signal_id="s1", symbol="NIFTY", direction=1,
            entry_price=100.0, target_price=102.0, stop_price=98.5,
            price_path=[(t0, 100.5), (t1, 101.0), (t2, 102.1)],
            max_holding_minutes=375,
        )
    """

    def __init__(self, cost_bps: float = 10.0) -> None:
        self.cost = cost_bps / 10_000.0

    def resolve(
        self,
        signal_id: str,
        symbol: str,
        direction: int,
        entry_price: float,
        target_price: float,
        stop_price: float,
        price_path: list[tuple[datetime, float]],
        max_holding_minutes: int | None = None,
        manual_exit_price: float | None = None,
        forced_exit: bool = False,
    ) -> OutcomeResolution:
        """
        Walk the price path and classify the first barrier touched.

        ``direction``: +1 long, -1 short.
        ``price_path``: chronological (timestamp, price) tuples AFTER entry.
        Returns an OutcomeResolution with the exact exit_reason.
        """
        if not price_path:
            return OutcomeResolution(
                signal_id=signal_id, symbol=symbol, resolved=False,
                exit_reason="MISSING_EXECUTION", entry_price=entry_price,
            )

        entry_ts = price_path[0][0]
        mae = 0.0
        mfe = 0.0
        exit_reason = "TIME_EXPIRY"
        exit_price = price_path[-1][1]
        exit_ts = price_path[-1][0]

        for ts, price in price_path:
            excursion = direction * (price - entry_price) / entry_price
            mfe = max(mfe, excursion)
            mae = min(mae, excursion)

            if direction > 0:
                if price >= target_price:
                    exit_reason, exit_price, exit_ts = "TARGET_HIT", target_price, ts
                    break
                if price <= stop_price:
                    exit_reason, exit_price, exit_ts = "STOP_HIT", stop_price, ts
                    break
            else:
                if price <= target_price:
                    exit_reason, exit_price, exit_ts = "TARGET_HIT", target_price, ts
                    break
                if price >= stop_price:
                    exit_reason, exit_price, exit_ts = "STOP_HIT", stop_price, ts
                    break

        # Manual / forced exit override the time-expiry default.
        if exit_reason == "TIME_EXPIRY":
            if forced_exit:
                exit_reason = "FORCED_EXIT"
                if manual_exit_price is not None:
                    exit_price = manual_exit_price
            elif manual_exit_price is not None:
                exit_reason = "MANUAL_EXIT"
                exit_price = manual_exit_price

        gross = direction * (exit_price - entry_price) / entry_price
        net = gross - self.cost
        holding_min = int((exit_ts - entry_ts).total_seconds() / 60)

        return OutcomeResolution(
            signal_id=signal_id,
            symbol=symbol,
            resolved=True,
            exit_reason=exit_reason,  # type: ignore[arg-type]
            entry_price=entry_price,
            exit_price=exit_price,
            realized_return=round(gross, 6),
            realized_return_net=round(net, 6),
            realized_cost=round(self.cost, 6),
            mae=round(mae, 6),
            mfe=round(mfe, 6),
            holding_period_minutes=max(0, holding_min),
            resolved_at=datetime.now(tz=UTC),
        )


class FeedbackStore:
    """
    Append-only immutable feedback store (JSONL).

    Every FeedbackRecord is written once and never modified. Reads reconstruct
    the full outcome history for a signal / symbol.
    """

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: FeedbackRecord) -> None:
        line = record.model_dump_json()
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        logger.info("feedback_recorded", signal_id=record.signal_id, symbol=record.symbol,
                    exit_reason=record.exit_reason)

    def all_records(self) -> list[FeedbackRecord]:
        if not self._path.exists():
            return []
        records: list[FeedbackRecord] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(FeedbackRecord.model_validate_json(line))
                except Exception as exc:
                    logger.warning("feedback_parse_error", error=str(exc))
        return records

    def count(self) -> int:
        return len(self.all_records())

    def for_symbol(self, symbol: str) -> list[FeedbackRecord]:
        return [r for r in self.all_records() if r.symbol == symbol.upper() or r.symbol == symbol]

    def resolved_records(self) -> list[FeedbackRecord]:
        return [r for r in self.all_records() if r.exit_reason != "UNRESOLVED"]

    def summary(self) -> dict[str, Any]:
        records = self.resolved_records()
        n = len(records)
        if n == 0:
            return {"n_resolved": 0}
        nets = [r.realized_return_net for r in records if r.realized_return_net is not None]
        wins = [x for x in nets if x > 0]
        return {
            "n_resolved": n,
            "n_with_net_return": len(nets),
            "win_rate": round(len(wins) / len(nets), 4) if nets else 0.0,
            "mean_net_return": round(sum(nets) / len(nets), 6) if nets else 0.0,
            "exit_reasons": _count_by(records, "exit_reason"),
        }


def _count_by(records: list[FeedbackRecord], attr: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in records:
        key = str(getattr(r, attr))
        out[key] = out.get(key, 0) + 1
    return out
