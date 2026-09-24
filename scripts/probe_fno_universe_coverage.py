"""
scripts/probe_fno_universe_coverage.py — F&O universe + OHLCV coverage probe
(mandate §7, §8, §11, §14, §15).

Now that GET /v1/instruments/fno-universe returns a real snapshot again, this
script answers the *next* gating question honestly:

    Of the F&O-eligible symbols the data-service reports, how many actually have
    usable REAL historical OHLCV — enough to attempt cross-sectional research?

It does NOT fabricate anything. For each reported symbol it asks data-service2.0
for daily bars and records the real bar count + date range. Symbols with no bars
are recorded as NO_DATA (not dropped silently, not zero-filled).

It also classifies the survivorship quality of the membership metadata per
mandate §11: single effective_from with open effective_to ⇒ CURRENT_UNIVERSE_ONLY,
never "survivorship-free".

Usage:
    PYTHONPATH=. python3 scripts/probe_fno_universe_coverage.py \
        [--interval 1d] [--min-bars 400] [--limit 0] \
        [--out reports/fno_universe_coverage.json]

Exit 2 if the universe endpoint is unavailable (no synthetic fallback, §97).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

_ENV_FILE = Path(__file__).resolve().parent.parent / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if not _line or _line.startswith("#") or "=" not in _line:
            continue
        _k, _, _v = _line.partition("=")
        _k, _v = _k.strip(), _v.strip().strip('"').strip("'")
        if _k and _k not in os.environ:
            os.environ[_k] = _v
os.environ.setdefault("ML_SERVICE_API_KEY", "coverage-probe-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "coverage-probe-key")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")


class UniverseUnavailableError(RuntimeError):
    """Raised when the F&O universe cannot be sourced (no synthetic fallback)."""


def _classify_survivorship(instrument_metadata: list[dict]) -> str:
    """Classify survivorship quality of membership metadata (mandate §11).

    - If every symbol has a single open membership window (one effective_from,
      no effective_to), the universe is a current-membership bootstrap, not a
      true point-in-time membership history ⇒ CURRENT_UNIVERSE_ONLY.
    - If some symbols carry closed windows (effective_to set) or multiple
      distinct effective_from dates exist ⇒ SURVIVORSHIP_LIMITED (partial
      history) — still not survivorship-free unless removals are tracked.
    """
    from_dates = {m.get("validFrom") for m in instrument_metadata if m.get("validFrom")}
    any_closed = any(m.get("validTo") for m in instrument_metadata)
    if not instrument_metadata:
        return "UNKNOWN"
    if len(from_dates) <= 1 and not any_closed:
        return "CURRENT_UNIVERSE_ONLY"
    if any_closed:
        return "SURVIVORSHIP_LIMITED"
    return "SURVIVORSHIP_LIMITED"


async def _run(interval: str, min_bars: int, limit: int, days: int) -> dict:
    from src.clients.data_service import (
        DataServiceClient,
        DataServiceUnavailableError,
        LowDataConfidenceError,
        SignalEngineNotAllowedError,
    )

    client = DataServiceClient()
    await client.connect()
    try:
        # ── 1. F&O universe ────────────────────────────────────────────────
        try:
            uni = await client.get_fno_universe()
        except DataServiceUnavailableError as exc:
            raise UniverseUnavailableError(
                f"data-service2.0 fno-universe unavailable: {exc}; no synthetic fallback (§97)."
            ) from exc

        data = uni.get("data", uni)
        symbols: list[str] = list(data.get("symbols", []))
        meta_list: list[dict] = list(data.get("instrumentMetadata", []))
        availability = data.get("availability") or data.get("status")
        if not symbols:
            raise UniverseUnavailableError(
                f"fno-universe returned availability={availability} with 0 symbols; "
                "refusing to proceed (§9, §97)."
            )

        survivorship = _classify_survivorship(meta_list)

        if limit and limit > 0:
            symbols = symbols[:limit]

        # ── 2. Per-symbol OHLCV coverage probe ─────────────────────────────
        to_date = datetime.now(tz=UTC).date().isoformat()
        from_date = (datetime.now(tz=UTC) - timedelta(days=days)).date().isoformat()

        per_symbol: dict[str, dict] = {}
        status_counts: Counter[str] = Counter()

        # Bounded concurrency so we don't hammer the data-service.
        sem = asyncio.Semaphore(6)

        async def probe(sym: str) -> None:
            async with sem:
                rec: dict = {"symbol": sym}
                try:
                    bars = await client.get_historical_ohlcv(
                        sym, interval=interval, from_date=from_date, to_date=to_date
                    )
                except (SignalEngineNotAllowedError, LowDataConfidenceError) as exc:
                    rec["status"] = "QUALITY_GATE_BLOCKED"
                    rec["detail"] = type(exc).__name__
                    per_symbol[sym] = rec
                    status_counts[rec["status"]] += 1
                    return
                except DataServiceUnavailableError as exc:
                    rec["status"] = "UNAVAILABLE"
                    rec["detail"] = str(exc)[:160]
                    per_symbol[sym] = rec
                    status_counts[rec["status"]] += 1
                    return
                except Exception as exc:  # noqa: BLE001
                    rec["status"] = "ERROR"
                    rec["detail"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                    per_symbol[sym] = rec
                    status_counts[rec["status"]] += 1
                    return

                n = len(bars) if bars else 0
                rec["n_bars"] = n
                if n == 0:
                    rec["status"] = "NO_DATA"
                elif n < min_bars:
                    rec["status"] = "PARTIAL_DATA"
                else:
                    rec["status"] = "VALID_DATA"
                # capture date range if present
                if n:
                    try:
                        times = [b.get("time") for b in bars if b.get("time") is not None]
                        if times:
                            rec["first_ts"] = min(times)
                            rec["last_ts"] = max(times)
                    except Exception:  # noqa: BLE001
                        pass
                per_symbol[sym] = rec
                status_counts[rec["status"]] += 1

        await asyncio.gather(*(probe(s) for s in symbols))

        valid = sorted(s for s, r in per_symbol.items() if r.get("status") == "VALID_DATA")
        partial = sorted(s for s, r in per_symbol.items() if r.get("status") == "PARTIAL_DATA")

        bar_counts = [r["n_bars"] for r in per_symbol.values() if r.get("n_bars")]
        median_bars = (
            sorted(bar_counts)[len(bar_counts) // 2] if bar_counts else 0
        )

        return {
            "generated_at": datetime.now(tz=UTC).isoformat(),
            "data_source_class": "HISTORICAL_REAL",
            "interval": interval,
            "min_bars": min_bars,
            "probe_window_days": days,
            "universe": {
                "availability": availability,
                "reported_symbol_count": len(data.get("symbols", [])),
                "probed_symbol_count": len(symbols),
                "survivorship_classification": survivorship,
                "survivorship_note": (
                    "Membership metadata carries a single open effective_from window; "
                    "this is a CURRENT-membership bootstrap, NOT point-in-time historical "
                    "F&O membership. Any cross-sectional result on this universe is "
                    "SURVIVORSHIP-LIMITED and must not be represented as unbiased "
                    "historical evidence (mandate §11, §54)."
                ),
            },
            "coverage": {
                "status_counts": dict(status_counts),
                "valid_symbol_count": len(valid),
                "partial_symbol_count": len(partial),
                "median_bars_per_symbol": median_bars,
                "valid_symbols": valid,
                "partial_symbols": partial,
            },
            "per_symbol": per_symbol,
            "research_gate": _gate(len(valid)),
        }
    finally:
        await client.disconnect()


def _gate(n_valid: int) -> dict:
    """Decide whether the universe supports cross-sectional research (§14)."""
    if n_valid >= 100:
        state = "CROSS_SECTIONAL_READY_PREFERRED"
    elif n_valid >= 50:
        state = "CROSS_SECTIONAL_READY_MIN"
    elif n_valid >= 10:
        state = "CROSS_SECTIONAL_LIMITED"
    else:
        state = "UNIVERSE_RESEARCH_BLOCKED"
    return {
        "n_valid_symbols": n_valid,
        "state": state,
        "thresholds": {"preferred": 100, "minimum": 50, "limited": 10},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", default="1d")
    ap.add_argument("--min-bars", type=int, default=400)
    ap.add_argument("--limit", type=int, default=0, help="probe only first N symbols (0=all)")
    ap.add_argument("--days", type=int, default=1200)
    ap.add_argument("--out", default="reports/fno_universe_coverage.json")
    args = ap.parse_args()
    try:
        result = asyncio.run(_run(args.interval, args.min_bars, args.limit, args.days))
    except UniverseUnavailableError as exc:
        print(f"[coverage][FAIL] {exc}", file=sys.stderr)
        sys.exit(2)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, default=str))
    g = result["research_gate"]
    c = result["coverage"]
    print(f"[coverage] report → {out}")
    print(f"[coverage] reported={result['universe']['reported_symbol_count']} "
          f"probed={result['universe']['probed_symbol_count']} "
          f"survivorship={result['universe']['survivorship_classification']}")
    print(f"[coverage] status_counts={c['status_counts']}")
    print(f"[coverage] VALID={c['valid_symbol_count']} PARTIAL={c['partial_symbol_count']} "
          f"median_bars={c['median_bars_per_symbol']}")
    print(f"[coverage] RESEARCH_GATE={g['state']} (n_valid={g['n_valid_symbols']})")


if __name__ == "__main__":
    main()
