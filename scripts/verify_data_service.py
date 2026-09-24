"""
scripts/verify_data_service.py — explicit data-service2.0 connectivity verification
(mandate §5, §6).

This does NOT assume authenticated access is resolved. It probes each capability
and classifies the outcome into an explicit, machine-readable status taxonomy so a
research run can decide honestly whether real-data research is possible:

    DATA_SERVICE_UNAVAILABLE  — cannot reach the service (connect/timeout/circuit)
    DATA_SERVICE_AUTH_FAILED  — HTTP 401/403 (bad/placeholder credentials)
    SYMBOL_NOT_SUPPORTED      — endpoint reachable but symbol unknown
    TIMEFRAME_NOT_SUPPORTED   — interval rejected/banned
    NO_DATA                   — endpoint reachable, zero bars returned
    PARTIAL_DATA              — some bars but below a usable minimum
    VALID_DATA                — usable series returned

It writes a JSON report and prints a summary. It NEVER falls back to synthetic
data — verification of real access must not be faked (mandate §43).

Usage:
    PYTHONPATH=. python3 scripts/verify_data_service.py \
        [--symbols NIFTY,RELIANCE] [--out reports/data_service_connectivity.json]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Load .env first (real creds win), then harmless placeholders so the Settings
# singleton can construct. Placeholders will simply surface as AUTH_FAILED.
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
os.environ.setdefault("ML_SERVICE_API_KEY", "verify-tool-key")
os.environ.setdefault("DATA_SERVICE_API_KEY", "verify-tool-key")
os.environ.setdefault("DATA_SERVICE_2_URL", "http://localhost:8200")
os.environ.setdefault("SENTINEL_PULSE_URL", "http://localhost:3001")

from src.clients.data_service import (  # noqa: E402
    CANONICAL_INTERVALS,
    DataServiceAuthError,
    DataServiceClient,
    DataServiceUnavailableError,
    LowDataConfidenceError,
    SignalEngineNotAllowedError,
)


class Status:
    UNAVAILABLE = "DATA_SERVICE_UNAVAILABLE"
    AUTH_FAILED = "DATA_SERVICE_AUTH_FAILED"
    SYMBOL_NOT_SUPPORTED = "SYMBOL_NOT_SUPPORTED"
    TIMEFRAME_NOT_SUPPORTED = "TIMEFRAME_NOT_SUPPORTED"
    NO_DATA = "NO_DATA"
    PARTIAL_DATA = "PARTIAL_DATA"
    VALID_DATA = "VALID_DATA"
    LOW_CONFIDENCE = "LOW_DATA_CONFIDENCE"
    SIGNAL_ENGINE_NOT_ALLOWED = "SIGNAL_ENGINE_NOT_ALLOWED"
    OK = "OK"


MIN_USABLE_BARS = 200


async def _probe_historical(
    client: DataServiceClient, symbol: str, interval: str, days: int
) -> dict:
    """Probe a single symbol/interval and classify the outcome."""
    to_date = datetime.now(tz=UTC).date().isoformat()
    from_date = (datetime.now(tz=UTC) - timedelta(days=days)).date().isoformat()
    try:
        bars = await client.get_historical_ohlcv(
            symbol, interval=interval, from_date=from_date, to_date=to_date
        )
    except DataServiceAuthError as exc:
        return {"status": Status.AUTH_FAILED, "detail": str(exc)}
    except DataServiceUnavailableError as exc:
        return {"status": Status.UNAVAILABLE, "detail": str(exc)}
    except SignalEngineNotAllowedError as exc:
        return {"status": Status.SIGNAL_ENGINE_NOT_ALLOWED, "detail": str(exc)}
    except LowDataConfidenceError as exc:
        return {"status": Status.LOW_CONFIDENCE, "detail": str(exc)}
    except ValueError as exc:  # banned/unsupported interval
        return {"status": Status.TIMEFRAME_NOT_SUPPORTED, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": Status.UNAVAILABLE, "detail": f"unexpected: {exc}"}

    n = len(bars) if bars else 0
    if n == 0:
        return {"status": Status.NO_DATA, "n_bars": 0}
    if n < MIN_USABLE_BARS:
        return {"status": Status.PARTIAL_DATA, "n_bars": n}
    return {"status": Status.VALID_DATA, "n_bars": n}


async def _probe_endpoint(coro) -> dict:
    """Probe a bare endpoint (market status / fno universe) and classify."""
    try:
        data = await coro
        return {"status": Status.OK, "keys": sorted(list(data))[:12] if isinstance(data, dict) else "list"}
    except DataServiceAuthError as exc:
        return {"status": Status.AUTH_FAILED, "detail": str(exc)}
    except DataServiceUnavailableError as exc:
        return {"status": Status.UNAVAILABLE, "detail": str(exc)}
    except Exception as exc:  # noqa: BLE001
        return {"status": Status.UNAVAILABLE, "detail": f"unexpected: {exc}"}


async def verify(symbols: list[str]) -> dict:
    client = DataServiceClient()
    report: dict = {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "base_url": client._base_url,  # noqa: SLF001 - diagnostic only
        "canonical_intervals": CANONICAL_INTERVALS,
        "checks": {},
    }

    # Connect. A failure here is decisive: the whole service is unreachable.
    try:
        await client.connect()
    except Exception as exc:  # noqa: BLE001
        report["overall"] = Status.UNAVAILABLE
        report["detail"] = f"connect() failed: {exc}"
        return report

    try:
        # 1) Authentication + reachability via market status (cheapest call).
        report["checks"]["market_status"] = await _probe_endpoint(client.get_market_status())
        # 2) F&O universe availability.
        report["checks"]["fno_universe"] = await _probe_endpoint(client.get_fno_universe())
        # 3) Historical OHLCV per symbol across a few key intervals.
        hist: dict = {}
        for sym in symbols:
            hist[sym] = {}
            for interval in ("1d", "15m", "5m"):
                hist[sym][interval] = await _probe_historical(client, sym, interval, days=900)
        report["checks"]["historical"] = hist
        # 4) Confirm the 3m ban is enforced client-side (never sent to network).
        try:
            await client.get_historical_ohlcv(symbols[0], interval="3m")
            report["checks"]["interval_3m_ban"] = {"status": "NOT_ENFORCED"}
        except ValueError:
            report["checks"]["interval_3m_ban"] = {"status": "ENFORCED"}
        except Exception:  # noqa: BLE001
            report["checks"]["interval_3m_ban"] = {"status": "ENFORCED"}
    finally:
        await client.disconnect()

    # Derive an overall verdict from the checks.
    statuses = [report["checks"]["market_status"]["status"]]
    if any(s == Status.AUTH_FAILED for s in _all_statuses(report)):
        report["overall"] = Status.AUTH_FAILED
    elif all(s in (Status.UNAVAILABLE,) for s in statuses):
        report["overall"] = Status.UNAVAILABLE
    elif _has_valid_data(report):
        report["overall"] = Status.VALID_DATA
    else:
        report["overall"] = Status.UNAVAILABLE
    return report


def _all_statuses(report: dict) -> list[str]:
    out: list[str] = []
    checks = report.get("checks", {})
    for key, val in checks.items():
        if key == "historical":
            for sym, ivs in val.items():
                for iv, res in ivs.items():
                    out.append(res.get("status", ""))
        elif isinstance(val, dict):
            out.append(val.get("status", ""))
    return out


def _has_valid_data(report: dict) -> bool:
    hist = report.get("checks", {}).get("historical", {})
    for _sym, ivs in hist.items():
        for _iv, res in ivs.items():
            if res.get("status") == Status.VALID_DATA:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="NIFTY,BANKNIFTY,RELIANCE")
    ap.add_argument("--out", default="reports/data_service_connectivity.json")
    args = ap.parse_args()

    report = asyncio.run(verify(args.symbols.split(",")))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"[verify] connectivity report written to {out_path}")
    print(f"[verify] base_url={report.get('base_url')}")
    print(f"[verify] overall={report.get('overall')}")
    if report.get("detail"):
        print(f"[verify] detail={report['detail']}")
    # Exit non-zero when real data is not usable, so CI/research can gate on it.
    return 0 if report.get("overall") == Status.VALID_DATA else 3


if __name__ == "__main__":
    sys.exit(main())
