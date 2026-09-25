"""
src.data.ingestion — historical market-data ingestion pipeline (P0-003).

DataIngestionPipeline retrieves historical OHLCV bars for an F&O universe from
data-service2.0 (the SOLE market-data authority) and persists them as an
immutable, PIT-preserving raw dataset on disk.

Design guarantees
-----------------
- Data comes EXCLUSIVELY from DataServiceClient — no direct provider access.
- Every bar preserves its source timestamp, instrument identity, and (when the
  upstream envelope supplies it) DataConfidenceScore / provider lineage.
- Missing values are NEVER silently replaced with zero — a gap is recorded as a
  gap and the affected bar is dropped from the clean set, not fabricated.
- Ingestion is RESUMABLE: a JSON checkpoint records the last completed (symbol,
  interval) and the last bar timestamp so re-running does not re-download.
- Bounded concurrency + a simple async rate limiter keep load on the data
  service predictable.
- Validation covers: duplicates, chronology, OHLC consistency, non-negative
  volume, and gap detection.

Requirements: P0-003, Req 2.x (PIT), 06_DATA_CONTRACTS, 08_LABEL_SPECIFICATION.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)


# ── Validation result ──────────────────────────────────────────────────────────


@dataclass
class IngestionValidationReport:
    """Per-(symbol, interval) validation summary produced during ingestion."""

    symbol: str
    interval: str
    rows_raw: int = 0
    rows_clean: int = 0
    duplicates_dropped: int = 0
    ohlc_violations: int = 0
    negative_volume: int = 0
    chronology_violations: int = 0
    gaps_detected: int = 0
    expected_closures: int = 0            # weekends + NSE holidays (mandate §4)
    true_missing_sessions: int = 0        # actual trading sessions with no data
    first_ts: str | None = None
    last_ts: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "interval": self.interval,
            "rows_raw": self.rows_raw,
            "rows_clean": self.rows_clean,
            "duplicates_dropped": self.duplicates_dropped,
            "ohlc_violations": self.ohlc_violations,
            "negative_volume": self.negative_volume,
            "chronology_violations": self.chronology_violations,
            "gaps_detected": self.gaps_detected,
            "expected_closures": self.expected_closures,
            "true_missing_sessions": self.true_missing_sessions,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
        }


@dataclass
class IngestionResult:
    """Aggregate result returned by ``DataIngestionPipeline.ingest``."""

    output_dir: Path
    symbols_ingested: list[str] = field(default_factory=list)
    symbols_failed: list[str] = field(default_factory=list)
    reports: list[IngestionValidationReport] = field(default_factory=list)
    total_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_dir": str(self.output_dir),
            "symbols_ingested": self.symbols_ingested,
            "symbols_failed": self.symbols_failed,
            "total_rows": self.total_rows,
            "reports": [r.to_dict() for r in self.reports],
        }


# ── Async rate limiter ─────────────────────────────────────────────────────────


class _RateLimiter:
    """Simple token-bucket-ish async limiter: at most ``rate`` calls per second."""

    def __init__(self, rate_per_sec: float) -> None:
        self._min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._last: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self._min_interval <= 0.0:
            return
        async with self._lock:
            now = asyncio.get_event_loop().time()
            wait = self._min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = asyncio.get_event_loop().time()


# ── Bar normalisation / validation helpers ─────────────────────────────────────

_OHLC_COLS = ("open", "high", "low", "close")


def _parse_bar_timestamp(bar: dict[str, Any]) -> datetime | None:
    """Extract a UTC-aware timestamp from a raw bar dict, or None if unparseable."""
    raw = (
        bar.get("timestamp")
        or bar.get("ts")
        or bar.get("time")
        or bar.get("date")
        or bar.get("t")
    )
    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        # Heuristic: ms epoch if large, else seconds.
        val = float(raw)
        if val > 1e12:
            val /= 1000.0
        return datetime.fromtimestamp(val, tz=UTC)
    if isinstance(raw, str):
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            return None
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=UTC)
    return None


def normalize_bars(
    raw_bars: list[dict[str, Any]],
    symbol: str,
    interval: str,
    report: IngestionValidationReport,
) -> pd.DataFrame:
    """
    Normalise a list of raw bar dicts into a clean, validated DataFrame.

    Validation performed (missing/invalid → dropped, NEVER zero-filled):
      - timestamp must parse and be UTC-aware
      - OHLC must be present, finite, and satisfy low <= open/close <= high
      - volume must be present and >= 0 (missing volume → dropped)
      - duplicates on timestamp removed (keep first)
      - rows sorted chronologically; out-of-order originals counted

    Returns a DataFrame indexed by UTC timestamp with columns
    [open, high, low, close, volume, vwap?, data_confidence, symbol].
    """
    report.rows_raw = len(raw_bars)
    records: list[dict[str, Any]] = []
    seen_ts: set[datetime] = set()

    for bar in raw_bars:
        ts = _parse_bar_timestamp(bar)
        if ts is None:
            continue

        # OHLC extraction — require all four present and finite.
        try:
            o = float(bar["open"]) if bar.get("open") is not None else None
            h = float(bar["high"]) if bar.get("high") is not None else None
            low_ = float(bar["low"]) if bar.get("low") is not None else None
            c = float(bar["close"]) if bar.get("close") is not None else None
        except (TypeError, ValueError, KeyError):
            continue

        if None in (o, h, low_, c):
            # Missing OHLC — do NOT fabricate; drop the bar.
            continue

        # OHLC consistency.
        if not (low_ <= o <= h and low_ <= c <= h and low_ <= h):
            report.ohlc_violations += 1
            continue

        # Volume — missing volume is a data gap, not zero.
        vol_raw = bar.get("volume")
        if vol_raw is None:
            continue
        try:
            vol = float(vol_raw)
        except (TypeError, ValueError):
            continue
        if vol < 0:
            report.negative_volume += 1
            continue

        if ts in seen_ts:
            report.duplicates_dropped += 1
            continue
        seen_ts.add(ts)

        vwap_raw = bar.get("vwap")
        try:
            vwap = float(vwap_raw) if vwap_raw is not None else None
        except (TypeError, ValueError):
            vwap = None

        conf = bar.get("data_confidence")
        if conf is None:
            meta = bar.get("metadata") or {}
            quality = meta.get("quality") if isinstance(meta, dict) else {}
            if isinstance(quality, dict):
                conf = quality.get("score")

        records.append(
            {
                "timestamp": ts,
                "open": o,
                "high": h,
                "low": low_,
                "close": c,
                "volume": vol,
                "vwap": vwap,
                "data_confidence": int(conf) if conf is not None else None,
                "provider": bar.get("provider") or bar.get("source"),
                "symbol": symbol,
            }
        )

    if not records:
        report.rows_clean = 0
        return pd.DataFrame(
            columns=["open", "high", "low", "close", "volume", "vwap",
                     "data_confidence", "provider", "symbol"]
        )

    df = pd.DataFrame(records)
    # Chronology check on original order before sorting.
    ts_series = df["timestamp"]
    report.chronology_violations = int((ts_series.diff().dt.total_seconds() < 0).sum())

    df = df.sort_values("timestamp").set_index("timestamp")
    report.rows_clean = len(df)
    report.first_ts = df.index.min().isoformat()
    report.last_ts = df.index.max().isoformat()

    # Gap detection for daily bars: use NSE exchange calendar (mandate §4).
    # IMPORTANT: weekends and NSE holidays are EXPECTED_MARKET_CLOSURE,
    # NOT true missing sessions.  Using pd.bdate_range (Mon–Fri only) was
    # already a partial fix but still over-counts by including NSE holidays.
    if interval == "1d" and len(df) > 1:
        try:
            from src.validation.calendar import classify_gaps_in_series, is_market_open
            present_dates = sorted({ts.date() for ts in df.index})
            from datetime import date as _date
            first = present_dates[0]
            last = present_dates[-1]
            # Build the set of expected NSE trading days in the range
            from datetime import timedelta as _td
            expected_trading_days: set[_date] = set()
            cur = first
            while cur <= last:
                if is_market_open(cur):
                    expected_trading_days.add(cur)
                cur += _td(days=1)
            missing_trading_days = expected_trading_days - set(present_dates)
            report.gaps_detected = len(missing_trading_days)
            # Store gap classification for observability
            gap_summary = classify_gaps_in_series(present_dates)
            report.expected_closures = gap_summary["n_expected_closures"]
            report.true_missing_sessions = gap_summary["n_true_missing_sessions"]
        except ImportError:
            # Fallback to bdate_range if calendar module not available
            expected = pd.bdate_range(df.index.min(), df.index.max())
            present_set = {d.normalize().tz_localize(None) for d in df.index}
            expected_naive = set(expected.tz_localize(None))
            report.gaps_detected = len(expected_naive - present_set)

    return df


# ── Ingestion pipeline ─────────────────────────────────────────────────────────


class DataIngestionPipeline:
    """
    Resumable historical OHLCV ingestion from data-service2.0.

    Usage::

        pipeline = DataIngestionPipeline(data_client, output_root=Path("./data_raw"))
        result = await pipeline.ingest(
            symbols=["NIFTY", "BANKNIFTY", "RELIANCE"],
            interval="1d",
            from_date="2020-01-01",
            to_date="2024-12-31",
        )

    Each (symbol, interval) is written to
    ``{output_root}/{interval}/{symbol}.parquet`` and a checkpoint at
    ``{output_root}/_checkpoint.json`` records completed work so a re-run
    skips already-ingested symbols.
    """

    def __init__(
        self,
        data_client: Any,
        output_root: Path,
        max_concurrency: int = 4,
        rate_per_sec: float = 8.0,
    ) -> None:
        self._data = data_client
        self._root = Path(output_root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._sem = asyncio.Semaphore(max_concurrency)
        self._limiter = _RateLimiter(rate_per_sec)
        self._checkpoint_path = self._root / "_checkpoint.json"

    # ── Checkpoint I/O ────────────────────────────────────────────────────────

    def _load_checkpoint(self) -> dict[str, Any]:
        if self._checkpoint_path.exists():
            try:
                return json.loads(self._checkpoint_path.read_text())
            except (ValueError, OSError):
                return {"completed": {}}
        return {"completed": {}}

    def _save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        tmp = self._checkpoint_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(checkpoint, indent=2))
        tmp.replace(self._checkpoint_path)

    def _output_path(self, symbol: str, interval: str) -> Path:
        d = self._root / interval
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{symbol.upper()}.parquet"

    # ── Public API ────────────────────────────────────────────────────────────

    async def ingest(
        self,
        symbols: list[str],
        interval: str = "1d",
        from_date: str | None = None,
        to_date: str | None = None,
        exchange: str = "NSE",
        resume: bool = True,
    ) -> IngestionResult:
        """
        Ingest historical bars for all *symbols* at *interval*.

        Args:
            symbols:   Instrument symbols to ingest.
            interval:  Canonical bar interval (e.g. "1d", "1h", "5m").
            from_date: ISO date string for range start.
            to_date:   ISO date string for range end.
            exchange:  Exchange code.
            resume:    When True, symbols already in the checkpoint are skipped.

        Returns:
            IngestionResult with per-symbol validation reports.
        """
        checkpoint = self._load_checkpoint()
        completed: dict[str, Any] = checkpoint.setdefault("completed", {})
        result = IngestionResult(output_dir=self._root)

        async def _run(sym: str) -> None:
            key = f"{interval}:{sym.upper()}"
            if resume and key in completed and self._output_path(sym, interval).exists():
                logger.info("ingestion_skip_resumed", symbol=sym, interval=interval)
                result.symbols_ingested.append(sym)
                return
            async with self._sem:
                await self._limiter.acquire()
                report = await self._ingest_one(sym, interval, from_date, to_date, exchange)
            if report is None:
                result.symbols_failed.append(sym)
                return
            result.reports.append(report)
            result.total_rows += report.rows_clean
            result.symbols_ingested.append(sym)
            completed[key] = {
                "last_ts": report.last_ts,
                "rows": report.rows_clean,
                "ingested_at": datetime.now(tz=UTC).isoformat(),
            }
            self._save_checkpoint(checkpoint)

        await asyncio.gather(*[_run(s) for s in symbols], return_exceptions=False)
        self._save_checkpoint(checkpoint)
        logger.info(
            "ingestion_complete",
            n_ingested=len(result.symbols_ingested),
            n_failed=len(result.symbols_failed),
            total_rows=result.total_rows,
        )
        return result

    async def _ingest_one(
        self,
        symbol: str,
        interval: str,
        from_date: str | None,
        to_date: str | None,
        exchange: str,
    ) -> IngestionValidationReport | None:
        """Fetch, validate, and persist a single symbol. Returns None on failure."""
        report = IngestionValidationReport(symbol=symbol, interval=interval)
        try:
            raw = await self._data.get_historical_ohlcv(
                symbol,
                exchange=exchange,
                interval=interval,
                from_date=from_date,
                to_date=to_date,
            )
        except Exception as exc:
            logger.warning(
                "ingestion_fetch_failed", symbol=symbol, interval=interval, error=str(exc)
            )
            return None

        df = normalize_bars(raw or [], symbol, interval, report)
        if df.empty:
            logger.warning("ingestion_empty_after_validation", symbol=symbol, interval=interval)
            return report

        out = self._output_path(symbol, interval)
        try:
            df.to_parquet(out)
        except Exception:
            df.to_csv(out.with_suffix(".csv"))

        logger.info(
            "ingestion_symbol_done",
            symbol=symbol,
            interval=interval,
            rows_clean=report.rows_clean,
            gaps=report.gaps_detected,
        )
        return report

    def load_symbol(self, symbol: str, interval: str = "1d") -> pd.DataFrame:
        """Load a previously-ingested symbol's clean DataFrame from disk."""
        out = self._output_path(symbol, interval)
        if out.exists():
            return pd.read_parquet(out)
        csv = out.with_suffix(".csv")
        if csv.exists():
            return pd.read_csv(csv, index_col=0, parse_dates=True)
        raise FileNotFoundError(f"No ingested data for {symbol} @ {interval}")
