"""
AlphaForge Normalized Indian Market Data Client.

Retrieves canonical OHLCV + derivative data for the ML training pipeline from
the AlphaForge normalized data layer:

  Tier 0 (primary) → data-service Scrapling (/scraping/historical)
                      exchange-official NSE/BSE Bhavcopy; no broker credentials required
  Tier 1           → AlphaForge Next.js API  (/api/in/historical, /api/in/option-chain)
                      backed by Angel One SmartAPI + Upstox Analytics API
  Tier 2           → PostgreSQL historical snapshots (OptionChainSnapshot table +
                      instruments canonical store)
  Tier 3 (fallback)→ Yahoo Finance / yfinance (Indian indices + OHLCV only)

Priority chain per symbol:
  0. data-service Scrapling  (credential-free, exchange-official Bhavcopy — always tried first)
  1. AlphaForge API  (Angel One primary, Upstox failover — already normalised server-side)
  2. PostgreSQL direct query (stored snapshots already in the database)
  3. yfinance (OHLCV only, no OI / derivatives — used when offline or API key absent)

Every record returned by this client is tagged with a DataQuality enum so the
pipeline can exclude STALE / INVALID / SUSPICIOUS rows before feature engineering.

Data structures produced here are drop-in compatible with the existing
compute_stock_features() / compute_regime_features() interface in
features/engineer.py — the downstream models do not need to change.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import Enum
from typing import Any

import httpx
import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# ─── Tier 0 data-service URL ──────────────────────────────────────────────────

DATA_SERVICE_URL: str = os.getenv("DATA_SERVICE_URL", "").rstrip("/")

# ─── Data-quality sentinel ─────────────────────────────────────────────────────


class DataQuality(str, Enum):
    """Record-level data quality classification."""

    GOOD = "GOOD"
    STALE = "STALE"          # Price / OI unchanged for ≥ 3 consecutive bars
    INVALID = "INVALID"      # Negative price/volume, zero close, schema mismatch
    SUSPICIOUS = "SUSPICIOUS"  # Extreme outlier: price move > 20 % in a single bar


# ─── Canonical data schemas ────────────────────────────────────────────────────


@dataclass
class OHLCVRecord:
    """Normalized daily OHLCV bar for a single instrument."""

    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: float
    # Derivatives fields — None when not available (e.g. pure index OHLCV)
    open_interest: float | None = None
    oi_change: float | None = None          # absolute OI change vs prior session
    oi_change_pct: float | None = None      # % OI change vs prior session
    delivery_pct: float | None = None       # NSE delivery %
    quality: DataQuality = DataQuality.GOOD
    source: str = "unknown"


@dataclass
class DerivativesSnapshot:
    """Normalized derivatives data for a single instrument at a given date."""

    symbol: str
    date: date
    pcr: float | None = None               # Put/Call Ratio (by OI)
    atm_iv: float | None = None            # ATM implied volatility (%)
    iv_percentile: float | None = None     # IV percentile (0-100) vs 1Y
    iv_rank: float | None = None           # IVR = (current - 52w low) / (52w high - 52w low)
    atm_skew: float | None = None          # ATM skew (25d call IV - 25d put IV)
    max_pain: float | None = None          # Max pain price for nearest expiry
    max_ce_oi_strike: float | None = None  # Strike with highest call OI
    max_pe_oi_strike: float | None = None  # Strike with highest put OI
    total_ce_oi: float | None = None
    total_pe_oi: float | None = None
    total_ce_oi_change: float | None = None
    total_pe_oi_change: float | None = None
    current_iv: float | None = None        # ATM IV alias (for compat with engineer.py)
    iv_history: list[float] = field(default_factory=list)  # Last 252 sessions ATM IV
    # F&O signal fields
    long_buildup: bool | None = None
    short_buildup: bool | None = None
    short_covering: bool | None = None
    long_unwinding: bool | None = None
    oi_concentration: float | None = None  # HHI of OI across strikes (0-1)
    source: str = "unknown"


@dataclass
class MarketBreadthRecord:
    """Daily market breadth snapshot for the NSE universe."""

    date: date
    advance_count: int = 0
    decline_count: int = 0
    unchanged_count: int = 0
    advance_decline_ratio: float = 0.0
    pct_above_sma20: float = 0.0
    pct_above_sma50: float = 0.0
    pct_above_sma200: float = 0.0
    india_vix: float | None = None
    source: str = "unknown"


@dataclass
class DatasetMetadata:
    """
    Dataset versioning record attached to every training artefact.

    Requirement #4: every training dataset must declare its provenance.
    """

    dataset_version: str
    provider_sources: list[str]
    date_range: tuple[str, str]              # (start_date, end_date) ISO strings
    instrument_universe: list[str]
    feature_version: str
    generated_at: str = field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    )
    quality_filter: list[str] = field(
        default_factory=lambda: ["STALE", "INVALID", "SUSPICIOUS"]
    )
    record_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "datasetVersion": self.dataset_version,
            "providerSources": self.provider_sources,
            "dateRange": {"start": self.date_range[0], "end": self.date_range[1]},
            "instrumentUniverse": self.instrument_universe,
            "featureVersion": self.feature_version,
            "generatedAt": self.generated_at,
            "qualityFilter": self.quality_filter,
            "recordCount": self.record_count,
        }


# ─── Data quality filters ──────────────────────────────────────────────────────


def _classify_quality(df: pd.DataFrame, symbol: str) -> pd.Series:
    """
    Assign a DataQuality tag to each row in an OHLCV DataFrame.

    Requirements:
      - INVALID  : close ≤ 0, open ≤ 0, volume < 0, high < low, NaN in OHLC
      - STALE    : close unchanged for ≥ 3 consecutive bars AND volume == 0
      - SUSPICIOUS: |pct_change(close)| > 20 % in one bar (single-bar move)
    """
    quality = pd.Series(DataQuality.GOOD, index=df.index)

    c = df["close"]
    v = df["volume"]
    h = df["high"]
    lo = df["low"]
    o = df["open"]

    # INVALID checks — highest priority, applied first
    invalid_mask = (
        c.isna() | o.isna() | h.isna() | lo.isna()
        | (c <= 0) | (o <= 0) | (h <= 0) | (lo <= 0)
        | (v < 0)
        | (h < lo)
    )
    quality[invalid_mask] = DataQuality.INVALID

    # STALE check: close unchanged for 3+ bars AND volume = 0
    # Only applied to rows that aren't already INVALID
    close_unchanged = (c == c.shift(1)) & (c == c.shift(2))
    zero_volume = v == 0
    stale_mask = close_unchanged & zero_volume & ~invalid_mask
    quality[stale_mask] = DataQuality.STALE

    # SUSPICIOUS check: |single-bar pct change| > 20 %
    # Only applied to rows that aren't already INVALID or STALE
    pct_chg = c.pct_change().abs()
    suspicious_mask = (pct_chg > 0.20) & ~invalid_mask & ~stale_mask
    quality[suspicious_mask] = DataQuality.SUSPICIOUS

    return quality


def filter_quality(df: pd.DataFrame, exclude: set[DataQuality] | None = None) -> pd.DataFrame:
    """
    Filter out rows whose DataQuality tag is in the exclude set.

    Default exclusion: {STALE, INVALID, SUSPICIOUS}  (Requirement #9)
    """
    if exclude is None:
        exclude = {DataQuality.STALE, DataQuality.INVALID, DataQuality.SUSPICIOUS}
    if "_quality" not in df.columns:
        return df
    mask = ~df["_quality"].isin(exclude)
    return df[mask].drop(columns=["_quality"])


# ─── OI buildup classification (Requirement #8) ───────────────────────────────


def classify_oi_buildup(price_chg_pct: float, oi_chg_pct: float) -> str:
    """
    Classify OI + price relationship into canonical F&O signal.

      Price ↑, OI ↑  → Long Buildup
      Price ↓, OI ↑  → Short Buildup
      Price ↑, OI ↓  → Short Covering
      Price ↓, OI ↓  → Long Unwinding
    """
    if price_chg_pct >= 0 and oi_chg_pct >= 0:
        return "LONG_BUILDUP"
    if price_chg_pct < 0 and oi_chg_pct >= 0:
        return "SHORT_BUILDUP"
    if price_chg_pct >= 0 and oi_chg_pct < 0:
        return "SHORT_COVERING"
    return "LONG_UNWINDING"


# ─── Scrapling data-service client (Tier 0) ───────────────────────────────────


class ScraplingDataClient:
    """
    HTTP client for the AlphaForge data-service Scrapling layer (Tier 0).

    Fetches exchange-official NSE/BSE Bhavcopy OHLCV from the credential-free
    data-service microservice at DATA_SERVICE_URL.  This is always the first
    source attempted; any failure (HTTP error, timeout, connection error, or
    empty response) falls through silently to Tier 1 so the downstream
    pipeline is never disrupted.

    The client is synchronous (uses httpx) to match the rest of the file.
    Timeout is fixed at 10 seconds per Requirement 11.3.
    """

    TIMEOUT: float = 10.0

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d",
    ) -> pd.DataFrame | None:
        """
        Fetch OHLCV candles from the data-service Scrapling endpoint.

        Endpoint:
          GET /scraping/historical?symbol=...&exchange=NSE&interval=1d&from=...&to=...

        Returns a normalised DataFrame on success (non-empty candles array),
        or None when:
          - The candles array is empty (HTTP 200 with no data) — caller proceeds
            to Tier 1 silently without any warning being logged here.
          - HTTP 5xx / connection error / timeout — caller logs a warning.

        The caller is responsible for distinguishing the two cases via the
        return value (None) and whether an error was already logged.
        """
        try:
            with httpx.Client(timeout=self.TIMEOUT) as client:
                resp = client.get(
                    f"{self.base_url}/scraping/historical",
                    params={
                        "symbol": symbol,
                        "exchange": "NSE",
                        "interval": interval,
                        "from": start_date,
                        "to": end_date,
                    },
                )
                resp.raise_for_status()
                payload = resp.json()

            candles: list[dict] = payload.get("candles") or []
            if not candles:
                # HTTP 200 but empty array — proceed to Tier 1 silently
                return None

            # Normalise candle list → DataFrame matching the canonical OHLCV shape
            df = pd.DataFrame(candles)

            # Map OHLCVCandle field names → internal column names
            col_map = {
                "time": "timestamp",   # UTC epoch seconds
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "volume": "volume",
                "oi": "open_interest",
            }
            df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

            # Build DatetimeIndex from UTC epoch seconds
            if "timestamp" in df.columns:
                df["date"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
                df = df.drop(columns=["timestamp"])
            elif "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], utc=True, errors="coerce")
            else:
                logger.warning("scrapling_no_timestamp_column", symbol=symbol)
                return None

            df = df.set_index("date").sort_index()

            # Ensure numeric types for required columns
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            if "open_interest" in df.columns:
                df["open_interest"] = pd.to_numeric(df["open_interest"], errors="coerce")

            required = ["open", "high", "low", "close", "volume"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                logger.warning(
                    "scrapling_missing_columns", symbol=symbol, cols=missing
                )
                return None

            df = df.dropna(subset=required)
            if df.empty:
                return None

            # Tag all rows as Tier 0 provenance — quality is GOOD (exchange-official data)
            df["_quality"] = DataQuality.GOOD
            df["_source"] = "scrapling_bhavcopy"

            return df

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "scrapling_fetch_failed",
                symbol=symbol,
                status=exc.response.status_code,
                error=str(exc),
            )
            return None
        except (httpx.TimeoutException, httpx.ConnectError) as exc:
            logger.warning(
                "scrapling_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )
            return None
        except Exception as exc:
            logger.warning(
                "scrapling_fetch_failed",
                symbol=symbol,
                error=str(exc),
            )
            return None


# ─── AlphaForge API client ─────────────────────────────────────────────────────


class AlphaForgeAPIClient:
    """
    HTTP client for the AlphaForge Next.js API layer.

    This is the primary data source. The Next.js API normalizes data from
    Angel One SmartAPI (primary) and Upstox Analytics (failover) into a
    canonical JSON shape before returning it to this client.

    The client is synchronous (uses httpx) so it works cleanly in both
    interactive training runs and CLI scripts without requiring an event loop.
    """

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": "application/json", "X-Source": "ml-training"},
        )

    # ── OHLCV historical data ─────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1day",
    ) -> pd.DataFrame | None:
        """
        Fetch normalized daily OHLCV from the AlphaForge historical endpoint.

        Endpoint contract (set by AlphaForge Next.js API):
          GET /api/in/historical?symbol=RELIANCE&from=2023-01-01&to=2026-07-31&interval=1day

        Returns a DataFrame with columns:
          [open, high, low, close, volume, open_interest, oi_change, delivery_pct]
          DatetimeIndex (Asia/Kolkata → UTC for consistency with feature layer)
        """
        try:
            resp = self._client.get(
                "/historical",
                params={
                    "symbol": symbol,
                    "from": start_date,
                    "to": end_date,
                    "interval": interval,
                },
            )
            resp.raise_for_status()
            payload = resp.json()

            candles = payload.get("data") or payload.get("candles") or []
            if not candles:
                logger.warning("api_no_candles", symbol=symbol)
                return None

            df = pd.DataFrame(candles)
            df = self._normalize_ohlcv_frame(df, symbol, source="alphaforge_api")
            return df if not df.empty else None

        except httpx.HTTPStatusError as exc:
            logger.warning(
                "api_http_error",
                symbol=symbol,
                status=exc.response.status_code,
            )
            return None
        except Exception as exc:
            logger.warning("api_fetch_failed", symbol=symbol, error=str(exc))
            return None

    def fetch_option_chain_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[DerivativesSnapshot]:
        """
        Fetch historical option chain summaries (PCR, IV, OI walls, max pain).

        Endpoint contract:
          GET /api/in/option-chain/history?symbol=NIFTY&from=...&to=...
        """
        try:
            resp = self._client.get(
                "/option-chain/history",
                params={"symbol": symbol, "from": start_date, "to": end_date},
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
            return [self._parse_deriv_snapshot(r, symbol) for r in rows]
        except Exception as exc:
            logger.warning("api_oc_history_failed", symbol=symbol, error=str(exc))
            return []

    def fetch_india_vix_history(
        self, start_date: str, end_date: str
    ) -> pd.Series | None:
        """
        Fetch India VIX daily series.
        Endpoint: GET /api/in/vix?from=...&to=...
        """
        try:
            resp = self._client.get(
                "/vix", params={"from": start_date, "to": end_date}
            )
            resp.raise_for_status()
            data = resp.json().get("data", [])
            if not data:
                return None
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            return df["vix"].astype(float)
        except Exception as exc:
            logger.warning("api_vix_failed", error=str(exc))
            return None

    def fetch_market_breadth(
        self, start_date: str, end_date: str
    ) -> list[MarketBreadthRecord]:
        """
        Fetch daily market breadth (advance/decline, % above SMAs).
        Endpoint: GET /api/in/market-breadth?from=...&to=...
        """
        try:
            resp = self._client.get(
                "/market-breadth", params={"from": start_date, "to": end_date}
            )
            resp.raise_for_status()
            rows = resp.json().get("data", [])
            results = []
            for r in rows:
                try:
                    results.append(
                        MarketBreadthRecord(
                            date=date.fromisoformat(r["date"]),
                            advance_count=int(r.get("advances", 0)),
                            decline_count=int(r.get("declines", 0)),
                            unchanged_count=int(r.get("unchanged", 0)),
                            advance_decline_ratio=float(r.get("adRatio", 0)),
                            pct_above_sma20=float(r.get("pctAboveSma20", 0)),
                            pct_above_sma50=float(r.get("pctAboveSma50", 0)),
                            pct_above_sma200=float(r.get("pctAboveSma200", 0)),
                            india_vix=float(r["vix"]) if r.get("vix") else None,
                            source="alphaforge_api",
                        )
                    )
                except (KeyError, ValueError):
                    continue
            return results
        except Exception as exc:
            logger.warning("api_breadth_failed", error=str(exc))
            return []

    # ── Internal normalization helpers ────────────────────────────────────

    @staticmethod
    def _normalize_ohlcv_frame(
        df: pd.DataFrame, symbol: str, source: str
    ) -> pd.DataFrame:
        """
        Normalize a raw API response DataFrame into the canonical OHLCV shape.

        The AlphaForge API can return various field name conventions depending on
        which broker backend responded (Angel One vs Upstox). We standardize here.
        """
        # Flexible column name mapping
        col_map = {
            # AlphaForge canonical names
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
            # Alternate names from various providers
            "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume",
            "ltp": "close",
            "openInterest": "open_interest", "oi": "open_interest",
            "oiChange": "oi_change", "oi_change": "oi_change",
            "oiChangePct": "oi_change_pct",
            "deliveryPct": "delivery_pct", "delivery_pct": "delivery_pct",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

        # Timestamp → DatetimeIndex
        for ts_col in ("timestamp", "date", "ts", "time", "candle_time"):
            if ts_col in df.columns:
                df["date"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
                df = df.drop(columns=[ts_col], errors="ignore")
                break
        if "date" not in df.columns:
            logger.warning("no_date_column", symbol=symbol)
            return pd.DataFrame()

        df = df.set_index("date").sort_index()
        df.index = df.index.tz_convert("UTC")

        # Ensure numeric types
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ["open_interest", "oi_change", "oi_change_pct", "delivery_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning("missing_columns", symbol=symbol, cols=missing)
            return pd.DataFrame()

        # Assign data quality tags (kept in hidden column, stripped by filter_quality)
        df["_quality"] = _classify_quality(df, symbol)
        df["_source"] = source

        return df.dropna(subset=required)

    @staticmethod
    def _parse_deriv_snapshot(r: dict, symbol: str) -> DerivativesSnapshot:
        def _f(key: str) -> float | None:
            v = r.get(key)
            try:
                return float(v) if v is not None else None
            except (TypeError, ValueError):
                return None

        return DerivativesSnapshot(
            symbol=symbol,
            date=date.fromisoformat(r["date"]) if r.get("date") else date.today(),
            pcr=_f("pcr"),
            atm_iv=_f("atmIv") or _f("atm_iv"),
            iv_percentile=_f("ivPercentile") or _f("iv_percentile"),
            iv_rank=_f("ivRank") or _f("iv_rank"),
            atm_skew=_f("atmSkew") or _f("atm_skew"),
            max_pain=_f("maxPain") or _f("max_pain"),
            max_ce_oi_strike=_f("maxCeOiStrike") or _f("max_ce_oi_strike"),
            max_pe_oi_strike=_f("maxPeOiStrike") or _f("max_pe_oi_strike"),
            total_ce_oi=_f("totalCeOi") or _f("total_ce_oi"),
            total_pe_oi=_f("totalPeOi") or _f("total_pe_oi"),
            total_ce_oi_change=_f("totalCeOiChange") or _f("total_ce_oi_change"),
            total_pe_oi_change=_f("totalPeOiChange") or _f("total_pe_oi_change"),
            current_iv=_f("atmIv") or _f("atm_iv"),
            source="alphaforge_api",
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "AlphaForgeAPIClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ─── PostgreSQL client ─────────────────────────────────────────────────────────


class PostgreSQLDataClient:
    """
    Direct PostgreSQL query client for AlphaForge historical data.

    Reads from the two relevant tables:
      - option_chain_snapshots : stored by the india-oc-capture worker
      - instruments             : NSE master (symbol, lot size, expiry dates)

    This is the secondary source — used when the AlphaForge API is unavailable
    but the database can be reached directly (e.g. same-host training runs).
    """

    def __init__(self, database_url: str):
        self._url = database_url
        self._engine = None

    def _get_engine(self):  # type: ignore[return]
        if self._engine is not None:
            return self._engine
        try:
            from sqlalchemy import create_engine

            self._engine = create_engine(self._url, pool_pre_ping=True)
            return self._engine
        except ImportError:
            logger.warning("sqlalchemy_not_installed")
            return None
        except Exception as exc:
            logger.warning("pg_engine_failed", error=str(exc))
            return None

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        """
        Query OHLCV from PostgreSQL.

        Expected table schema (price_history or candles):
          symbol TEXT, date DATE, open NUMERIC, high NUMERIC, low NUMERIC,
          close NUMERIC, volume BIGINT, open_interest BIGINT, oi_change BIGINT,
          delivery_pct NUMERIC
        """
        engine = self._get_engine()
        if engine is None:
            return None
        try:
            query = """
                SELECT
                    date,
                    open, high, low, close, volume,
                    open_interest, oi_change,
                    oi_change_pct,
                    delivery_pct
                FROM price_history
                WHERE symbol = %(symbol)s
                  AND date >= %(start)s
                  AND date <= %(end)s
                ORDER BY date ASC
            """
            df = pd.read_sql(
                query,
                engine,
                params={"symbol": symbol, "start": start_date, "end": end_date},
                parse_dates=["date"],
            )
            if df.empty:
                return None
            df = df.set_index("date").sort_index()
            df.index = pd.to_datetime(df.index, utc=True)
            df["_quality"] = _classify_quality(df, symbol)
            df["_source"] = "postgresql"
            return df
        except Exception as exc:
            logger.warning("pg_ohlcv_failed", symbol=symbol, error=str(exc))
            return None

    def fetch_option_chain_history(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> list[DerivativesSnapshot]:
        """
        Query stored option chain snapshots from PostgreSQL.

        Expected table: option_chain_snapshots
          symbol TEXT, snapshot_date DATE, pcr NUMERIC, atm_iv NUMERIC,
          iv_percentile NUMERIC, max_pain NUMERIC, max_ce_oi_strike NUMERIC,
          max_pe_oi_strike NUMERIC, total_ce_oi BIGINT, total_pe_oi BIGINT,
          total_ce_oi_change BIGINT, total_pe_oi_change BIGINT
        """
        engine = self._get_engine()
        if engine is None:
            return []
        try:
            query = """
                SELECT *
                FROM option_chain_snapshots
                WHERE symbol = %(symbol)s
                  AND snapshot_date >= %(start)s
                  AND snapshot_date <= %(end)s
                ORDER BY snapshot_date ASC
            """
            df = pd.read_sql(
                query,
                engine,
                params={"symbol": symbol, "start": start_date, "end": end_date},
                parse_dates=["snapshot_date"],
            )
            results = []
            for _, row in df.iterrows():
                results.append(
                    DerivativesSnapshot(
                        symbol=symbol,
                        date=row["snapshot_date"].date()
                        if hasattr(row["snapshot_date"], "date")
                        else date.fromisoformat(str(row["snapshot_date"])[:10]),
                        pcr=float(row["pcr"]) if pd.notna(row.get("pcr")) else None,
                        atm_iv=float(row["atm_iv"])
                        if pd.notna(row.get("atm_iv"))
                        else None,
                        iv_percentile=float(row["iv_percentile"])
                        if pd.notna(row.get("iv_percentile"))
                        else None,
                        max_pain=float(row["max_pain"])
                        if pd.notna(row.get("max_pain"))
                        else None,
                        max_ce_oi_strike=float(row["max_ce_oi_strike"])
                        if pd.notna(row.get("max_ce_oi_strike"))
                        else None,
                        max_pe_oi_strike=float(row["max_pe_oi_strike"])
                        if pd.notna(row.get("max_pe_oi_strike"))
                        else None,
                        total_ce_oi=float(row["total_ce_oi"])
                        if pd.notna(row.get("total_ce_oi"))
                        else None,
                        total_pe_oi=float(row["total_pe_oi"])
                        if pd.notna(row.get("total_pe_oi"))
                        else None,
                        total_ce_oi_change=float(row["total_ce_oi_change"])
                        if pd.notna(row.get("total_ce_oi_change"))
                        else None,
                        total_pe_oi_change=float(row["total_pe_oi_change"])
                        if pd.notna(row.get("total_pe_oi_change"))
                        else None,
                        current_iv=float(row["atm_iv"])
                        if pd.notna(row.get("atm_iv"))
                        else None,
                        source="postgresql",
                    )
                )
            return results
        except Exception as exc:
            logger.warning("pg_oc_failed", symbol=symbol, error=str(exc))
            return []


# ─── yfinance fallback client ──────────────────────────────────────────────────


class YFinanceFallbackClient:
    """
    Yahoo Finance fallback: OHLCV only, no OI / derivatives.

    Used when both the AlphaForge API and PostgreSQL are unavailable.
    Clearly tagged with source='yfinance' so the pipeline can distinguish
    enriched vs. unenriched training samples.

    Maps NSE symbol → yfinance ticker:
      RELIANCE  → RELIANCE.NS
      ^NSEI     → ^NSEI   (no suffix for indices)
      NIFTY     → ^NSEI
      BANKNIFTY → ^NSEBANK
    """

    _NSE_INDEX_MAP = {
        "NIFTY": "^NSEI",
        "BANKNIFTY": "^NSEBANK",
        "FINNIFTY": "^CNXFIN",
        "MIDCPNIFTY": "^NSEMDCP50",
        "INDIA_VIX": "^INDIAVIX",
    }

    def _to_yf_ticker(self, symbol: str) -> str:
        if symbol in self._NSE_INDEX_MAP:
            return self._NSE_INDEX_MAP[symbol]
        if symbol.startswith("^"):
            return symbol
        return f"{symbol}.NS"

    def fetch_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame | None:
        try:
            import yfinance as yf  # noqa: PLC0415 — import deferred intentionally
        except ImportError:
            logger.error("yfinance_not_installed")
            return None

        ticker = self._to_yf_ticker(symbol)
        try:
            df = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                interval="1d",
                progress=False,
                auto_adjust=True,
            )
            if df is None or df.empty:
                logger.warning("yfinance_no_data", symbol=symbol, ticker=ticker)
                return None

            # Normalize MultiIndex columns (yfinance ≥ 1.0)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0].lower() for c in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]

            required = ["open", "high", "low", "close", "volume"]
            if not all(c in df.columns for c in required):
                return None

            df = df[required].copy()
            df.index = pd.to_datetime(df.index, utc=True)
            df["_quality"] = _classify_quality(df, symbol)
            df["_source"] = "yfinance"
            return df.dropna(subset=required)

        except Exception as exc:
            logger.warning("yfinance_failed", symbol=symbol, error=str(exc))
            return None


# ─── Unified data client (priority chain) ─────────────────────────────────────


class MarketDataClient:
    """
    Unified AlphaForge normalized Indian market data client.

    Implements the three-tier priority chain:
      1. AlphaForge API (Angel One primary / Upstox failover, normalized server-side)
      2. PostgreSQL direct (stored snapshots, same schema)
      3. yfinance fallback (OHLCV only, no derivatives enrichment)

    Callers interact only with this class — the source selection is transparent.
    The returned DataFrames always carry a `_source` column and are quality-tagged.
    The caller passes the result through `filter_quality()` before feature computation.
    """

    def __init__(
        self,
        *,
        api_base_url: str | None = None,
        database_url: str | None = None,
        request_delay_s: float = 0.2,
    ):
        self._scrapling = (
            ScraplingDataClient(DATA_SERVICE_URL) if DATA_SERVICE_URL else None
        )
        self._api = (
            AlphaForgeAPIClient(api_base_url) if api_base_url else None
        )
        self._pg = (
            PostgreSQLDataClient(database_url) if database_url else None
        )
        self._yf = YFinanceFallbackClient()
        self._delay = request_delay_s
        self._sources_used: list[str] = []

    # ── OHLCV ─────────────────────────────────────────────────────────────

    def get_ohlcv(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        *,
        exclude_quality: set[DataQuality] | None = None,
    ) -> pd.DataFrame | None:
        """
        Fetch OHLCV for a symbol, trying each source in priority order.

        Args:
            symbol: NSE symbol (e.g. "RELIANCE", "NIFTY", "^NSEI")
            start_date: ISO date string "YYYY-MM-DD"
            end_date:   ISO date string "YYYY-MM-DD"
            exclude_quality: Set of DataQuality values to filter out.
                             Defaults to {STALE, INVALID, SUSPICIOUS}.

        Returns:
            DataFrame with columns [open, high, low, close, volume,
            open_interest*, oi_change*, delivery_pct*]  (* when available)
            DatetimeIndex in UTC. Returns None only when all sources fail.
        """
        df: pd.DataFrame | None = None
        source_used = "none"

        # 0. Scrapling data-service (exchange-official Bhavcopy — Requirement 11.1–11.5)
        if self._scrapling is not None:
            df = self._scrapling.fetch_ohlcv(symbol, start_date, end_date)
            if df is not None and not df.empty:
                source_used = "scrapling_bhavcopy"
                logger.debug("ohlcv_source", symbol=symbol, source=source_used)

        # 1. AlphaForge API
        if (df is None or df.empty) and self._api is not None:
            df = self._api.fetch_ohlcv(symbol, start_date, end_date)
            if df is not None and not df.empty:
                source_used = "alphaforge_api"
                logger.debug("ohlcv_source", symbol=symbol, source=source_used)

        # 2. PostgreSQL fallback
        if (df is None or df.empty) and self._pg is not None:
            time.sleep(self._delay)
            df = self._pg.fetch_ohlcv(symbol, start_date, end_date)
            if df is not None and not df.empty:
                source_used = "postgresql"
                logger.debug("ohlcv_source", symbol=symbol, source=source_used)

        # 3. yfinance final fallback
        if df is None or df.empty:
            time.sleep(self._delay)
            df = self._yf.fetch_ohlcv(symbol, start_date, end_date)
            if df is not None and not df.empty:
                source_used = "yfinance"
                logger.info(
                    "ohlcv_yfinance_fallback",
                    symbol=symbol,
                    note="No derivative enrichment available from yfinance",
                )

        if df is None or df.empty:
            logger.warning("ohlcv_all_sources_failed", symbol=symbol)
            return None

        if source_used not in self._sources_used:
            self._sources_used.append(source_used)

        return filter_quality(df, exclude=exclude_quality)

    # ── Derivatives / option chain ─────────────────────────────────────────

    def get_derivatives(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
    ) -> dict[date, DerivativesSnapshot]:
        """
        Fetch historical derivatives snapshots, keyed by date.

        Returns an empty dict (not None) when no derivative data is available,
        so callers can degrade gracefully (yfinance-sourced rows will simply
        have no OI / PCR / IV enrichment).
        """
        snapshots: list[DerivativesSnapshot] = []

        # 1. AlphaForge API
        if self._api is not None:
            snapshots = self._api.fetch_option_chain_history(
                symbol, start_date, end_date
            )
            if snapshots:
                logger.debug(
                    "deriv_source",
                    symbol=symbol,
                    count=len(snapshots),
                    source="alphaforge_api",
                )

        # 2. PostgreSQL fallback
        if not snapshots and self._pg is not None:
            snapshots = self._pg.fetch_option_chain_history(
                symbol, start_date, end_date
            )
            if snapshots:
                logger.debug(
                    "deriv_source",
                    symbol=symbol,
                    count=len(snapshots),
                    source="postgresql",
                )

        # Build IV history for iv_rank computation
        iv_vals = [s.atm_iv for s in snapshots if s.atm_iv is not None]
        for snap in snapshots:
            snap.iv_history = iv_vals
            snap.current_iv = snap.atm_iv

        return {s.date: s for s in snapshots}

    # ── Market breadth + India VIX ─────────────────────────────────────────

    def get_market_breadth(
        self, start_date: str, end_date: str
    ) -> dict[date, MarketBreadthRecord]:
        """Return daily market breadth records keyed by date."""
        if self._api is None:
            return {}
        records = self._api.fetch_market_breadth(start_date, end_date)
        return {r.date: r for r in records}

    def get_india_vix(
        self, start_date: str, end_date: str
    ) -> pd.Series | None:
        """Return India VIX daily series. Falls back to yfinance ^INDIAVIX."""
        if self._api is not None:
            vix = self._api.fetch_india_vix_history(start_date, end_date)
            if vix is not None and not vix.empty:
                return vix

        # yfinance fallback for India VIX
        df = self._yf.fetch_ohlcv("INDIA_VIX", start_date, end_date)
        if df is not None and "close" in df.columns:
            return df["close"].rename("vix")
        return None

    # ── Batch fetching ─────────────────────────────────────────────────────

    def get_universe_ohlcv(
        self,
        symbols: list[str],
        start_date: str,
        end_date: str,
        *,
        exclude_quality: set[DataQuality] | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for a full universe of symbols.

        Returns a dict {symbol: DataFrame}. Symbols that fail all sources are
        silently omitted from the result (logged at WARNING level).
        """
        result: dict[str, pd.DataFrame] = {}
        for i, symbol in enumerate(symbols):
            df = self.get_ohlcv(
                symbol, start_date, end_date, exclude_quality=exclude_quality
            )
            if df is not None and not df.empty:
                result[symbol] = df
            if i > 0 and i % 10 == 0:
                logger.info(
                    "universe_fetch_progress",
                    done=i,
                    total=len(symbols),
                    fetched=len(result),
                )
            time.sleep(self._delay)
        logger.info(
            "universe_fetch_complete",
            symbols_requested=len(symbols),
            symbols_fetched=len(result),
            sources=self._sources_used,
        )
        return result

    # ── Source provenance ──────────────────────────────────────────────────

    @property
    def sources_used(self) -> list[str]:
        """List of data sources actually used during this session."""
        return list(self._sources_used)

    def close(self) -> None:
        if self._api:
            self._api.close()

    def __enter__(self) -> "MarketDataClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()


# ─── Factory helper ────────────────────────────────────────────────────────────


def create_client_from_env() -> MarketDataClient:
    """
    Build a MarketDataClient from environment variables.

    Environment variables read:
      DATA_SERVICE_URL         — AlphaForge data-service base URL (Tier 0 Scrapling)
                                 When set, Bhavcopy data is used as the primary source.
      ALPHAFORGE_API_BASE_URL  — AlphaForge Next.js API base URL
                                 (default: http://localhost:3000/api/in)
      DATABASE_URL             — PostgreSQL connection string
      ML_DATA_REQUEST_DELAY    — seconds between requests (default: 0.2)

    When DATA_SERVICE_URL is unset, Tier 0 is skipped and Tier 1 (AlphaForge API) is used first.
    When ALPHAFORGE_API_BASE_URL is unset, the API tier is skipped.
    When DATABASE_URL is unset, the PostgreSQL tier is skipped.
    yfinance fallback is always available (requires yfinance package).
    """
    api_url = os.getenv(
        "ALPHAFORGE_API_BASE_URL",
        os.getenv("NEXTJS_API_BASE_URL", "http://localhost:3000/api/in"),
    )
    db_url = os.getenv("DATABASE_URL")
    delay = float(os.getenv("ML_DATA_REQUEST_DELAY", "0.2"))

    client = MarketDataClient(
        api_base_url=api_url or None,
        database_url=db_url or None,
        request_delay_s=delay,
    )
    logger.info(
        "market_data_client_created",
        scrapling_configured=bool(DATA_SERVICE_URL),
        api_configured=api_url is not None,
        pg_configured=db_url is not None,
        yfinance_fallback=True,
    )
    return client
