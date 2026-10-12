"""
data_loader.py — loads NSE F&O 1d OHLCV data from local parquet files.

Only symbols in ``src.scanners.universe.FO_UNIVERSE`` (NSE indices + F&O stocks)
are loaded.  Non-F&O symbols present in the parquet directory are silently skipped.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import pandas as pd

from src.scanners.universe import is_fo_symbol

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DATA_DIR      = _PROJECT_ROOT / "data" / "1d" / "1d"

_lock:      Lock                     = Lock()
_cache:     dict[str, pd.DataFrame] = {}
_cache_at:  datetime | None          = None
_CACHE_TTL  = 300   # seconds — refresh every 5 minutes


def load_1d_data(force: bool = False) -> dict[str, pd.DataFrame]:
    """
    Return a dict mapping trading symbol → daily OHLCV DataFrame.

    DataFrame index is a tz-aware UTC DatetimeIndex; columns are
    ``open``, ``high``, ``low``, ``close``, ``volume`` (all float64).
    Symbols with fewer than 50 bars are skipped.

    Results are cached for ``_CACHE_TTL`` seconds.  Pass ``force=True``
    to bypass the cache and re-read all parquet files from disk.
    """
    global _cache, _cache_at

    with _lock:
        now = datetime.now(timezone.utc)
        age = (now - _cache_at).total_seconds() if _cache_at else float("inf")

        if not force and age < _CACHE_TTL and _cache:
            return _cache

        if not _DATA_DIR.exists():
            logger.warning("data_loader: parquet directory not found: %s", _DATA_DIR)
            return {}

        dfs: dict[str, pd.DataFrame] = {}
        for pf in sorted(_DATA_DIR.glob("*.parquet")):
            sym = pf.stem
            # Skip non-F&O symbols (e.g. DVR variants, legacy tickers)
            if not is_fo_symbol(sym):
                logger.debug("data_loader: skipping non-F&O symbol %s", sym)
                continue
            try:
                df = pd.read_parquet(pf).drop(columns=["provider"], errors="ignore")
                df = df[["open", "high", "low", "close", "volume"]].dropna()
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
                if len(df) >= 50:
                    dfs[sym] = df
            except Exception as exc:
                logger.debug("data_loader: failed to load %s: %s", sym, exc)

        _cache    = dfs
        _cache_at = now
        logger.info("data_loader: loaded %d symbols from %s", len(dfs), _DATA_DIR)
        return dfs
