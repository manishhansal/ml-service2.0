"""
MSScanner — top-level orchestrator for the Market Structure Scanner.

Pipeline per symbol:
  detect_market_structure → rank_ms_signals

Supported timeframes: "5m", "15m", "1h", "1d"
Per-timeframe swing_lookback overrides are applied before detection.
"""
from __future__ import annotations
import time
from datetime import datetime, timezone
import pandas as pd
import structlog
from src.scanners.market_structure.config import MSConfig
from src.scanners.market_structure.detector import detect_market_structure
from src.scanners.market_structure.scorer import rank_ms_signals
from src.scanners.market_structure.schemas import MSScanResult, MSSignal

_logger = structlog.get_logger(__name__)
SUPPORTED_TIMEFRAMES = frozenset({"5m", "15m", "1h", "1d"})


class MSScanner:
    def __init__(self, config: MSConfig | None = None) -> None:
        self.config = config or MSConfig()

    def scan(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        timeframe: str,
        trend_states: dict[str, object] | None = None,
    ) -> MSScanResult:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Valid values: 5m, 15m, 1h, 1d")

        # Per-timeframe swing_lookback override
        tf_lookback = {
            "5m":  self.config.tf_5m_swing_lookback,
            "15m": self.config.tf_15m_swing_lookback,
            "1h":  self.config.tf_1h_swing_lookback,
            "1d":  self.config.tf_1d_swing_lookback,
        }
        effective_config = self.config.model_copy(
            update={"swing_lookback": tf_lookback[timeframe]}
        )

        start_ts = time.time()
        all_signals: list[MSSignal] = []
        df_by_symbol: dict[str, pd.DataFrame] = {}

        for symbol, df in ohlcv_by_symbol.items():
            try:
                signals = detect_market_structure(df, symbol, timeframe, effective_config)
                all_signals.extend(signals)
                df_by_symbol[symbol] = df
            except Exception as exc:
                _logger.error("ms_scanner_symbol_error", symbol=symbol, error=str(exc))

        all_signals = rank_ms_signals(all_signals, effective_config, df_by_symbol)

        # Compute intraday trading levels
        from src.scanners.intraday import compute_ms_intraday
        from src.scanners.market_structure.schemas import MSSignalType

        intraday_signals: list[MSSignal] = []
        for s in all_signals:
            df = df_by_symbol.get(s.symbol)
            if df is not None and len(df) > 0:
                current_close = float(df["close"].iloc[-1])
                prev_close = df["close"].shift(1)
                tr = pd.concat(
                    [
                        df["high"] - df["low"],
                        (df["high"] - prev_close).abs(),
                        (df["low"] - prev_close).abs(),
                    ],
                    axis=1,
                ).max(axis=1)
                atr = float(tr.rolling(14, min_periods=1).mean().iloc[-1])
                is_bullish = s.signal_type in (MSSignalType.BOS_BULLISH, MSSignalType.CHOCH_BULLISH)
                levels = compute_ms_intraday(is_bullish, s.broken_level, current_close, atr, s.is_choch)
                intraday_signals.append(s.model_copy(update=levels))
            else:
                intraday_signals.append(s)
        all_signals = intraday_signals

        # ── Scanner-specific pre-filter (break_magnitude + invalidated) ───────
        all_signals = [
            s for s in all_signals
            if s.break_magnitude >= self.config.min_break_magnitude and not s.invalidated
        ]

        # ── Quality gate (trend + liquidity + R:R) ────────────────────────────
        from src.scanners.quality import (
            compute_market_direction,
            compute_symbol_trend,
            passes_liquidity_filter,
            passes_quality_filters,
        )
        _market_dir = compute_market_direction(ohlcv_by_symbol)
        _qf: list[MSSignal] = []
        for _s in all_signals:
            _df = ohlcv_by_symbol.get(_s.symbol)
            if _df is None:
                continue
            if not passes_liquidity_filter(_df, self.config.min_daily_value_crore):
                continue
            if self.config.require_trend_alignment:
                _trend = compute_symbol_trend(_df)
                if not passes_quality_filters(_s, _trend, _market_dir, self.config.min_rr_ratio):
                    continue
            elif (_s.intraday_rr_ratio or 0.0) < self.config.min_rr_ratio:
                continue
            _qf.append(_s)
        all_signals = _qf

        _logger.info(
            "ms_scan_complete",
            symbols=len(ohlcv_by_symbol),
            n_signals=len(all_signals),
            elapsed_ms=int((time.time() - start_ts) * 1000),
        )

        return MSScanResult(
            signals=all_signals,
            scanned_at=datetime.now(tz=timezone.utc),
            n_symbols_scanned=len(ohlcv_by_symbol),
        )
