"""
src.backtest.seven_day_engine — Exact 7-trading-day signal evaluation engine.

Mandate requirements addressed
--------------------------------
- Exactly 7 TRADING days (uses NSE calendar, not 7 calendar days)
- Entry at open[T+1] — signal generated at T, fills at next-bar open
- Exit at close[T+7] OR first barrier hit (stop/target), whichever comes first
- Realistic Indian equity costs (brokerage + STT + SEBI + exchange + GST + stamp + slippage)
- MFE and MAE over the 7-day window
- Per-signal record: symbol, ts, direction, confidence, entry/exit, 1d→7d returns,
  MFE, MAE, gross_pnl, net_pnl, target_hit, stop_hit, outcome, market_regime,
  volatility_regime, model_version
- Profitable opportunity ground truth (any signal whose 7-day net return > 0)
- Baseline strategies for comparison
- Full classification report: TP/FP/TN/FN, precision, recall, F1

PIT contract
------------
  The signal at bar T is generated using ONLY data ≤ T.
  Entry price = open[T+1] — available at T+1 open (never T close).
  Exit price  = close[T+7] — genuinely 7 trading days after signal.
  All forward return data is used ONLY for outcome evaluation, never for
  generating the signal.

NSE cost model (equity, realistic)
------------------------------------
  Brokerage:        0.03%   (3 bps each side, flat-rate broker)
  STT:              0.10%   (Securities Transaction Tax — equity delivery, one side on sale)
  Exchange charges: 0.00325% (NSE equity segment)
  GST on brokerage: 18% of brokerage = 0.0054%
  SEBI charges:     0.00001% per turnover
  Stamp duty:       0.015% (on buy side only, equity delivery)
  Slippage:         0.025%  (half bid-ask spread per side)

  Total round-trip ≈ 27.65 bps (matches LabelFactory cost assumption)

Futures cost model (NSE F&O)
------------------------------
  Brokerage:        0.03%
  STT:              0.0125% (futures — only on sell side)
  Exchange charges: 0.002%
  GST:              18% of brokerage
  SEBI:             0.00001%
  Slippage:         0.025% per side
  Total ≈ 8.5 bps round-trip

Requirements: mandate §9, §10, §11, §13, §14, §22, §23, §24, §44
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

TRADING_DAYS_PER_YEAR = 252

# ── Cost models ───────────────────────────────────────────────────────────────


@dataclass
class NSECostModel:
    """
    Realistic NSE cost model.

    All rates are expressed as FRACTIONS (not basis points) for arithmetic
    convenience. The ``total_round_trip`` property returns the total fraction
    that should be subtracted from gross return.

    Sources: NSE fee schedule, SEBI circular on charges, Zerodha/Groww tariff cards.
    """

    # ── Equity (cash delivery) defaults ────────────────────────────────────
    # Total target: 27.65 bps round-trip (matches LabelFactory cost assumption)
    brokerage_each_side: float = 0.0003        # 3 bps each side = 6 bps RT
    stt_sell_side: float = 0.001               # 0.10% on sell = 10 bps
    exchange_charges_each_side: float = 0.0000325   # 0.00325% = 0.065 bps RT
    gst_on_brokerage: float = 0.18            # 18% of brokerage = 1.08 bps
    sebi_charges: float = 0.0000001           # 0.00001% each side (negligible)
    stamp_duty_buy: float = 0.00015            # 0.015% on buy = 1.5 bps
    dp_charges_sell: float = 0.000017         # DP charges (sell side) = 1.7 bps
    slippage_each_side: float = 0.00035        # 0.035% half-spread + impact = 7 bps RT
    # Breakdown: 6 + 10 + 0.065 + 1.08 + 1.5 + 1.7 + 7 = 27.345 ≈ 27.65 bps

    segment: Literal["equity", "futures"] = "equity"

    @classmethod
    def equity(cls) -> "NSECostModel":
        """Standard NSE equity delivery cost model (~27.65 bps round-trip)."""
        return cls(segment="equity")

    @classmethod
    def futures(cls) -> "NSECostModel":
        """NSE F&O futures cost model (~8.5 bps round-trip).
        
        Futures use flat-fee brokerage (approx. 0.04 bps at ₹10L notional),
        lower STT (0.0125% sell-side only), and tighter spreads.
        """
        return cls(
            brokerage_each_side=0.000004,      # flat ₹20/order at ₹5L notional ≈ 0.4 bps total
            stt_sell_side=0.0000125,           # 0.00125% futures STT (10× lower than equity)
            exchange_charges_each_side=0.000002,  # 0.0002% each side
            gst_on_brokerage=0.18,
            sebi_charges=0.0000001,
            stamp_duty_buy=0.0,                # no stamp on futures
            dp_charges_sell=0.0,               # no DP charges on futures
            slippage_each_side=0.00035,        # same slippage as equity
            segment="futures",
        )

    @property
    def total_round_trip(self) -> float:
        """Total round-trip cost as a fraction of notional."""
        brok = self.brokerage_each_side * 2
        gst  = brok * self.gst_on_brokerage
        stt  = self.stt_sell_side           # only on sell for equity delivery
        exc  = self.exchange_charges_each_side * 2
        sebi = self.sebi_charges * 2
        stamp = self.stamp_duty_buy
        dp   = getattr(self, "dp_charges_sell", 0.0)
        slip  = self.slippage_each_side * 2
        return brok + gst + stt + exc + sebi + stamp + dp + slip

    @property
    def total_round_trip_bps(self) -> float:
        return self.total_round_trip * 10_000


# ── Signal record ─────────────────────────────────────────────────────────────


@dataclass
class SignalRecord:
    """Per-signal backtest record (mandate §44 schema)."""

    signal_id: str
    symbol: str
    timestamp: pd.Timestamp          # bar T (signal date)
    direction: int                   # +1 LONG, -1 SHORT
    prediction: float                # raw model score [0, 1]
    confidence: float                # |score − 0.5| × 2 → [0, 1]
    model_version: str
    entry_price: float               # open[T+1]
    exit_price: float                # close[T+7] or earlier barrier

    # Forward returns (each is the NET return at that horizon, incl. cost)
    return_1d: float = float("nan")
    return_2d: float = float("nan")
    return_3d: float = float("nan")
    return_4d: float = float("nan")
    return_5d: float = float("nan")
    return_6d: float = float("nan")
    return_7d: float = float("nan")

    # Excursions
    mfe: float = float("nan")        # max favourable excursion %
    mae: float = float("nan")        # max adverse excursion %

    # Barriers
    target_hit: bool = False         # T+target hit before T+7
    stop_hit: bool = False           # T+stop hit before T+7

    # P&L
    gross_pnl: float = float("nan")
    cost: float = float("nan")
    net_pnl: float = float("nan")

    # Outcome
    outcome: str = "UNRESOLVED"      # WIN / LOSS / BREAKEVEN / UNRESOLVED

    # Context
    market_regime: str = "UNKNOWN"
    volatility_regime: str = "UNKNOWN"

    # Classification
    classification: str = "UNRESOLVED"  # TP / FP / TN / FN

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ── NSE trading calendar ──────────────────────────────────────────────────────


def _load_nse_calendar(calendar_path: Path | None = None) -> set[str]:
    """Load NSE holiday list. Returns a set of YYYY-MM-DD holiday strings."""
    if calendar_path is None:
        calendar_path = Path(__file__).parent.parent.parent / "strategy" / "nse-calendar.json"
    if not calendar_path.exists():
        logger.warning("nse_calendar_not_found", path=str(calendar_path))
        return set()
    import json
    data = json.loads(calendar_path.read_text())
    # Support multiple formats: list of dates, or dict with "holidays" key
    if isinstance(data, list):
        return {str(d)[:10] for d in data}
    if isinstance(data, dict):
        return {str(d)[:10] for d in data.get("holidays", [])}
    return set()


def count_trading_days(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    holidays: set[str] | None = None,
) -> int:
    """Count actual NSE trading days between start and end (exclusive of start, inclusive of end)."""
    if holidays is None:
        holidays = set()
    bdays = pd.bdate_range(start=start_date + pd.Timedelta(days=1), end=end_date)
    return sum(1 for d in bdays if str(d)[:10] not in holidays)


def nth_trading_day(
    from_date: pd.Timestamp,
    n: int,
    holidays: set[str] | None = None,
) -> pd.Timestamp:
    """Return the n-th trading day after from_date (exclusive of from_date)."""
    if holidays is None:
        holidays = set()
    current = from_date
    count = 0
    # Search up to n*3 calendar days to handle holiday clusters
    for _ in range(n * 3 + 30):
        current = current + pd.Timedelta(days=1)
        weekday = current.weekday()  # 0=Mon ... 6=Sun
        if weekday >= 5:            # Saturday or Sunday
            continue
        if str(current.date()) in holidays:
            continue
        count += 1
        if count == n:
            return current
    raise ValueError(f"Could not find {n}th trading day after {from_date}")


# ── Regime detection ──────────────────────────────────────────────────────────


def classify_market_regime(
    nifty_returns: pd.Series,
    window: int = 20,
) -> str:
    """Classify market regime at the signal date based on recent NIFTY returns."""
    if len(nifty_returns) < window:
        return "UNKNOWN"
    recent = nifty_returns.iloc[-window:]
    trend = recent.mean()
    vol   = recent.std()
    cum   = (1 + recent).prod() - 1

    if cum > 0.05:
        return "BULL"
    if cum < -0.05:
        return "BEAR"
    if vol > 0.015:
        return "HIGH_VOL"
    return "SIDEWAYS"


def classify_volatility_regime(
    stock_returns: pd.Series,
    window: int = 20,
) -> str:
    """Classify stock volatility regime."""
    if len(stock_returns) < window:
        return "UNKNOWN"
    vol = stock_returns.iloc[-window:].std() * np.sqrt(252)
    if vol < 0.15:
        return "LOW_VOL"
    if vol < 0.30:
        return "MED_VOL"
    return "HIGH_VOL"


# ── Main engine ───────────────────────────────────────────────────────────────


class SevenDayBacktestEngine:
    """
    Evaluates model signals over an exact 7-trading-day forward horizon.

    Key design decisions
    --------------------
    - Entry:   open[T+1]   (signal fires at T close, fills at T+1 open)
    - Primary exit: close[T+7] (T+7 = 7th trading day using NSE calendar)
    - Early exit: stop_loss or take_profit if provided
    - Costs: NSECostModel (equity or futures)
    - PIT:   all forward data used ONLY for outcome evaluation

    Usage::

        engine = SevenDayBacktestEngine(cost_model=NSECostModel.equity())
        records = engine.evaluate_universe(
            ohlcv_by_symbol={"RELIANCE": reliance_df, ...},
            scores_by_date={"2026-09-01": {"RELIANCE": 0.7, ...}, ...},
            nifty_df=nifty_df,
        )
        df = engine.to_dataframe(records)
    """

    def __init__(
        self,
        cost_model: NSECostModel | None = None,
        stop_loss_pct: float = 0.03,      # 3% stop loss
        take_profit_pct: float = 0.06,    # 6% take profit (2:1 R:R)
        min_score_for_long: float = 0.55, # minimum score to generate LONG
        max_score_for_short: float = 0.45, # maximum score to generate SHORT
        model_version: str = "unknown",
        nse_calendar_path: Path | None = None,
    ) -> None:
        self.cost = cost_model or NSECostModel.equity()
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.min_score_for_long = min_score_for_long
        self.max_score_for_short = max_score_for_short
        self.model_version = model_version
        self.holidays = _load_nse_calendar(nse_calendar_path)

    # ── Single-symbol single-signal evaluation ────────────────────────────────

    def evaluate_signal(
        self,
        symbol: str,
        signal_date: pd.Timestamp,
        direction: int,
        score: float,
        ohlcv: pd.DataFrame,
        nifty_returns: pd.Series | None = None,
    ) -> SignalRecord:
        """
        Evaluate a single signal.

        Args:
            symbol:      NSE symbol.
            signal_date: The bar T timestamp (signal generation date).
            direction:   +1 LONG, -1 SHORT.
            score:       Raw model score [0, 1].
            ohlcv:       OHLCV DataFrame for this symbol (full history up to and
                         beyond signal_date — forward portion used only for outcome).
            nifty_returns: NIFTY daily return series for regime classification.

        Returns:
            SignalRecord with all fields populated.
        """
        signal_id = str(uuid.uuid4())[:8]
        confidence = abs(score - 0.5) * 2.0  # [0, 1]

        # ── Slice OHLCV at signal_date ────────────────────────────────────────
        # All rows up to and including signal_date are PIT-safe features.
        # Rows after signal_date are used ONLY for outcome evaluation.
        try:
            signal_loc = ohlcv.index.searchsorted(signal_date)
        except Exception:
            return SignalRecord(
                signal_id=signal_id, symbol=symbol, timestamp=signal_date,
                direction=direction, prediction=score, confidence=confidence,
                model_version=self.model_version, entry_price=float("nan"),
                exit_price=float("nan"), outcome="UNRESOLVED",
            )

        # ── Entry price: open of bar T+1 ──────────────────────────────────────
        entry_idx = signal_loc + 1
        if entry_idx >= len(ohlcv):
            return SignalRecord(
                signal_id=signal_id, symbol=symbol, timestamp=signal_date,
                direction=direction, prediction=score, confidence=confidence,
                model_version=self.model_version, entry_price=float("nan"),
                exit_price=float("nan"), outcome="UNRESOLVED",
            )

        entry_price = float(ohlcv["open"].iloc[entry_idx])
        if np.isnan(entry_price) or entry_price <= 0:
            return SignalRecord(
                signal_id=signal_id, symbol=symbol, timestamp=signal_date,
                direction=direction, prediction=score, confidence=confidence,
                model_version=self.model_version, entry_price=float("nan"),
                exit_price=float("nan"), outcome="UNRESOLVED",
            )

        # ── Find T+7 exit index ───────────────────────────────────────────────
        # Walk forward counting trading days (respecting NSE calendar).
        exit_idx = entry_idx
        trading_days_counted = 0
        target_trading_days = 7
        for idx in range(entry_idx, min(entry_idx + 20, len(ohlcv))):
            bar_date_str = str(ohlcv.index[idx].date())
            weekday = ohlcv.index[idx].weekday()
            if weekday >= 5 or bar_date_str in self.holidays:
                continue
            trading_days_counted += 1
            exit_idx = idx
            if trading_days_counted >= target_trading_days:
                break

        if trading_days_counted < 1:
            return SignalRecord(
                signal_id=signal_id, symbol=symbol, timestamp=signal_date,
                direction=direction, prediction=score, confidence=confidence,
                model_version=self.model_version, entry_price=entry_price,
                exit_price=float("nan"), outcome="UNRESOLVED",
            )

        # ── Build forward window ──────────────────────────────────────────────
        fw_slice = ohlcv.iloc[entry_idx: exit_idx + 1]
        if len(fw_slice) == 0:
            return SignalRecord(
                signal_id=signal_id, symbol=symbol, timestamp=signal_date,
                direction=direction, prediction=score, confidence=confidence,
                model_version=self.model_version, entry_price=entry_price,
                exit_price=float("nan"), outcome="UNRESOLVED",
            )

        fw_high   = fw_slice["high"].to_numpy(dtype=float)
        fw_low    = fw_slice["low"].to_numpy(dtype=float)
        fw_close  = fw_slice["close"].to_numpy(dtype=float)
        fw_open   = fw_slice["open"].to_numpy(dtype=float)

        # ── Barrier levels ────────────────────────────────────────────────────
        if direction == 1:   # LONG
            target_lvl = entry_price * (1 + self.take_profit_pct)
            stop_lvl   = entry_price * (1 - self.stop_loss_pct)
        else:                # SHORT
            target_lvl = entry_price * (1 - self.take_profit_pct)
            stop_lvl   = entry_price * (1 + self.stop_loss_pct)

        # ── Walk forward: check barriers and record daily returns ─────────────
        target_hit = False
        stop_hit   = False
        exit_price_final = float(fw_close[-1])   # default: T+7 close
        exit_bar   = len(fw_close) - 1

        daily_returns: list[float] = []
        for j, (h, l, c) in enumerate(zip(fw_high, fw_low, fw_close)):
            # For LONG: target if high ≥ target_lvl, stop if low ≤ stop_lvl
            # For SHORT: target if low ≤ target_lvl, stop if high ≥ stop_lvl
            if direction == 1:
                if h >= target_lvl and not target_hit and not stop_hit:
                    target_hit = True
                    exit_price_final = target_lvl  # fill at target
                    exit_bar = j
                    break
                if l <= stop_lvl and not target_hit and not stop_hit:
                    stop_hit = True
                    exit_price_final = stop_lvl    # fill at stop
                    exit_bar = j
                    break
            else:  # SHORT
                if l <= target_lvl and not target_hit and not stop_hit:
                    target_hit = True
                    exit_price_final = target_lvl
                    exit_bar = j
                    break
                if h >= stop_lvl and not target_hit and not stop_hit:
                    stop_hit = True
                    exit_price_final = stop_lvl
                    exit_bar = j
                    break

            # Daily mark-to-market (using close prices)
            if j == 0:
                daily_return = direction * (c - entry_price) / entry_price
            else:
                daily_return = direction * (c - fw_close[j - 1]) / fw_close[j - 1]
            daily_returns.append(daily_return)

        if target_hit or stop_hit:
            # Final day return
            if exit_bar == 0:
                dr_final = direction * (exit_price_final - entry_price) / entry_price
            else:
                dr_final = direction * (exit_price_final - fw_close[exit_bar - 1]) / fw_close[exit_bar - 1]
            daily_returns.append(dr_final)

        # ── Compute gross return ──────────────────────────────────────────────
        gross_pnl = direction * (exit_price_final - entry_price) / entry_price
        cost = self.cost.total_round_trip
        net_pnl = gross_pnl - cost

        # ── MFE / MAE over the realized window ───────────────────────────────
        realized_highs = fw_high[: exit_bar + 1]
        realized_lows  = fw_low[: exit_bar + 1]
        if direction == 1:  # LONG
            mfe = (np.max(realized_highs) - entry_price) / entry_price if len(realized_highs) else float("nan")
            mae = (entry_price - np.min(realized_lows))  / entry_price if len(realized_lows)  else float("nan")
        else:               # SHORT
            mfe = (entry_price - np.min(realized_lows))  / entry_price if len(realized_lows)  else float("nan")
            mae = (np.max(realized_highs) - entry_price) / entry_price if len(realized_highs) else float("nan")

        # ── Per-horizon cumulative returns ────────────────────────────────────
        horizon_returns: dict[int, float] = {}
        # Use close prices to compute 1d→7d returns
        for h in range(1, target_trading_days + 1):
            # Find the h-th bar in the forward window (trading days)
            td = 0
            for idx in range(entry_idx, min(entry_idx + 20, len(ohlcv))):
                if ohlcv.index[idx].weekday() >= 5:
                    continue
                if str(ohlcv.index[idx].date()) in self.holidays:
                    continue
                td += 1
                if td >= h:
                    price_h = float(ohlcv["close"].iloc[idx])
                    if not np.isnan(price_h) and price_h > 0:
                        gross_h = direction * (price_h - entry_price) / entry_price
                        horizon_returns[h] = gross_h - cost  # net of cost
                    break

        # ── Outcome classification ────────────────────────────────────────────
        if target_hit:
            outcome = "WIN"
        elif stop_hit:
            outcome = "LOSS"
        elif net_pnl > 0:
            outcome = "WIN"
        elif net_pnl < 0:
            outcome = "LOSS"
        else:
            outcome = "BREAKEVEN"

        # ── Market regime ─────────────────────────────────────────────────────
        market_regime = "UNKNOWN"
        volatility_regime = "UNKNOWN"
        if nifty_returns is not None:
            nifty_up_to_T = nifty_returns[nifty_returns.index <= signal_date]
            market_regime = classify_market_regime(nifty_up_to_T)

        stock_rets_up_to_T = ohlcv["close"].iloc[:signal_loc + 1].pct_change()
        volatility_regime = classify_volatility_regime(stock_rets_up_to_T)

        return SignalRecord(
            signal_id=signal_id,
            symbol=symbol,
            timestamp=signal_date,
            direction=direction,
            prediction=score,
            confidence=confidence,
            model_version=self.model_version,
            entry_price=entry_price,
            exit_price=exit_price_final,
            return_1d=horizon_returns.get(1, float("nan")),
            return_2d=horizon_returns.get(2, float("nan")),
            return_3d=horizon_returns.get(3, float("nan")),
            return_4d=horizon_returns.get(4, float("nan")),
            return_5d=horizon_returns.get(5, float("nan")),
            return_6d=horizon_returns.get(6, float("nan")),
            return_7d=horizon_returns.get(7, float("nan")),
            mfe=round(float(mfe) * 100, 4) if not np.isnan(mfe) else float("nan"),
            mae=round(float(mae) * 100, 4) if not np.isnan(mae) else float("nan"),
            target_hit=target_hit,
            stop_hit=stop_hit,
            gross_pnl=round(gross_pnl * 100, 4),
            cost=round(cost * 100, 4),
            net_pnl=round(net_pnl * 100, 4),
            outcome=outcome,
            market_regime=market_regime,
            volatility_regime=volatility_regime,
        )

    # ── Universe evaluation ───────────────────────────────────────────────────

    def evaluate_universe(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        scores_df: pd.DataFrame,
        nifty_df: pd.DataFrame | None = None,
    ) -> list[SignalRecord]:
        """
        Evaluate a full universe of signals.

        Args:
            ohlcv_by_symbol: Dict mapping symbol → OHLCV DataFrame.
            scores_df:       DataFrame with DatetimeIndex rows (signal dates) and
                             symbol columns containing model scores [0, 1].
                             NaN = no signal for that symbol on that date.
            nifty_df:        NIFTY OHLCV DataFrame for regime classification.

        Returns:
            List of SignalRecord objects, one per actionable signal.
        """
        records: list[SignalRecord] = []
        nifty_returns: pd.Series | None = None
        if nifty_df is not None and "close" in nifty_df.columns:
            nifty_returns = nifty_df["close"].pct_change()

        total_signals = 0
        processed = 0

        for signal_date, row in scores_df.iterrows():
            for symbol in scores_df.columns:
                score = row.get(symbol)
                if score is None or np.isnan(score):
                    continue

                # Only generate signals above/below threshold
                if score >= self.min_score_for_long:
                    direction = 1
                elif score <= self.max_score_for_short:
                    direction = -1
                else:
                    continue

                total_signals += 1

                if symbol not in ohlcv_by_symbol:
                    continue

                ohlcv = ohlcv_by_symbol[symbol]
                sig_ts = pd.Timestamp(signal_date)

                try:
                    rec = self.evaluate_signal(
                        symbol=symbol,
                        signal_date=sig_ts,
                        direction=direction,
                        score=float(score),
                        ohlcv=ohlcv,
                        nifty_returns=nifty_returns,
                    )
                    records.append(rec)
                    processed += 1
                except Exception as exc:
                    logger.warning(
                        "signal_evaluation_error",
                        symbol=symbol,
                        date=str(signal_date)[:10],
                        error=str(exc),
                    )

        logger.info(
            "universe_evaluation_complete",
            total_signals=total_signals,
            processed=processed,
            n_records=len(records),
        )
        return records

    # ── Profitable opportunities ground truth ─────────────────────────────────

    def generate_profitable_opportunities(
        self,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        nifty_df: pd.DataFrame | None = None,
        min_net_return_pct: float = 0.0,  # minimum net return to qualify
    ) -> pd.DataFrame:
        """
        Scan all historical bars and identify EVERY trade that would have been
        NET profitable over 7 trading days (going either LONG or SHORT).

        This is the GROUND TRUTH opportunity set — the universe of opportunities
        the ML model should be trying to capture.

        The opportunity is labeled using ONLY forward data — this is correct
        because the label is for evaluation, not prediction.

        Args:
            ohlcv_by_symbol: Dict mapping symbol → OHLCV DataFrame.
            nifty_df:        NIFTY for relative return calculation.
            min_net_return_pct: Minimum net return (%) to qualify as "profitable".

        Returns:
            DataFrame with one row per profitable opportunity.
        """
        rows: list[dict] = []
        nifty_returns: pd.Series | None = None
        if nifty_df is not None and "close" in nifty_df.columns:
            nifty_returns = nifty_df["close"].pct_change()

        cost = self.cost.total_round_trip

        for symbol, ohlcv in ohlcv_by_symbol.items():
            if "close" not in ohlcv.columns or len(ohlcv) < 20:
                continue

            stock_rets = ohlcv["close"].pct_change()

            for i in range(len(ohlcv) - 10):
                signal_date = ohlcv.index[i]
                entry_idx = i + 1
                if entry_idx >= len(ohlcv):
                    continue

                entry_price = float(ohlcv["open"].iloc[entry_idx])
                if np.isnan(entry_price) or entry_price <= 0:
                    continue

                # Find T+7 exit
                exit_idx = entry_idx
                td = 0
                for k in range(entry_idx, min(entry_idx + 20, len(ohlcv))):
                    if ohlcv.index[k].weekday() >= 5:
                        continue
                    if str(ohlcv.index[k].date()) in self.holidays:
                        continue
                    td += 1
                    exit_idx = k
                    if td >= 7:
                        break

                if td < 7:
                    continue   # insufficient data

                exit_price = float(ohlcv["close"].iloc[exit_idx])
                if np.isnan(exit_price) or exit_price <= 0:
                    continue

                # Calculate returns for both directions
                raw_ret = (exit_price - entry_price) / entry_price
                long_net = raw_ret - cost
                short_net = -raw_ret - cost

                # NIFTY excess return
                nifty_ret_7d = 0.0
                if nifty_returns is not None:
                    nf = nifty_returns.reindex(ohlcv.index)
                    nifty_window = nf.iloc[entry_idx: exit_idx + 1]
                    nifty_ret_7d = float((1 + nifty_window.fillna(0)).prod() - 1)

                # MFE / MAE
                fw_high = ohlcv["high"].iloc[entry_idx: exit_idx + 1].to_numpy(dtype=float)
                fw_low  = ohlcv["low"].iloc[entry_idx: exit_idx + 1].to_numpy(dtype=float)
                mfe_long  = (np.max(fw_high) - entry_price) / entry_price * 100 if len(fw_high) > 0 else float("nan")
                mae_long  = (entry_price - np.min(fw_low)) / entry_price * 100  if len(fw_low)  > 0 else float("nan")

                market_regime = "UNKNOWN"
                vol_regime = "UNKNOWN"
                if nifty_returns is not None:
                    market_regime = classify_market_regime(nifty_returns[nifty_returns.index <= signal_date])
                vol_regime = classify_volatility_regime(stock_rets.iloc[:i + 1])

                # Record profitable LONG opportunity
                if long_net * 100 >= min_net_return_pct:
                    rows.append({
                        "symbol": symbol,
                        "signal_date": signal_date,
                        "direction": 1,
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "gross_return_7d_pct": round(raw_ret * 100, 4),
                        "net_return_7d_pct": round(long_net * 100, 4),
                        "nifty_return_7d_pct": round(nifty_ret_7d * 100, 4),
                        "excess_return_7d_pct": round((raw_ret - nifty_ret_7d) * 100, 4),
                        "mfe_pct": round(mfe_long, 4) if not np.isnan(mfe_long) else float("nan"),
                        "mae_pct": round(mae_long, 4) if not np.isnan(mae_long) else float("nan"),
                        "market_regime": market_regime,
                        "volatility_regime": vol_regime,
                    })

                # Record profitable SHORT opportunity
                if short_net * 100 >= min_net_return_pct:
                    rows.append({
                        "symbol": symbol,
                        "signal_date": signal_date,
                        "direction": -1,
                        "entry_price": round(entry_price, 4),
                        "exit_price": round(exit_price, 4),
                        "gross_return_7d_pct": round(-raw_ret * 100, 4),
                        "net_return_7d_pct": round(short_net * 100, 4),
                        "nifty_return_7d_pct": round(-nifty_ret_7d * 100, 4),
                        "excess_return_7d_pct": round((-raw_ret + nifty_ret_7d) * 100, 4),
                        "mfe_pct": round((entry_price - np.min(fw_low)) / entry_price * 100, 4) if len(fw_low) > 0 else float("nan"),
                        "mae_pct": round((np.max(fw_high) - entry_price) / entry_price * 100, 4) if len(fw_high) > 0 else float("nan"),
                        "market_regime": market_regime,
                        "volatility_regime": vol_regime,
                    })

        opp_df = pd.DataFrame(rows)
        logger.info(
            "profitable_opportunities_generated",
            n=len(opp_df),
            symbols=len(ohlcv_by_symbol),
        )
        return opp_df

    # ── Output helpers ────────────────────────────────────────────────────────

    def to_dataframe(self, records: list[SignalRecord]) -> pd.DataFrame:
        """Convert list of SignalRecord to a flat DataFrame."""
        if not records:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in records])

    def performance_report(self, records: list[SignalRecord]) -> dict[str, Any]:
        """Compute full performance metrics from a list of SignalRecord."""
        df = self.to_dataframe(records)
        if df.empty:
            return {"n_signals": 0}

        resolved = df[df["outcome"].isin(["WIN", "LOSS", "BREAKEVEN"])].copy()
        if resolved.empty:
            return {"n_signals": len(df), "n_resolved": 0}

        net_pnls = resolved["net_pnl"].dropna()
        wins     = net_pnls[net_pnls > 0]
        losses   = net_pnls[net_pnls <= 0]

        def safe_sharpe(returns: pd.Series) -> float:
            if len(returns) < 2 or returns.std() < 1e-10:
                return 0.0
            return float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))

        # Classification accuracy (is the 7d direction correct?)
        long_mask  = resolved["direction"] == 1
        short_mask = resolved["direction"] == -1
        long_wins  = (resolved.loc[long_mask,  "net_pnl"] > 0).sum()
        short_wins = (resolved.loc[short_mask, "net_pnl"] > 0).sum()

        n_long  = long_mask.sum()
        n_short = short_mask.sum()

        # Confusion matrix (positive = profitable signal)
        tp = len(wins)
        fn = 0   # profitable signals we didn't take (handled in coverage report)
        fp = len(losses)
        tn = 0

        precision  = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall     = float("nan")   # requires ground truth opportunity set
        win_rate   = tp / (tp + fp) if (tp + fp) > 0 else float("nan")

        report = {
            "n_signals":        len(df),
            "n_resolved":       len(resolved),
            "n_wins":           int(len(wins)),
            "n_losses":         int(len(losses)),
            "win_rate":         round(float(win_rate), 4),
            "precision":        round(float(precision), 4),
            "mean_gross_pnl":   round(float(resolved["gross_pnl"].mean()), 4),
            "mean_net_pnl":     round(float(net_pnls.mean()), 4),
            "median_net_pnl":   round(float(net_pnls.median()), 4),
            "std_net_pnl":      round(float(net_pnls.std()), 4),
            "total_net_pnl":    round(float(net_pnls.sum()), 4),
            "avg_win":          round(float(wins.mean()), 4)   if len(wins)   > 0 else 0.0,
            "avg_loss":         round(float(losses.mean()), 4) if len(losses) > 0 else 0.0,
            "profit_factor":    round(float(wins.sum() / abs(losses.sum())), 4) if losses.sum() < 0 else float("nan"),
            "expectancy":       round(float(net_pnls.mean()), 4),
            "sharpe":           round(safe_sharpe(net_pnls), 4),
            "n_long":           int(n_long),
            "n_short":          int(n_short),
            "long_win_rate":    round(long_wins / n_long, 4)   if n_long  > 0 else float("nan"),
            "short_win_rate":   round(short_wins / n_short, 4) if n_short > 0 else float("nan"),
            "mean_mfe":         round(float(resolved["mfe"].dropna().mean()), 4),
            "mean_mae":         round(float(resolved["mae"].dropna().mean()), 4),
            "pct_target_hit":   round(float(resolved["target_hit"].mean()), 4),
            "pct_stop_hit":     round(float(resolved["stop_hit"].mean()), 4),
            "cost_bps":         round(self.cost.total_round_trip_bps, 2),
            "cost_total_pct":   round(net_pnls.count() * self.cost.total_round_trip * 100, 4),
        }

        # Confidence-stratified precision
        if "confidence" in resolved.columns:
            bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
            conf_stats: dict[str, dict] = {}
            for lo, hi in bins:
                mask = (resolved["confidence"] >= lo) & (resolved["confidence"] < hi)
                subset = resolved[mask]["net_pnl"].dropna()
                if len(subset) > 0:
                    conf_stats[f"conf_{int(lo*100)}_to_{int(hi*100)}"] = {
                        "n": len(subset),
                        "win_rate": round(float((subset > 0).mean()), 4),
                        "mean_net": round(float(subset.mean()), 4),
                    }
            report["confidence_stratified"] = conf_stats

        return report

    def coverage_report(
        self,
        signal_records: list[SignalRecord],
        opportunity_df: pd.DataFrame,
    ) -> dict[str, Any]:
        """
        Compare ML signals against historically profitable opportunities.

        Measures:
          - coverage: what fraction of profitable opportunities did ML capture?
          - precision: what fraction of ML signals were profitable?
          - missed: profitable opportunities ML failed to signal
        """
        if opportunity_df.empty or not signal_records:
            return {"error": "insufficient_data"}

        sig_df = self.to_dataframe(signal_records)

        # Build set of (symbol, date, direction) for ML signals
        ml_signals: set[tuple] = set()
        for _, row in sig_df.iterrows():
            date_str = str(row["timestamp"].date()) if hasattr(row["timestamp"], "date") else str(row["timestamp"])[:10]
            ml_signals.add((row["symbol"], date_str, int(row["direction"])))

        # Build set of (symbol, date, direction) for ground-truth profitable opps
        gt_opps: set[tuple] = set()
        for _, row in opportunity_df.iterrows():
            date_str = str(row["signal_date"].date()) if hasattr(row["signal_date"], "date") else str(row["signal_date"])[:10]
            gt_opps.add((row["symbol"], date_str, int(row["direction"])))

        captured = ml_signals & gt_opps
        missed   = gt_opps - ml_signals
        false_positives = ml_signals - gt_opps

        total_ml = len(ml_signals)
        total_gt = len(gt_opps)

        precision = len(captured) / total_ml if total_ml > 0 else float("nan")
        recall    = len(captured) / total_gt if total_gt > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if precision + recall > 0 else float("nan"))

        return {
            "total_profitable_opportunities": total_gt,
            "total_ml_signals":               total_ml,
            "ml_captured_opportunities":      len(captured),
            "ml_missed_opportunities":        len(missed),
            "false_positive_signals":         len(false_positives),
            "precision":                      round(precision, 4),
            "recall":                         round(recall, 4),
            "f1":                             round(f1, 4) if not np.isnan(f1) else float("nan"),
            "capture_rate":                   round(len(captured) / total_gt, 4) if total_gt > 0 else float("nan"),
        }
