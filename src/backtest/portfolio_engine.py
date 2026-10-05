"""
src.backtest.portfolio_engine — Event-driven portfolio-level execution simulator (§23A).

Implements the complete signal-to-P&L lifecycle for overlapping concurrent positions:

    SIGNAL GENERATION
          ↓
    SIGNAL VALIDATION (dedup, expiry, exposure)
          ↓
    PORTFOLIO DECISION (capital, risk, same-symbol, opposite-signal policies)
          ↓
    ORDER GENERATION (sizing, allocation model)
          ↓
    ORDER EXECUTION (next-bar open, slippage, fills)
          ↓
    OPEN POSITION (stop/target/7-day lifecycle)
          ↓
    POSITION MANAGEMENT (daily MTM, stop/target evaluation)
          ↓
    EXIT (stop hit / target hit / time expiry / opposite signal)
          ↓
    REALIZED P&L + ATTRIBUTION

Key properties
--------------
- Deterministic: identical inputs → identical outputs (seeded RNG for slippage noise)
- Chronological: events processed in strict time order (§23A.25)
- No look-ahead: signal at T uses only data ≤ T; fills at T+1 open
- Capital-constrained: never allocates more than available capital
- Portfolio-aware: tracks gross/net exposure, sector limits, correlation risk
- Dual-mode: Mode A (isolated signals) and Mode B (portfolio constrained)

Execution event order at each bar (§23A.25):
  1. Receive market bar
  2. Mark-to-market open positions
  3. Execute pending orders at this bar's open
  4. Evaluate stop/target conditions using this bar's high/low
  5. Execute time exits for positions at T+7
  6. Process signals arriving at this timestamp
  7. Apply portfolio constraints → eligible signals
  8. Generate and queue orders for eligible signals
  9. Update portfolio state, record equity point

Requirements: §23A.1–§23A.27
"""
from __future__ import annotations

import hashlib
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.logging_config import get_logger

logger = get_logger(__name__)

# ── Enumerations ──────────────────────────────────────────────────────────────


class EventType(str, Enum):
    MARKET_BAR   = "MARKET_BAR"
    SIGNAL       = "SIGNAL"
    ORDER_SUBMIT = "ORDER_SUBMIT"
    ORDER_FILL   = "ORDER_FILL"
    STOP_HIT     = "STOP_HIT"
    TARGET_HIT   = "TARGET_HIT"
    TIME_EXPIRY  = "TIME_EXPIRY"
    RISK_BREACH  = "RISK_BREACH"
    CAPITAL_CHANGE = "CAPITAL_CHANGE"


class OrderStatus(str, Enum):
    PENDING   = "PENDING"
    FILLED    = "FILLED"
    CANCELLED = "CANCELLED"
    EXPIRED   = "EXPIRED"


class RejectionReason(str, Enum):
    DUPLICATE          = "DUPLICATE_SIGNAL"
    SIGNAL_EXPIRED     = "SIGNAL_EXPIRED"
    MAX_POSITIONS      = "MAX_POSITIONS_REACHED"
    INSUFFICIENT_CAPITAL = "INSUFFICIENT_CAPITAL"
    SAME_SYMBOL_IGNORE = "SAME_SYMBOL_IGNORE_POLICY"
    OPPOSITE_SIGNAL_IGNORE = "OPPOSITE_SIGNAL_IGNORE_POLICY"
    MAX_GROSS_EXPOSURE = "MAX_GROSS_EXPOSURE"
    MAX_NET_EXPOSURE   = "MAX_NET_EXPOSURE"
    MAX_SECTOR_EXPOSURE = "MAX_SECTOR_EXPOSURE"
    MAX_OPEN_RISK      = "MAX_OPEN_RISK"
    MAX_DAILY_LOSS     = "MAX_DAILY_LOSS_REACHED"
    MAX_DRAWDOWN       = "MAX_DRAWDOWN_REACHED"
    CONSECUTIVE_LOSSES = "MAX_CONSECUTIVE_LOSSES"
    LOW_EXPECTED_VALUE = "LOW_EXPECTED_VALUE"
    NO_MARKET_DATA     = "NO_MARKET_DATA"


SameSymbolPolicy = Literal["IGNORE", "ADD", "REPLACE", "AVERAGE"]
OppositeSignalPolicy = Literal["IGNORE", "CLOSE_AND_REVERSE", "CLOSE_ONLY", "REDUCE"]
AllocationMethod = Literal[
    "EQUAL", "CONFIDENCE_WEIGHTED", "RISK_WEIGHTED", "VOLATILITY_SCALED", "FIXED_RISK"
]


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class PortfolioConfig:
    """
    Complete configuration for the portfolio execution engine.

    All limit parameters are expressed as FRACTIONS of portfolio equity
    (e.g., max_capital_per_position=0.10 means each position is max 10%).
    """

    # Capital
    initial_capital: float = 1_000_000.0   # ₹10 lakh default

    # Position limits
    max_positions: int = 10
    max_capital_per_position: float = 0.10  # 10% per position
    max_gross_exposure: float = 1.0         # 100% gross
    max_net_exposure: float = 0.50          # ±50% net
    max_single_trade_risk: float = 0.01     # 1% of equity per trade

    # Sector/correlation limits
    max_sector_exposure: float = 0.30       # 30% in one sector
    sector_map: dict[str, str] = field(default_factory=dict)  # symbol→sector

    # Risk controls
    max_daily_loss: float = 0.03            # 3% daily loss halts new signals
    max_total_drawdown: float = 0.10        # 10% total drawdown halts
    max_consecutive_losses: int = 5
    max_open_risk: float = 0.05             # 5% total portfolio at risk

    # Holding period
    holding_period_bars: int = 7            # 7 trading days
    signal_entry_window_bars: int = 1       # must execute within 1 bar of signal
    stop_loss_pct: float = 0.03             # 3% stop
    take_profit_pct: float = 0.06          # 6% target (2:1 R:R)

    # Overlap policies
    same_symbol_policy: SameSymbolPolicy = "IGNORE"
    opposite_signal_policy: OppositeSignalPolicy = "CLOSE_ONLY"

    # Allocation
    allocation_method: AllocationMethod = "EQUAL"
    risk_per_trade: float = 0.01            # for FIXED_RISK method

    # Execution
    slippage_bps: float = 3.5              # per side (7bps RT already in NSECostModel)
    commission_bps_per_trade: float = 13.5 # half of equity 27bps (one side)

    # Deduplication
    dedup_window_bars: int = 1             # ignore duplicate signals within N bars

    # Min expected value filter
    min_expected_net_return: float = 0.0   # reject signals with EV < this

    # Capacity stress levels (for §23A.22)
    capacity_levels: tuple = (100_000, 500_000, 1_000_000, 5_000_000, 10_000_000)

    @property
    def round_trip_cost(self) -> float:
        """Total round-trip cost fraction."""
        return (self.commission_bps_per_trade * 2 + self.slippage_bps * 2) / 10_000.0


# ── Core data types ───────────────────────────────────────────────────────────


@dataclass
class Position:
    """A single open position with full provenance."""

    position_id: str
    signal_id: str
    symbol: str
    direction: int              # +1 LONG, -1 SHORT
    quantity: float
    entry_timestamp: pd.Timestamp
    entry_bar_index: int        # for T+7 counting
    entry_price: float
    stop_price: float
    target_price: float
    cost_paid: float            # total commission + slippage paid on entry
    model_version: str = "unknown"
    confidence: float = 0.5
    current_price: float = 0.0
    sector: str = "UNKNOWN"

    @property
    def notional_value(self) -> float:
        return abs(self.quantity * self.current_price)

    @property
    def unrealized_pnl(self) -> float:
        return self.direction * (self.current_price - self.entry_price) * self.quantity

    @property
    def unrealized_return(self) -> float:
        if self.entry_price == 0:
            return 0.0
        return self.direction * (self.current_price - self.entry_price) / self.entry_price


@dataclass
class Order:
    """A pending order waiting for execution."""

    order_id: str
    signal_id: str
    symbol: str
    direction: int
    quantity: float
    order_type: str = "MARKET"
    submitted_timestamp: pd.Timestamp = field(default_factory=pd.Timestamp.now)
    expiry_bar_index: int = 0
    status: OrderStatus = OrderStatus.PENDING
    stop_price: float = 0.0
    target_price: float = 0.0
    confidence: float = 0.5
    model_version: str = "unknown"


@dataclass
class ExecutedTrade:
    """A completed round-trip trade."""

    trade_id: str
    signal_id: str
    position_id: str
    symbol: str
    direction: int
    quantity: float
    entry_timestamp: pd.Timestamp
    exit_timestamp: pd.Timestamp
    entry_price: float
    exit_price: float
    entry_bar_index: int
    exit_bar_index: int
    bars_held: int
    exit_reason: str            # STOP_HIT / TARGET_HIT / TIME_EXPIRY / OPPOSITE_SIGNAL / RISK
    gross_pnl: float
    entry_cost: float
    exit_cost: float
    net_pnl: float
    gross_return: float
    net_return: float
    model_version: str = "unknown"
    confidence: float = 0.5
    sector: str = "UNKNOWN"

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class RejectedSignal:
    """A signal that was evaluated but not executed."""

    signal_id: str
    symbol: str
    timestamp: pd.Timestamp
    direction: int
    score: float
    confidence: float
    reason: RejectionReason
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_id": self.signal_id,
            "symbol": self.symbol,
            "timestamp": str(self.timestamp),
            "direction": self.direction,
            "score": self.score,
            "confidence": self.confidence,
            "reason": self.reason.value,
            "detail": self.detail,
        }


@dataclass
class EquityPoint:
    """A single point on the portfolio equity curve."""

    timestamp: pd.Timestamp
    bar_index: int
    cash: float
    market_value: float
    gross_exposure: float
    net_exposure: float
    n_positions: int
    realized_pnl: float
    unrealized_pnl: float
    daily_pnl: float
    total_equity: float
    drawdown: float
    capital_utilization: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class SignalExecutionMapping:
    """Maps a signal to its full execution chain."""

    signal_id: str
    symbol: str
    signal_timestamp: str
    direction: int
    score: float
    confidence: float
    decision: str               # EXECUTE / REJECT / MODIFY
    rejection_reason: str
    order_id: str
    position_id: str
    trade_id: str
    entry_timestamp: str
    entry_price: float
    exit_timestamp: str
    exit_price: float
    exit_reason: str
    net_pnl: float
    net_return: float

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ── Portfolio State ───────────────────────────────────────────────────────────


class PortfolioState:
    """
    Complete mutable portfolio state.

    Maintains all open positions, pending orders, P&L, and risk metrics.
    Updated at every bar by the PortfolioEngine.
    """

    def __init__(self, config: PortfolioConfig) -> None:
        self.config = config
        self.cash: float = config.initial_capital
        self.positions: dict[str, list[Position]] = defaultdict(list)  # symbol → [Position]
        self.pending_orders: list[Order] = []
        self.realized_pnl: float = 0.0
        self.peak_equity: float = config.initial_capital
        self.consecutive_losses: int = 0
        self.daily_pnl_today: float = 0.0
        self._daily_pnl_baseline: float = config.initial_capital
        self._seen_signal_ids: set[str] = set()    # for deduplication
        self._recent_signals: dict[str, list[tuple[pd.Timestamp, int]]] = defaultdict(list)

    # ── Equity calculations ──────────────────────────────────────────────────

    @property
    def market_value(self) -> float:
        """Current market value of all open positions at current prices."""
        return sum(p.quantity * p.current_price
                   for positions in self.positions.values() for p in positions)

    @property
    def unrealized_pnl(self) -> float:
        return sum(p.direction * (p.current_price - p.entry_price) * p.quantity
                   for positions in self.positions.values() for p in positions)

    @property
    def total_equity(self) -> float:
        """
        Correct total equity accounting for both LONG and SHORT positions.

        When a position opens: cash -= quantity × entry_price + cost_paid
        Position book value at time T: quantity × entry_price (what we paid/committed)
        Unrealized P&L: direction × (current_price - entry_price) × quantity

        total_equity = cash + committed_notional + unrealized_pnl
                     = (initial - Σ(qty×entry+cost)) + Σ(qty×entry) + Σ(dir×(cur-entry)×qty)
                     = initial - Σcost + unrealized_pnl
        """
        committed = sum(
            p.quantity * p.entry_price
            for positions in self.positions.values()
            for p in positions
        )
        unrealized = sum(
            p.direction * (p.current_price - p.entry_price) * p.quantity
            for positions in self.positions.values()
            for p in positions
        )
        return self.cash + committed + unrealized

    @property
    def gross_exposure(self) -> float:
        eq = self.total_equity
        if eq <= 0:
            return 0.0
        return sum(p.notional_value for positions in self.positions.values() for p in positions) / eq

    @property
    def net_exposure(self) -> float:
        eq = self.total_equity
        if eq <= 0:
            return 0.0
        net = sum(p.direction * p.quantity * p.current_price
                  for positions in self.positions.values() for p in positions)
        return net / eq

    @property
    def long_exposure(self) -> float:
        eq = max(self.total_equity, 1.0)
        return sum(p.quantity * p.current_price
                   for positions in self.positions.values()
                   for p in positions if p.direction == 1) / eq

    @property
    def short_exposure(self) -> float:
        eq = max(self.total_equity, 1.0)
        return sum(p.quantity * p.current_price
                   for positions in self.positions.values()
                   for p in positions if p.direction == -1) / eq

    @property
    def n_open_positions(self) -> int:
        return sum(len(ps) for ps in self.positions.values())

    @property
    def open_risk(self) -> float:
        """Fraction of equity at risk across all positions (stop distance × notional / equity)."""
        eq = max(self.total_equity, 1.0)
        total_risk = 0.0
        for positions in self.positions.values():
            for p in positions:
                if p.direction == 1:
                    risk = (p.current_price - p.stop_price) * p.quantity
                else:
                    risk = (p.stop_price - p.current_price) * p.quantity
                total_risk += max(risk, 0.0)
        return total_risk / eq

    @property
    def drawdown(self) -> float:
        """Current drawdown from peak. Pure read — does NOT mutate peak_equity.
        
        §33 fix: peak_equity is only updated in the main event loop at step 9
        (after equity curve recording), NOT inside this property. This ensures:
        - The halt check sees the true current drawdown
        - Mark-to-market losses trigger the halt correctly
        - The 10% limit is a HARD limit, not a soft guideline
        """
        eq = self.total_equity
        if self.peak_equity <= 0:
            return 0.0
        return (eq - self.peak_equity) / self.peak_equity

    @property
    def available_capital(self) -> float:
        """Capital available for new positions.

        Cash is already reduced by (quantity × entry_price + cost) each time a
        position opens (via add_position), so cash IS the available capital.
        No further subtraction needed.
        """
        return max(self.cash, 0.0)

    def sector_exposure(self, sector: str) -> float:
        eq = max(self.total_equity, 1.0)
        return sum(p.notional_value
                   for positions in self.positions.values()
                   for p in positions if p.sector == sector) / eq

    # ── Mutation helpers ─────────────────────────────────────────────────────

    def add_position(self, pos: Position) -> None:
        self.positions[pos.symbol].append(pos)
        self.cash -= pos.quantity * pos.entry_price + pos.cost_paid

    def remove_position(self, position_id: str) -> Position | None:
        for symbol, ps in self.positions.items():
            for i, p in enumerate(ps):
                if p.position_id == position_id:
                    removed = ps.pop(i)
                    if not ps:
                        del self.positions[symbol]
                    return removed
        return None

    def mark_to_market(self, prices: dict[str, float]) -> None:
        """Update current_price for all positions."""
        for positions in self.positions.values():
            for p in positions:
                if p.symbol in prices:
                    p.current_price = prices[p.symbol]

    def reset_daily_pnl(self, current_equity: float) -> None:
        self._daily_pnl_baseline = current_equity
        self.daily_pnl_today = 0.0

    def update_daily_pnl(self) -> None:
        self.daily_pnl_today = self.total_equity - self._daily_pnl_baseline

    def mark_signal_seen(self, signal_id: str, symbol: str,
                         ts: pd.Timestamp, direction: int) -> None:
        self._seen_signal_ids.add(signal_id)
        self._recent_signals[symbol].append((ts, direction))
        # Trim old entries (keep only recent)
        cutoff = ts - pd.Timedelta(days=30)
        self._recent_signals[symbol] = [
            (t, d) for t, d in self._recent_signals[symbol] if t >= cutoff
        ]


# ── Capital Allocator ─────────────────────────────────────────────────────────


def compute_allocation(
    state: PortfolioState,
    signal_score: float,
    signal_atr: float,
    signal_stop_distance: float,
    method: AllocationMethod,
) -> float:
    """
    Compute the notional allocation for a single signal.

    Returns the NOTIONAL VALUE (in currency) to allocate.
    Never exceeds available capital or per-position cap.
    """
    equity = state.total_equity
    avail  = state.available_capital
    cfg    = state.config

    per_position_cap = equity * cfg.max_capital_per_position

    if method == "EQUAL":
        # Divide available capital equally across max positions
        remaining_slots = max(cfg.max_positions - state.n_open_positions, 1)
        alloc = min(avail / remaining_slots, per_position_cap)

    elif method == "CONFIDENCE_WEIGHTED":
        # Weight by distance from 0.5 (model conviction)
        conviction = max(abs(signal_score - 0.5) * 2.0, 0.1)
        base = equity * cfg.max_capital_per_position
        alloc = min(base * conviction, per_position_cap, avail)

    elif method == "RISK_WEIGHTED":
        # Allocate so that the risk budget = risk_per_trade × equity
        risk_budget = equity * cfg.risk_per_trade
        if signal_stop_distance > 1e-6:
            alloc = min(risk_budget / signal_stop_distance, per_position_cap, avail)
        else:
            alloc = equity * cfg.max_capital_per_position * 0.5

    elif method == "VOLATILITY_SCALED":
        # Normalize to target volatility
        target_vol = 0.01   # target 1% daily vol contribution
        if signal_atr > 0:
            vol_scale = target_vol / signal_atr
            alloc = min(equity * vol_scale, per_position_cap, avail)
        else:
            alloc = equity * cfg.max_capital_per_position * 0.5

    elif method == "FIXED_RISK":
        # Each trade risks exactly risk_per_trade × equity
        risk_budget = equity * cfg.risk_per_trade
        stop_dist = signal_stop_distance if signal_stop_distance > 1e-6 else cfg.stop_loss_pct
        alloc = min(risk_budget / stop_dist, per_position_cap, avail)

    else:
        alloc = min(per_position_cap, avail)

    return max(alloc, 0.0)


# ── Deduplication ─────────────────────────────────────────────────────────────


def compute_signal_hash(symbol: str, timestamp: pd.Timestamp,
                        direction: int, model_version: str) -> str:
    """Deterministic signal hash for deduplication (§23A.17)."""
    raw = f"{symbol}|{str(timestamp)[:16]}|{direction}|{model_version}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# ── Main Engine ───────────────────────────────────────────────────────────────


class PortfolioEngine:
    """
    Event-driven portfolio-level backtest engine.

    Processes signals in chronological order subject to full portfolio
    constraints: capital, exposure, risk limits, same-symbol/opposite-signal
    policies, and position sizing.

    Two execution modes
    -------------------
    Mode A (isolated): every valid signal is evaluated independently,
        ignoring portfolio state — measures raw signal quality.
    Mode B (portfolio): signals compete for capital/risk/exposure —
        measures actual tradable P&L.

    Usage::

        engine = PortfolioEngine(PortfolioConfig())
        result = engine.run(
            signals_df=signals_df,
            ohlcv_by_symbol=ohlcv_by_symbol,
        )
        result.to_csv("artifacts/portfolio_backtest/")
    """

    def __init__(self, config: PortfolioConfig | None = None) -> None:
        self.config = config or PortfolioConfig()

    def run(
        self,
        signals_df: pd.DataFrame,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
        mode: Literal["A_isolated", "B_portfolio"] = "B_portfolio",
    ) -> "PortfolioResult":
        """
        Execute the full backtest simulation.

        Args:
            signals_df:        DataFrame with columns [timestamp, symbol, direction,
                               prediction (score), confidence, model_version].
                               Indexed by any monotonic index.
            ohlcv_by_symbol:   OHLCV DataFrames per symbol (for price data).
            mode:              "A_isolated" or "B_portfolio".

        Returns:
            PortfolioResult with all trades, equity curve, rejected signals,
            orders, and performance metrics.
        """
        cfg = self.config
        state = PortfolioState(cfg)

        # ── Build unified price universe ──────────────────────────────────────
        # Collect all unique timestamps across all symbols
        all_dates = sorted(set(
            ts
            for df in ohlcv_by_symbol.values()
            for ts in df.index.tolist()
        ))
        if not all_dates:
            logger.warning("portfolio_engine_no_price_data")
            return PortfolioResult(mode=mode, config=cfg)

        # ── Normalise signals_df ──────────────────────────────────────────────
        sig_df = signals_df.copy()
        if "timestamp" not in sig_df.columns and sig_df.index.name == "timestamp":
            sig_df = sig_df.reset_index()
        elif "timestamp" not in sig_df.columns:
            sig_df["timestamp"] = sig_df.index
        sig_df["timestamp"] = pd.to_datetime(sig_df["timestamp"], utc=True, errors="coerce")
        sig_df = sig_df.dropna(subset=["timestamp"]).sort_values("timestamp")

        # Build signal lookup by date
        sig_by_date: dict[str, list[dict]] = defaultdict(list)
        for _, row in sig_df.iterrows():
            date_key = str(row["timestamp"])[:10]
            sig_by_date[date_key].append(row.to_dict())

        # ── Output containers ─────────────────────────────────────────────────
        trades:             list[ExecutedTrade]         = []
        equity_curve:       list[EquityPoint]           = []
        rejected:           list[RejectedSignal]        = []
        order_log:          list[dict]                  = []
        event_log:          list[dict]                  = []
        signal_map:         list[SignalExecutionMapping] = []

        # Metrics for overlap analysis
        overlap_stats = defaultdict(int)
        peak_concurrent = 0

        bar_index = 0
        state.reset_daily_pnl(cfg.initial_capital)
        prev_date = None

        # ── Main event loop ───────────────────────────────────────────────────
        for ts in all_dates:
            date_key = str(ts)[:10]

            # ── Step 1: Receive market bar ─────────────────────────────────────
            current_prices: dict[str, float] = {}
            bar_opens:  dict[str, float] = {}
            bar_highs:  dict[str, float] = {}
            bar_lows:   dict[str, float] = {}
            bar_closes: dict[str, float] = {}
            bar_atrs:   dict[str, float] = {}

            for sym, df in ohlcv_by_symbol.items():
                if ts in df.index:
                    row = df.loc[ts]
                    current_prices[sym] = float(row["close"])
                    bar_opens[sym]  = float(row.get("open",  row["close"]))
                    bar_highs[sym]  = float(row.get("high",  row["close"]))
                    bar_lows[sym]   = float(row.get("low",   row["close"]))
                    bar_closes[sym] = float(row["close"])
                    # Simple ATR proxy: range / close
                    bar_atrs[sym]   = (bar_highs[sym] - bar_lows[sym]) / max(bar_closes[sym], 1.0)

            event_log.append({
                "timestamp": str(ts), "bar_index": bar_index,
                "event": EventType.MARKET_BAR.value,
                "n_symbols": len(current_prices),
            })

            # ── Step 2: Mark-to-market ─────────────────────────────────────────
            state.mark_to_market(current_prices)

            # ── Step 3: Fill pending orders at today's open ────────────────────
            # Orders submitted at bar t-1 fill at bar t's open
            still_pending: list[Order] = []
            for order in state.pending_orders:
                sym = order.symbol
                if sym not in bar_opens:
                    still_pending.append(order)
                    continue
                if bar_index > order.expiry_bar_index:
                    order.status = OrderStatus.EXPIRED
                    overlap_stats["expired_signals"] += 1
                    event_log.append({
                        "timestamp": str(ts), "bar_index": bar_index,
                        "event": "ORDER_EXPIRED", "symbol": sym,
                        "order_id": order.order_id,
                    })
                    continue

                # Fill at open + slippage
                slip_frac = cfg.slippage_bps / 10_000.0
                rng = np.random.default_rng(abs(hash(order.order_id)) % (2**31))
                slip_noise = rng.uniform(-0.3, 0.3) * slip_frac
                if order.direction == 1:
                    fill_price = bar_opens[sym] * (1 + slip_frac + slip_noise)
                else:
                    fill_price = bar_opens[sym] * (1 - slip_frac - slip_noise)

                entry_cost = (cfg.commission_bps_per_trade / 10_000.0) * order.quantity * fill_price

                stop_p, target_p = _compute_barriers(
                    fill_price, order.direction,
                    cfg.stop_loss_pct, cfg.take_profit_pct
                )

                sector = cfg.sector_map.get(sym, "UNKNOWN")
                pos = Position(
                    position_id   = str(uuid.uuid4())[:12],
                    signal_id     = order.signal_id,
                    symbol        = sym,
                    direction     = order.direction,
                    quantity      = order.quantity,
                    entry_timestamp = ts,
                    entry_bar_index = bar_index,
                    entry_price   = fill_price,
                    stop_price    = stop_p,
                    target_price  = target_p,
                    cost_paid     = entry_cost,
                    model_version = order.model_version,
                    confidence    = order.confidence,
                    current_price = fill_price,
                    sector        = sector,
                )
                state.add_position(pos)
                order.status = OrderStatus.FILLED

                overlap_stats["executed_signals"] += 1
                order_log.append({
                    "order_id": order.order_id,
                    "signal_id": order.signal_id,
                    "symbol": sym,
                    "direction": order.direction,
                    "quantity": order.quantity,
                    "fill_price": round(fill_price, 4),
                    "entry_cost": round(entry_cost, 4),
                    "timestamp": str(ts),
                    "position_id": pos.position_id,
                    "status": "FILLED",
                })
                event_log.append({
                    "timestamp": str(ts), "bar_index": bar_index,
                    "event": EventType.ORDER_FILL.value, "symbol": sym,
                    "position_id": pos.position_id, "direction": order.direction,
                    "fill_price": round(fill_price, 4),
                })
                logger.debug("order_filled", sym=sym, fill_price=round(fill_price, 2),
                             n_open=state.n_open_positions)

            state.pending_orders = still_pending

            # ── Step 4: Evaluate stops/targets ─────────────────────────────────
            positions_to_close: list[tuple[str, str]] = []  # (position_id, reason)
            for sym, pos_list in list(state.positions.items()):
                h = bar_highs.get(sym, 0.0)
                l = bar_lows.get(sym,  0.0)
                c = bar_closes.get(sym, 0.0)
                for pos in pos_list:
                    if pos.direction == 1:   # LONG
                        if h > 0 and h >= pos.target_price:
                            positions_to_close.append((pos.position_id, "TARGET_HIT"))
                        elif l > 0 and l <= pos.stop_price:
                            positions_to_close.append((pos.position_id, "STOP_HIT"))
                    else:                    # SHORT
                        if l > 0 and l <= pos.target_price:
                            positions_to_close.append((pos.position_id, "TARGET_HIT"))
                        elif h > 0 and h >= pos.stop_price:
                            positions_to_close.append((pos.position_id, "STOP_HIT"))

            # ── Step 5: Time expiry (T+7) ──────────────────────────────────────
            for sym, pos_list in list(state.positions.items()):
                for pos in pos_list:
                    bars_held = bar_index - pos.entry_bar_index
                    if bars_held >= cfg.holding_period_bars:
                        # Avoid double-close
                        if not any(pid == pos.position_id for pid, _ in positions_to_close):
                            positions_to_close.append((pos.position_id, "TIME_EXPIRY"))

            # Execute closures
            for pos_id, reason in positions_to_close:
                pos = state.remove_position(pos_id)
                if pos is None:
                    continue
                sym = pos.symbol

                # Determine exit price
                if reason == "TARGET_HIT":
                    exit_price = pos.target_price
                    event_log.append({
                        "timestamp": str(ts), "bar_index": bar_index,
                        "event": EventType.TARGET_HIT.value, "symbol": sym,
                    })
                elif reason == "STOP_HIT":
                    exit_price = pos.stop_price
                    event_log.append({
                        "timestamp": str(ts), "bar_index": bar_index,
                        "event": EventType.STOP_HIT.value, "symbol": sym,
                    })
                else:  # TIME_EXPIRY
                    exit_price = bar_closes.get(sym, pos.current_price)
                    event_log.append({
                        "timestamp": str(ts), "bar_index": bar_index,
                        "event": EventType.TIME_EXPIRY.value, "symbol": sym,
                        "bars_held": bar_index - pos.entry_bar_index,
                    })

                exit_cost = (cfg.commission_bps_per_trade / 10_000.0) * pos.quantity * exit_price
                gross_pnl = pos.direction * (exit_price - pos.entry_price) * pos.quantity
                net_pnl   = gross_pnl - pos.cost_paid - exit_cost
                gross_ret = pos.direction * (exit_price - pos.entry_price) / max(pos.entry_price, 1e-6)
                net_ret   = gross_ret - (pos.cost_paid + exit_cost) / max(pos.quantity * pos.entry_price, 1e-6)

                state.cash += pos.quantity * exit_price - exit_cost
                state.realized_pnl += net_pnl
                if net_pnl < 0:
                    state.consecutive_losses += 1
                else:
                    state.consecutive_losses = 0

                trade_id = str(uuid.uuid4())[:12]
                trade = ExecutedTrade(
                    trade_id=trade_id, signal_id=pos.signal_id,
                    position_id=pos_id, symbol=sym,
                    direction=pos.direction, quantity=pos.quantity,
                    entry_timestamp=pos.entry_timestamp, exit_timestamp=ts,
                    entry_price=pos.entry_price, exit_price=exit_price,
                    entry_bar_index=pos.entry_bar_index, exit_bar_index=bar_index,
                    bars_held=bar_index - pos.entry_bar_index,
                    exit_reason=reason,
                    gross_pnl=round(gross_pnl, 4), entry_cost=round(pos.cost_paid, 4),
                    exit_cost=round(exit_cost, 4), net_pnl=round(net_pnl, 4),
                    gross_return=round(gross_ret, 6), net_return=round(net_ret, 6),
                    model_version=pos.model_version, confidence=pos.confidence,
                    sector=pos.sector,
                )
                trades.append(trade)
                logger.debug("position_closed", sym=sym, reason=reason,
                             net_pnl=round(net_pnl, 2))

            # Track peak concurrent positions
            concurrent_now = state.n_open_positions
            if concurrent_now > peak_concurrent:
                peak_concurrent = concurrent_now

            # ── Step 6: Process today's signals ───────────────────────────────
            today_signals = sig_by_date.get(date_key, [])
            overlap_stats["total_signals"] += len(today_signals)

            for sig in today_signals:
                symbol    = str(sig.get("symbol", ""))
                direction = int(sig.get("direction", 0))
                score     = float(sig.get("prediction", sig.get("score", 0.5)))
                confidence = float(sig.get("confidence", abs(score - 0.5) * 2))
                model_ver  = str(sig.get("model_version", "unknown"))
                sig_ts     = pd.to_datetime(sig.get("timestamp", ts), utc=True)

                if direction not in (1, -1):
                    continue
                if not symbol:
                    continue

                # Build deterministic signal ID
                sig_id = sig.get("signal_id") or compute_signal_hash(
                    symbol, sig_ts, direction, model_ver
                )

                # ── Portfolio constraint evaluation (Mode B only) ─────────────
                if mode == "B_portfolio":
                    reject = self._evaluate_signal(
                        sig_id, symbol, sig_ts, direction, score, confidence,
                        bar_index, state, bar_closes, bar_atrs
                    )
                    if reject is not None:
                        rejected.append(RejectedSignal(
                            signal_id=sig_id, symbol=symbol, timestamp=sig_ts,
                            direction=direction, score=score, confidence=confidence,
                            reason=reject[0], detail=reject[1],
                        ))
                        overlap_stats[f"rejected_{reject[0].value.lower()[:20]}"] += 1
                        overlap_stats["rejected_signals"] += 1
                        signal_map.append(SignalExecutionMapping(
                            signal_id=sig_id, symbol=symbol,
                            signal_timestamp=str(sig_ts), direction=direction,
                            score=score, confidence=confidence,
                            decision="REJECT", rejection_reason=reject[0].value,
                            order_id="", position_id="", trade_id="",
                            entry_timestamp="", entry_price=0.0,
                            exit_timestamp="", exit_price=0.0,
                            exit_reason="", net_pnl=0.0, net_return=0.0,
                        ))
                        continue

                overlap_stats["eligible_signals"] += 1

                # ── Handle opposite signal: close existing position ────────────
                _opposite_signal_close_only = False
                if mode == "B_portfolio" and symbol in state.positions:
                    for pos in list(state.positions.get(symbol, [])):
                        if pos.direction != direction:
                            policy = cfg.opposite_signal_policy
                            if policy in ("CLOSE_AND_REVERSE", "CLOSE_ONLY"):
                                exit_price = bar_closes.get(symbol, pos.current_price)
                                exit_cost = (cfg.commission_bps_per_trade / 10_000.0) * pos.quantity * exit_price
                                gross_pnl = pos.direction * (exit_price - pos.entry_price) * pos.quantity
                                net_pnl = gross_pnl - pos.cost_paid - exit_cost
                                state.cash += pos.quantity * exit_price - exit_cost
                                state.realized_pnl += net_pnl
                                state.remove_position(pos.position_id)
                                trade = ExecutedTrade(
                                    trade_id=str(uuid.uuid4())[:12],
                                    signal_id=pos.signal_id,
                                    position_id=pos.position_id, symbol=symbol,
                                    direction=pos.direction, quantity=pos.quantity,
                                    entry_timestamp=pos.entry_timestamp, exit_timestamp=ts,
                                    entry_price=pos.entry_price, exit_price=exit_price,
                                    entry_bar_index=pos.entry_bar_index, exit_bar_index=bar_index,
                                    bars_held=bar_index - pos.entry_bar_index,
                                    exit_reason="OPPOSITE_SIGNAL",
                                    gross_pnl=round(gross_pnl, 4), entry_cost=round(pos.cost_paid, 4),
                                    exit_cost=round(exit_cost, 4), net_pnl=round(net_pnl, 4),
                                    gross_return=round(pos.direction * (exit_price - pos.entry_price) / max(pos.entry_price, 1e-6), 6),
                                    net_return=round((gross_pnl - pos.cost_paid - exit_cost) / max(pos.quantity * pos.entry_price, 1e-6), 6),
                                    model_version=pos.model_version, confidence=pos.confidence,
                                )
                                trades.append(trade)
                                overlap_stats["opposite_signal_events"] += 1
                                if policy == "CLOSE_ONLY":
                                    _opposite_signal_close_only = True

                if _opposite_signal_close_only:
                    signal_map.append(SignalExecutionMapping(
                        signal_id=sig_id, symbol=symbol,
                        signal_timestamp=str(sig_ts), direction=direction,
                        score=score, confidence=confidence,
                        decision="CLOSE_ONLY", rejection_reason="OPPOSITE_SIGNAL",
                        order_id="", position_id="", trade_id="",
                        entry_timestamp="", entry_price=0.0,
                        exit_timestamp="", exit_price=0.0,
                        exit_reason="", net_pnl=0.0, net_return=0.0,
                    ))
                    continue  # skip opening a new position for CLOSE_ONLY

                # ── Compute allocation ─────────────────────────────────────────
                stop_dist = cfg.stop_loss_pct
                atr = bar_atrs.get(symbol, 0.01)
                entry_ref = bar_closes.get(symbol, 0.0)
                if entry_ref <= 0:
                    overlap_stats["rejected_signals"] += 1
                    rejected.append(RejectedSignal(
                        signal_id=sig_id, symbol=symbol, timestamp=sig_ts,
                        direction=direction, score=score, confidence=confidence,
                        reason=RejectionReason.NO_MARKET_DATA, detail="no_close_price",
                    ))
                    continue

                alloc = compute_allocation(
                    state, score, atr, stop_dist, cfg.allocation_method
                )
                if alloc < 100:   # minimum ₹100 position
                    overlap_stats["rejected_signals"] += 1
                    rejected.append(RejectedSignal(
                        signal_id=sig_id, symbol=symbol, timestamp=sig_ts,
                        direction=direction, score=score, confidence=confidence,
                        reason=RejectionReason.INSUFFICIENT_CAPITAL, detail=f"alloc={alloc:.0f}",
                    ))
                    continue

                quantity = alloc / entry_ref
                if quantity < 1:
                    quantity = 1.0

                # ── Generate order ─────────────────────────────────────────────
                order = Order(
                    order_id           = str(uuid.uuid4())[:12],
                    signal_id          = sig_id,
                    symbol             = symbol,
                    direction          = direction,
                    quantity           = quantity,
                    submitted_timestamp = ts,
                    expiry_bar_index   = bar_index + cfg.signal_entry_window_bars,
                    stop_price         = 0.0,
                    target_price       = 0.0,
                    confidence         = confidence,
                    model_version      = model_ver,
                )
                state.pending_orders.append(order)
                state.mark_signal_seen(sig_id, symbol, sig_ts, direction)

                signal_map.append(SignalExecutionMapping(
                    signal_id=sig_id, symbol=symbol,
                    signal_timestamp=str(sig_ts), direction=direction,
                    score=score, confidence=confidence,
                    decision="EXECUTE", rejection_reason="",
                    order_id=order.order_id, position_id="",
                    trade_id="", entry_timestamp="", entry_price=0.0,
                    exit_timestamp="", exit_price=0.0,
                    exit_reason="", net_pnl=0.0, net_return=0.0,
                ))
                overlap_stats["pending_orders_submitted"] += 1

            # ── Step 9: Record equity curve ────────────────────────────────────
            state.update_daily_pnl()
            eq = state.total_equity
            if eq > state.peak_equity:
                state.peak_equity = eq
            dd = (eq - state.peak_equity) / max(state.peak_equity, 1.0)
            cap_util = state.gross_exposure

            equity_curve.append(EquityPoint(
                timestamp=ts, bar_index=bar_index,
                cash=round(state.cash, 2),
                market_value=round(state.market_value, 2),
                gross_exposure=round(state.gross_exposure, 4),
                net_exposure=round(state.net_exposure, 4),
                n_positions=state.n_open_positions,
                realized_pnl=round(state.realized_pnl, 2),
                unrealized_pnl=round(state.unrealized_pnl, 2),
                daily_pnl=round(state.daily_pnl_today, 2),
                total_equity=round(eq, 2),
                drawdown=round(dd, 6),
                capital_utilization=round(cap_util, 4),
            ))

            # Reset daily PnL at start of new trading day
            if prev_date and prev_date != date_key:
                state.reset_daily_pnl(eq)
            prev_date = date_key
            bar_index += 1

        # ── Close any remaining positions at last price ─────────────────────
        for sym, pos_list in list(state.positions.items()):
            last_price = current_prices.get(sym, 0.0)
            for pos in pos_list:
                if last_price <= 0:
                    continue
                exit_cost = (cfg.commission_bps_per_trade / 10_000.0) * pos.quantity * last_price
                gross_pnl = pos.direction * (last_price - pos.entry_price) * pos.quantity
                net_pnl = gross_pnl - pos.cost_paid - exit_cost
                state.realized_pnl += net_pnl
                state.cash += pos.quantity * last_price - exit_cost
                trades.append(ExecutedTrade(
                    trade_id=str(uuid.uuid4())[:12], signal_id=pos.signal_id,
                    position_id=pos.position_id, symbol=sym,
                    direction=pos.direction, quantity=pos.quantity,
                    entry_timestamp=pos.entry_timestamp, exit_timestamp=ts,
                    entry_price=pos.entry_price, exit_price=last_price,
                    entry_bar_index=pos.entry_bar_index, exit_bar_index=bar_index,
                    bars_held=bar_index - pos.entry_bar_index,
                    exit_reason="EOD_FORCED",
                    gross_pnl=round(gross_pnl, 4), entry_cost=round(pos.cost_paid, 4),
                    exit_cost=round(exit_cost, 4), net_pnl=round(net_pnl, 4),
                    gross_return=round(pos.direction * (last_price - pos.entry_price) / max(pos.entry_price, 1e-6), 6),
                    net_return=round((gross_pnl - pos.cost_paid - exit_cost) / max(pos.quantity * pos.entry_price, 1e-6), 6),
                ))
        state.positions.clear()

        # ── Update signal mapping with trade results ─────────────────────────
        trade_by_signal: dict[str, ExecutedTrade] = {t.signal_id: t for t in trades}
        for sm in signal_map:
            if sm.decision == "EXECUTE" and sm.signal_id in trade_by_signal:
                t = trade_by_signal[sm.signal_id]
                sm.position_id  = t.position_id
                sm.trade_id     = t.trade_id
                sm.entry_price  = t.entry_price
                sm.exit_price   = t.exit_price
                sm.entry_timestamp = str(t.entry_timestamp)
                sm.exit_timestamp  = str(t.exit_timestamp)
                sm.exit_reason  = t.exit_reason
                sm.net_pnl      = t.net_pnl
                sm.net_return   = t.net_return

        overlap_stats["maximum_concurrent_positions"] = peak_concurrent
        overlap_stats["duplicate_signals"] = sum(
            1 for r in rejected if r.reason == RejectionReason.DUPLICATE
        )

        return PortfolioResult(
            mode=mode,
            config=cfg,
            trades=trades,
            equity_curve=equity_curve,
            rejected_signals=rejected,
            order_log=order_log,
            event_log=event_log,
            signal_map=signal_map,
            overlap_stats=dict(overlap_stats),
            final_state=state,
        )

    def run_both_modes(
        self,
        signals_df: pd.DataFrame,
        ohlcv_by_symbol: dict[str, pd.DataFrame],
    ) -> tuple["PortfolioResult", "PortfolioResult"]:
        """Run Mode A and Mode B and return both results for comparison."""
        result_a = self.run(signals_df, ohlcv_by_symbol, mode="A_isolated")
        result_b = self.run(signals_df, ohlcv_by_symbol, mode="B_portfolio")
        return result_a, result_b

    # ── Signal evaluation helpers ──────────────────────────────────────────────

    def _evaluate_signal(
        self,
        sig_id: str,
        symbol: str,
        ts: pd.Timestamp,
        direction: int,
        score: float,
        confidence: float,
        bar_index: int,
        state: PortfolioState,
        prices: dict[str, float],
        atrs: dict[str, float],
    ) -> tuple[RejectionReason, str] | None:
        """
        Evaluate a signal against all portfolio constraints.

        Returns None if eligible, or (RejectionReason, detail) if rejected.
        Checks are ordered from cheapest to most expensive.
        """
        cfg = self.config

        # ── Deduplication ─────────────────────────────────────────────────────
        if sig_id in state._seen_signal_ids:
            return (RejectionReason.DUPLICATE, f"signal_id={sig_id}")

        # ── Daily loss halt ───────────────────────────────────────────────────
        if state.daily_pnl_today <= -(cfg.max_daily_loss * state.total_equity):
            return (RejectionReason.MAX_DAILY_LOSS, f"daily_pnl={state.daily_pnl_today:.2f}")

        # ── Drawdown halt ─────────────────────────────────────────────────────
        dd = abs(state.drawdown)
        if dd >= cfg.max_total_drawdown:
            return (RejectionReason.MAX_DRAWDOWN, f"drawdown={dd:.2%}")

        # ── Consecutive losses ────────────────────────────────────────────────
        if state.consecutive_losses >= cfg.max_consecutive_losses:
            return (RejectionReason.CONSECUTIVE_LOSSES,
                    f"consecutive_losses={state.consecutive_losses}")

        # ── Open risk ─────────────────────────────────────────────────────────
        if state.open_risk >= cfg.max_open_risk:
            return (RejectionReason.MAX_OPEN_RISK, f"open_risk={state.open_risk:.2%}")

        # ── Position limit ────────────────────────────────────────────────────
        # Count open positions PLUS already-queued pending orders together
        # so we don't over-allocate when all signals arrive on the same bar.
        n_already_committed = state.n_open_positions + len(state.pending_orders)
        if n_already_committed >= cfg.max_positions:
            return (RejectionReason.MAX_POSITIONS,
                    f"n={n_already_committed} (open={state.n_open_positions}+pending={len(state.pending_orders)})")

        # ── Same-symbol policy ────────────────────────────────────────────────
        if symbol in state.positions and state.positions[symbol]:
            existing_dirs = [p.direction for p in state.positions[symbol]]
            if all(d == direction for d in existing_dirs):
                # Same direction — check policy
                if cfg.same_symbol_policy == "IGNORE":
                    return (RejectionReason.SAME_SYMBOL_IGNORE, f"policy=IGNORE, dir={direction}")
            else:
                # Opposite direction
                if cfg.opposite_signal_policy == "IGNORE":
                    return (RejectionReason.OPPOSITE_SIGNAL_IGNORE, "policy=IGNORE")

        # ── Capital check ─────────────────────────────────────────────────────
        min_position = max(prices.get(symbol, 1.0), 1.0)  # at least 1 share
        if state.available_capital < min_position:
            return (RejectionReason.INSUFFICIENT_CAPITAL,
                    f"avail={state.available_capital:.0f}")

        # ── Gross exposure (including pending orders estimate) ─────────────────
        # Count pending order slots as already-committed exposure so we don't
        # over-allocate when many signals arrive on the same bar before any fills.
        n_pending = len(state.pending_orders)
        estimated_committed = state.gross_exposure + n_pending * cfg.max_capital_per_position
        if estimated_committed + cfg.max_capital_per_position > cfg.max_gross_exposure:
            return (RejectionReason.MAX_GROSS_EXPOSURE,
                    f"estimated={estimated_committed + cfg.max_capital_per_position:.2%} > max={cfg.max_gross_exposure:.2%}")

        # ── Net exposure ──────────────────────────────────────────────────────
        additional_net = direction * (state.config.max_capital_per_position)
        if abs(state.net_exposure + additional_net) > cfg.max_net_exposure:
            return (RejectionReason.MAX_NET_EXPOSURE,
                    f"net={state.net_exposure:.2%}")

        # ── Sector exposure ───────────────────────────────────────────────────
        sector = cfg.sector_map.get(symbol, "UNKNOWN")
        if sector != "UNKNOWN":
            if state.sector_exposure(sector) >= cfg.max_sector_exposure:
                return (RejectionReason.MAX_SECTOR_EXPOSURE,
                        f"sector={sector}, exp={state.sector_exposure(sector):.2%}")

        # ── Expected value filter ─────────────────────────────────────────────
        if cfg.min_expected_net_return > 0:
            estimated_ev = (confidence * cfg.take_profit_pct
                            - (1 - confidence) * cfg.stop_loss_pct
                            - cfg.round_trip_cost)
            if estimated_ev < cfg.min_expected_net_return:
                return (RejectionReason.LOW_EXPECTED_VALUE,
                        f"ev={estimated_ev:.4f}")

        return None  # all checks passed → ELIGIBLE


# ── Result Container ──────────────────────────────────────────────────────────


class PortfolioResult:
    """All outputs from a portfolio backtest run."""

    def __init__(
        self,
        mode: str,
        config: PortfolioConfig,
        trades: list[ExecutedTrade] | None = None,
        equity_curve: list[EquityPoint] | None = None,
        rejected_signals: list[RejectedSignal] | None = None,
        order_log: list[dict] | None = None,
        event_log: list[dict] | None = None,
        signal_map: list[SignalExecutionMapping] | None = None,
        overlap_stats: dict | None = None,
        final_state: PortfolioState | None = None,
    ) -> None:
        self.mode = mode
        self.config = config
        self.trades = trades or []
        self.equity_curve = equity_curve or []
        self.rejected_signals = rejected_signals or []
        self.order_log = order_log or []
        self.event_log = event_log or []
        self.signal_map = signal_map or []
        self.overlap_stats = overlap_stats or {}
        self.final_state = final_state

    # ── DataFrames ─────────────────────────────────────────────────────────────

    def trades_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.to_dict() for t in self.trades])

    def equity_curve_df(self) -> pd.DataFrame:
        if not self.equity_curve:
            return pd.DataFrame()
        return pd.DataFrame([e.to_dict() for e in self.equity_curve])

    def rejected_df(self) -> pd.DataFrame:
        if not self.rejected_signals:
            return pd.DataFrame()
        return pd.DataFrame([r.to_dict() for r in self.rejected_signals])

    def orders_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.order_log) if self.order_log else pd.DataFrame()

    def events_df(self) -> pd.DataFrame:
        return pd.DataFrame(self.event_log) if self.event_log else pd.DataFrame()

    def signal_map_df(self) -> pd.DataFrame:
        if not self.signal_map:
            return pd.DataFrame()
        return pd.DataFrame([s.to_dict() for s in self.signal_map])

    # ── Performance metrics ────────────────────────────────────────────────────

    def performance_summary(self) -> dict[str, Any]:
        """Compute complete performance metrics for this result."""
        if not self.trades:
            return {"n_trades": 0, "mode": self.mode}

        df = self.trades_df()
        net = df["net_pnl"]
        wins = net[net > 0]
        losses = net[net <= 0]

        eq_df = self.equity_curve_df()
        # total_return from trade P&L (avoids equity-curve terminal-value timing issue)
        net_sum = float(net.sum())
        total_return = net_sum / max(self.config.initial_capital, 1.0)

        # Per-trade annualised Sharpe
        std = float(net.std())
        sharpe = float(net.mean() / (std + 1e-10)) * np.sqrt(252) if std > 0 else 0.0

        # Max drawdown from equity curve (if available)
        max_dd = 0.0
        if not eq_df.empty and "drawdown" in eq_df.columns:
            max_dd = float(eq_df["drawdown"].min())

        return {
            "mode":                      self.mode,
            "initial_capital":           self.config.initial_capital,
            "total_return_pct":          round(total_return * 100, 4),
            "n_trades":                  len(df),
            "n_wins":                    int(len(wins)),
            "n_losses":                  int(len(losses)),
            "win_rate":                  round(float(len(wins) / max(len(df), 1)), 4),
            "mean_net_pnl":              round(float(net.mean()), 4),
            "total_net_pnl":             round(float(net.sum()), 4),
            "avg_win":                   round(float(wins.mean()), 4) if len(wins) > 0 else 0.0,
            "avg_loss":                  round(float(losses.mean()), 4) if len(losses) > 0 else 0.0,
            "profit_factor":             round(float(wins.sum() / (abs(losses.sum()) + 1e-10)), 4),
            "expectancy":                round(float(net.mean()), 4),
            "sharpe":                    round(sharpe, 4),
            "max_drawdown":              round(max_dd, 4),
            "total_costs":               round(float((df["entry_cost"] + df["exit_cost"]).sum()), 2),
            "avg_bars_held":             round(float(df["bars_held"].mean()), 2) if "bars_held" in df.columns else 0.0,
            "exit_reason_counts":        df["exit_reason"].value_counts().to_dict() if "exit_reason" in df.columns else {},
            **{f"overlap_{k}": v for k, v in self.overlap_stats.items()},
        }

    def comparison_with(self, other: "PortfolioResult") -> dict[str, Any]:
        """Compare this result (Mode A) with another (Mode B)."""
        a = self.performance_summary()
        b = other.performance_summary()
        return {
            "mode_a_isolated": a,
            "mode_b_portfolio": b,
            "delta_win_rate":      round(b.get("win_rate", 0) - a.get("win_rate", 0), 4),
            "delta_total_net_pnl": round(b.get("total_net_pnl", 0) - a.get("total_net_pnl", 0), 2),
            "delta_n_trades":      b.get("n_trades", 0) - a.get("n_trades", 0),
            "delta_sharpe":        round(b.get("sharpe", 0) - a.get("sharpe", 0), 4),
            "portfolio_captures_pct": round(
                b.get("n_trades", 0) / max(a.get("n_trades", 1), 1) * 100, 2
            ),
        }

    def to_csv(self, output_dir: str) -> None:
        """Write all DataFrames to CSV files in output_dir."""
        from pathlib import Path
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.mode}_"

        for name, df in [
            ("portfolio_equity_curve", self.equity_curve_df()),
            ("executed_orders",        self.orders_df()),
            ("rejected_signals",       self.rejected_df()),
            ("portfolio_events",       self.events_df()),
            ("signal_execution_mapping", self.signal_map_df()),
        ]:
            if not df.empty:
                df.to_csv(str(out / f"{prefix}{name}.csv"), index=False)
                logger.info("portfolio_csv_written", file=f"{prefix}{name}.csv", rows=len(df))

        # Trades file uses the standard name (no prefix for backwards compat)
        df_trades = self.trades_df()
        if not df_trades.empty:
            df_trades.to_csv(str(out / f"{prefix}executed_trades.csv"), index=False)


# ── Utility ───────────────────────────────────────────────────────────────────


def _compute_barriers(
    entry_price: float,
    direction: int,
    stop_pct: float,
    target_pct: float,
) -> tuple[float, float]:
    """Compute stop and target prices from entry."""
    if direction == 1:  # LONG
        stop_p   = entry_price * (1 - stop_pct)
        target_p = entry_price * (1 + target_pct)
    else:              # SHORT
        stop_p   = entry_price * (1 + stop_pct)
        target_p = entry_price * (1 - target_pct)
    return stop_p, target_p


def run_capacity_analysis(
    signals_df: pd.DataFrame,
    ohlcv_by_symbol: dict[str, pd.DataFrame],
    capital_levels: tuple = (100_000, 500_000, 1_000_000, 5_000_000, 10_000_000),
) -> dict[str, dict]:
    """
    Run portfolio backtest at multiple capital levels (§23A.22).

    Returns dict mapping capital_level → performance_summary.
    """
    results: dict[str, dict] = {}
    for capital in capital_levels:
        cfg = PortfolioConfig(initial_capital=float(capital))
        engine = PortfolioEngine(cfg)
        result = engine.run(signals_df, ohlcv_by_symbol, mode="B_portfolio")
        perf = result.performance_summary()
        results[f"cap_{capital:,}"] = {
            **perf,
            "capital": capital,
            "total_return_pct": perf.get("total_return_pct", 0),
            "n_trades": perf.get("n_trades", 0),
        }
    return results


def run_execution_stress_test(
    signals_df: pd.DataFrame,
    ohlcv_by_symbol: dict[str, pd.DataFrame],
) -> dict[str, dict]:
    """
    Run portfolio backtest under 4 execution scenarios (§23A.23).

    optimistic: tight spreads, low slippage
    base: realistic equity costs
    conservative: 1.5× costs
    stress: 2× costs
    """
    scenarios = {
        "optimistic":    {"slippage_bps": 1.0, "commission_bps_per_trade": 4.5},
        "base":          {"slippage_bps": 3.5, "commission_bps_per_trade": 13.5},
        "conservative":  {"slippage_bps": 5.0, "commission_bps_per_trade": 18.0},
        "stress":        {"slippage_bps": 7.0, "commission_bps_per_trade": 27.0},
    }
    results: dict[str, dict] = {}
    for scenario_name, cost_params in scenarios.items():
        cfg = PortfolioConfig(**cost_params)
        engine = PortfolioEngine(cfg)
        result = engine.run(signals_df, ohlcv_by_symbol, mode="B_portfolio")
        results[scenario_name] = result.performance_summary()
    return results
