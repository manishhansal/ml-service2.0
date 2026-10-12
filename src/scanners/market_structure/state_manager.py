"""
TrendStateTracker — maintains the Trend_State machine for a single symbol-timeframe pair.

Tracks the sequence of swing highs and swing lows, classifies each as HH/LH or HL/LL,
and determines the current trend state (UNDEFINED → BULLISH/BEARISH).

State transitions:
  UNDEFINED → BULLISH: when HH + HL sequence is confirmed
  UNDEFINED → BEARISH: when LL + LH sequence is confirmed
  Any state → opposite: on ChoCh (Change of Character)
  Any state → same: on BoS (Break of Structure — does NOT change state)
"""
from __future__ import annotations
from src.scanners.market_structure.schemas import TrendState


class TrendStateTracker:
    """Maintains trend state, swing sequences, and swing counts."""

    def __init__(self) -> None:
        self.trend_state: TrendState = TrendState.UNDEFINED
        self.swing_highs: list[tuple[int, float]] = []  # (bar_index, price)
        self.swing_lows:  list[tuple[int, float]] = []  # (bar_index, price)
        self.swing_count: int = 0

    def register_swing_high(self, bar_index: int, price: float) -> None:
        """Register a new swing high and classify it as HH or LH."""
        self.swing_highs.append((bar_index, price))

    def register_swing_low(self, bar_index: int, price: float) -> None:
        """Register a new swing low and classify it as HL or LL."""
        self.swing_lows.append((bar_index, price))

    def last_swing_high_type(self) -> str:
        """Return 'HH' if the last high is higher than the prior, else 'LH'."""
        if len(self.swing_highs) < 2:
            return "FIRST"
        return "HH" if self.swing_highs[-1][1] > self.swing_highs[-2][1] else "LH"

    def last_swing_low_type(self) -> str:
        """Return 'HL' if the last low is higher than the prior, else 'LL'."""
        if len(self.swing_lows) < 2:
            return "FIRST"
        return "HL" if self.swing_lows[-1][1] > self.swing_lows[-2][1] else "LL"

    def update_trend_state(self) -> None:
        """Re-evaluate and update trend_state from swing point sequence."""
        if len(self.swing_highs) >= 2 and len(self.swing_lows) >= 2:
            hh = self.swing_highs[-1][1] > self.swing_highs[-2][1]
            hl = self.swing_lows[-1][1]  > self.swing_lows[-2][1]
            ll = self.swing_lows[-1][1]  < self.swing_lows[-2][1]
            lh = self.swing_highs[-1][1] < self.swing_highs[-2][1]
            if hh and hl:
                self.trend_state = TrendState.BULLISH
            elif ll and lh:
                self.trend_state = TrendState.BEARISH

    def on_choch_bullish(self) -> None:
        """ChoCh to bullish — trend reverses, swing count resets."""
        self.trend_state = TrendState.BULLISH
        self.swing_count = 0

    def on_choch_bearish(self) -> None:
        """ChoCh to bearish — trend reverses, swing count resets."""
        self.trend_state = TrendState.BEARISH
        self.swing_count = 0

    def on_bos(self) -> None:
        """BoS — trend continues, swing count increments."""
        self.swing_count += 1
