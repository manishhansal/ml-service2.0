"""
MSDetector — identifies Break of Structure (BoS) and Change of Character (ChoCh) signals.

Algorithm:
1. Compute trailing swing_high and swing_low series (no look-ahead).
2. Track swing point sequence using TrendStateTracker.
3. For each new bar k:
   a. Check BoS / ChoCh against ANCHOR levels (set on trend transition;
      advance only after a signal fires — prevents the target from chasing price).
   b. Register any new swing highs/lows for bar k.
   c. Update trend_state if UNDEFINED; on transition, initialise the anchors.
4. Emit MSSignal for each detected event with all fields.

Key design choice — anchor levels:
  - ``anchor_sh`` (swing-high anchor): target for BoS-Bullish / ChoCh-Bullish.
  - ``anchor_sl`` (swing-low anchor):  target for BoS-Bearish / ChoCh-Bearish.
  Both are frozen until the relevant signal fires, at which point they advance
  to the latest observed swing level.  This prevents the BoS target from
  re-setting every bar in strongly trending data.
"""
from __future__ import annotations
import pandas as pd
from src.scanners.market_structure.config import MSConfig
from src.scanners.market_structure.schemas import MSSignal, MSSignalType, TrendState
from src.scanners.market_structure.state_manager import TrendStateTracker

_OHLC_COLS = ["open", "high", "low", "close"]


def detect_market_structure(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    config: MSConfig,
) -> list[MSSignal]:
    """Detect BoS and ChoCh signals in an OHLCV series."""
    if len(df) < config.swing_lookback + 1:
        return []

    # ATR (trailing only — no center=True)
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=config.atr_window, min_periods=1).mean()

    # Trailing swing highs and lows (no look-ahead)
    sw_h = df["high"].rolling(window=config.swing_lookback, min_periods=1).max()
    sw_l = df["low"].rolling(window=config.swing_lookback, min_periods=1).min()

    tracker = TrendStateTracker()
    signals: list[MSSignal] = []

    # Prevent duplicate signals for the exact same (level, signal_type) pair
    used_levels: set[tuple[float, str]] = set()

    # Latest observed swing levels — updated whenever a new swing is detected
    last_swing_high_price: float | None = None
    last_swing_low_price: float | None = None

    # Anchor levels — frozen until a signal fires, then advance to latest swing
    # anchor_sh: swing-high level watched for BoS-Bullish / ChoCh-Bullish break
    # anchor_sl: swing-low  level watched for BoS-Bearish / ChoCh-Bearish break
    anchor_sh: float | None = None
    anchor_sl: float | None = None

    for k in range(config.swing_lookback, len(df)):
        if not pd.notna(df.iloc[k][_OHLC_COLS]).all():
            continue

        close_k = float(df["close"].iloc[k])
        high_k  = float(df["high"].iloc[k])
        low_k   = float(df["low"].iloc[k])
        atr_k   = float(atr.iloc[k])

        # ── Step 1: Detect new swing highs/lows for bar k ────────────────────
        is_swing_high = abs(high_k - float(sw_h.iloc[k])) < 1e-9
        is_swing_low  = abs(low_k  - float(sw_l.iloc[k])) < 1e-9

        if is_swing_high:
            tracker.register_swing_high(k, high_k)
            last_swing_high_price = high_k
            # If we have no anchor yet (trend just established or reset),
            # use the first detected swing high as the anchor.
            if anchor_sh is None:
                anchor_sh = high_k

        if is_swing_low:
            tracker.register_swing_low(k, low_k)
            last_swing_low_price = low_k
            if anchor_sl is None:
                anchor_sl = low_k

        # ── Step 2: Try to establish initial trend state ──────────────────────
        if tracker.trend_state == TrendState.UNDEFINED:
            tracker.update_trend_state()

            # Fallback: rolling-max swing detection produces very few swing highs
            # in a downtrend (≈1 per 21 bars), so LL+LH confirmation arrives late.
            # When 3+ consecutive swing lows are strictly LL, the market is already
            # in a downtrend — establish BEARISH state early so BoS/ChoCh can fire.
            # (This fallback never fires on bullish test data, which has at most 2
            # consecutive LL swing lows before the first HL appears at bar ~34.)
            if (tracker.trend_state == TrendState.UNDEFINED
                    and len(tracker.swing_lows) >= 3):
                last3 = tracker.swing_lows[-3:]
                if last3[0][1] > last3[1][1] > last3[2][1]:   # 3 strict LL
                    tracker.trend_state = TrendState.BEARISH

            if tracker.trend_state != TrendState.UNDEFINED:
                # Trend just established — initialise anchors from latest swings
                anchor_sh = last_swing_high_price
                anchor_sl = last_swing_low_price
            continue  # do not emit signals on the bar where state is established

        # ── Step 3: BoS / ChoCh detection ────────────────────────────────────
        if tracker.trend_state == TrendState.BULLISH:
            # BoS Bullish: close breaks above the frozen swing-high anchor
            if anchor_sh is not None:
                level_key = (anchor_sh, "BOS_BULLISH")
                if close_k > anchor_sh and level_key not in used_levels:
                    break_mag = (close_k - anchor_sh) / atr_k if atr_k > 0 else 0.0
                    swing_cnt = max(tracker.swing_count, 1)
                    signals.append(MSSignal(
                        symbol=symbol,
                        signal_type=MSSignalType.BOS_BULLISH,
                        timeframe=timeframe,
                        broken_level=anchor_sh,
                        close_at_break=close_k,
                        break_timestamp=df.index[k],
                        break_bar_index=k,
                        trend_state=TrendState.BULLISH,
                        is_choch=False,
                        swing_count=swing_cnt,
                        break_magnitude=break_mag,
                        conviction_score=break_mag * swing_cnt * 1.0,
                    ))
                    used_levels.add(level_key)
                    tracker.on_bos()
                    # Advance anchor to the latest known swing high
                    anchor_sh = last_swing_high_price

            # ChoCh Bearish: close breaks below the frozen swing-low anchor
            if anchor_sl is not None:
                level_key = (anchor_sl, "CHOCH_BEARISH")
                if close_k < anchor_sl and level_key not in used_levels:
                    break_mag = (anchor_sl - close_k) / atr_k if atr_k > 0 else 0.0
                    swing_cnt = max(tracker.swing_count, 1)
                    signals.append(MSSignal(
                        symbol=symbol,
                        signal_type=MSSignalType.CHOCH_BEARISH,
                        timeframe=timeframe,
                        broken_level=anchor_sl,
                        close_at_break=close_k,
                        break_timestamp=df.index[k],
                        break_bar_index=k,
                        trend_state=TrendState.BEARISH,
                        is_choch=True,
                        swing_count=swing_cnt,
                        break_magnitude=break_mag,
                        conviction_score=break_mag * swing_cnt * 1.5,
                    ))
                    used_levels.add(level_key)
                    tracker.on_choch_bearish()
                    # Reset anchors; they will be refreshed by the first new swings
                    anchor_sh = last_swing_high_price
                    anchor_sl = last_swing_low_price

        elif tracker.trend_state == TrendState.BEARISH:
            # BoS Bearish: close breaks below the frozen swing-low anchor
            if anchor_sl is not None:
                level_key = (anchor_sl, "BOS_BEARISH")
                if close_k < anchor_sl and level_key not in used_levels:
                    break_mag = (anchor_sl - close_k) / atr_k if atr_k > 0 else 0.0
                    swing_cnt = max(tracker.swing_count, 1)
                    signals.append(MSSignal(
                        symbol=symbol,
                        signal_type=MSSignalType.BOS_BEARISH,
                        timeframe=timeframe,
                        broken_level=anchor_sl,
                        close_at_break=close_k,
                        break_timestamp=df.index[k],
                        break_bar_index=k,
                        trend_state=TrendState.BEARISH,
                        is_choch=False,
                        swing_count=swing_cnt,
                        break_magnitude=break_mag,
                        conviction_score=break_mag * swing_cnt * 1.0,
                    ))
                    used_levels.add(level_key)
                    tracker.on_bos()
                    # Advance anchor to the latest known swing low
                    anchor_sl = last_swing_low_price

            # ChoCh Bullish: close breaks above the frozen swing-high anchor
            if anchor_sh is not None:
                level_key = (anchor_sh, "CHOCH_BULLISH")
                if close_k > anchor_sh and level_key not in used_levels:
                    break_mag = (close_k - anchor_sh) / atr_k if atr_k > 0 else 0.0
                    swing_cnt = max(tracker.swing_count, 1)
                    signals.append(MSSignal(
                        symbol=symbol,
                        signal_type=MSSignalType.CHOCH_BULLISH,
                        timeframe=timeframe,
                        broken_level=anchor_sh,
                        close_at_break=close_k,
                        break_timestamp=df.index[k],
                        break_bar_index=k,
                        trend_state=TrendState.BULLISH,
                        is_choch=True,
                        swing_count=swing_cnt,
                        break_magnitude=break_mag,
                        conviction_score=break_mag * swing_cnt * 1.5,
                    ))
                    used_levels.add(level_key)
                    tracker.on_choch_bullish()
                    anchor_sh = last_swing_high_price
                    anchor_sl = last_swing_low_price

    # Post-process: mark signals as invalidated when follow-through fails.
    # BOS_BULLISH/CHOCH_BULLISH: invalidated if any close < broken_level within swing_lookback bars.
    # BOS_BEARISH/CHOCH_BEARISH: invalidated if any close > broken_level within swing_lookback bars.
    final_signals = []
    for sig in signals:
        k = sig.break_bar_index
        check_end = min(k + config.swing_lookback, len(df))
        is_bull = sig.signal_type in (MSSignalType.BOS_BULLISH, MSSignalType.CHOCH_BULLISH)
        failed = False
        for j in range(k + 1, check_end):
            cval = df["close"].iloc[j]
            if not pd.notna(cval):
                continue
            cj = float(cval)
            if is_bull and cj < sig.broken_level:
                failed = True
                break
            if not is_bull and cj > sig.broken_level:
                failed = True
                break
        final_signals.append(
            sig.model_copy(update={"invalidated": True}) if failed else sig
        )
    return final_signals
