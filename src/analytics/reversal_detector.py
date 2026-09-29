"""
src.analytics.reversal_detector — Real-time Reversal/Mean-Reversion Scanner.

Problem
-------
The LightGBM momentum model predicts trends continuing.  When a stock has
been falling for 3-5 days and is deeply oversold (RSI < 30, at lower Bollinger
Band, 3+ consecutive down days), it is a candidate for a sharp REVERSAL rather
than continued decline.  The momentum model would still score it SHORT —
exactly wrong for the reversal day.

Solution
--------
Run a PARALLEL reversal scan alongside the momentum model.  For every symbol:

1. Compute reversal readiness score (0–1):
   - 0   = no reversal signal
   - 0.5 = moderate oversold/overbought
   - 1.0 = extreme — multiple confirming signals

2. Flag symbols as REVERSAL_LONG (oversold bounce candidate) or
   REVERSAL_SHORT (overbought drop candidate)

3. In autorun_till_close.py, OVERRIDE the momentum signal with the reversal
   signal when reversal_score > override_threshold (default: 0.70)

Evidence from Sep 29 session:
  ADANIENT:  RSI~28, 5 consec down bars, at lower BB → REVERSAL_LONG → +5.01% actual
  PAYTM:     RSI~32, 4 consec down bars → REVERSAL_LONG → +2.89% actual
  IEX:       RSI~72, 4 consec up bars → REVERSAL_SHORT → -4.45% actual
  PATANJALI: RSI~74, bb_above_upper → REVERSAL_SHORT → -4.96% actual

Usage::

    detector = ReversalDetector()
    reversals = detector.scan(scores)  # scores = output of score_all()
    enhanced = detector.merge_with_momentum(scores, reversals)

Requirements: momentum model complement, not replacement.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

PARQUET_DIR = Path("data/1d/1d")


@dataclass
class ReversalSignal:
    """Single reversal signal for one symbol."""
    symbol: str
    reversal_type: str        # "REVERSAL_LONG" or "REVERSAL_SHORT"
    reversal_score: float     # 0-1; how strong the reversal setup is
    momentum_score: float     # original LightGBM score
    momentum_direction: int   # original LightGBM direction
    override_direction: int   # +1 LONG or -1 SHORT
    rsi: float
    consec_bars: int
    bb_position: float        # bb_pct_b: 0=lower band, 1=upper
    vol_spike: float
    evidence: str             # human-readable explanation


class ReversalDetector:
    """
    Scans the full 218-symbol universe for reversal setups in real time.

    Reversal setup conditions (configurable via thresholds):
      Oversold bounce (→ LONG override):
        - RSI < rsi_oversold_threshold (default 32)
        - AND consecutive down bars >= consec_bars_threshold (default 3)
        - OR bb_pct_b < bb_oversold_threshold (at/below lower Bollinger Band)
        - OR price_from_20d_low_pct < 0.02 (within 2% of 20-day low)

      Overbought drop (→ SHORT override):
        - RSI > rsi_overbought_threshold (default 68)
        - AND consecutive up bars >= consec_bars_threshold (default 3)
        - OR bb_pct_b > bb_overbought_threshold (at/above upper Bollinger Band)
        - OR stoch_reversal_signal == -1 (stochastic bearish cross)

    The reversal_score (0-1) reflects the number and strength of confirming
    conditions.  Only signals above override_threshold replace the momentum
    signal; lower-scoring reversals are reported but not acted on.
    """

    def __init__(
        self,
        rsi_oversold_threshold: float = 32.0,
        rsi_overbought_threshold: float = 68.0,
        consec_bars_threshold: int = 3,
        bb_oversold_threshold: float = 0.10,
        bb_overbought_threshold: float = 0.90,
        override_threshold: float = 0.60,
        parquet_dir: Path = PARQUET_DIR,
    ) -> None:
        self.rsi_os  = rsi_oversold_threshold
        self.rsi_ob  = rsi_overbought_threshold
        self.consec  = consec_bars_threshold
        self.bb_os   = bb_oversold_threshold
        self.bb_ob   = bb_overbought_threshold
        self.override_threshold = override_threshold
        self.parquet_dir = Path(parquet_dir)

    def _compute_reversal_score(
        self,
        rsi: float,
        consec_down: float,
        consec_up: float,
        bb_pct_b: float,
        price_from_low: float,
        price_from_high: float,
        vol_spike: float,
        stoch_signal: float,
    ) -> tuple[str, float, str]:
        """
        Returns (reversal_type, score, evidence_text).
        reversal_type: "REVERSAL_LONG", "REVERSAL_SHORT", or "NONE"
        score: 0-1 (higher = stronger setup)
        """
        long_score = 0.0
        short_score = 0.0
        long_ev = []
        short_ev = []

        # ── Oversold (LONG reversal) signals ─────────────────────────────────
        if rsi < self.rsi_os:
            pts = (self.rsi_os - rsi) / self.rsi_os   # deeper = stronger
            long_score += 0.30 * pts
            long_ev.append(f"RSI={rsi:.1f}<{self.rsi_os}")
        if consec_down >= self.consec:
            pts = min(consec_down / 5.0, 1.0)
            long_score += 0.25 * pts
            long_ev.append(f"{int(consec_down)} consec down")
        if bb_pct_b < self.bb_os:
            pts = (self.bb_os - bb_pct_b) / self.bb_os
            long_score += 0.25 * pts
            long_ev.append(f"BB%B={bb_pct_b:.2f}<{self.bb_os}")
        if price_from_low < 0.02:
            long_score += 0.15
            long_ev.append(f"at 20d low (+{price_from_low*100:.1f}%)")
        if stoch_signal > 0:
            long_score += 0.10
            long_ev.append("stoch bullish cross")
        if vol_spike > 2.0:
            long_score += 0.05 * min(vol_spike / 3.0, 1.0)
            long_ev.append(f"vol spike {vol_spike:.1f}x")

        # ── Overbought (SHORT reversal) signals ───────────────────────────────
        if rsi > self.rsi_ob:
            pts = (rsi - self.rsi_ob) / (100 - self.rsi_ob)
            short_score += 0.30 * pts
            short_ev.append(f"RSI={rsi:.1f}>{self.rsi_ob}")
        if consec_up >= self.consec:
            pts = min(consec_up / 5.0, 1.0)
            short_score += 0.25 * pts
            short_ev.append(f"{int(consec_up)} consec up")
        if bb_pct_b > self.bb_ob:
            pts = (bb_pct_b - self.bb_ob) / (1.0 - self.bb_ob)
            short_score += 0.25 * pts
            short_ev.append(f"BB%B={bb_pct_b:.2f}>{self.bb_ob}")
        if price_from_high > -0.02:
            short_score += 0.15
            short_ev.append(f"at 20d high ({price_from_high*100:.1f}%)")
        if stoch_signal < 0:
            short_score += 0.10
            short_ev.append("stoch bearish cross")
        if vol_spike > 2.0:
            short_score += 0.05 * min(vol_spike / 3.0, 1.0)
            short_ev.append(f"vol spike {vol_spike:.1f}x")

        if long_score > short_score and long_score > 0.1:
            return "REVERSAL_LONG", round(min(long_score, 1.0), 4), " | ".join(long_ev)
        if short_score > long_score and short_score > 0.1:
            return "REVERSAL_SHORT", round(min(short_score, 1.0), 4), " | ".join(short_ev)
        return "NONE", 0.0, ""

    def scan_symbol(self, sym: str) -> ReversalSignal | None:
        """Compute reversal signal for a single symbol from its parquet."""
        pf = self.parquet_dir / f"{sym}.parquet"
        if not pf.exists():
            return None
        try:
            df = pd.read_parquet(pf)
            df.columns = [c.lower() for c in df.columns]
            if len(df) < 30:
                return None

            from src.features.families.reversal import compute_reversal_features
            close  = df["close"].astype(float)
            high   = df.get("high", close).astype(float)
            low    = df.get("low",  close).astype(float)
            volume = df.get("volume", pd.Series(1, index=df.index)).astype(float)

            rev = compute_reversal_features(close, high, low, volume)
            last = {k: float(v.iloc[-1]) if hasattr(v, 'iloc') else float(v)
                    for k, v in rev.items()}

            rtype, rscore, evidence = self._compute_reversal_score(
                rsi          = last.get("rsi_extreme_distance", 0) * -50 + 50,   # convert back from normalized
                consec_down  = last.get("consec_down_bars", 0),
                consec_up    = last.get("consec_up_bars", 0),
                bb_pct_b     = last.get("bb_pct_b", 0.5),
                price_from_low  = last.get("price_from_20d_low_pct", 1.0),
                price_from_high = last.get("price_from_20d_high_pct", -1.0),
                vol_spike    = last.get("vol_spike_ratio", 1.0),
                stoch_signal = last.get("stoch_reversal_signal", 0),
            )

            if rtype == "NONE":
                return None

            override_dir = 1 if rtype == "REVERSAL_LONG" else -1

            return ReversalSignal(
                symbol=sym,
                reversal_type=rtype,
                reversal_score=rscore,
                momentum_score=0.0,        # filled in merge_with_momentum
                momentum_direction=0,
                override_direction=override_dir,
                rsi=round(last.get("rsi_extreme_distance", 0) * -50 + 50, 1),
                consec_bars=int(last.get("consec_down_bars", 0) or last.get("consec_up_bars", 0)),
                bb_position=round(last.get("bb_pct_b", 0.5), 3),
                vol_spike=round(last.get("vol_spike_ratio", 1.0), 2),
                evidence=evidence,
            )
        except Exception as e:
            logger.debug("reversal_scan_error", symbol=sym, error=str(e))
            return None

    def scan(self, symbols: list[str] | None = None) -> list[ReversalSignal]:
        """
        Scan all symbols (or provided list) for reversal setups.

        Returns only signals with reversal_score > 0 (all detected setups).
        """
        if symbols is None:
            symbols = [p.stem for p in sorted(self.parquet_dir.glob("*.parquet"))]

        signals = []
        for sym in symbols:
            sig = self.scan_symbol(sym)
            if sig is not None:
                signals.append(sig)

        # Sort by reversal_score descending
        signals.sort(key=lambda s: -s.reversal_score)
        logger.info(
            "reversal_scan_complete",
            n_symbols=len(symbols),
            n_reversal_signals=len(signals),
            n_strong=sum(1 for s in signals if s.reversal_score >= self.override_threshold),
        )
        return signals

    def merge_with_momentum(
        self,
        momentum_scores: list[dict],
        reversals: list[ReversalSignal],
    ) -> list[dict]:
        """
        Merge reversal signals with momentum scores.

        For symbols with a strong reversal signal (score >= override_threshold)
        that CONTRADICTS the momentum signal, the direction is OVERRIDDEN.

        A reversal ONLY overrides momentum when:
          - reversal_score >= override_threshold (default 0.60)
          - reversal direction OPPOSES momentum direction (this is the key case)

        If momentum and reversal AGREE, the signal is strengthened (no change needed).

        Returns:
            Updated momentum_scores list with reversal_override and
            reversal_score fields added to each entry.
        """
        rev_by_symbol = {s.symbol: s for s in reversals}
        result = []
        n_overridden = 0

        for score_dict in momentum_scores:
            sym = score_dict.get("symbol", "")
            rev = rev_by_symbol.get(sym)
            entry = dict(score_dict)

            if rev is not None:
                entry["reversal_score"]    = rev.reversal_score
                entry["reversal_type"]     = rev.reversal_type
                entry["reversal_evidence"] = rev.evidence

                # Fill in momentum info
                rev.momentum_score     = entry.get("score", 0.5)
                rev.momentum_direction = entry.get("direction", 0)

                # Override only when reversal is strong AND contradicts momentum
                momentum_dir = entry.get("direction", 0)
                if (
                    rev.reversal_score >= self.override_threshold
                    and momentum_dir != 0
                    and rev.override_direction != momentum_dir
                ):
                    old_dir = momentum_dir
                    entry["direction"] = rev.override_direction
                    entry["reversal_override"] = True
                    n_overridden += 1
                    logger.info(
                        "reversal_override_applied",
                        symbol=sym,
                        old_direction=old_dir,
                        new_direction=rev.override_direction,
                        reversal_score=rev.reversal_score,
                        evidence=rev.evidence[:80],
                    )
                else:
                    entry["reversal_override"] = False
            else:
                entry["reversal_score"] = 0.0
                entry["reversal_type"]  = "NONE"
                entry["reversal_override"] = False

            result.append(entry)

        logger.info(
            "reversal_merge_complete",
            n_total=len(momentum_scores),
            n_overridden=n_overridden,
        )
        return result

    def top_setups(
        self,
        reversals: list[ReversalSignal],
        n: int = 10,
    ) -> dict[str, list[dict]]:
        """Return top N oversold and top N overbought setups."""
        long_setups  = [s for s in reversals if s.reversal_type == "REVERSAL_LONG"]
        short_setups = [s for s in reversals if s.reversal_type == "REVERSAL_SHORT"]
        return {
            "oversold_bounce": [
                {"symbol": s.symbol, "score": s.reversal_score, "rsi": s.rsi,
                 "consec_down": s.consec_bars, "bb": s.bb_position,
                 "evidence": s.evidence}
                for s in long_setups[:n]
            ],
            "overbought_drop": [
                {"symbol": s.symbol, "score": s.reversal_score, "rsi": s.rsi,
                 "consec_up": s.consec_bars, "bb": s.bb_position,
                 "evidence": s.evidence}
                for s in short_setups[:n]
            ],
        }
