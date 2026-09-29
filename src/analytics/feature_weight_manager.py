"""
src.analytics.feature_weight_manager — Agent-editable signal filters.

Reads strategy/feature_weights.json and applies sector-regime filters,
score thresholds, and symbol overrides to live signal scores AFTER the
LightGBM model has scored them.

This is a portfolio construction layer — it does NOT touch the model or
features.  The JSON file is committed to git after every SelfLearningLoop
retro, making the git log a learnable history of filter adjustments.

Key operations:
    1. Score threshold filter — remove low-conviction signals
    2. Sector-regime dimmer — reduce conviction for defensive sectors in bear days
    3. Symbol overrides — disable specific symbols (e.g. DQ issues)
    4. Long book limit — cap LONG count in bearish regimes

Usage::

    mgr = FeatureWeightManager()

    # After score_all() in autorun_till_close.py:
    regime = mgr.detect_regime(nifty_chg=-1.52)
    filtered = mgr.apply(scores, regime=regime)
    # filtered has some direction=0 (FLAT) entries where signals were suppressed

    # In SelfLearningLoop, after a resolved batch:
    mgr.update_from_sweep(optimal_threshold=0.15, evidence="...")
    mgr.update_sector_filter("HIGH_CORR_BEAR", "PHARMA", multiplier=0.10, evidence="...")

Requirements: Phil's strategy/playbook.md + risk.json pattern, mandate §4.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)

WEIGHTS_PATH = Path("strategy/feature_weights.json")
NEUTRAL      = 0.50


def _load_weights(path: Path) -> dict:
    if not path.exists():
        logger.warning("feature_weights_missing", path=str(path))
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning("feature_weights_load_failed", error=str(e))
        return {}


class FeatureWeightManager:
    """
    Applies strategy/feature_weights.json filters to live signal scores.

    The weights file is agent-editable: after each resolved batch the
    SelfLearningLoop may call update_* methods which update the JSON and
    produce a git-committable change.  The git log of feature_weights.json
    IS the learning record (Phil's pattern).
    """

    def __init__(self, weights_path: Path = WEIGHTS_PATH) -> None:
        self._path = weights_path
        self._weights = _load_weights(self._path)

    def reload(self) -> None:
        """Re-read from disk (call after SelfLearningLoop writes)."""
        self._weights = _load_weights(self._path)

    # ── Regime detection ──────────────────────────────────────────────────────

    def detect_regime(
        self,
        nifty_chg: float = 0.0,
        banknifty_chg: float = 0.0,
        vol_regime: str = "NORMAL",
    ) -> str:
        """
        Map intraday market conditions to a regime label used in sector filters.

        Returns one of: HIGH_CORR_BEAR, TRENDING_BEAR, LOW_VOL, NORMAL.
        """
        if nifty_chg <= -1.5:
            return "HIGH_CORR_BEAR"
        if nifty_chg <= -0.8:
            return "TRENDING_BEAR"
        if vol_regime == "LOW":
            return "LOW_VOL"
        return "NORMAL"

    # ── Main filter pipeline ──────────────────────────────────────────────────

    def apply(
        self,
        scores: list[dict],
        regime: str = "NORMAL",
        nifty_chg: float = 0.0,
    ) -> tuple[list[dict], dict[str, Any]]:
        """
        Apply all filters to scored signals.

        Returns:
            (filtered_scores, filter_summary)
            filtered_scores: same list with direction=0 for suppressed signals
            filter_summary:  dict with counts/reasons for logging
        """
        threshold   = self._get_threshold()
        sector_map  = self._weights.get("sector_map", {})
        overrides   = self._weights.get("symbol_overrides", {})
        regime_cfg  = self._weights.get("sector_regime_filters", {}).get(regime, {})
        long_limits = self._weights.get("long_book_limits", {})
        max_long    = long_limits.get(regime, long_limits.get("NORMAL", {})).get("max_positions", 99)

        filtered        = []
        n_threshold     = 0
        n_sector_dimmed = 0
        n_disabled      = 0
        n_long_capped   = 0
        long_count      = 0

        for s in scores:
            sym   = s.get("symbol", "")
            score = float(s.get("score", NEUTRAL))
            orig_dir = int(s.get("direction", 0))
            new_dir  = orig_dir
            conviction = abs(score - NEUTRAL)

            # ── 1. Symbol override (e.g. TATAMOTORS disabled) ─────────────
            override = overrides.get(sym, {})
            if not override.get("enabled", True):
                new_dir = 0
                n_disabled += 1
                filtered.append({**s, "direction": 0,
                                  "filter_reason": f"DISABLED:{override.get('reason','')}"})
                continue

            # ── 2. Score threshold ─────────────────────────────────────────
            if conviction < threshold and threshold > 0:
                new_dir = 0
                n_threshold += 1
                filtered.append({**s, "direction": 0,
                                  "filter_reason": f"THRESHOLD:{threshold:.2f}"})
                continue

            # ── 3. Sector-regime dimmer ────────────────────────────────────
            sector   = sector_map.get(sym, "")
            all_cfg  = regime_cfg.get("ALL", {})
            sec_cfg  = regime_cfg.get(sector, {})
            dim_cfg  = sec_cfg if sec_cfg else all_cfg

            if dim_cfg:
                multiplier = float(dim_cfg.get("multiplier", 1.0))
                filter_dir = int(dim_cfg.get("direction", 0))
                # direction=0 → apply dimmer regardless of signal direction
                # direction=-1 → only dim SHORT signals on this sector
                if filter_dir == 0 or filter_dir == orig_dir:
                    dimmed_conviction = conviction * multiplier
                    if dimmed_conviction < 0.05:   # conviction → near-zero → flat
                        new_dir = 0
                        n_sector_dimmed += 1
                        filtered.append({**s, "direction": 0,
                                          "filter_reason": f"SECTOR_DIM:{sector}:{regime}:{multiplier}"})
                        continue
                    # Partial dim: reduce score towards neutral
                    if orig_dir == -1:
                        new_score = NEUTRAL - dimmed_conviction
                    else:
                        new_score = NEUTRAL + dimmed_conviction
                    s = {**s, "score": round(new_score, 6)}
                    if sec_cfg:
                        n_sector_dimmed += 1
                        s["filter_reason"] = f"SECTOR_DIM_PARTIAL:{sector}:{multiplier}"

            # ── 4. Long book cap ───────────────────────────────────────────
            if new_dir == 1:
                if long_count >= max_long:
                    new_dir = 0
                    n_long_capped += 1
                    filtered.append({**s, "direction": 0,
                                      "filter_reason": f"LONG_CAP:{max_long}:{regime}"})
                    continue
                long_count += 1

            filtered.append({**s, "direction": new_dir})

        n_active = sum(1 for f in filtered if f["direction"] != 0)
        summary = {
            "regime":          regime,
            "threshold":       threshold,
            "n_total":         len(scores),
            "n_active":        n_active,
            "n_filtered":      len(scores) - n_active,
            "n_threshold":     n_threshold,
            "n_sector_dimmed": n_sector_dimmed,
            "n_disabled":      n_disabled,
            "n_long_capped":   n_long_capped,
            "max_long_regime": max_long,
        }
        logger.info("feature_weight_manager_applied", **summary)
        return filtered, summary

    # ── Updates (called by SelfLearningLoop after retro) ─────────────────────

    def update_from_sweep(
        self,
        optimal_threshold: float,
        evidence: str = "",
        win_rate: float = 0.0,
    ) -> None:
        """
        Update score_threshold from ScoreThresholdSweep result.
        Writes to feature_weights.json — commit to git to preserve the learning.
        """
        self._weights.setdefault("score_threshold", {})
        self._weights["score_threshold"]["current"]          = round(optimal_threshold, 4)
        self._weights["score_threshold"]["optimal_from_sweep"] = round(optimal_threshold, 4)
        self._weights["score_threshold"]["sweep_win_rate"]   = round(win_rate, 4)
        self._weights["score_threshold"]["sweep_evidence"]   = evidence
        self._weights["_last_updated"]    = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        self._weights["_last_updated_by"] = "SelfLearningLoop.update_from_sweep"
        self._save()
        logger.info(
            "feature_weights_threshold_updated",
            threshold=optimal_threshold, win_rate=round(win_rate, 4),
        )

    def update_sector_filter(
        self,
        regime: str,
        sector: str,
        multiplier: float,
        direction: int = -1,
        evidence: str = "",
    ) -> None:
        """
        Update a sector-regime dimmer.
        multiplier=0.0 → fully suppress;  1.0 → no change.
        """
        self._weights.setdefault("sector_regime_filters", {})
        self._weights["sector_regime_filters"].setdefault(regime, {})
        self._weights["sector_regime_filters"][regime][sector] = {
            "multiplier": round(multiplier, 3),
            "direction":  direction,
            "_evidence":  evidence,
            "_updated":   datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        }
        self._weights["_last_updated"]    = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        self._weights["_last_updated_by"] = "SelfLearningLoop.update_sector_filter"
        self._save()
        logger.info(
            "feature_weights_sector_filter_updated",
            regime=regime, sector=sector, multiplier=multiplier,
        )

    def update_long_limit(self, regime: str, max_positions: int, evidence: str = "") -> None:
        """Update LONG position cap for a given regime."""
        self._weights.setdefault("long_book_limits", {})
        self._weights["long_book_limits"][regime] = {
            "max_positions": max_positions,
            "_evidence": evidence,
            "_updated": datetime.now(tz=timezone.utc).strftime("%Y-%m-%d"),
        }
        self._weights["_last_updated"] = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
        self._save()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_threshold(self) -> float:
        cfg = self._weights.get("score_threshold", {})
        return float(cfg.get("current", 0.0))

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(self._weights, indent=2) + "\n")
        logger.info("feature_weights_saved", path=str(self._path))
