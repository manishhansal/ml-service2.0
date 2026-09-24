"""Tests for the intraday research-state classifier (mandate §46, §48).

Locks the policy that a statistically interesting signal which does NOT survive
costs is ECONOMICALLY_UNVIABLE, never a cost-surviving candidate.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "run_intraday_research",
    Path(__file__).resolve().parent.parent / "scripts" / "run_intraday_research.py",
)
_mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_mod)  # type: ignore[union-attr]

_classify = _mod._classify
S = _mod.ResearchState


class TestResearchStateClassifier:
    def test_below_ic_threshold_is_no_signal(self):
        assert _classify(ic_mean=0.0, pbo=0.1, net_sharpe=1.0) == S.NO_SIGNAL
        assert _classify(ic_mean=-0.07, pbo=0.8, net_sharpe=-3.0) == S.NO_SIGNAL

    def test_high_pbo_is_weak_signal(self):
        assert _classify(ic_mean=0.05, pbo=0.6, net_sharpe=1.0) == S.WEAK_SIGNAL

    def test_positive_ic_low_pbo_but_negative_sharpe_is_uneconomic(self):
        # The 5m real-data case: IC +0.063, PBO 0.0, net Sharpe -3.04.
        assert _classify(
            ic_mean=0.0627, pbo=0.0, net_sharpe=-3.04
        ) == S.ECONOMICALLY_UNVIABLE

    def test_positive_ic_low_pbo_positive_sharpe_is_candidate(self):
        assert _classify(
            ic_mean=0.05, pbo=0.1, net_sharpe=0.8
        ) == S.COST_SURVIVING_RESEARCH_CANDIDATE

    def test_cost_gate_is_decisive_over_statistics(self):
        # Even a strong IC / zero PBO cannot become a candidate if it loses money.
        assert _classify(
            ic_mean=0.20, pbo=0.0, net_sharpe=-0.01
        ) == S.ECONOMICALLY_UNVIABLE
