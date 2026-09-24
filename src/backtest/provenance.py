"""
src.backtest.provenance — enforce ``PROXY_BACKTEST != REAL_MARKET_PNL`` (mandate §2.1).

The certification harness builds a backtest whose price path is *reconstructed
from the dataset's realized (label-horizon) returns* when tick-accurate market
bars are not replayed. That path is contaminated by construction: the label is
a forward-looking quantity, so any P&L derived from it is a pipeline-illustration
proxy, never economic evidence.

The forensic audit (see ML_SERVICE_FINAL_CERTIFICATION.md → "Proxy-Backtest
Contradiction") confirmed that NO shadow/production eligibility gate consumes
this proxy P&L today — shadow eligibility keys only on walk-forward OOS IC /
CPCV PBO / walk-forward net Sharpe / calibration ECE, and the six promotion
gates read held-out *validation-partition* metrics. This module turns that
property from an incidental fact into an explicitly asserted, testable invariant
so a future edit cannot silently wire proxy P&L into an eligibility decision.

Usage:
    from src.backtest.provenance import BacktestProvenance, classify_backtest, \
        assert_not_economic_evidence

    prov = classify_backtest(price_path_reconstructed=True, in_sample_signals=True)
    backtest_artifact = {**bt.to_dict(), **prov.to_dict()}
    # Before any eligibility decision that might read P&L:
    assert_not_economic_evidence(prov)   # raises unless prov permits it
"""
from __future__ import annotations

from dataclasses import dataclass


class ProxyPnLAsEvidenceError(RuntimeError):
    """Raised if proxy / reconstructed-path backtest P&L is used as economic evidence."""


# Machine-readable P&L provenance classes (distinct from data_source_class).
class PnLProvenance:
    #: Price path reconstructed from realized (forward) label returns. Proxy only.
    RECONSTRUCTED_PROXY = "RECONSTRUCTED_PROXY"
    #: Backtest over ACTUAL historical OHLCV bars, out-of-sample signals, next-bar exec.
    REAL_OHLCV = "REAL_OHLCV"
    #: Forward paper: signal at T, outcome strictly after T on live data.
    FORWARD_PAPER = "FORWARD_PAPER"


@dataclass(frozen=True)
class BacktestProvenance:
    """Immutable classification of how a backtest's P&L was produced."""

    pnl_provenance: str
    price_path_reconstructed: bool
    in_sample_signals: bool
    #: True only when this P&L is admissible as economic evidence for eligibility.
    is_economic_evidence: bool
    reason: str

    def to_dict(self) -> dict:
        return {
            "pnl_provenance": self.pnl_provenance,
            "price_path_reconstructed": self.price_path_reconstructed,
            "in_sample_signals": self.in_sample_signals,
            "is_economic_evidence": self.is_economic_evidence,
            "provenance_reason": self.reason,
            # A loud, human-readable invariant restated in every artifact.
            "invariant": "PROXY_BACKTEST != REAL_MARKET_PNL",
        }


def classify_backtest(
    *,
    price_path_reconstructed: bool,
    in_sample_signals: bool,
    real_ohlcv: bool = False,
    forward_paper: bool = False,
) -> BacktestProvenance:
    """Classify a backtest's P&L provenance and whether it is economic evidence.

    A backtest is economic evidence ONLY when it is run on real OHLCV bars with
    out-of-sample signals and no reconstructed price path. Any reconstruction or
    in-sample signal contamination downgrades it to a proxy.
    """
    if forward_paper:
        return BacktestProvenance(
            pnl_provenance=PnLProvenance.FORWARD_PAPER,
            price_path_reconstructed=False,
            in_sample_signals=False,
            is_economic_evidence=True,
            reason="Forward-paper P&L (signal at T, outcome after T on live data).",
        )
    if price_path_reconstructed or not real_ohlcv:
        return BacktestProvenance(
            pnl_provenance=PnLProvenance.RECONSTRUCTED_PROXY,
            price_path_reconstructed=True,
            in_sample_signals=in_sample_signals,
            is_economic_evidence=False,
            reason=(
                "Price path reconstructed from realized (forward) label returns; "
                "P&L is a pipeline-illustration proxy, NOT tick-accurate market P&L. "
                "Forbidden as shadow/production economic evidence (mandate §2.1)."
            ),
        )
    if in_sample_signals:
        return BacktestProvenance(
            pnl_provenance=PnLProvenance.REAL_OHLCV,
            price_path_reconstructed=False,
            in_sample_signals=True,
            is_economic_evidence=False,
            reason=(
                "Real OHLCV bars but signals are in-sample fit-and-predict; "
                "not out-of-sample, so not admissible as economic evidence."
            ),
        )
    return BacktestProvenance(
        pnl_provenance=PnLProvenance.REAL_OHLCV,
        price_path_reconstructed=False,
        in_sample_signals=False,
        is_economic_evidence=True,
        reason="Real OHLCV bars, out-of-sample signals, next-bar execution.",
    )


def assert_not_economic_evidence(prov: BacktestProvenance) -> None:
    """Guard: raise if a proxy backtest is about to be used as economic evidence.

    Call this at any site that would let backtest P&L influence shadow or
    production eligibility. It is a no-op for genuine REAL_OHLCV / FORWARD_PAPER
    provenance and a hard failure for a RECONSTRUCTED_PROXY.
    """
    if not prov.is_economic_evidence:
        raise ProxyPnLAsEvidenceError(
            f"Refusing to use {prov.pnl_provenance} backtest P&L as economic "
            f"evidence: {prov.reason}"
        )
