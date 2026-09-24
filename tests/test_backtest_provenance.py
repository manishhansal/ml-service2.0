"""Tests for the PROXY_BACKTEST != REAL_MARKET_PNL invariant (mandate §2.1)."""
from __future__ import annotations

import pytest

from src.backtest.provenance import (
    BacktestProvenance,
    PnLProvenance,
    ProxyPnLAsEvidenceError,
    assert_not_economic_evidence,
    classify_backtest,
)


class TestClassifyBacktest:
    def test_reconstructed_path_is_proxy_not_evidence(self):
        prov = classify_backtest(
            price_path_reconstructed=True, in_sample_signals=True, real_ohlcv=False
        )
        assert prov.pnl_provenance == PnLProvenance.RECONSTRUCTED_PROXY
        assert prov.is_economic_evidence is False
        assert "proxy" in prov.reason.lower()

    def test_no_real_ohlcv_is_proxy_even_if_not_reconstructed_flag(self):
        # Absence of real OHLCV alone downgrades to proxy.
        prov = classify_backtest(
            price_path_reconstructed=False, in_sample_signals=False, real_ohlcv=False
        )
        assert prov.pnl_provenance == PnLProvenance.RECONSTRUCTED_PROXY
        assert prov.is_economic_evidence is False

    def test_real_ohlcv_in_sample_signals_not_evidence(self):
        prov = classify_backtest(
            price_path_reconstructed=False, in_sample_signals=True, real_ohlcv=True
        )
        assert prov.pnl_provenance == PnLProvenance.REAL_OHLCV
        assert prov.is_economic_evidence is False
        assert "in-sample" in prov.reason.lower()

    def test_real_ohlcv_oos_signals_is_evidence(self):
        prov = classify_backtest(
            price_path_reconstructed=False, in_sample_signals=False, real_ohlcv=True
        )
        assert prov.pnl_provenance == PnLProvenance.REAL_OHLCV
        assert prov.is_economic_evidence is True

    def test_forward_paper_is_evidence(self):
        prov = classify_backtest(
            price_path_reconstructed=False, in_sample_signals=False, forward_paper=True
        )
        assert prov.pnl_provenance == PnLProvenance.FORWARD_PAPER
        assert prov.is_economic_evidence is True


class TestAssertGuard:
    def test_guard_raises_on_proxy(self):
        prov = classify_backtest(
            price_path_reconstructed=True, in_sample_signals=True, real_ohlcv=False
        )
        with pytest.raises(ProxyPnLAsEvidenceError):
            assert_not_economic_evidence(prov)

    def test_guard_noop_on_real_evidence(self):
        prov = classify_backtest(
            price_path_reconstructed=False, in_sample_signals=False, real_ohlcv=True
        )
        # Must not raise.
        assert_not_economic_evidence(prov)

    def test_guard_raises_on_real_ohlcv_but_in_sample(self):
        prov = classify_backtest(
            price_path_reconstructed=False, in_sample_signals=True, real_ohlcv=True
        )
        with pytest.raises(ProxyPnLAsEvidenceError):
            assert_not_economic_evidence(prov)


class TestArtifactStamp:
    def test_to_dict_carries_invariant_and_flags(self):
        prov = classify_backtest(
            price_path_reconstructed=True, in_sample_signals=True, real_ohlcv=False
        )
        d = prov.to_dict()
        assert d["invariant"] == "PROXY_BACKTEST != REAL_MARKET_PNL"
        assert d["is_economic_evidence"] is False
        assert d["pnl_provenance"] == PnLProvenance.RECONSTRUCTED_PROXY
        assert d["price_path_reconstructed"] is True

    def test_provenance_is_frozen(self):
        prov = BacktestProvenance(
            pnl_provenance=PnLProvenance.RECONSTRUCTED_PROXY,
            price_path_reconstructed=True,
            in_sample_signals=True,
            is_economic_evidence=False,
            reason="x",
        )
        with pytest.raises(Exception):
            prov.pnl_provenance = PnLProvenance.REAL_OHLCV  # type: ignore[misc]
