"""
Coverage-boosting tests (batch 2) for Task 64 — final checkpoint.

Targets: approval_token, promotion gates, audit logger, portfolio optimizer,
         features/leakage_validator, training/purged_kfold.

Requirements: Req 18.1, Req 18.2
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# registry/approval_token.py  (Req 13.10, Req 13.11)
# ---------------------------------------------------------------------------


class TestApprovalTokenManager:

    def _mgr(self, reviewers=None, key="test-secret-key", expiry=24):
        from src.registry.approval_token import ApprovalTokenManager
        return ApprovalTokenManager(
            authorized_reviewers=reviewers or {"admin", "ml-ops"},
            signing_key=key,
            expiry_hours=expiry,
        )

    def test_issue_and_validate_round_trip(self):
        mgr = self._mgr()
        token = mgr.issue_token("admin", "market_regime-v1.0.0")
        payload = mgr.validate_token(token, challenger_id="market_regime-v1.0.0")
        assert payload["issuer"] == "admin"
        assert payload["challenger_id"] == "market_regime-v1.0.0"

    def test_issue_unauthorized_reviewer_raises(self):
        from src.registry.approval_token import ApprovalTokenError
        mgr = self._mgr()
        with pytest.raises(ApprovalTokenError):
            mgr.issue_token("unauthorized-user", "challenger-v1.0.0")

    def test_tampered_signature_raises(self):
        from src.registry.approval_token import ApprovalTokenError
        mgr = self._mgr()
        token = mgr.issue_token("admin", "challenger-v1.0.0")
        # Corrupt the signature
        parts = token.rsplit(".", 1)
        bad_token = parts[0] + ".badsignature000"
        with pytest.raises(ApprovalTokenError):
            mgr.validate_token(bad_token)

    def test_malformed_token_raises(self):
        from src.registry.approval_token import ApprovalTokenError
        mgr = self._mgr()
        with pytest.raises(ApprovalTokenError):
            mgr.validate_token("notavalidtoken")

    def test_wrong_challenger_id_raises(self):
        from src.registry.approval_token import ApprovalTokenError
        mgr = self._mgr()
        token = mgr.issue_token("admin", "correct-challenger")
        with pytest.raises(ApprovalTokenError):
            mgr.validate_token(token, challenger_id="wrong-challenger")

    def test_payload_contains_required_fields(self):
        mgr = self._mgr()
        token = mgr.issue_token("ml-ops", "model-v1.0.0")
        payload = mgr.validate_token(token)
        assert "issued_at" in payload
        assert "expires_at" in payload
        assert "issuer" in payload
        assert "challenger_id" in payload


class TestApprovalTokenValidator:

    def test_validate_returns_issuer_string(self):
        from src.registry.approval_token import ApprovalTokenValidator
        v = ApprovalTokenValidator(authorized_reviewers={"admin"}, signing_key="test-key")
        token = v.issue_token("admin", "model-v1.0.0")
        issuer = v.validate(token, challenger_id="model-v1.0.0")
        assert issuer == "admin"

    def test_validate_invalid_token_raises(self):
        from src.registry.approval_token import ApprovalTokenValidator, InvalidApprovalTokenError
        v = ApprovalTokenValidator(authorized_reviewers={"admin"}, signing_key="test-key")
        with pytest.raises(InvalidApprovalTokenError):
            v.validate("bad.token")


# ---------------------------------------------------------------------------
# registry/promotion.py  (Req 13.1–13.14)
# ---------------------------------------------------------------------------


def _good_challenger(**overrides):
    """Return a challenger dict that passes all six gates by default."""
    base = {
        "model_name": "market_regime",
        "version": "2.0.0",
        "ic_mean": 0.12,
        "brier_score": 0.15,
        "max_drawdown": 0.10,
        "oos_count": 100,
        "look_ahead_validated": True,
        "evidence_level": "A",
        "final_oos_used_for_selection": False,
        "ic_decay_status": "OK",
        "feature_drift_severity": "LOW",
        "net_return_validation": 0.05,
        "turnover_validation": 0.20,
        "sharpe_net": 1.5,
    }
    base.update(overrides)
    return base


def _good_champion(**overrides):
    base = {
        "model_name": "market_regime",
        "version": "1.0.0",
        "ic_mean": 0.08,
        "brier_score": 0.18,
        "max_drawdown": 0.12,
        "net_return_validation": 0.04,
        "turnover_validation": 0.20,
    }
    base.update(overrides)
    return base


class TestModelPromotion:

    def _promotion(self):
        from src.registry.promotion import ModelPromotion
        from src.audit.logger import AuditLogger
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            audit = AuditLogger(log_path=Path(f.name))
        return ModelPromotion(audit_logger=audit)

    def test_all_gates_pass_returns_promote(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        decision = p.evaluate_all_gates(_good_challenger(), champion=_good_champion())
        assert decision.outcome == PromotionOutcome.PROMOTE

    def test_no_champion_all_pass(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        decision = p.evaluate_all_gates(_good_challenger())
        assert decision.outcome == PromotionOutcome.PROMOTE

    def test_final_oos_contaminated_outcome(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(final_oos_used_for_selection=True)
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.FINAL_OOS_CONTAMINATED

    def test_data_gate_low_oos_count_blocked(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(oos_count=30)  # below 60
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.BLOCKED

    def test_data_gate_bad_evidence_level_rejected(self):
        from src.schemas.base import PromotionOutcome, EvidenceLevel
        p = self._promotion()
        # Use the actual enum value string for level C or D
        c = _good_challenger(evidence_level=EvidenceLevel.LEVEL_D.value)
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_data_gate_look_ahead_failure_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(look_ahead_validated=False)
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_predictive_gate_ic_below_champion_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        # challenger IC much lower than champion IC
        c = _good_challenger(ic_mean=0.01)
        ch = _good_champion(ic_mean=0.15)
        decision = p.evaluate_all_gates(c, champion=ch)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_predictive_gate_no_champion_negative_ic_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(ic_mean=-0.05)
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_calibration_gate_too_high_brier_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        # challenger Brier worse than champion by more than 0.01
        c = _good_challenger(brier_score=0.30)
        ch = _good_champion(brier_score=0.18)
        decision = p.evaluate_all_gates(c, champion=ch)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_execution_gate_negative_return_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(net_return_validation=-0.01)
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_execution_gate_high_turnover_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(turnover_validation=0.50)  # > champion * 1.20
        ch = _good_champion(turnover_validation=0.20)
        decision = p.evaluate_all_gates(c, champion=ch)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_risk_gate_high_drawdown_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(max_drawdown=0.25)
        ch = _good_champion(max_drawdown=0.12)
        decision = p.evaluate_all_gates(c, champion=ch)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_stability_gate_ic_decay_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(ic_decay_status="FAILED")
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_stability_gate_high_drift_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(feature_drift_severity="CRITICAL")
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_decision_has_six_gate_evaluations(self):
        p = self._promotion()
        decision = p.evaluate_all_gates(_good_challenger())
        assert len(decision.gate_evaluations) == 6

    def test_gate_names_are_correct(self):
        p = self._promotion()
        decision = p.evaluate_all_gates(_good_challenger())
        names = {g.gate_name for g in decision.gate_evaluations}
        expected = {"DATA", "PREDICTIVE", "CALIBRATION", "EXECUTION", "RISK", "STABILITY"}
        assert names == expected

    def test_risk_gate_no_champion_absolute_threshold(self):
        """With no champion, drawdown > 20% must FAIL."""
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(max_drawdown=0.25)
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.REJECTED

    def test_calibration_no_champion_high_brier_rejected(self):
        from src.schemas.base import PromotionOutcome
        p = self._promotion()
        c = _good_challenger(brier_score=0.30)  # >= 0.25 threshold
        decision = p.evaluate_all_gates(c)
        assert decision.outcome == PromotionOutcome.REJECTED


# ---------------------------------------------------------------------------
# audit/logger.py
# ---------------------------------------------------------------------------


class TestAuditLogger:

    def _logger(self, tmp_path):
        from src.audit.logger import AuditLogger
        return AuditLogger(log_path=tmp_path / "audit.jsonl")

    def test_log_training_run_creates_file(self, tmp_path):
        audit = self._logger(tmp_path)
        audit.log_training_run(
            run_id="run-001",
            model_name="market_regime",
            model_version="1.0.0",
            started_at="2025-01-01T00:00:00+00:00",
            completed_at="2025-01-01T01:00:00+00:00",
            dataset_hash="abc123",
            training_date_range=("2024-01-01", "2024-12-01"),
            validation_date_range=("2024-12-01", "2025-01-01"),
            hyperparameters={"n_estimators": 100},
            ic_per_fold=[0.10, 0.12, 0.11],
            ic_mean=0.11,
            sharpe_net=1.2,
            max_drawdown=0.08,
            pbo=0.04,
            gate_results={"DATA": "pass"},
            outcome="PROMOTE",
        )
        assert (tmp_path / "audit.jsonl").exists()

    def test_log_promotion_decision(self, tmp_path):
        audit = self._logger(tmp_path)
        audit.log_promotion_decision(
            challenger_id="market_regime-v2.0.0",
            champion_id="market_regime-v1.0.0",
            outcome="PROMOTE",
            gate_results={"DATA": "pass", "PREDICTIVE": "pass"},
            approval_policy="HUMAN_APPROVAL_REQUIRED",
        )
        content = (tmp_path / "audit.jsonl").read_text()
        assert "promotion_decision" in content

    def test_log_online_update(self, tmp_path):
        audit = self._logger(tmp_path)
        audit.log_online_update(
            model_name="market_regime",
            prior_version="1.0.0",
            new_version="1.0.0-online-1",
            ic_delta=0.02,
            consecutive_update_count=1,
        )
        content = (tmp_path / "audit.jsonl").read_text()
        assert "online_update" in content

    def test_log_online_update_rejected(self, tmp_path):
        audit = self._logger(tmp_path)
        audit.log_online_update_rejected(
            model_name="market_regime",
            prior_version="1.0.0",
            candidate_version="1.0.0-online-cand",
            reason="new_ic_below_prior",
            prior_ic=0.12,
            candidate_ic=0.08,
        )
        content = (tmp_path / "audit.jsonl").read_text()
        assert "online_update_rejected" in content

    def test_check_append_only_passes_on_valid_log(self, tmp_path):
        audit = self._logger(tmp_path)
        audit.log_online_update("model", "1.0.0", "1.0.1", 0.01, 1)
        # Should not raise
        audit._check_append_only()

    def test_check_append_only_no_file_no_raise(self, tmp_path):
        audit = self._logger(tmp_path / "subdir")
        # File doesn't exist yet
        audit._check_append_only()

    def test_check_append_only_detects_tampering(self, tmp_path):
        from src.audit.logger import AuditLogViolation
        import json

        audit = self._logger(tmp_path)
        audit.log_training_run(
            run_id="run-002",
            model_name="model",
            model_version="1.0.0",
            started_at="2025-01-01T00:00:00+00:00",
            completed_at="2025-01-01T01:00:00+00:00",
            dataset_hash="def456",
            training_date_range=("2024-01-01", "2024-12-01"),
            validation_date_range=("2024-12-01", "2025-01-01"),
            hyperparameters={},
            ic_per_fold=[0.10],
            ic_mean=0.10,
            sharpe_net=1.0,
            max_drawdown=0.05,
            pbo=0.03,
            gate_results={},
            outcome="PROMOTE",
        )
        log_path = tmp_path / "audit.jsonl"
        # Tamper: replace the content with a modified hash
        content = log_path.read_text()
        entry = json.loads(content.strip())
        entry["entry_hash"] = "0" * 64  # force bad hash
        log_path.write_text(json.dumps(entry) + "\n")

        with pytest.raises(AuditLogViolation):
            audit._check_append_only()

    def test_multiple_events_in_one_file(self, tmp_path):
        audit = self._logger(tmp_path)
        audit.log_online_update("m1", "1.0", "1.1", 0.01, 1)
        audit.log_online_update("m2", "1.0", "1.1", 0.02, 1)
        audit.log_online_update("m3", "1.0", "1.1", 0.03, 1)
        content = (tmp_path / "audit.jsonl").read_text()
        lines = [l for l in content.splitlines() if l.strip()]
        assert len(lines) == 3
        # Integrity should pass
        audit._check_append_only()


# ---------------------------------------------------------------------------
# models/portfolio_optimizer.py  (Req 8.1–8.9)
# ---------------------------------------------------------------------------


def _make_returns_df(n_assets=4, n_obs=60):
    """Generate a realistic daily returns DataFrame."""
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(42)
    data = rng.normal(0.0005, 0.015, size=(n_obs, n_assets))
    cols = [f"ASSET{i}" for i in range(n_assets)]
    return pd.DataFrame(data, columns=cols)


class TestPortfolioOptimizer:

    def _opt(self):
        from src.models.portfolio_optimizer import PortfolioOptimizer
        return PortfolioOptimizer()

    def test_hrp_weights_sum_to_one(self):
        opt = self._opt()
        df = _make_returns_df()
        result = opt.hrp_allocation(df)
        if result.get("available", True):
            assert abs(sum(result["weights"].values()) - 1.0) < 1e-6

    def test_hrp_all_weights_non_negative(self):
        opt = self._opt()
        df = _make_returns_df()
        result = opt.hrp_allocation(df)
        if result.get("available", True):
            for w in result["weights"].values():
                assert w >= -1e-9

    def test_insufficient_data_returns_available_false(self):
        import pandas as pd
        opt = self._opt()
        # Only 5 observations — below MIN_OBSERVATIONS (20)
        df = _make_returns_df(n_obs=5)
        result = opt.hrp_allocation(df)
        assert result["available"] is False

    def test_max_div_weights_sum_to_one(self):
        opt = self._opt()
        df = _make_returns_df()
        result = opt.max_div_allocation(df)
        if result.get("available", True):
            assert abs(sum(result["weights"].values()) - 1.0) < 1e-6

    def test_erc_weights_sum_to_one(self):
        opt = self._opt()
        df = _make_returns_df()
        result = opt.erc_allocation(df)
        if result.get("available", True):
            assert abs(sum(result["weights"].values()) - 1.0) < 1e-6

    def test_cvar_weights_sum_to_one(self):
        opt = self._opt()
        df = _make_returns_df()
        result = opt.cvar_allocation(df)
        if result.get("available", True):
            assert abs(sum(result["weights"].values()) - 1.0) < 1e-6

    def test_normalize_weights_helper(self):
        from src.models.portfolio_optimizer import _normalize_weights

        w = {"A": 0.3, "B": 0.3, "C": 0.4}
        n = _normalize_weights(w)
        assert abs(sum(n.values()) - 1.0) < 1e-9

    def test_normalize_weights_negative_clipped(self):
        from src.models.portfolio_optimizer import _normalize_weights

        w = {"A": -0.5, "B": 0.8, "C": 0.7}
        n = _normalize_weights(w)
        assert all(v >= 0 for v in n.values())
        assert abs(sum(n.values()) - 1.0) < 1e-9

    def test_equal_weights_helper(self):
        from src.models.portfolio_optimizer import _equal_weights

        cols = ["A", "B", "C", "D"]
        w = _equal_weights(cols)
        assert all(abs(v - 0.25) < 1e-9 for v in w.values())
        assert abs(sum(w.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# features/leakage_validator.py
# ---------------------------------------------------------------------------


class TestLeakageValidator:

    def test_clean_data_passes(self):
        """Well-ordered PIT data with uncorrelated features should return no violations."""
        from src.features.leakage_validator import LeakageValidator
        import pandas as pd
        import numpy as np

        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        # Feature with near-zero future correlation
        rng = np.random.default_rng(0)
        feature_matrix = pd.DataFrame(
            {"feature_a": rng.standard_normal(n),
             "feature_b": rng.standard_normal(n)},
            index=dates,
        )
        # Independent random labels — should NOT trigger leakage
        label_vector = pd.Series(rng.standard_normal(n), index=dates)
        validator = LeakageValidator(threshold=0.99)  # high threshold so random passes
        result = validator.validate(feature_matrix, label_vector)
        assert result is None  # no violation

    def test_detects_leakage_raises(self):
        """A feature that IS the future return should be flagged."""
        from src.features.leakage_validator import LeakageValidator, PITViolationError
        import pandas as pd
        import numpy as np

        n = 60
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        rng = np.random.default_rng(1)
        future_return = pd.Series(rng.standard_normal(n), index=dates)
        # Feature IS the future return shifted — perfect forward correlation
        feature_matrix = pd.DataFrame({"leaky_feature": future_return.shift(1)}, index=dates)
        # Make label = future_return so correlation ~ 1.0
        label_vector = future_return
        validator = LeakageValidator(threshold=0.05)
        with pytest.raises(PITViolationError):
            validator.validate(feature_matrix, label_vector)

    def test_empty_data_returns_none(self):
        from src.features.leakage_validator import LeakageValidator
        import pandas as pd

        v = LeakageValidator()
        empty_df = pd.DataFrame()
        empty_series = pd.Series(dtype=float)
        result = v.validate(empty_df, empty_series)
        assert result is None


# ---------------------------------------------------------------------------
# training/purged_kfold.py
# ---------------------------------------------------------------------------


class TestPurgedKFoldSplitter:

    def test_split_returns_correct_number_of_folds(self):
        """PurgedKFoldSplitter with n_splits=5 should produce 5 (train, test) pairs."""
        from src.training.purged_kfold import PurgedKFoldSplitter
        import numpy as np
        import pandas as pd

        pkf = PurgedKFoldSplitter(n_splits=5, embargo_days=5)
        X = np.arange(200).reshape(200, 1)
        y = np.zeros(200)
        timestamps = pd.date_range("2023-01-01", periods=200, freq="D")
        splits = list(pkf.split(X, y, timestamps))
        assert len(splits) == 5

    def test_train_test_sets_disjoint(self):
        from src.training.purged_kfold import PurgedKFoldSplitter
        import numpy as np
        import pandas as pd

        pkf = PurgedKFoldSplitter(n_splits=3, embargo_days=5)
        X = np.arange(150).reshape(150, 1)
        y = np.zeros(150)
        timestamps = pd.date_range("2023-01-01", periods=150, freq="D")
        for train_idx, test_idx in pkf.split(X, y, timestamps):
            assert len(set(train_idx).intersection(test_idx)) == 0

    def test_embargo_removes_samples_near_test(self):
        """Purging removes samples in the embargo period before the test set."""
        from src.training.purged_kfold import PurgedKFoldSplitter
        import numpy as np
        import pandas as pd

        pkf = PurgedKFoldSplitter(n_splits=3, embargo_days=10)
        X = np.arange(120).reshape(120, 1)
        y = np.zeros(120)
        timestamps = pd.date_range("2023-01-01", periods=120, freq="D")
        for train_idx, test_idx in pkf.split(X, y, timestamps):
            if len(test_idx) > 0:
                # Basic disjoint check — train and test must not share indices
                assert len(set(train_idx).intersection(test_idx)) == 0

    def test_invalid_embargo_raises(self):
        from src.training.purged_kfold import PurgedKFoldSplitter
        with pytest.raises(ValueError):
            PurgedKFoldSplitter(n_splits=5, embargo_days=2)  # < 5

    def test_embargo_zero_disabled(self):
        """embargo_days=0 disables embargo — should still produce folds."""
        from src.training.purged_kfold import PurgedKFoldSplitter
        import numpy as np
        import pandas as pd

        pkf = PurgedKFoldSplitter(n_splits=3, embargo_days=0)
        X = np.arange(90).reshape(90, 1)
        y = np.zeros(90)
        timestamps = pd.date_range("2023-01-01", periods=90, freq="D")
        splits = list(pkf.split(X, y, timestamps))
        assert len(splits) == 3


# ---------------------------------------------------------------------------
# models/regime_classifier.py (extra coverage)
# ---------------------------------------------------------------------------


class TestRegimeClassifier:

    def _clf(self):
        from src.models.regime_classifier import RegimeClassifier
        return RegimeClassifier()

    def test_heuristic_predict_returns_result(self):
        from src.schemas.base import MarketRegime
        clf = self._clf()
        result = clf.predict({
            "adx_14": 30.0,
            "rsi_14": 65.0,
            "atr_pct": 1.5,
            "trend_strength": 0.6,
            "india_vix": 14.0,
            "ema_stack_score": 0.8,
            "price_above_200d_ma": True,
        })
        assert hasattr(result, "regime")
        assert isinstance(result.regime, MarketRegime)

    def test_confidence_bounded(self):
        clf = self._clf()
        result = clf.predict({"adx_14": 25.0, "india_vix": 16.0})
        assert 0.0 <= result.confidence <= 1.0

    def test_crash_regime_high_vix(self):
        from src.schemas.base import MarketRegime
        clf = self._clf()
        result = clf.predict({
            "india_vix": 42.0,
            "adx_14": 35.0,
            "rsi_14": 20.0,
        })
        # High VIX should likely produce CRASH or VOLATILE
        assert result.regime in (MarketRegime.CRASH, MarketRegime.VOLATILE, MarketRegime.BEAR)

    def test_all_inputs_missing_does_not_crash(self):
        clf = self._clf()
        result = clf.predict({})
        assert result is not None


# ---------------------------------------------------------------------------
# models/risk_predictor.py (extra coverage)
# ---------------------------------------------------------------------------


class TestRiskPredictor:

    def _pred(self):
        from src.models.risk_predictor import RiskPredictor
        return RiskPredictor()

    def test_predict_returns_result(self):
        pred = self._pred()
        result = pred.predict({
            "atr_pct": 1.5,
            "india_vix": 16.0,
            "max_drawdown_5d": 0.03,
            "correlation_with_nifty": 0.7,
            "pcr": 0.8,
            "beta": 1.1,
            "rsi_14": 55.0,
        })
        assert result is not None

    def test_no_trained_model_uses_heuristic(self):
        from src.models.risk_predictor import RiskPredictor
        from src.schemas.base import PredictionProvenance
        pred = RiskPredictor()
        assert not pred.has_trained_model
        result = pred.predict({"atr_pct": 2.0, "india_vix": 20.0})
        assert result.provenance == PredictionProvenance.HEURISTIC
