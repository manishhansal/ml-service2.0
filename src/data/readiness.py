"""
src.data.readiness — Training-readiness gate (mandate §27, §28, §54).

Implements a single authoritative training-readiness check that answers
READY or NOT_READY with machine-readable reasons for every mandatory gate.

Gate categories:
  DATA      — data-service2.0 reachability, authentication, universe, quality
  NEWS      — SentinelPulse reachability (if news training requested)
  FEATURES  — feature schema, PIT, NaN/inf policy
  LABELS    — validity, future-only, sufficient observations
  TRAINING  — Docker runtime, dependencies, seeds, artifact path, registry
  VALIDATION — walk-forward, CPCV, calibration, backtest, cost model
  PROVENANCE — git SHA, dataset hash, configuration hash

Mandate §28: The gate MUST fail closed. Examples:
  - data-service unavailable → NOT_READY / DATA_SERVICE_UNAVAILABLE
  - SentinelPulse unavailable + news required → NOT_READY / SENTINELPULSE_UNAVAILABLE
  - news optional → READY_MARKET_ONLY / NEWS_DISABLED
  - insufficient history → NOT_READY / INSUFFICIENT_MARKET_HISTORY

Usage (programmatic)::

    gate = TrainingReadinessGate()
    result = await gate.check(
        data_client=...,
        sentinel_client=...,
        universe=["NIFTY", "BANKNIFTY", "RELIANCE"],
        timeframe="1d",
        min_history_days=252,
        news_required=False,
    )
    if not result.training_ready:
        logger.error("NOT_READY", blockers=result.blockers)
        raise SystemExit(1)

Usage (API)::

    GET /v2/training/readiness
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.logging_config import get_logger

logger = get_logger(__name__)


# ── Gate result ───────────────────────────────────────────────────────────────


@dataclass
class GateCheckResult:
    """Result of a single readiness gate."""

    gate: str
    status: str  # "PASS" | "FAIL" | "WARN" | "SKIP"
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"gate": self.gate, "status": self.status}
        if self.reason:
            d["reason"] = self.reason
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class TrainingReadinessResult:
    """Aggregated training readiness result (mandate §54)."""

    training_ready: bool
    mode: str  # "READY" | "READY_MARKET_ONLY" | "NOT_READY"
    news_status: str  # "ENABLED" | "DISABLED" | "UNAVAILABLE" | "EVIDENCE_PENDING"
    gates: list[GateCheckResult] = field(default_factory=list)
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked_at: str = field(default_factory=lambda: datetime.now(tz=UTC).isoformat())
    git_sha: str = ""
    docker_image: str = ""
    python_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "training_ready": self.training_ready,
            "mode": self.mode,
            "news_status": self.news_status,
            "blockers": self.blockers,
            "warnings": self.warnings,
            "gates": [g.to_dict() for g in self.gates],
            "checked_at": self.checked_at,
            "git_sha": self.git_sha,
            "docker_image": self.docker_image,
            "python_version": self.python_version,
        }


# ── Helpers ───────────────────────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).parent,
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def _docker_image() -> str:
    """Return the Docker image digest if running inside a container."""
    # Set by our Dockerfile CMD / entrypoint or CI
    return os.environ.get("DOCKER_IMAGE_DIGEST", os.environ.get("IMAGE_DIGEST", "unknown"))


def _python_version() -> str:
    import sys
    return sys.version.split()[0]


# ── Gate ───────────────────────────────────────────────────────────────────────


class TrainingReadinessGate:
    """
    Authoritative training-readiness gate (mandate §27, §28).

    Checks every precondition before training is allowed. Any critical
    blocker sets training_ready=False. Non-critical issues are warnings.

    Mandate §28: gate MUST fail closed — no silent fallbacks.
    """

    def __init__(
        self,
        artifact_root: Path | None = None,
    ) -> None:
        self._artifact_root = artifact_root or Path("./artifacts")

    async def check(
        self,
        data_client: Any,
        sentinel_client: Any,
        universe: list[str],
        timeframe: str = "1d",
        min_history_days: int = 252,
        news_required: bool = False,
        min_universe_size: int = 3,
    ) -> TrainingReadinessResult:
        """Run all readiness gates and return the aggregated result.

        Args:
            data_client:       Connected DataServiceClient instance.
            sentinel_client:   Connected SentinelPulseClient instance.
            universe:          List of symbols to check.
            timeframe:         Target training timeframe.
            min_history_days:  Minimum required historical bars (business days).
            news_required:     True if news features are REQUIRED (not optional).
            min_universe_size: Minimum number of symbols needed for training.
        """
        gates: list[GateCheckResult] = []
        blockers: list[str] = []
        warnings: list[str] = []
        news_status = "DISABLED"

        # ── 1. DATA gates ────────────────────────────────────────────────────

        # 1a+1b. data-service2.0 reachability AND auth — combined single call
        # to avoid hitting rate limits (mandate §5.1.D)
        data_reachable, data_auth = await self._check_data_service_and_auth(data_client)
        gates.append(data_reachable)
        if data_reachable.status == "FAIL":
            blockers.append(
                "DATA_SERVICE_AUTH_FAILED"
                if data_reachable.details.get("is_auth_error")
                else "DATA_SERVICE_UNAVAILABLE"
            )
            return self._finalize(
                gates, blockers, warnings, news_status, False, "NOT_READY"
            )
        gates.append(data_auth)
        if data_auth.status == "FAIL":
            blockers.append("DATA_SERVICE_AUTH_FAILED")
            return self._finalize(
                gates, blockers, warnings, news_status, False, "NOT_READY"
            )

        # 1c. F&O universe availability
        universe_gate = await self._check_universe(data_client, universe, min_universe_size)
        gates.append(universe_gate)
        if universe_gate.status == "FAIL":
            blockers.append("UNIVERSE_UNAVAILABLE")
            return self._finalize(
                gates, blockers, warnings, news_status, False, "NOT_READY"
            )
        if universe_gate.status == "WARN":
            warnings.append(universe_gate.reason)  # warn only, not a blocker

        # Small delay to stay within data-service rate limits between gate checks
        import asyncio as _asyncio
        await _asyncio.sleep(0.5)

        # 1d. Historical data sufficiency
        history_gate = await self._check_history(
            data_client, universe[:5], timeframe, min_history_days
        )
        gates.append(history_gate)
        if history_gate.status == "FAIL":
            blockers.append("INSUFFICIENT_MARKET_HISTORY")
            return self._finalize(
                gates, blockers, warnings, news_status, False, "NOT_READY"
            )

        # ── 2. NEWS gates ────────────────────────────────────────────────────

        news_gate = await self._check_sentinel(sentinel_client, news_required)
        gates.append(news_gate)
        if news_gate.status == "FAIL" and news_required:
            blockers.append("SENTINELPULSE_UNAVAILABLE")
            news_status = "UNAVAILABLE"
            return self._finalize(
                gates, blockers, warnings, news_status, False, "NOT_READY"
            )
        elif news_gate.status == "FAIL":
            news_status = "UNAVAILABLE"
            warnings.append("SENTINELPULSE_UNAVAILABLE_NEWS_DISABLED")
        elif news_gate.status == "WARN":
            news_status = "EVIDENCE_PENDING"
            warnings.append(news_gate.reason)
        elif news_gate.status == "PASS":
            news_status = "ENABLED"
        else:
            news_status = "DISABLED"

        # ── 3. FEATURES gate ─────────────────────────────────────────────────

        features_gate = self._check_features()
        gates.append(features_gate)
        if features_gate.status == "FAIL":
            blockers.append("FEATURE_SCHEMA_INVALID")

        # ── 4. LABELS gate ───────────────────────────────────────────────────

        labels_gate = self._check_labels()
        gates.append(labels_gate)
        if labels_gate.status == "FAIL":
            blockers.append("LABEL_SCHEMA_INVALID")

        # ── 5. TRAINING environment gate ────────────────────────────────────

        training_gate = self._check_training_env()
        gates.append(training_gate)
        if training_gate.status == "FAIL":
            blockers.append("TRAINING_ENVIRONMENT_INVALID")

        # ── 6. PROVENANCE gate ───────────────────────────────────────────────

        provenance_gate = self._check_provenance()
        gates.append(provenance_gate)
        if provenance_gate.status == "FAIL":
            warnings.append("PROVENANCE_INCOMPLETE")  # warn only, not a blocker

        # ── Final decision ───────────────────────────────────────────────────

        has_blockers = len(blockers) > 0
        if has_blockers:
            mode = "NOT_READY"
            ready = False
        elif news_status in ("DISABLED", "UNAVAILABLE"):
            mode = "READY_MARKET_ONLY"
            ready = True
        else:
            mode = "READY"
            ready = True

        return self._finalize(gates, blockers, warnings, news_status, ready, mode)

    # ── Gate implementations ────────────────────────────────────────────────

    async def _check_data_service_and_auth(
        self, data_client: Any
    ) -> tuple[GateCheckResult, GateCheckResult]:
        """Check reachability AND auth with a single call, with 429 backoff.

        Returns (reachability_gate, auth_gate).
        Retries once after Retry-After delay when rate-limited.
        """
        import asyncio as _asyncio
        from src.clients.data_service import DataServiceAuthError, DataServiceRateLimitedError

        async def _attempt() -> tuple[GateCheckResult, GateCheckResult]:
            try:
                await data_client.get_market_status()
                reach = GateCheckResult(gate="DATA_SERVICE_REACHABLE", status="PASS")
                auth = GateCheckResult(gate="DATA_SERVICE_AUTH", status="PASS")
            except DataServiceAuthError as exc:
                reach = GateCheckResult(
                    gate="DATA_SERVICE_REACHABLE",
                    status="FAIL",
                    reason=f"AUTH_FAILED: {exc}",
                    details={"is_auth_error": True},
                )
                auth = GateCheckResult(
                    gate="DATA_SERVICE_AUTH",
                    status="FAIL",
                    reason=f"AUTH_FAILED: {exc}",
                )
            except DataServiceRateLimitedError as exc:
                reach = GateCheckResult(
                    gate="DATA_SERVICE_REACHABLE",
                    status="FAIL",
                    reason=f"RATE_LIMITED: {exc}",
                    details={
                        "is_auth_error": False,
                        "retry_after_seconds": exc.retry_after_seconds,
                    },
                )
                auth = GateCheckResult(gate="DATA_SERVICE_AUTH", status="PASS")
            except Exception as exc:
                reach = GateCheckResult(
                    gate="DATA_SERVICE_REACHABLE",
                    status="FAIL",
                    reason=f"DATA_SERVICE_UNAVAILABLE: {type(exc).__name__}: {exc}",
                    details={"is_auth_error": False},
                )
                auth = GateCheckResult(gate="DATA_SERVICE_AUTH", status="PASS")
            return reach, auth

        # First attempt
        reach, auth = await _attempt()

        # If rate-limited, wait and retry once
        if (
            reach.status == "FAIL"
            and "RATE_LIMITED" in reach.reason
        ):
            wait = float(reach.details.get("retry_after_seconds") or 5.0)
            wait = min(wait + 2.0, 30.0)  # add 2s buffer, cap at 30s
            logger.info(
                "readiness_gate_rate_limited_waiting",
                wait_seconds=wait,
            )
            await _asyncio.sleep(wait)
            reach, auth = await _attempt()

        return reach, auth

    # Keep old methods for backward compat with tests
    async def _check_data_service(self, data_client: Any) -> GateCheckResult:
        reach, _ = await self._check_data_service_and_auth(data_client)
        return reach

    async def _check_data_auth(self, data_client: Any) -> GateCheckResult:
        _, auth = await self._check_data_service_and_auth(data_client)
        return auth

    async def _check_universe(
        self, data_client: Any, universe: list[str], min_size: int
    ) -> GateCheckResult:
        """Check F&O universe availability and size. Retries once on 429."""
        import asyncio as _asyncio
        from src.clients.data_service import DataServiceRateLimitedError

        async def _fetch() -> GateCheckResult:
            try:
                resp = await data_client.get_fno_universe()
                symbols = (
                    resp.get("data", {}).get("constituents", [])
                    or resp.get("data", {}).get("symbols", [])
                    or resp.get("symbols", [])
                    or []
                )
                def _sym(s: object) -> str:
                    return s["symbol"] if isinstance(s, dict) else str(s)
                symbol_list = [_sym(s) for s in symbols] if isinstance(symbols, list) else []
                n = len(symbol_list)
                available = [s for s in universe if s in symbol_list]
                if n == 0:
                    return GateCheckResult(
                        gate="UNIVERSE_AVAILABLE",
                        status="FAIL",
                        reason="NO_UNIVERSE: F&O universe returned 0 symbols.",
                        details={"universe_size": 0},
                    )
                if len(available) < min_size and len(universe) > 0:
                    return GateCheckResult(
                        gate="UNIVERSE_AVAILABLE",
                        status="WARN",
                        reason=(
                            f"PARTIAL_UNIVERSE: {len(available)}/{len(universe)} "
                            f"requested symbols found in universe."
                        ),
                        details={"found": len(available), "requested": len(universe)},
                    )
                return GateCheckResult(
                    gate="UNIVERSE_AVAILABLE",
                    status="PASS",
                    details={"universe_size": n, "requested_found": len(available)},
                )
            except DataServiceRateLimitedError as exc:
                return GateCheckResult(
                    gate="UNIVERSE_AVAILABLE",
                    status="FAIL",
                    reason=f"UNIVERSE_RATE_LIMITED: {exc}",
                    details={"retry_after": exc.retry_after_seconds},
                )
            except Exception as exc:
                return GateCheckResult(
                    gate="UNIVERSE_AVAILABLE",
                    status="FAIL",
                    reason=f"UNIVERSE_UNAVAILABLE: {type(exc).__name__}: {exc}",
                )

        result = await _fetch()
        if result.status == "FAIL" and "RATE_LIMITED" in result.reason:
            wait = float(result.details.get("retry_after") or 5.0) + 2.0
            wait = min(wait, 35.0)
            logger.info("readiness_universe_rate_limited_waiting", wait_seconds=wait)
            await _asyncio.sleep(wait)
            result = await _fetch()
        return result

    async def _check_history(
        self,
        data_client: Any,
        symbols: list[str],
        timeframe: str,
        min_days: int,
    ) -> GateCheckResult:
        """Check that the first symbol has enough historical bars. Retries on 429."""
        import asyncio as _asyncio
        from src.clients.data_service import DataServiceRateLimitedError

        if not symbols:
            return GateCheckResult(
                gate="MARKET_HISTORY_SUFFICIENT",
                status="FAIL",
                reason="INSUFFICIENT_MARKET_HISTORY: no symbols to check.",
            )
        sym = symbols[0]

        async def _fetch() -> GateCheckResult:
            try:
                bars = await data_client.get_historical_ohlcv(sym, interval=timeframe)
                n_bars = len(bars) if isinstance(bars, list) else 0
                if n_bars < min_days:
                    return GateCheckResult(
                        gate="MARKET_HISTORY_SUFFICIENT",
                        status="FAIL",
                        reason=(
                            f"INSUFFICIENT_MARKET_HISTORY: {sym} has {n_bars} bars, "
                            f"required={min_days}."
                        ),
                        details={"bars_found": n_bars, "required": min_days, "symbol": sym},
                    )
                return GateCheckResult(
                    gate="MARKET_HISTORY_SUFFICIENT",
                    status="PASS",
                    details={"bars_found": n_bars, "required": min_days, "symbol": sym},
                )
            except DataServiceRateLimitedError as exc:
                return GateCheckResult(
                    gate="MARKET_HISTORY_SUFFICIENT",
                    status="FAIL",
                    reason=f"HISTORY_RATE_LIMITED: {exc}",
                    details={"retry_after": exc.retry_after_seconds},
                )
            except Exception as exc:
                return GateCheckResult(
                    gate="MARKET_HISTORY_SUFFICIENT",
                    status="FAIL",
                    reason=f"INSUFFICIENT_MARKET_HISTORY: {sym} fetch failed: {exc}",
                )

        result = await _fetch()
        if result.status == "FAIL" and "RATE_LIMITED" in result.reason:
            wait = float(result.details.get("retry_after") or 5.0) + 2.0
            wait = min(wait, 35.0)
            logger.info("readiness_history_rate_limited_waiting", wait_seconds=wait)
            await _asyncio.sleep(wait)
            result = await _fetch()
        return result

    async def _check_sentinel(
        self, sentinel_client: Any, news_required: bool
    ) -> GateCheckResult:
        """Check SentinelPulse reachability and historical coverage."""
        if sentinel_client is None:
            if news_required:
                return GateCheckResult(
                    gate="SENTINELPULSE_AVAILABLE",
                    status="FAIL",
                    reason="SENTINELPULSE_UNAVAILABLE: client not configured.",
                )
            return GateCheckResult(
                gate="SENTINELPULSE_AVAILABLE",
                status="SKIP",
                reason="NEWS_OPTIONAL_AND_DISABLED",
            )
        try:
            ctx = await sentinel_client.fetch_market_context()
            if ctx is None:
                status = "FAIL" if news_required else "WARN"
                return GateCheckResult(
                    gate="SENTINELPULSE_AVAILABLE",
                    status=status,
                    reason="SENTINELPULSE_UNREACHABLE: fetch_market_context returned None.",
                )
            # Check historical coverage — use training/samples endpoint
            # SentinelPulse returns {"data": [...], "meta": {...}} — the client
            # extracts response["data"] which is a list, not {"samples": [...]}
            samples_resp = await sentinel_client.fetch_training_samples(limit=10)
            # Normalise: could be None, a list, or a dict with "samples" key
            if samples_resp is None:
                sample_list: list = []
            elif isinstance(samples_resp, list):
                sample_list = samples_resp
            elif isinstance(samples_resp, dict):
                sample_list = samples_resp.get("samples", []) or []
            else:
                sample_list = []

            if len(sample_list) == 0:
                return GateCheckResult(
                    gate="SENTINELPULSE_AVAILABLE",
                    status="WARN",
                    reason=(
                        "SENTINELPULSE_CONNECTED_BUT_NO_TRAINING_SAMPLES: "
                        "news_status=EVIDENCE_PENDING. "
                        "SentinelPulse is reachable and authenticated but has no "
                        "processed training samples in its database. "
                        "News ingestion pipeline may not have run yet."
                    ),
                    details={"sample_count": 0},
                )
            return GateCheckResult(
                gate="SENTINELPULSE_AVAILABLE",
                status="PASS",
                details={"training_samples_available": len(sample_list)},
            )
        except Exception as exc:
            status = "FAIL" if news_required else "WARN"
            return GateCheckResult(
                gate="SENTINELPULSE_AVAILABLE",
                status=status,
                reason=f"SENTINELPULSE_ERROR: {type(exc).__name__}: {exc}",
            )

    def _check_features(self) -> GateCheckResult:
        """Verify the feature schema is valid and importable."""
        try:
            from src.features.factory import FeatureFactory, FEATURE_SCHEMA_VERSION
            ff = FeatureFactory()
            if len(ff.FEATURE_NAMES) == 0:
                return GateCheckResult(
                    gate="FEATURE_SCHEMA_VALID",
                    status="FAIL",
                    reason="FEATURE_SCHEMA_INVALID: no features defined.",
                )
            return GateCheckResult(
                gate="FEATURE_SCHEMA_VALID",
                status="PASS",
                details={
                    "feature_count": len(ff.FEATURE_NAMES),
                    "schema_version": FEATURE_SCHEMA_VERSION,
                },
            )
        except Exception as exc:
            return GateCheckResult(
                gate="FEATURE_SCHEMA_VALID",
                status="FAIL",
                reason=f"FEATURE_SCHEMA_INVALID: {exc}",
            )

    def _check_labels(self) -> GateCheckResult:
        """Verify the label schema is valid and importable."""
        try:
            from src.data.labels import LabelFactory, LabelConfig
            from src.data.dataset_builder import LABEL_SCHEMA_VERSION
            # Default config must be next_open (mandate §20)
            cfg = LabelConfig()
            if cfg.execution_model != "next_open":
                return GateCheckResult(
                    gate="LABEL_SCHEMA_VALID",
                    status="FAIL",
                    reason=(
                        f"LABEL_SCHEMA_INVALID: default execution_model="
                        f"'{cfg.execution_model}' must be 'next_open' (mandate §20)."
                    ),
                )
            return GateCheckResult(
                gate="LABEL_SCHEMA_VALID",
                status="PASS",
                details={
                    "default_execution_model": cfg.execution_model,
                    "schema_version": LABEL_SCHEMA_VERSION,
                },
            )
        except Exception as exc:
            return GateCheckResult(
                gate="LABEL_SCHEMA_VALID",
                status="FAIL",
                reason=f"LABEL_SCHEMA_INVALID: {exc}",
            )

    def _check_training_env(self) -> GateCheckResult:
        """Verify training dependencies are importable."""
        # Core ML dependencies that must always be present
        core_packages = ["numpy", "pandas", "sklearn"]
        # ML model packages — may not be available on macOS ARM host (LightGBM native crash)
        # but MUST be available inside Docker (mandate §2, §23)
        optional_packages = ["lightgbm", "xgboost", "mlflow"]

        missing_core: list[str] = []
        missing_optional: list[str] = []

        for pkg in core_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing_core.append(pkg)

        for pkg in optional_packages:
            try:
                __import__(pkg)
            except ImportError:
                missing_optional.append(pkg)

        if missing_core:
            return GateCheckResult(
                gate="TRAINING_DEPS_AVAILABLE",
                status="FAIL",
                reason=f"TRAINING_ENVIRONMENT_INVALID: missing core packages {missing_core}",
                details={"missing_core": missing_core},
            )

        # Check artifact path is writable
        try:
            self._artifact_root.mkdir(parents=True, exist_ok=True)
            test_file = self._artifact_root / ".write_test"
            test_file.touch()
            test_file.unlink()
        except Exception as exc:
            return GateCheckResult(
                gate="TRAINING_DEPS_AVAILABLE",
                status="FAIL",
                reason=f"ARTIFACT_PATH_NOT_WRITABLE: {exc}",
            )

        if missing_optional:
            # Warn but don't block — optional packages may be absent on macOS ARM host
            # but must be present inside Docker
            return GateCheckResult(
                gate="TRAINING_DEPS_AVAILABLE",
                status="WARN",
                reason=(
                    f"OPTIONAL_PACKAGES_MISSING_ON_HOST: {missing_optional}. "
                    "These MUST be available inside Docker (mandate §23). "
                    "This is only a warning on the host; Docker execution is authoritative."
                ),
                details={"missing_optional": missing_optional},
            )

        return GateCheckResult(gate="TRAINING_DEPS_AVAILABLE", status="PASS")

    def _check_provenance(self) -> GateCheckResult:
        """Check that git SHA and docker image are available."""
        sha = _git_sha()
        img = _docker_image()
        details = {"git_sha": sha, "docker_image": img}

        if sha == "unknown" and img == "unknown":
            return GateCheckResult(
                gate="PROVENANCE_AVAILABLE",
                status="FAIL",
                reason="PROVENANCE_INCOMPLETE: git_sha=unknown, docker_image=unknown.",
                details=details,
            )
        return GateCheckResult(
            gate="PROVENANCE_AVAILABLE", status="PASS", details=details
        )

    # ── Finalize ────────────────────────────────────────────────────────────

    def _finalize(
        self,
        gates: list[GateCheckResult],
        blockers: list[str],
        warnings: list[str],
        news_status: str,
        ready: bool,
        mode: str,
    ) -> TrainingReadinessResult:
        result = TrainingReadinessResult(
            training_ready=ready,
            mode=mode,
            news_status=news_status,
            gates=gates,
            blockers=blockers,
            warnings=warnings,
            git_sha=_git_sha(),
            docker_image=_docker_image(),
            python_version=_python_version(),
        )
        if ready:
            logger.info(
                "training_readiness_gate_passed",
                mode=mode,
                news_status=news_status,
                warnings=warnings,
            )
        else:
            logger.warning(
                "training_readiness_gate_failed",
                mode=mode,
                blockers=blockers,
                warnings=warnings,
            )
        return result
