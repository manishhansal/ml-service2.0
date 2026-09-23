"""
ApprovalToken — signed token for human-in-the-loop model promotion approval.

Design:
- Token format: base64url(JSON payload) + "." + HMAC-SHA256 signature
- Payload: {issued_at: ISO-8601, issuer: str, challenger_id: str, expires_at: ISO-8601}
- Expiry: settings.approval_token_expiry_hours (default 24 hours)
- Signing key: HMAC-SHA256 using ML_SERVICE_API_KEY as the secret
- Authorized reviewers: configurable list (default: ["admin", "ml-ops", "quant-lead"])

Requirements: Req 13.10, Req 13.11
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from src.config import settings
from src.logging_config import get_logger

logger = get_logger(__name__)

# Default authorized reviewers — override via env var AUTHORIZED_REVIEWERS (comma-separated)
DEFAULT_AUTHORIZED_REVIEWERS = {"admin", "ml-ops", "quant-lead"}


class ApprovalTokenError(Exception):
    """Raised when an approval token is invalid, expired, or from an unauthorized issuer."""


# Alias used in tasks.md API contract: ApprovalTokenValidator.validate() raises InvalidApprovalTokenError
InvalidApprovalTokenError = ApprovalTokenError


class ApprovalTokenManager:
    """
    Issues and validates HMAC-signed approval tokens for model promotion.

    The token lifecycle:
    1. Issue: ml-ops engineer calls ``issue_token(issuer="quant-lead", challenger_id="...")``
    2. Validate: promotion pipeline calls ``validate_token(token, challenger_id="...")``
       - Signature must match (HMAC-SHA256 with ML_SERVICE_API_KEY)
       - Token must not be expired (configurable, default < 24 h old)
       - Issuer must be in the authorized_reviewers set

    Usage::

        mgr = ApprovalTokenManager()
        token = mgr.issue_token("admin", "market_regime-v1.0.0")
        payload = mgr.validate_token(token, "market_regime-v1.0.0")
    """

    def __init__(
        self,
        authorized_reviewers: set[str] | None = None,
        signing_key: str | None = None,
        expiry_hours: int | None = None,
    ) -> None:
        self._authorized_reviewers: set[str] = (
            authorized_reviewers if authorized_reviewers is not None
            else DEFAULT_AUTHORIZED_REVIEWERS
        )
        self._signing_key: str = signing_key or settings.ml_service_api_key
        self._expiry_hours: int = (
            expiry_hours if expiry_hours is not None
            else settings.approval_token_expiry_hours
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def issue_token(self, issuer: str, challenger_id: str) -> str:
        """
        Issue a signed approval token.

        Args:
            issuer:        Identity of the reviewer (must be in authorized_reviewers).
            challenger_id: The model challenger being approved.

        Returns:
            Signed token string (``payload_b64.signature_hex``).

        Raises:
            ApprovalTokenError: If issuer is not in the authorized-reviewer registry.
        """
        if issuer not in self._authorized_reviewers:
            raise ApprovalTokenError(
                f"Issuer {issuer!r} is not in the authorized-reviewer registry. "
                f"Authorized reviewers: {sorted(self._authorized_reviewers)}"
            )

        now = datetime.now(tz=timezone.utc)
        expires_at = now + timedelta(hours=self._expiry_hours)

        payload: dict[str, str] = {
            "issued_at": now.isoformat(),
            "expires_at": expires_at.isoformat(),
            "issuer": issuer,
            "challenger_id": challenger_id,
        }

        payload_json = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload_b64 = base64.urlsafe_b64encode(payload_json.encode()).decode()

        signature = self._sign(payload_b64)
        token = f"{payload_b64}.{signature}"

        logger.info(
            "approval_token_issued",
            issuer=issuer,
            challenger_id=challenger_id,
            expires_at=expires_at.isoformat(),
        )

        return token

    def validate_token(self, token: str, challenger_id: str | None = None) -> dict[str, Any]:
        """
        Validate an approval token and return its decoded payload.

        Checks (in order):
        1. Token format — must be ``payload_b64.signature``
        2. Signature integrity — HMAC-SHA256 must match
        3. Payload decodability
        4. Expiry — ``expires_at`` must be in the future
        5. Issuer authorization — issuer must be in authorized_reviewers
        6. Challenger match — if ``challenger_id`` is provided, must equal payload value

        Args:
            token:         The token string to validate.
            challenger_id: Optional expected challenger_id to match against payload.

        Returns:
            Decoded payload dict with keys: ``issued_at``, ``expires_at``,
            ``issuer``, ``challenger_id``.

        Raises:
            ApprovalTokenError: On invalid format, bad signature, expiry, or
                                 unauthorized issuer.
        """
        # (1) Format check
        try:
            payload_b64, signature = token.rsplit(".", 1)
        except ValueError:
            raise ApprovalTokenError(
                "Token format is invalid (expected payload_b64.signature_hex)"
            )

        # (2) Signature verification — constant-time comparison prevents timing attacks
        expected_sig = self._sign(payload_b64)
        if not hmac.compare_digest(expected_sig, signature):
            raise ApprovalTokenError("Token signature is invalid")

        # (3) Payload decoding
        try:
            payload_json = base64.urlsafe_b64decode(payload_b64.encode()).decode()
            payload: dict[str, Any] = json.loads(payload_json)
        except Exception as exc:
            raise ApprovalTokenError(f"Token payload could not be decoded: {exc}") from exc

        # (4) Expiry check
        expires_at_str = payload.get("expires_at", "")
        try:
            expires_at = datetime.fromisoformat(expires_at_str)
        except (ValueError, TypeError) as exc:
            raise ApprovalTokenError(
                f"Token expires_at field is invalid: {expires_at_str!r}"
            ) from exc

        if expires_at.tzinfo is None:
            expires_at = expires_at.replace(tzinfo=timezone.utc)

        now = datetime.now(tz=timezone.utc)
        if now > expires_at:
            raise ApprovalTokenError(
                f"Token expired at {expires_at.isoformat()} "
                f"(current time: {now.isoformat()})"
            )

        # (5) Issuer authorization
        issuer: str = payload.get("issuer", "")
        if issuer not in self._authorized_reviewers:
            raise ApprovalTokenError(
                f"Token issuer {issuer!r} is not in the authorized-reviewer registry"
            )

        # (6) Challenger ID match (optional)
        if challenger_id is not None:
            token_challenger: str = payload.get("challenger_id", "")
            if token_challenger != challenger_id:
                raise ApprovalTokenError(
                    f"Token challenger_id {token_challenger!r} does not match "
                    f"expected {challenger_id!r}"
                )

        logger.info(
            "approval_token_validated",
            issuer=issuer,
            challenger_id=payload.get("challenger_id"),
        )

        return payload

    # ── Private helpers ────────────────────────────────────────────────────────

    def _sign(self, payload_b64: str) -> str:
        """Compute HMAC-SHA256 hex-digest of the base64-encoded payload."""
        key = self._signing_key.encode("utf-8")
        message = payload_b64.encode("utf-8")
        return hmac.new(key, message, hashlib.sha256).hexdigest()


class ApprovalTokenValidator:
    """
    Thin façade around :class:`ApprovalTokenManager` that exposes the
    ``validate(token) -> ReviewerIdentity`` interface required by
    Task 43 and used by ``ModelRegistry.register_champion()``.

    ``ReviewerIdentity`` is a plain ``str`` (the issuer field from the
    token payload).  The caller is responsible for binding the
    ``challenger_id`` before calling :meth:`validate` if per-challenger
    scoping is needed; otherwise the token is validated for any challenger.
    """

    def __init__(
        self,
        authorized_reviewers: set[str] | None = None,
        signing_key: str | None = None,
        expiry_hours: int | None = None,
    ) -> None:
        self._manager = ApprovalTokenManager(
            authorized_reviewers=authorized_reviewers,
            signing_key=signing_key,
            expiry_hours=expiry_hours,
        )

    def validate(self, token: str, challenger_id: str | None = None) -> str:
        """
        Validate *token* and return the reviewer identity (issuer string).

        Args:
            token:         The signed approval token.
            challenger_id: If provided, the token's challenger_id field must
                           match this value exactly.

        Returns:
            The issuer string from the token payload (i.e. the reviewer identity).

        Raises:
            InvalidApprovalTokenError: If the token is invalid, expired, tampered,
                                        or from an unauthorized issuer.
        """
        payload = self._manager.validate_token(token, challenger_id=challenger_id)
        return str(payload["issuer"])

    def issue_token(self, issuer: str, challenger_id: str) -> str:
        """Convenience wrapper — delegates to the underlying manager."""
        return self._manager.issue_token(issuer, challenger_id)
