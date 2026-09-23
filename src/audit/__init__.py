"""
Audit package for ml-service2.0.

Exports:
- AuditLogger          — append-only JSON Lines audit logger
- AuditLogViolation    — raised when a prior entry has been tampered with
- AuditLogWriteFailure — raised when an entry cannot be written to disk
"""
from __future__ import annotations

from src.audit.logger import AuditLogViolation, AuditLogWriteFailure, AuditLogger

__all__ = [
    "AuditLogger",
    "AuditLogViolation",
    "AuditLogWriteFailure",
]
