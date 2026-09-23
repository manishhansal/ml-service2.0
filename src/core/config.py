"""
src.core.config — Pydantic BaseSettings for ml-service2.0.

This module is the canonical Phase-2 settings interface.  It re-exports the
production ``Settings`` singleton from ``src.config`` under the Phase-2 name
``CoreSettings`` so that Phase-2 code and tests can use:

    from src.core.config import settings, CoreSettings

without duplicating the full settings definition (which lives in src/config.py).

Phase 3 will migrate everything here and remove the root-level src/config.py.
"""
from __future__ import annotations

# Re-export the production singleton so Phase-2 stubs and tests import from
# a consistent location.
from src.config import Settings as CoreSettings
from src.config import settings  # noqa: F401  (re-export)

__all__ = ["CoreSettings", "settings"]
