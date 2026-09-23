"""
src.meta_engine — Phase 2 stub package for the LangChain-based meta-decision interface.

This package exposes the ``MetaEngine`` interface defined in Phase 2.
The production implementation (delegating to ``src.meta.engine``) is
wired up in Phase 3.

Re-exports:
    MetaEngine — Phase 2 stub that raises NotImplementedError
"""
from src.meta_engine.engine import MetaEngine

__all__ = ["MetaEngine"]
