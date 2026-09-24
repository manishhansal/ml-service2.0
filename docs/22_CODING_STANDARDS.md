# Coding Standards
**ml-service2.0**

*Date: 2026-09-24*

---

## Language and Runtime
- Python 3.11+ required
- Type hints on all function signatures
- `from __future__ import annotations` in all files

## Schema Design
- Pydantic V2 strict mode (`ConfigDict(strict=True, frozen=True)`) for all schemas
- No bare dicts as return types from public APIs
- All enums as `(str, Enum)` for JSON serialization

## Numerical Code
- All financial calculations must have dedicated unit tests with known analytical examples
- Named constants for all thresholds (no magic numbers)
- Explicit handling of null/NaN (never substitute 0 for missing derivatives data)
- Document units in variable names: `price_pct` not `price`; `vol_annualized` not `vol`

## Error Handling
- Never silently catch and ignore exceptions in financial logic
- Specific exception types over bare `except Exception`
- Logging before re-raise: `logger.error("context", exc_info=True); raise`

## Module Organization
- One class per file for major components
- Pure functions for quantitative calculations (no side effects)
- Dependency injection over global singletons
- No circular imports

## Documentation
- Module-level docstring explaining purpose, requirements covered, and usage example
- Public method docstrings with Args, Returns, Raises
- Math formulas in docstrings for quantitative functions

## Dependencies
- Exact version pinning in pyproject.toml
- No new dependencies without documented justification
- No GPL/AGPL dependencies

---

*End of Coding Standards*


---

## IMPLEMENTATION STATUS (2026-09-24, post-execution)

New modules follow the project standards: typed signatures, docstrings, structlog logging, Pydantic V2 schemas, and no silent failures on external calls. ruff auto-fixes were applied to the new source. Remaining ruff N806 warnings (uppercase X/y) are intentionally retained as the standard ML/sklearn convention, consistent with the pre-existing codebase. Authoritative status: reports/ML_SERVICE_FINAL_CERTIFICATION.md.
