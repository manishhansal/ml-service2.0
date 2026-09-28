"""
tests/test_static_leakage_audit.py
------------------------------------
FIX NEW-P0-003: Static leakage audit integrated into CI.

Scans all feature-computation source files for patterns that constitute
forward-looking bias:
  - shift(-N) in non-label, non-test, non-training code
  - center=True in rolling windows

This test asserts ZERO "INVALID" findings in the feature code paths.
LABEL_ONLY and CAUSAL findings are acceptable — they occur in files
that legitimately look forward (label construction) or in economically
valid contexts (A/D line volume deltas).

Any INVALID finding must either be:
  (a) Fixed (remove the forward-looking expression), OR
  (b) Documented with an explicit exception added to _KNOWN_EXCEPTIONS below.

Adding an exception requires a written justification reviewed by a human.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.features.leakage_validator import AuditFinding, run_static_leakage_audit

# ── Directories to scan (feature + model + backtest code) ────────────────────
_SCAN_DIRS = [
    Path("src/features"),
    Path("src/models"),
    Path("src/backtest"),
    Path("src/analytics"),
    Path("src/meta"),
]

# ── Known exceptions (must be empty — add only after human review) ───────────
# Format: (relative_file_path_substring, line_number_approx, pattern)
# A finding is exempt if its file path contains the substring AND the pattern matches.
_KNOWN_EXCEPTIONS: list[tuple[str, str]] = [
    # Example (do not add without justification):
    # ("analytics/forward_paper.py", "shift(-1)"),  # explained: paper mode only
]


# ── Helper ────────────────────────────────────────────────────────────────────

def _is_exempt(finding: AuditFinding) -> bool:
    """Return True if this finding is in the known-exceptions allowlist."""
    for path_substr, pattern in _KNOWN_EXCEPTIONS:
        if path_substr in finding.file and pattern in finding.pattern:
            return True
    return False


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.pit
def test_no_invalid_shift_in_feature_code() -> None:
    """No INVALID shift(-N) patterns in feature/model/backtest source code.

    FIX NEW-P0-003: This test enforces that no feature-computation code
    contains forward-looking shifts that could introduce look-ahead bias.

    Allowed: LABEL_ONLY (in label-construction paths), CAUSAL (A/D volume).
    Forbidden: INVALID (forward shift in feature code).
    """
    all_findings: list[AuditFinding] = []
    for scan_dir in _SCAN_DIRS:
        if scan_dir.exists():
            all_findings.extend(run_static_leakage_audit([scan_dir]))

    invalid_findings = [
        f for f in all_findings
        if f.classification == "INVALID" and not _is_exempt(f)
    ]

    if invalid_findings:
        details = "\n".join(
            f"  {f.file}:{f.line} [{f.pattern}] — {f.code_snippet[:80]}"
            for f in invalid_findings
        )
        pytest.fail(
            f"Static leakage audit found {len(invalid_findings)} INVALID finding(s).\n"
            f"These are forward-looking expressions in feature/model code — FIX or add "
            f"to _KNOWN_EXCEPTIONS with written justification.\n\n{details}"
        )


@pytest.mark.pit
def test_static_leakage_audit_runs_without_error() -> None:
    """The static leakage audit itself must complete without exceptions."""
    for scan_dir in _SCAN_DIRS:
        if scan_dir.exists():
            findings = run_static_leakage_audit([scan_dir])
            assert isinstance(findings, list), f"Expected list, got {type(findings)}"


@pytest.mark.pit
def test_static_leakage_audit_classifies_label_dirs_correctly() -> None:
    """Negative shifts in src/labels/ must be classified LABEL_ONLY, not INVALID."""
    label_dir = Path("src/labels")
    if not label_dir.exists():
        pytest.skip("src/labels not found — skipping classification check")

    findings = run_static_leakage_audit([label_dir])
    for f in findings:
        if "shift" in f.pattern.lower():
            assert f.classification in ("LABEL_ONLY", "CAUSAL"), (
                f"Unexpected classification '{f.classification}' for shift in label path: "
                f"{f.file}:{f.line}"
            )


@pytest.mark.pit
def test_static_leakage_audit_classifies_data_labels_correctly() -> None:
    """Negative shifts in src/data/labels.py must be classified LABEL_ONLY."""
    data_dir = Path("src/data")
    if not data_dir.exists():
        pytest.skip("src/data not found")

    findings = run_static_leakage_audit([data_dir])
    for f in findings:
        if "shift" in f.pattern.lower() and "labels.py" in f.file:
            assert f.classification == "LABEL_ONLY", (
                f"labels.py shift should be LABEL_ONLY, got '{f.classification}': "
                f"{f.file}:{f.line}"
            )
