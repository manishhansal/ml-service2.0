"""
tests/phase3_archived/conftest.py
----------------------------------
These tests were written for an earlier architecture phase where several
modules had different names or paths (e.g. ``src.data_reliability`` has been
folded into ``src.data.reliability``).

The entire directory is **archived** — tests here are kept for historical
reference only and are explicitly skipped at collection time so they do not
break the default ``pytest`` run.

To view what was tested: read individual test files.
To re-enable a specific file: copy it to ``tests/`` and update imports.
"""
import pytest

collect_ignore_glob = ["test_phase3*.py"]
