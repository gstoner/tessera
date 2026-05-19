"""Lock the static-analysis baseline so future commits don't drift.

  * **ruff** — must report zero errors against ``python/tessera/``
    with the configuration in ``pyproject.toml``.  A regression
    fails this test immediately.
  * **mypy** — must report a count at or below the ratchet
    baseline in ``scripts/mypy_baseline.txt``.  Use
    ``scripts/mypy_ratchet.sh --update-baseline`` to commit a
    decrease after a cleanup sprint.

Both are skipped cleanly when the tool isn't installed in the
host Python — tooling-aware CI lanes still pass elsewhere.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
RUFF = shutil.which("ruff") or os.path.join(os.path.dirname(sys.executable), "ruff")
MYPY = shutil.which("mypy") or os.path.join(os.path.dirname(sys.executable), "mypy")


def _have(binary: str) -> bool:
    return os.path.isfile(binary) and os.access(binary, os.X_OK)


@pytest.mark.skipif(not _have(RUFF), reason="ruff not installed in this Python")
def test_ruff_is_clean_against_pyproject_config() -> None:
    """``ruff check python/tessera/`` must report zero errors.

    The first day-1 invocation landed at zero after applying the
    auto-fixable rule set and configuring the ``ignore`` block in
    ``pyproject.toml``.  A regression — a new error code, a newly
    drifted file — fails this test loudly.  Track new errors via:

        ruff check python/tessera/

    If a rule is too noisy for the codebase, add it to the
    ``ignore`` list in ``pyproject.toml`` with a one-line comment
    explaining why; do not blanket-ignore individual files.
    """
    proc = subprocess.run(
        [RUFF, "check", "python/tessera/"],
        capture_output=True, text=True, timeout=60, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"ruff reported errors (rc={proc.returncode}).  Output:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


@pytest.mark.skipif(not _have(MYPY), reason="mypy not installed in this Python")
def test_mypy_count_is_at_or_below_baseline() -> None:
    """``mypy`` error count must not exceed the ratchet baseline.

    Driven via ``scripts/mypy_ratchet.sh`` so the same logic runs
    here and in CI.  Update the baseline (after a focused cleanup
    sprint) via:

        scripts/mypy_ratchet.sh --update-baseline
    """
    script = REPO_ROOT / "scripts" / "mypy_ratchet.sh"
    assert script.is_file(), f"missing ratchet script: {script}"
    proc = subprocess.run(
        ["bash", str(script)],
        capture_output=True, text=True, timeout=180, cwd=str(REPO_ROOT),
        env={**os.environ, "MYPY": MYPY},
    )
    # rc=0 → at-or-below baseline; rc=1 → regression.
    assert proc.returncode == 0, (
        f"mypy ratchet reports an error-count regression "
        f"(rc={proc.returncode}).\n\nratchet stdout:\n{proc.stdout}\n"
        f"\nratchet stderr:\n{proc.stderr}"
    )
