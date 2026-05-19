"""Regression: lit ``TESSERA_OPT`` / ``FILECHECK`` env overrides
absolutize against the repo root.

Findings audit (2026-05-19) flagged that a relative env var like
``TESSERA_OPT=build/tools/tessera-opt/tessera-opt`` would fail
because lit runs each fixture's script from its ``Output/``
subdirectory — the relative path no longer resolves against the
repo root from there.

The fix lives in ``_resolve()`` inside the two lit configs:

  * ``tests/tessera-ir/lit.cfg.py``
  * ``src/solvers/tpp/test/TPP/lit.cfg.py``

Both files now expand ``~``, accept absolute paths verbatim, and
join relative env-var values against the lit-config-anchored repo
root before substituting them into the test command line.
"""

from __future__ import annotations

import os
import subprocess
import shutil
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
TESSERA_OPT_ABS = (
    REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"
)
TESSERA_OPT_REL = "build/tools/tessera-opt/tessera-opt"
FIXTURE = (
    REPO_ROOT / "tests" / "tessera-ir" / "phase7" / "neighbors_halo_infer.mlir"
)


def _have_tessera_opt() -> bool:
    return TESSERA_OPT_ABS.is_file() and os.access(TESSERA_OPT_ABS, os.X_OK)


def _have_lit() -> str | None:
    lit = shutil.which("lit") or os.path.join(
        os.path.dirname(sys.executable), "lit",
    )
    if os.path.isfile(lit) and os.access(lit, os.X_OK):
        return lit
    return None


@pytest.mark.skipif(
    not _have_tessera_opt() or _have_lit() is None,
    reason="tessera-opt or lit not available",
)
def test_relative_tessera_opt_override_resolves_against_repo_root() -> None:
    """A relative ``TESSERA_OPT=build/...`` override must not break
    when lit chdir's into a fixture's ``Output/`` subdirectory.

    Pre-fix, the literal relative path was substituted into the lit
    script and ld-not-found from ``Output/...``.  Post-fix, the lit
    config absolutizes the override before substitution.
    """
    lit = _have_lit()
    assert lit is not None
    env = dict(os.environ)
    env["TESSERA_OPT"] = TESSERA_OPT_REL  # the deliberately-fragile form
    proc = subprocess.run(
        [lit, str(FIXTURE), "-v"],
        capture_output=True, text=True, timeout=60,
        env=env, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"relative TESSERA_OPT override broke lit (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )
    assert "Passed: 1" in proc.stdout, proc.stdout


@pytest.mark.skipif(
    not _have_tessera_opt() or _have_lit() is None,
    reason="tessera-opt or lit not available",
)
def test_absolute_tessera_opt_override_still_honored() -> None:
    """Absolute paths must pass through verbatim — same shape as
    before the fix, just locked here to catch any future regression
    that mishandles them."""
    lit = _have_lit()
    assert lit is not None
    env = dict(os.environ)
    env["TESSERA_OPT"] = str(TESSERA_OPT_ABS)
    proc = subprocess.run(
        [lit, str(FIXTURE), "-v"],
        capture_output=True, text=True, timeout=60,
        env=env, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"absolute TESSERA_OPT override broke lit (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )


@pytest.mark.skipif(
    not _have_tessera_opt() or _have_lit() is None,
    reason="tessera-opt or lit not available",
)
def test_tilde_tessera_opt_override_is_expanded() -> None:
    """``~``-prefixed paths are common in shell aliases; the lit
    config must expand them so a value like
    ``TESSERA_OPT=~/dev_project/.../tessera-opt`` works."""
    home = Path.home()
    try:
        # Compute a ``~/...`` form of the absolute path.
        relative_to_home = TESSERA_OPT_ABS.relative_to(home)
    except ValueError:
        pytest.skip(
            f"tessera-opt is not under $HOME (got {TESSERA_OPT_ABS}); "
            f"tilde-expansion lane untestable from here"
        )
    lit = _have_lit()
    assert lit is not None
    env = dict(os.environ)
    env["TESSERA_OPT"] = f"~/{relative_to_home}"
    proc = subprocess.run(
        [lit, str(FIXTURE), "-v"],
        capture_output=True, text=True, timeout=60,
        env=env, cwd=str(REPO_ROOT),
    )
    assert proc.returncode == 0, (
        f"~-prefixed TESSERA_OPT override broke lit (rc={proc.returncode}):\n"
        f"stdout:\n{proc.stdout}\n\nstderr:\n{proc.stderr}"
    )
