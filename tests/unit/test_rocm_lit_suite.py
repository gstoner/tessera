"""Durable CI guard: run the ROCm backend lit suite from pytest.

The C++ ``check-tessera-rocm`` target only runs under a CMake build; this brings
the same lit fixtures into the normal pytest lane so the lit-runner wiring (site
config → ``load_config`` → ``lit.cfg.py`` → FileCheck/not on PATH) and the
fixtures themselves stay green in ordinary CI.  Skips cleanly when the ROCm
backend hasn't been configured/built or when ``lit`` isn't installed.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[2]

# Candidate build dirs that may hold the configured ROCm lit site config.
_SITE_CANDIDATES = [
    _REPO / "build-rocm/src/compiler/codegen/Tessera_ROCM_Backend/test/lit.site.cfg.py",
    _REPO / "build/src/compiler/codegen/Tessera_ROCM_Backend/test/lit.site.cfg.py",
]


def _lit() -> str | None:
    return shutil.which("lit") or shutil.which("llvm-lit") or shutil.which("lit.py")


def _site_dir() -> Path | None:
    for c in _SITE_CANDIDATES:
        if c.exists():
            return c.parent
    return None


def test_rocm_lit_suite_passes() -> None:
    lit = _lit()
    if lit is None:
        pytest.skip("lit not installed")
    test_dir = _site_dir()
    if test_dir is None:
        pytest.skip(
            "ROCm backend not configured (no generated lit.site.cfg.py); "
            "configure -DTESSERA_BUILD_ROCM_BACKEND=ON and build tessera-rocm-opt")
    res = subprocess.run(
        [lit, "-q", str(test_dir)], capture_output=True, text=True,
    )
    # lit returns non-zero if any fixture fails; surface its report on failure.
    assert res.returncode == 0, (
        f"ROCm lit suite failed (rc={res.returncode}):\n"
        f"--- stdout ---\n{res.stdout}\n--- stderr ---\n{res.stderr}")
    # Sanity: lit actually discovered tests (guards against a silent
    # "contained no tests" regression in the lit wiring).
    out = res.stdout + res.stderr
    m = re.search(r"Total Discovered Tests:\s*(\d+)", out)
    assert m is not None and int(m.group(1)) > 0, (
        f"lit discovered no tests (lit-runner wiring regressed):\n{out}")
