"""Gap 2 — BoundaryConditionLowerPass (Neighbors dialect).

Structural + behavioral guards for the new pass that lowers the
``stencil.bc`` string carried by ``tessera.neighbors.stencil.define``
into per-axis structured attributes (``stencil.bc.modes`` /
``stencil.bc.values`` / ``stencil.bc.has_value``) on each ``stencil.apply``.

The structural tests catch wiring regressions without a C++ build.  The
behavioral test runs ``tessera-opt -tessera-stencil-lower
-tessera-boundary-condition-lower`` against the lit fixture and is skipped
when the binary is not on PATH.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NEIGHBORS_ROOT = REPO_ROOT / "src" / "compiler" / "tessera_neighbors"
BC_CPP = (
    NEIGHBORS_ROOT
    / "lib" / "Dialect" / "Neighbors" / "Transforms"
    / "BoundaryConditionLowerPass.cpp"
)
LIT_FIXTURE = (
    REPO_ROOT
    / "tests" / "tessera-ir" / "phase7"
    / "neighbors_boundary_condition_lower.mlir"
)


# --------------------------------------------------------------------------- #
# Structural wiring
# --------------------------------------------------------------------------- #


def test_bc_pass_source_file_exists() -> None:
    assert BC_CPP.exists(), f"missing pass source: {BC_CPP}"


def test_bc_pass_declares_registration_fn() -> None:
    text = BC_CPP.read_text()
    assert "void registerBoundaryConditionLowerPass()" in text
    assert 'getArgument' in text and 'tessera-boundary-condition-lower' in text


def test_bc_pass_lists_four_supported_modes() -> None:
    text = BC_CPP.read_text()
    # Each of the four BC modes must be a recognised token.
    for mode in ("periodic", "reflect", "dirichlet", "neumann"):
        assert f'"{mode}"' in text, f"BC mode '{mode}' missing from pass"


def test_bc_pass_emits_three_structured_attributes() -> None:
    text = BC_CPP.read_text()
    for attr in (
        "stencil.bc.modes",
        "stencil.bc.values",
        "stencil.bc.has_value",
        "stencil.bc.lowered",
    ):
        assert attr in text, f"BC pass does not emit attribute '{attr}'"


def test_bc_pass_registered_in_cmake() -> None:
    cmake = NEIGHBORS_ROOT / "CMakeLists.txt"
    assert "BoundaryConditionLowerPass.cpp" in cmake.read_text()


def test_bc_pass_registered_in_passes_header() -> None:
    header = (
        NEIGHBORS_ROOT
        / "include" / "tessera" / "Dialect" / "Neighbors" / "Transforms"
        / "Passes.h"
    )
    assert "registerBoundaryConditionLowerPass" in header.read_text()


def test_bc_pass_registered_in_tessera_opt() -> None:
    opt = REPO_ROOT / "tools" / "tessera-opt" / "tessera-opt.cpp"
    assert "registerBoundaryConditionLowerPass" in opt.read_text()


def test_lit_fixture_exists_and_runs_pipeline() -> None:
    text = LIT_FIXTURE.read_text()
    # The RUN: line must chain stencil-lower → bc-lower (StencilLowerPass is
    # what sets the ``stencil.bc`` annotation that this pass consumes).
    assert "-tessera-stencil-lower" in text
    assert "-tessera-boundary-condition-lower" in text
    # And it must cover all four supported BC modes across functions.
    for mode in ("periodic", "reflect", "dirichlet", "neumann"):
        assert mode in text


# --------------------------------------------------------------------------- #
# Behavioral contract — skipped if the binary is not available
# --------------------------------------------------------------------------- #


def _find_tessera_opt() -> str | None:
    for candidate in (
        os.environ.get("TESSERA_OPT"),
        shutil.which("tessera-opt"),
        str(REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"),
        str(REPO_ROOT / "build" / "bin" / "tessera-opt"),
    ):
        if candidate and Path(candidate).exists():
            return candidate
    return None


def test_bc_pass_runs_against_lit_fixture() -> None:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built — skipping behavioral contract test")

    result = subprocess.run(
        [
            binary,
            "-tessera-stencil-lower",
            "-tessera-boundary-condition-lower",
            str(LIT_FIXTURE),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    # Built binary may predate this pass — skip rather than fail, matching
    # the test_neighbors_dialect.py convention.  CI rebuilds on every
    # change in src/compiler/tessera_neighbors/ and exercises the fresh
    # binary against this same fixture under the lit lane.
    if (
        result.returncode != 0
        and "Unknown command line argument" in result.stderr
        and "tessera-boundary-condition-lower" in result.stderr
    ):
        pytest.skip(
            "tessera-opt binary predates BoundaryConditionLowerPass — "
            "rebuild build/tools/tessera-opt to exercise this contract"
        )
    assert result.returncode == 0, (
        f"tessera-opt failed running boundary-condition-lower pass:\n"
        f"stderr:\n{result.stderr}"
    )
    # All four fixtures should carry the lowered sentinel.
    out = result.stdout
    assert out.count("stencil.bc.lowered = true") >= 4
    # And each BC mode must appear in the structured ArrayAttr.
    assert '"periodic"' in out
    assert '"reflect"' in out
    assert '"dirichlet"' in out
    assert '"neumann"' in out
