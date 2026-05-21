"""Ask 1+2 — StencilLoopMaterializePass.

Materializes a BC-lowered ``tessera.neighbors.stencil.apply`` into a real
``scf.for``-nested loop body with per-axis BC fixups consuming the
``stencil.bc.modes`` / ``stencil.bc.values`` attributes emitted by
``BoundaryConditionLowerPass``.

This file ships structural guards that always run + a behavioral
contract that subprocesses against ``tessera-opt`` (skipped if the
binary predates the pass).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NEIGHBORS_ROOT = REPO_ROOT / "src" / "compiler" / "tessera_neighbors"
MAT_CPP = (
    NEIGHBORS_ROOT
    / "lib" / "Dialect" / "Neighbors" / "Transforms"
    / "StencilLoopMaterializePass.cpp"
)
LIT_FIXTURE = (
    REPO_ROOT
    / "tests" / "tessera-ir" / "phase7"
    / "neighbors_stencil_materialize.mlir"
)


# --------------------------------------------------------------------------- #
# Structural wiring
# --------------------------------------------------------------------------- #


def test_pass_source_exists() -> None:
    assert MAT_CPP.exists()


def test_pass_declares_registration_fn() -> None:
    text = MAT_CPP.read_text()
    assert "void registerStencilLoopMaterializePass()" in text
    assert "tessera-stencil-loop-materialize" in text


def test_pass_emits_loop_primitives() -> None:
    """Every primitive the architecture doc names must appear in the
    pass source.  These are the building blocks the lit fixture
    FileChecks against — a regression in one of them is the kind of
    thing a refactor would silently break, so we lock the source."""
    text = MAT_CPP.read_text()
    primitives = (
        "scf::ForOp",
        "tensor::DimOp",
        "tensor::EmptyOp",
        "tensor::ExtractOp",
        "tensor::InsertOp",
        "scf::YieldOp",
        "arith::RemSIOp",      # periodic
        "arith::MaxSIOp",      # reflect/dirichlet/neumann clamp lower
        "arith::MinSIOp",      # reflect/dirichlet/neumann clamp upper
        "arith::CmpIOp",       # in-bounds test
        "arith::AndIOp",       # combine geZero + ltN
        "arith::XOrIOp",       # invert in-bounds to oob
        "arith::AddIOp",       # raw index = base + delta
        "arith::AddFOp",       # tap accumulation
        "arith::SelectOp",     # BC value-side selects
        "arith::ConstantOp",   # BC float constants
    )
    for p in primitives:
        assert p in text, f"primitive '{p}' missing from materialize pass"


def test_pass_documents_four_bc_modes() -> None:
    text = MAT_CPP.read_text()
    # Three BC modes are explicitly dispatched on; reflect is the implicit
    # else branch (clamp-only).  All four must still appear in the source —
    # the explicit three as quoted string comparisons, reflect in the
    # documented mode block.
    for quoted in ('"periodic"', '"dirichlet"', '"neumann"'):
        assert quoted in text, f"BC mode {quoted} missing from materialize pass"
    # Reflect appears in the docstring + comment that names the fall-through.
    assert "reflect" in text, "reflect BC must be at least documented in source"


def test_pass_writes_materialized_sentinel() -> None:
    text = MAT_CPP.read_text()
    assert "stencil.materialized" in text


def test_pass_registered_in_passes_header() -> None:
    header = (
        NEIGHBORS_ROOT
        / "include" / "tessera" / "Dialect" / "Neighbors" / "Transforms"
        / "Passes.h"
    )
    assert "registerStencilLoopMaterializePass" in header.read_text()


def test_pass_registered_in_cmake() -> None:
    cmake = NEIGHBORS_ROOT / "CMakeLists.txt"
    assert "StencilLoopMaterializePass.cpp" in cmake.read_text()


def test_pass_registered_in_tessera_opt() -> None:
    opt = REPO_ROOT / "tools" / "tessera-opt" / "tessera-opt.cpp"
    assert "registerStencilLoopMaterializePass" in opt.read_text()


def test_pass_supports_rank_n_via_recursive_helper() -> None:
    """Sub-3 — the pass must use a recursive ``buildNest`` helper so
    rank-3 / rank-4 stencils materialize through the same machinery.
    Pinning the helper's name keeps refactors honest about the contract."""
    text = MAT_CPP.read_text()
    assert "buildNest" in text
    # The pass advertises rank-N capability in its description.
    assert "rank-N" in text or "Rank-N" in text


def test_rank3_lit_fixture_exists() -> None:
    rank3_fixture = (
        REPO_ROOT / "tests" / "tessera-ir" / "phase7"
        / "neighbors_stencil_materialize_rank3.mlir"
    )
    assert rank3_fixture.exists()
    text = rank3_fixture.read_text()
    # Functions covering rank-3 + rank-4 + mixed-BC.
    assert "test_materialize_rank3_periodic_7pt" in text
    assert "test_materialize_rank3_mixed_bc" in text
    assert "test_materialize_rank4_periodic" in text


def test_lit_fixture_chains_three_passes() -> None:
    """The fixture has to chain stencil-lower → bc-lower → materialize."""
    text = LIT_FIXTURE.read_text()
    for arg in (
        "-tessera-stencil-lower",
        "-tessera-boundary-condition-lower",
        "-tessera-stencil-loop-materialize",
    ):
        assert arg in text, f"fixture RUN line missing {arg}"
    # All four BC modes are exercised somewhere in the fixture.
    for mode in ("periodic", "dirichlet", "neumann"):
        assert mode in text


# --------------------------------------------------------------------------- #
# Behavioral contract — skipped if binary predates the pass
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


def test_materialize_pass_runs_against_lit_fixture() -> None:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built — skipping behavioral contract test")
    result = subprocess.run(
        [
            binary,
            "-tessera-stencil-lower",
            "-tessera-boundary-condition-lower",
            "-tessera-stencil-loop-materialize",
            str(LIT_FIXTURE),
        ],
        capture_output=True, text=True, timeout=30,
    )
    if (
        result.returncode != 0
        and "Unknown command line argument" in result.stderr
        and "tessera-stencil-loop-materialize" in result.stderr
    ):
        pytest.skip(
            "tessera-opt binary predates StencilLoopMaterializePass — "
            "rebuild build/tools/tessera-opt to exercise this contract"
        )
    assert result.returncode == 0, (
        f"materialize pass failed: {result.stderr}"
    )
    out = result.stdout
    # Three functions, three materialized sentinels.
    assert out.count("stencil.materialized = true") >= 3
    # Periodic must use arith.remsi.
    assert "arith.remsi" in out
    # Dirichlet must emit the BC constant.
    assert "2.500000e+00" in out
    # Neumann must emit the negative-one constant.
    assert "-1.000000e+00" in out
    # All three functions must produce an scf.for body.
    assert out.count("scf.for") >= 3
    # Every materialized loop must reach a tensor.insert / tensor.extract.
    assert "tensor.extract" in out
    assert "tensor.insert" in out
