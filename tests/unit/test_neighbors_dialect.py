"""Phase 7 — Neighbors dialect wiring tests.

These tests validate two things:

1. Structural wiring: the source files declare and register the dialect
   and the four Phase 7 passes (HaloInfer, StencilLower, PipelineOverlap,
   DynamicTopology). This catches regressions in the registration plumbing
   without requiring a C++ build.

2. Behavioral contract (skipped if `tessera-opt` is not on PATH or not yet
   built): runs `tessera-opt -tessera-halo-infer` against a minimal stencil
   and asserts the expected `halo.width` annotation appears.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
NEIGHBORS_ROOT = REPO_ROOT / "src" / "compiler" / "tessera_neighbors"
TESSERA_OPT_CPP = REPO_ROOT / "tools" / "tessera-opt" / "tessera-opt.cpp"
TESSERA_OPT_CMAKE = REPO_ROOT / "tools" / "tessera-opt" / "CMakeLists.txt"

PASS_REGISTRATION_FNS = (
    "registerHaloInferPass",
    "registerStencilLowerPass",
    "registerPipelineOverlapPass",
    "registerDynamicTopologyPass",
)


# --------------------------------------------------------------------------- #
# Structural wiring
# --------------------------------------------------------------------------- #


def test_neighbors_passes_header_declares_all_four_registration_fns() -> None:
    header = NEIGHBORS_ROOT / "include" / "tessera" / "Dialect" / "Neighbors" / "Transforms" / "Passes.h"
    text = header.read_text()
    for fn in PASS_REGISTRATION_FNS:
        assert fn in text, f"{fn} missing from Passes.h"


def test_neighbors_dialect_header_declares_register_fn() -> None:
    header = NEIGHBORS_ROOT / "include" / "tessera" / "Dialect" / "Neighbors" / "IR" / "NeighborsDialect.h"
    assert header.exists(), "NeighborsDialect.h registration header missing"
    assert "registerNeighborsDialect" in header.read_text()


def test_each_pass_cpp_defines_its_registration_fn() -> None:
    pass_cpp_files = {
        "HaloInferPass.cpp": "registerHaloInferPass",
        "StencilLowerPass.cpp": "registerStencilLowerPass",
        "PipelineOverlapPass.cpp": "registerPipelineOverlapPass",
        "DynamicTopologyPass.cpp": "registerDynamicTopologyPass",
    }
    transforms_dir = NEIGHBORS_ROOT / "lib" / "Dialect" / "Neighbors" / "Transforms"
    for filename, fn_name in pass_cpp_files.items():
        text = (transforms_dir / filename).read_text()
        assert f"void {fn_name}()" in text, f"{fn_name} not defined in {filename}"


def test_dialect_cpp_no_longer_includes_missing_tablegen_inc() -> None:
    """The hand-written dialect must not reference NeighborsOps.cpp.inc — that
    file is never generated because TableGen is not wired in CMakeLists."""
    cpp = NEIGHBORS_ROOT / "lib" / "Dialect" / "Neighbors" / "IR" / "TesseraNeighbors.cpp"
    text = cpp.read_text()
    assert 'NeighborsOps.cpp.inc' not in text, (
        "TesseraNeighbors.cpp still references the missing TableGen output"
    )


def test_dialect_cpp_registers_all_seven_ops() -> None:
    cpp = NEIGHBORS_ROOT / "lib" / "Dialect" / "Neighbors" / "IR" / "TesseraNeighbors.cpp"
    text = cpp.read_text()
    expected_ops = (
        "CreateTopologyOp",
        "HaloRegionOp",
        "HaloExchangeOp",
        "NeighborReadOp",
        "StencilDefineOp",
        "StencilApplyOp",
        "PipelineConfigOp",
    )
    for op in expected_ops:
        assert f"struct {op}" in text, f"{op} struct definition missing"
    # And they must all appear in the addOperations<...> list.
    add_ops_line = next(
        (line for line in text.splitlines() if "addOperations<" in line),
        None,
    )
    add_ops_block_start = text.find("addOperations<")
    assert add_ops_block_start != -1, "addOperations<...> call missing"
    add_ops_block = text[add_ops_block_start : add_ops_block_start + 500]
    for op in expected_ops:
        assert op in add_ops_block, f"{op} not listed in addOperations<>"


def test_tessera_opt_cpp_registers_neighbors_dialect_and_passes() -> None:
    text = TESSERA_OPT_CPP.read_text()
    assert "NeighborsDialect.h" in text, "tessera-opt does not include the dialect header"
    assert "registerNeighborsDialect(registry)" in text
    for fn in PASS_REGISTRATION_FNS:
        assert f"tessera::neighbors::{fn}" in text, (
            f"tessera-opt does not call {fn}"
        )


def test_tessera_opt_cmake_links_tesseraneighbors() -> None:
    text = TESSERA_OPT_CMAKE.read_text()
    assert "TesseraNeighbors" in text, (
        "tools/tessera-opt/CMakeLists.txt does not link TesseraNeighbors"
    )


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


_HALO_INFER_INPUT = """\
func.func @test_stencil_halo_infer(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %topo = "tessera.neighbors.topology.create"() {
      kind = "2d_mesh", defaults = "von_neumann"
  } : () -> !tessera.neighbors.topology

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]> : tensor<2xi64>,
              dense<[1, 0]> : tensor<2xi64>,
              dense<[-1, 0]> : tensor<2xi64>,
              dense<[0, 1]> : tensor<2xi64>,
              dense<[0, -1]> : tensor<2xi64>],
      bc = "periodic"
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?x?xf32>, !tessera.neighbors.topology) -> tensor<?x?xf32>

  return %out : tensor<?x?xf32>
}
"""


def test_halo_infer_pass_annotates_stencil_apply() -> None:
    binary = _find_tessera_opt()
    if binary is None:
        pytest.skip("tessera-opt not built — skipping behavioral contract test")

    result = subprocess.run(
        [binary, "-tessera-halo-infer"],
        input=_HALO_INFER_INPUT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"tessera-opt -tessera-halo-infer failed: {result.stderr}"
    )
    assert "tessera.neighbors.stencil.apply" in result.stdout
    assert "halo.width" in result.stdout, (
        f"HaloInferPass did not annotate halo.width.\nOutput:\n{result.stdout}"
    )
