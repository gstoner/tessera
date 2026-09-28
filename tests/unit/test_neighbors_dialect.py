"""Phase 7 — Neighbors dialect wiring tests.

These tests validate two things:

1. Structural wiring: the `tessera.neighbors.*` ops have one declaration
   (core `TesseraOps.td`), and the Phase 7 passes (HaloInfer, StencilLower,
   PipelineOverlap, DynamicTopology, ...) are registered. This catches
   regressions in the registration plumbing without requiring a C++ build.

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
    "registerBoundaryConditionLowerPass",
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


NEIGHBORS_OPS = (
    "topology.create",
    "halo.region",
    "halo.exchange",
    "halo.pack",
    "halo.transport",
    "halo.unpack",
    "neighbor.read",
    "stencil.define",
    "stencil.apply",
    "pipeline.config",
)
TESSERA_OPS_TD = REPO_ROOT / "src" / "compiler" / "ir" / "TesseraOps.td"


def test_neighbors_ops_have_exactly_one_authority() -> None:
    """Decision #31: the `tessera.neighbors.*` ops are declared once, in the core
    `tessera` dialect ODS -- the declaration MLIR's parser actually resolves.

    An unbuilt `tessera_neighbors.td` and a hand-written C++ `tessera.neighbors`
    dialect re-declared these names until 2026-09-27
    (SMALL-CORRECTNESS-GAPS-2026-09-27). The ODS side is gated for every op by
    `test_ods_op_has_consumer.py`; this pins the neighbors files specifically.
    """
    td = TESSERA_OPS_TD.read_text()
    for op in NEIGHBORS_OPS:
        assert f'"neighbors.{op}"' in td, f"TesseraOps.td lost neighbors.{op}"
    ir_dir = NEIGHBORS_ROOT / "include" / "tessera" / "Dialect" / "Neighbors" / "IR"
    lib_ir = NEIGHBORS_ROOT / "lib" / "Dialect" / "Neighbors" / "IR"
    for stale in (ir_dir, lib_ir):
        leftovers = sorted(stale.rglob("*")) if stale.exists() else []
        assert not leftovers, (
            f"{stale} holds a second neighbors op declaration again: {leftovers}")


def test_each_pass_cpp_defines_its_registration_fn() -> None:
    pass_cpp_files = {
        "HaloInferPass.cpp": "registerHaloInferPass",
        "StencilLowerPass.cpp": "registerStencilLowerPass",
        "BoundaryConditionLowerPass.cpp": "registerBoundaryConditionLowerPass",
        "PipelineOverlapPass.cpp": "registerPipelineOverlapPass",
        "DynamicTopologyPass.cpp": "registerDynamicTopologyPass",
    }
    transforms_dir = NEIGHBORS_ROOT / "lib" / "Dialect" / "Neighbors" / "Transforms"
    for filename, fn_name in pass_cpp_files.items():
        text = (transforms_dir / filename).read_text()
        assert f"void {fn_name}()" in text, f"{fn_name} not defined in {filename}"


def test_tessera_opt_cpp_registers_neighbors_passes_but_no_second_dialect() -> None:
    text = TESSERA_OPT_CPP.read_text()
    assert "registerNeighborsDialect" not in text, (
        "tessera-opt registers a second neighbors dialect; the ops are core "
        "`tessera` dialect ops")
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
  } : () -> index

  %st = "tessera.neighbors.stencil.define"() {
      taps = [dense<[0, 0]> : tensor<2xi64>, dense<[1, 0]> : tensor<2xi64>,
              dense<[-1, 0]> : tensor<2xi64>, dense<[0, 1]> : tensor<2xi64>,
              dense<[0, -1]> : tensor<2xi64>],
      coeffs = [1.0 : f64, 1.0 : f64, 1.0 : f64, 1.0 : f64, 1.0 : f64],
      bc = "periodic"
  } : () -> index

  %out = "tessera.neighbors.stencil.apply"(%st, %arg0, %topo) :
      (index, tensor<?x?xf32>, index) -> tensor<?x?xf32>

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
