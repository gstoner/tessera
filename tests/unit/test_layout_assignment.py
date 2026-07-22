"""LayoutAssignmentPass v1 — the assignment half of the layout contract.

Seed kernel-producer layouts (matmul→row_major, flash_attn→bhsd,
conv2d_nhwc→nhwc), propagate through pointwise ops, and insert
tessera.cast{layout} markers at consumer accept-set boundaries. Paired with
LayoutLegalityPass as its verifier. See COMPILER_AUDIT (Phase 1).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PASS_SRC = REPO / "src" / "transforms" / "lib" / "LayoutAssignmentPass.cpp"
_CANDIDATES = (
    REPO / "build" / "tools" / "tessera-opt" / "tessera-opt",
    REPO / "build-llvm23" / "tools" / "tessera-opt" / "tessera-opt",
)


def _find_opt():
    if explicit := os.environ.get("TESSERA_OPT_PATH"):
        if Path(explicit).is_file():
            return explicit
    for c in _CANDIDATES:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return shutil.which("tessera-opt")


_OPT = _find_opt()
_needs_opt = pytest.mark.skipif(_OPT is None, reason="tessera-opt not built")


def test_pass_source_has_three_phases():
    src = PASS_SRC.read_text()
    # Seed kernel producers with their natural layouts.
    assert "producerLayout" in src
    assert '"row_major"' in src and '"bhsd"' in src and '"nhwc"' in src
    # Propagate through pointwise.
    assert "isPointwise" in src
    # Insert cast{layout} at consumer accept-set boundaries.
    assert "consumerAcceptSet" in src and "tessera.cast" in src


@_needs_opt
def test_seed_and_propagate(tmp_path):
    fixture = '''
func.func @f(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %0 = "tessera.matmul"(%a, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %1 = "tessera.relu"(%0) : (tensor<4x16xf32>) -> tensor<4x16xf32>
  return %1 : tensor<4x16xf32>
}
'''
    f = tmp_path / "seed.mlir"
    f.write_text(fixture)
    out = subprocess.run(
        [_OPT, str(f), "--tessera-layout-assignment"],
        capture_output=True, text=True, check=True).stdout
    # matmul seeded + relu propagated, both row_major.
    assert out.count('tessera.layout = "row_major"') == 2


@_needs_opt
def test_insert_cast_at_consumer_boundary_then_legal(tmp_path):
    # lhs producer tagged "tile" (not in matmul accept-set) → a row_major cast is
    # spliced before the matmul lhs; the result is then layout-legal.
    fixture = '''
func.func @f(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
  %at = "tessera.cast"(%a) {tessera.layout = "tile"} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  %0 = "tessera.matmul"(%at, %b) : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  return %0 : tensor<4x16xf32>
}
'''
    f = tmp_path / "insert.mlir"
    f.write_text(fixture)
    out = subprocess.run(
        [_OPT, str(f), "--tessera-layout-assignment"],
        capture_output=True, text=True, check=True).stdout
    assert 'tessera.layout = "tile"' in out          # original marker kept
    assert 'tessera.layout = "row_major"' in out      # inserted cast + seeded matmul
    assert out.count("tessera.cast") == 2             # original + inserted

    # The assignment output must be layout-legal (assign + verify).
    legal = subprocess.run(
        [_OPT, str(f), "--tessera-layout-assignment", "--tessera-layout-legality"],
        capture_output=True, text=True)
    assert legal.returncode == 0, legal.stderr


@_needs_opt
def test_matmul_epilogue_last_axis_reduce_propagates_and_preserves_packing(tmp_path):
    fixture = '''
func.func @f(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4xf32> {
  %0 = "tessera.matmul"(%a, %b) {tessera.storage_packed = true, tessera.storage_container = "int8"} : (tensor<4x8xf32>, tensor<8x16xf32>) -> tensor<4x16xf32>
  %1 = "tessera.gelu"(%0) : (tensor<4x16xf32>) -> tensor<4x16xf32>
  %2 = "tessera.reduce"(%1) {kind = "sum", axis = -1 : i64} : (tensor<4x16xf32>) -> tensor<4xf32>
  return %2 : tensor<4xf32>
}
'''
    f = tmp_path / "matmul_epilogue_reduce.mlir"
    f.write_text(fixture)
    out = subprocess.run(
        [_OPT, str(f), "--tessera-layout-assignment"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert out.count('tessera.layout = "row_major"') == 3
    assert "tessera.storage_packed = true" in out
    assert 'tessera.storage_container = "int8"' in out


@_needs_opt
def test_inserted_cast_records_source_layout(tmp_path):
    fixture = '''
func.func @f(%x: tensor<4x16xf32>) -> tensor<4xf32> {
  %p = "tessera.cast"(%x) {tessera.layout = "packed"} : (tensor<4x16xf32>) -> tensor<4x16xf32>
  %r = "tessera.reduce"(%p) {kind = "sum", axis = -1 : i64} : (tensor<4x16xf32>) -> tensor<4xf32>
  return %r : tensor<4xf32>
}
'''
    f = tmp_path / "packed_reduce.mlir"
    f.write_text(fixture)
    out = subprocess.run(
        [_OPT, str(f), "--tessera-layout-assignment"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert 'tessera.source_layout = "packed"' in out
    assert 'tessera.layout = "row_major"' in out
