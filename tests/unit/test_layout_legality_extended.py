"""LayoutLegalityPass — producer/consumer accept-set extended to conv2d + attn.

The pass now enforces a per-consumer-op layout accept-set on the operands that
carry the contract: tessera.matmul (row_major/col_major, kept verbatim),
tessera.conv2d_nhwc (nhwc on the data operand), tessera.flash_attn (bhsd on
Q/K/V). See COMPILER_AUDIT "layout and binding contracts are uneven".
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PASS_SRC = REPO / "src" / "transforms" / "lib" / "LayoutLegalityPass.cpp"
_CANDIDATES = (
    REPO / "build" / "tools" / "tessera-opt" / "tessera-opt",
    REPO / "build-llvm22" / "tools" / "tessera-opt" / "tessera-opt",
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


def test_pass_source_covers_conv_and_attn():
    src = PASS_SRC.read_text()
    # matmul rule preserved (existing fixtures + verifier-sprint test depend on it)
    assert "matmulAcceptSet" in src
    # extended consumers with their accept-sets
    assert 'tessera.conv2d_nhwc' in src and '"nhwc"' in src
    assert 'tessera.flash_attn' in src and '"bhsd"' in src
    assert "checkTensorOpLayouts" in src


@_needs_opt
def test_conv_and_attn_layout_mismatch_diagnostics(tmp_path):
    fixture = '''
func.func @conv_bad(%x: tensor<1x8x8x3xf32>, %w: tensor<3x3x3x16xf32>) -> tensor<1x8x8x16xf32> {
  %xc = "tessera.cast"(%x) {tessera.layout = "nchw"} : (tensor<1x8x8x3xf32>) -> tensor<1x8x8x3xf32>
  %o = "tessera.conv2d_nhwc"(%xc, %w) {strides = [1, 1], dilations = [1, 1]} : (tensor<1x8x8x3xf32>, tensor<3x3x3x16xf32>) -> tensor<1x8x8x16xf32>
  return %o : tensor<1x8x8x16xf32>
}
func.func @attn_bad(%q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>, %v: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
  %qc = "tessera.cast"(%q) {tessera.layout = "row_major"} : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  %o = "tessera.flash_attn"(%qc, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {head_dim = 4 : i64} : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  return %o : tensor<1x2x8x4xf32>
}
'''
    f = tmp_path / "ll.mlir"
    f.write_text(fixture)
    out = subprocess.run(
        [_OPT, str(f), "-tessera-layout-legality", "--allow-unregistered-dialect"],
        capture_output=True, text=True, timeout=60)
    # Both consumers reject their bad operand layouts.
    err = out.stderr
    assert 'tessera.conv2d_nhwc operand #0 has layout "nchw" but its accept-set is {nhwc}' in err
    assert 'tessera.flash_attn operand #0 has layout "row_major" but its accept-set is {bhsd}' in err


@_needs_opt
def test_correct_layouts_pass_clean(tmp_path):
    fixture = '''
func.func @ok(%q: tensor<1x2x8x4xf32>, %k: tensor<1x2x8x4xf32>, %v: tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32> {
  %qc = "tessera.cast"(%q) {tessera.layout = "bhsd"} : (tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  %o = "tessera.flash_attn"(%qc, %k, %v) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>}> {head_dim = 4 : i64} : (tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>, tensor<1x2x8x4xf32>) -> tensor<1x2x8x4xf32>
  return %o : tensor<1x2x8x4xf32>
}
'''
    f = tmp_path / "ok.mlir"
    f.write_text(fixture)
    out = subprocess.run(
        [_OPT, str(f), "-tessera-layout-legality", "--allow-unregistered-dialect"],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert "LAYOUT_LEGALITY_PRODUCER_CONSUMER_MISMATCH" not in out.stderr
