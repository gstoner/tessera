"""Apple Target IR fusion-descriptor emit + consume (Decision #19, slice 1).

The matmul→softmax→matmul Apple fusion pass now emits a first-class fusion
descriptor on the fused call (`tessera.fusion.kernel` + `tessera.fusion.source`)
and consumes the compiler's `tessera.fusion.intent` when present, instead of
only re-discovering the chain structurally. See COMPILER_AUDIT "fusion intent is
too late".
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PASS_SRC = (REPO / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend" /
            "lib" / "Target" / "Apple" / "Lowering" /
            "MatmulSoftmaxMatmulFusionToAppleGPU.cpp")
# A2b (2026-07): the intent-attr lookup + Decision #21 mismatch warning moved
# into this shared chain-walk helper; each pass names its intent via
# fusionDescriptorDriven(op, "<kernel>").
CHAIN_UTILS = (REPO / "src" / "compiler" / "codegen" / "Tessera_Apple_Backend" /
               "include" / "Tessera" / "Target" / "Apple" / "FusionChainUtils.h")
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

_FIXTURE = '''
func.func @descriptor_driven(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>, %C: tensor<32x8xf32>) -> tensor<8x8xf32> {
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  %o  = "tessera.matmul"(%p, %C) {tessera.fusion.intent = "matmul_softmax_matmul"} : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
  return %o : tensor<8x8xf32>
}
func.func @rediscovered(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>, %C: tensor<32x8xf32>) -> tensor<8x8xf32> {
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  %o  = "tessera.matmul"(%p, %C) : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
  return %o : tensor<8x8xf32>
}
'''


def test_pass_emits_and_consumes_the_descriptor_in_source():
    src = PASS_SRC.read_text()
    # The pass emits the first-class fusion descriptor on the fused call.
    assert 'tessera.fusion.kernel' in src           # emits the descriptor
    assert 'tessera.fusion.source' in src
    assert '"descriptor"' in src and '"rediscovered"' in src
    # It consumes the compiler intent via the shared chain-walk helper (A2b):
    # the pass names the intent, the helper reads the attr + warns on mismatch.
    assert 'fusionDescriptorDriven' in src          # consumes the compiler intent
    util = CHAIN_UTILS.read_text()
    assert 'tessera.fusion.intent' in util          # helper reads the intent attr
    assert 'descriptor/IR mismatch' in util         # Decision #21 diagnostic


@_needs_opt
def test_descriptor_vs_rediscovered_source_tag(tmp_path):
    f = tmp_path / "fd.mlir"
    f.write_text(_FIXTURE)
    out = subprocess.run(
        [_OPT, str(f),
         "--pass-pipeline=builtin.module(tessera-lower-to-apple_gpu-runtime)",
         "--allow-unregistered-dialect"],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    text = out.stdout
    # Both functions fuse to the same kernel ...
    assert text.count('tessera.fusion.kernel = "matmul_softmax_matmul"') == 2
    # ... but the descriptor-annotated chain is tagged source="descriptor" and
    # the bare chain source="rediscovered".
    assert 'tessera.fusion.source = "descriptor"' in text
    assert 'tessera.fusion.source = "rediscovered"' in text
