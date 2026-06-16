"""Producer-covers-consumer parity guard for the Apple Target IR fusion passes.

The C++ analogue of the Phase 0c equivalence oracle. The Python producer
(`canonical_compile.stamp_fusion_intents`) and the C++ Apple fusion passes are
two *independent* fusion recognizers over two representations (GraphIRModule vs.
MLIR). The descriptor (`tessera.fusion.intent`) is the bridge: the producer
stamps it, the C++ pass consumes it and tags the fused call `source="descriptor"`.

This guard proves the two recognizers are in sync on the stamped path: for every
chain the C++ passes fuse, the producer stamps the intent the consumer expects,
so the lowered call is `source="descriptor"` — never `rediscovered`. A
`rediscovered` leak here would mean the C++ recognizes a chain the producer
doesn't (e.g. the matmul_rmsnorm_safe drift closed in Phase 0c, where the
producer maps that kernel to the "matmul_rmsnorm" intent the C++ reads).

See docs/audit/compiler/COMPILER_AUDIT.md (Library → Optimizing Compiler).
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
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


def _two_op_fixture(tail_op: str, intent: str) -> str:
    # matmul (8x16 @ 16x32 -> 8x32) -> tail (N=32 <= 256), f32, rank-2, static.
    return f'''
func.func @f(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>) -> tensor<8x32xf32> {{
  %m = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %o = "{tail_op}"(%m) {{tessera.fusion.intent = "{intent}"}} : (tensor<8x32xf32>) -> tensor<8x32xf32>
  return %o : tensor<8x32xf32>
}}
'''


def _msm_fixture() -> str:
    return '''
func.func @f(%A: tensor<8x16xf32>, %B: tensor<16x32xf32>, %C: tensor<32x8xf32>) -> tensor<8x8xf32> {
  %m1 = "tessera.matmul"(%A, %B) : (tensor<8x16xf32>, tensor<16x32xf32>) -> tensor<8x32xf32>
  %p  = "tessera.softmax"(%m1) : (tensor<8x32xf32>) -> tensor<8x32xf32>
  %o  = "tessera.matmul"(%p, %C) {tessera.fusion.intent = "matmul_softmax_matmul"} : (tensor<8x32xf32>, tensor<32x8xf32>) -> tensor<8x8xf32>
  return %o : tensor<8x8xf32>
}
'''


# (case-id, fixture, optional epilogue tag the synth path should carry). The
# 2-op chains now lower through the generic `synth_matmul_epilogue` synthesizer
# (Optimizing-Compiler Plan F2b), which still consumes the intent; the 3-op
# attention block keeps its dedicated `matmul_softmax_matmul` kernel.
_CASES = [
    ("matmul_softmax", _two_op_fixture("tessera.softmax", "matmul_softmax"), None),
    ("matmul_gelu", _two_op_fixture("tessera.gelu", "matmul_gelu"), "gelu"),
    ("matmul_rmsnorm", _two_op_fixture("tessera.rmsnorm", "matmul_rmsnorm"),
     "rmsnorm"),
    # The Phase 0c drift case: rmsnorm_safe carries the "matmul_rmsnorm" intent
    # the C++ consumer reads for both variants.
    ("matmul_rmsnorm_safe",
     _two_op_fixture("tessera.rmsnorm_safe", "matmul_rmsnorm"), "rmsnorm"),
    ("matmul_softmax_matmul", _msm_fixture(), None),
]


@_needs_opt
@pytest.mark.parametrize("case_id,fixture,epilogue",
                         _CASES, ids=[c[0] for c in _CASES])
def test_stamped_chain_lowers_as_descriptor(tmp_path, case_id, fixture, epilogue):
    """A producer-stamped chain must lower as a fused descriptor with
    source="descriptor" (the C++ consumed the intent) — and must NOT leak a
    "rediscovered" tag. The parity invariant; the specific fused kernel
    (dedicated vs. generic synth epilogue) is an implementation choice."""
    f = tmp_path / f"{case_id}.mlir"
    f.write_text(fixture)
    out = subprocess.run(
        [_OPT, str(f),
         "--pass-pipeline=builtin.module(tessera-lower-to-apple_gpu-runtime)",
         "--allow-unregistered-dialect"],
        capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    text = out.stdout
    # It fused (a descriptor was emitted) and the compiler intent drove it.
    assert 'tessera.fusion.kernel = ' in text, text
    assert 'tessera.fusion.source = "descriptor"' in text, text
    assert 'tessera.fusion.source = "rediscovered"' not in text, (
        f"{case_id}: producer-stamped chain lowered as rediscovered — the "
        f"Python producer and the C++ consumer have drifted.\n{text}")
    if epilogue is not None:
        assert f'tessera.fusion.epilogue = "{epilogue}"' in text, text
