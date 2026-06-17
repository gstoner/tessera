"""Phase 4 GPU emission — the linalg→gpu→NVVM spine in tessera-opt.

`--tessera-emit-nvvm` lowers a tessera kernel through linalg → scf.parallel →
gpu.launch → outlined gpu.func → NVVM IR text, retargeting the tessera→linalg
codegen spine to the GPU. EMISSION ONLY: GPU launch (cuLaunchKernel) is
hardware-gated. See COMPILER_AUDIT (Phase 4) + tests/tessera-ir/phase8/gpu_emit_nvvm.mlir.
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

_KERNEL = '''
func.func @ew(%a: tensor<64xf32>, %b: tensor<64xf32>) -> tensor<64xf32> {
  %0 = "tessera.add"(%a, %b) : (tensor<64xf32>, tensor<64xf32>) -> tensor<64xf32>
  return %0 : tensor<64xf32>
}
'''


@_needs_opt
def test_tessera_add_emits_nvvm(tmp_path):
    f = tmp_path / "ew.mlir"
    f.write_text(_KERNEL)
    out = subprocess.run(
        [_OPT, str(f), "--tessera-emit-nvvm"],
        capture_output=True, text=True, check=True).stdout
    # The outlined kernel is real NVVM: a gpu.module with an llvm.func kernel and
    # NVVM special-register reads (blockIdx etc.).
    assert "gpu.module" in out
    assert "nvvm.kernel" in out
    assert "nvvm.read.ptx.sreg" in out
    # The original tessera op is fully lowered (no tensor-level tessera.add left).
    assert "tessera.add" not in out


@_needs_opt
def test_emit_nvvm_is_emission_only_no_launch(tmp_path):
    """Honest-scope guard: emission produces the kernel, not a host launch — no
    cuLaunchKernel/gpu.launch_func resolution to a runtime is implied here."""
    f = tmp_path / "ew.mlir"
    f.write_text(_KERNEL)
    out = subprocess.run(
        [_OPT, str(f), "--tessera-emit-nvvm"],
        capture_output=True, text=True, check=True).stdout
    # A gpu.launch_func host stub remains (the kernel is emitted, not launched).
    assert "gpu.launch_func" in out
