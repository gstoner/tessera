"""Stage J — `tessera_rocm.wmma` lowers to a REAL `rocdl.wmma` intrinsic.

Before Stage J the `lower-tessera-target-to-rocdl` pass replaced every
`tessera_rocm.wmma` with a void *artifact marker* (`llvm.amdgcn.wmma.contract`)
— a placeholder that computed nothing. Stage J makes a WMMA op that carries real
RDNA fragment vectors lower to the real `rocdl.wmma.f32.16x16x16.{f16,bf16}` op,
which translates to `llvm.amdgcn.wmma.f32.16x16x16.*` — the **same instruction**
the standalone `rocdl_emit.py` emitter produces and `llc` proves. This folds the
Python side-emitter (compiler path 4) into the MLIR Target IR pass (path 3).

The golden intrinsic name is taken from `rocdl_emit.wmma_intrinsic(dtype)` so the
two emitters can never silently diverge.

Abstract / scalar WMMA (contract-level IR, no real fragments) still lowers to the
marker — fragment materialization is a separate (Stage K) lowering step; the
marker is the honest lowering at that abstraction level.

Skip-clean: tessera-opt / mlir-translate not built or found.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tessera.compiler import rocdl_emit

REPO = Path(__file__).resolve().parents[2]
TESSERA_OPT = REPO / "build" / "tools" / "tessera-opt" / "tessera-opt"


def _find(tool: str, *cands: str):
    if env := os.environ.get(f"TESSERA_{tool.upper().replace('-', '_')}"):
        return env if Path(env).is_file() else None
    for c in cands:
        if Path(c).is_file():
            return c
    return shutil.which(tool)


def _mlir_translate():
    return _find("mlir-translate", "/usr/lib/llvm-23/bin/mlir-translate",
                 "/opt/homebrew/opt/llvm/bin/mlir-translate")


def _need_opt():
    if not TESSERA_OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")


def _run(cmd, src):
    return subprocess.run(cmd, input=src, capture_output=True, text=True)


def _lower(src: str) -> str:
    r = _run([str(TESSERA_OPT), "-", "--lower-tessera-target-to-rocdl"], src)
    assert r.returncode == 0, f"lowering failed: {r.stderr}"
    return r.stdout


def _to_llvmir(rocdl_mlir: str) -> str:
    mt = _mlir_translate()
    if mt is None:
        pytest.skip("mlir-translate not found (install LLVM 23)")
    r = _run([mt, "--mlir-to-llvmir"], rocdl_mlir)
    assert r.returncode == 0, f"mlir-translate failed: {r.stderr}"
    return r.stdout


_FRAG = {
    "f16": "vector<16xf16>",
    "bf16": "vector<16xbf16>",
}


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_wmma_with_fragments_lowers_to_real_rocdl_wmma(dtype):
    _need_opt()
    frag = _FRAG[dtype]
    src = f'''
llvm.func @w(%a: {frag}, %b: {frag}, %c: vector<8xf32>) -> vector<8xf32> {{
  %d = "tessera_rocm.wmma"(%a, %b, %c)
       : ({frag}, {frag}, vector<8xf32>) -> vector<8xf32>
  llvm.return %d : vector<8xf32>
}}
'''
    rocdl = _lower(src)
    # The real op, not the marker.
    assert f"rocdl.wmma.f32.16x16x16.{dtype}" in rocdl, rocdl
    assert "wmma.contract" not in rocdl


@pytest.mark.parametrize("dtype", ["f16", "bf16"])
def test_lowered_intrinsic_matches_rocdl_emit_golden(dtype):
    """The LLVM IR the MLIR pass produces must carry the EXACT intrinsic name
    that the standalone rocdl_emit.py emitter declares — paths 3 and 4 agree."""
    _need_opt()
    frag = _FRAG[dtype]
    src = f'''
llvm.func @w(%a: {frag}, %b: {frag}, %c: vector<8xf32>) -> vector<8xf32> {{
  %d = "tessera_rocm.wmma"(%a, %b, %c)
       : ({frag}, {frag}, vector<8xf32>) -> vector<8xf32>
  llvm.return %d : vector<8xf32>
}}
'''
    llvmir = _to_llvmir(_lower(src))
    golden = rocdl_emit.wmma_intrinsic(dtype)   # e.g. llvm.amdgcn.wmma.f32.16x16x16.f16
    assert golden in llvmir, f"{golden!r} not in MLIR-lowered LLVM IR:\n{llvmir}"
    # bf16 takes the bit-pattern as <16 x i16> (RDNA ABI), f16 native <16 x half>.
    if dtype == "bf16":
        assert "<16 x i16>" in llvmir
    else:
        assert "<16 x half>" in llvmir


def test_abstract_scalar_wmma_still_lowers_to_marker():
    """Contract-level IR (scalar operands, no real fragments) keeps the artifact
    marker — fragment materialization is a separate lowering step."""
    _need_opt()
    src = '''
func.func @w(%a: f16, %b: f16, %c: f16) -> f16 {
  %d = "tessera_rocm.wmma"(%a, %b, %c) : (f16, f16, f16) -> f16
  return %d : f16
}
'''
    rocdl = _lower(src)
    assert "llvm.amdgcn.wmma.contract" in rocdl
    assert "rocdl.wmma" not in rocdl
