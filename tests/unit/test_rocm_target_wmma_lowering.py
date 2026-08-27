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

Abstract / scalar WMMA is Target IR only.  The executable conversion fails
closed until fragment materialization has produced the hardware vector ABI;
it never manufactures a void marker or an undefined result.

Skip-clean: tessera-opt / mlir-translate not built or found.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tessera.compiler import rocdl_emit
from tests._support.compiler_tool import require_tessera_opt, run_tessera_opt

REPO = Path(__file__).resolve().parents[2]


def _find(tool: str, *cands: str):
    if env := os.environ.get(f"TESSERA_{tool.upper().replace('-', '_')}"):
        return env if Path(env).is_file() else None
    for c in cands:
        if Path(c).is_file():
            return c
    return shutil.which(tool)


def _mlir_translate():
    return _find("mlir-translate", "/usr/lib/llvm-23/bin/mlir-translate",
                 "/opt/homebrew/opt/llvm@23/bin/mlir-translate")


def _need_opt():
    """Resolve the driver, skipping when it lacks the ROCm target lowering."""
    require_tessera_opt("--lower-tessera-target-to-rocdl")


def _run(cmd, src):
    return subprocess.run(cmd, input=src, capture_output=True, text=True)


def _lower(src: str) -> str:
    r = run_tessera_opt(src, "--lower-tessera-target-to-rocdl")
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


def test_abstract_scalar_wmma_fails_closed_at_executable_boundary():
    """Scalar contracts cannot masquerade as executable ROCm artifacts."""
    _need_opt()
    src = '''
func.func @w(%a: f16, %b: f16, %c: f16) -> f16 {
  %d = "tessera_rocm.wmma"(%a, %b, %c) : (f16, f16, f16) -> f16
  return %d : f16
}
'''
    result = run_tessera_opt(src, "--lower-tessera-target-to-rocdl")
    assert result.returncode != 0
    assert "requires typed hardware fragment vectors" in result.stderr
    assert "wmma.contract" not in result.stdout


def test_f16_accumulator_fragments_lower_to_real_opsel_intrinsic():
    _need_opt()
    src = '''
llvm.func @w(%a: vector<16xf16>, %b: vector<16xf16>,
             %c: vector<16xf16>) -> vector<16xf16> {
  %d = "tessera_rocm.wmma"(%a, %b, %c)
       : (vector<16xf16>, vector<16xf16>, vector<16xf16>) -> vector<16xf16>
  llvm.return %d : vector<16xf16>
}
'''
    rocdl = _lower(src)
    assert "rocdl.wmma.f16.16x16x16.f16" in rocdl
    # MLIR elides the default-valued false attribute.  An explicit true would
    # select the high accumulator half and violate this storage contract.
    assert "opsel = true" not in rocdl
