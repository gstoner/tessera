"""Single-source enforcer: the C++ ROCm lowering's emitted ``fp8_flavor`` must
match the Python source of truth ``rocm_target.fp8_dtype_flavor`` for every
arch (B4 / A6).

Mirrors the ``test_apple_gpu_tile_pass_status_matches_envelope`` pattern: runs
the real ``tessera-rocm-opt`` over an FP8 ``tile.mma`` and asserts the C++ table
in ``TileToROCM.cpp`` and the Python ``_FP8_SEMANTICS`` table never drift.
Skips cleanly when the binary has not been built.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

from tessera.compiler.rocm_target import (
    AMDArch,
    fp8_dtype_flavor,
    fp8_semantics,
    rocm_arch_string,
)

_REPO = Path(__file__).resolve().parents[2]
_CANDIDATES = [
    _REPO / "build-rocm/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt",
    _REPO / "build/src/compiler/codegen/Tessera_ROCM_Backend/tools/tessera-rocm-opt",
]


def _trop() -> str:
    for c in _CANDIDATES:
        if c.exists():
            return str(c)
    found = shutil.which("tessera-rocm-opt")
    if found:
        return found
    pytest.skip("tessera-rocm-opt not built (configure -DTESSERA_BUILD_ROCM_BACKEND=ON)")


# fp8 textual MLIR element-type spellings for the two canonical fp8 dtypes.
_MLIR_FP8 = {"fp8_e4m3": "f8E4M3FN", "fp8_e5m2": "f8E5M2"}

# Every arch the Python table knows about.
_ALL_ARCHES = [
    AMDArch.GFX_90A, AMDArch.GFX_940, AMDArch.GFX_942, AMDArch.GFX_950,
    AMDArch.GFX_1100, AMDArch.GFX_1151, AMDArch.GFX_1200,
    AMDArch.GFX_1250, AMDArch.GFX_1251,
]


def _run(arch_str: str, mlir_ty: str) -> subprocess.CompletedProcess[str]:
    src = (
        "module {\n"
        f"  func.func @k(%a: {mlir_ty}, %b: {mlir_ty}) -> {mlir_ty} {{\n"
        f'    %m = "tile.mma"(%a, %b) : ({mlir_ty}, {mlir_ty}) -> {mlir_ty}\n'
        f"    return %m : {mlir_ty}\n"
        "  }\n}\n"
    )
    return subprocess.run(
        [
            _trop(), "--allow-unregistered-dialect",
            f"--pass-pipeline=builtin.module(lower-tile-to-rocm{{arch={arch_str}}})",
        ],
        input=src, capture_output=True, text=True,
    )


_FLAVOR_RE = re.compile(r'fp8_flavor = "([^"]+)"')


@pytest.mark.parametrize("arch", _ALL_ARCHES)
@pytest.mark.parametrize("dtype", ["fp8_e4m3", "fp8_e5m2"])
def test_cpp_fp8_flavor_matches_python(arch: AMDArch, dtype: str) -> None:
    arch_str = rocm_arch_string(arch)
    res = _run(arch_str, _MLIR_FP8[dtype])
    sem = fp8_semantics(arch)

    if sem == "none":
        # No FP8 path → the pass must fail loudly (Decision #21), never emit a flavor.
        assert res.returncode != 0, (
            f"{arch_str} has no FP8 path but the pass succeeded:\n{res.stdout}")
        assert "no FP8 matrix path" in res.stderr
        return

    assert res.returncode == 0, f"pass failed for {arch_str}:\n{res.stderr}"
    m = _FLAVOR_RE.search(res.stdout)
    assert m is not None, f"no fp8_flavor emitted for {arch_str}:\n{res.stdout}"
    emitted = m.group(1)
    expected = fp8_dtype_flavor(arch, dtype)
    assert emitted == expected, (
        f"C++/Python FP8 flavor drift on {arch_str}/{dtype}: "
        f"C++ emitted {emitted!r}, Python says {expected!r}")
