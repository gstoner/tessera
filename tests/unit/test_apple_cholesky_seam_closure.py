"""L-series linalg pilot — L6: seam-closure execution for cholesky.

The whole point of the pilot is to vet *every IR layer to the backend* end to
end.  That only holds if the executed numeric result is produced by consuming
the **lowered Target IR** — specifically, by invoking the C ABI ``symbol`` that
the compiler wrote into the ``tessera_apple.*`` op — rather than a parallel
Python dispatcher that pattern-matches the op name independently.

This harness closes that seam for cholesky:

  1. Lower a ``tile.cholesky`` through the real ``tessera-opt`` Tile→Apple pass.
  2. Parse the ``symbol`` attribute straight out of the emitted Target IR.
  3. Compile the Apple CPU runtime, resolve *that exact symbol* via ctypes, run
     it on a random SPD matrix, and assert it matches ``numpy.linalg.cholesky``.

Because the symbol name is read from the IR (never hardcoded in the test), a
compiler change that emitted a different / wrong symbol would make the harness
execute the wrong thing and fail — which is exactly the end-to-end guarantee.

The CPU lane is the executable oracle on this machine (Accelerate LAPACK vs
numpy).  The GPU lane is asserted structurally: the IR must name the GPU MSL
symbol and tag it ``metal_runtime`` (executing the MSL kernel needs the Metal
``.mm`` runtime + a GPU build, covered elsewhere).
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
_OPT_DEFAULT = REPO_ROOT / "build" / "tools" / "tessera-opt" / "tessera-opt"
_CPU_RUNTIME_SRC = (
    REPO_ROOT
    / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_cpu_runtime.cpp"
)

_TILE_CHOLESKY = (
    'module {\n'
    '  "tile.cholesky"() {source = "tessera.cholesky", result = "v0", '
    'ordinal = 0 : i64, lower = true} : () -> ()\n'
    '}\n'
)


def _find_opt() -> str | None:
    if _OPT_DEFAULT.is_file() and os.access(_OPT_DEFAULT, os.X_OK):
        return str(_OPT_DEFAULT)
    return shutil.which("tessera-opt")


_OPT = _find_opt()
pytestmark = pytest.mark.skipif(
    _OPT is None, reason="tessera-opt not built; run `cmake --build build --target tessera-opt`")


def _lower(target_pipeline: str) -> str:
    proc = subprocess.run(
        [_OPT, "-", f"-{target_pipeline}", "--allow-unregistered-dialect"],
        input=_TILE_CHOLESKY, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def _symbol_from_ir(ir: str) -> str:
    m = re.search(r'symbol = "([^"]+)"', ir)
    assert m, f"no `symbol` attribute found in lowered Target IR:\n{ir}"
    return m.group(1)


def test_cpu_seam_closure_executes_ir_named_symbol(tmp_path):
    """End-to-end: the symbol named in the lowered Apple CPU Target IR is the
    one executed, and it computes the correct Cholesky factor."""
    cxx = shutil.which("c++") or shutil.which("clang++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("C++ compiler is not available")

    # 1. + 2. Lower through the real pass and read the symbol from the IR.
    ir = _lower("tessera-lower-to-apple_cpu")
    assert "tessera_apple.cpu.vector_op" in ir
    assert 'abi = "lapack_spotrf"' in ir
    symbol = _symbol_from_ir(ir)
    assert symbol == "tessera_apple_cpu_cholesky_f32"

    # 3. Compile the runtime and resolve *that* symbol dynamically.
    lib = tmp_path / ("librt.dylib" if sys.platform == "darwin" else "librt.so")
    cmd = [cxx, "-std=c++17", "-shared", "-fPIC", str(_CPU_RUNTIME_SRC), "-o", str(lib)]
    if sys.platform == "darwin":
        cmd.extend(["-framework", "Accelerate"])
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    runtime = ctypes.CDLL(str(lib))
    fn = getattr(runtime, symbol)  # resolved from the IR, not hardcoded
    fn.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
    ]
    fn.restype = ctypes.c_int32

    rng = np.random.default_rng(7)
    for n in (3, 8, 32):
        m = rng.standard_normal((n, n)).astype(np.float32)
        a = (m @ m.T + n * np.eye(n, dtype=np.float32)).astype(np.float32)
        out = np.zeros((n, n), dtype=np.float32)
        info = fn(
            a.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            n,
        )
        assert info == 0
        np.testing.assert_allclose(out, np.linalg.cholesky(a), rtol=1e-4, atol=1e-4)
        np.testing.assert_allclose(out @ out.T, a, rtol=1e-4, atol=1e-4)


def test_gpu_ir_names_metal_runtime_symbol():
    """The Apple GPU lowering must name the MSL kernel symbol and tag it
    metal_runtime (executable on the GPU build)."""
    ir = _lower("tessera-lower-to-apple_gpu")
    assert "tessera_apple.gpu.metal_kernel" in ir
    assert 'status = "metal_runtime"' in ir
    assert _symbol_from_ir(ir) == "tessera_apple_gpu_cholesky_f32"


def test_gpu_runtime_declares_cholesky_symbol():
    """Ground the GPU claim: the named symbol exists in the Metal runtime
    source (real MSL kernel from the batched-cholesky work)."""
    mm = list(
        (REPO_ROOT / "src/compiler/codegen/Tessera_Apple_Backend/runtime").glob("*.mm")
    )
    stub = (
        REPO_ROOT
        / "src/compiler/codegen/Tessera_Apple_Backend/runtime/apple_gpu_runtime_stub.cpp"
    )
    sources = mm + ([stub] if stub.is_file() else [])
    assert sources, "no Apple GPU runtime source found"
    blob = "\n".join(p.read_text(errors="ignore") for p in sources)
    assert "tessera_apple_gpu_cholesky_f32" in blob, (
        "GPU runtime must declare the symbol named by the Target IR")
