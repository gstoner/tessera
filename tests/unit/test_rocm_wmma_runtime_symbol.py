"""Execute-compare fixture for the shipped ROCm WMMA GEMM runtime symbol.

This is the numerical proof behind the `backend_manifest` `hardware_verified`
row for `tessera.matmul` on `rocm_gfx1151`: it dlopens the **shipped**
`libtessera_rocm_gemm.so`, calls the C-ABI symbol `tessera_rocm_wmma_gemm_f16`
(which HIPRTC-compiles the RDNA WMMA kernel for the device arch and launches it),
and compares the GPU result to a numpy reference GEMM.

Unlike the Stage C/D harness tests (launcher in the test), this validates the
*production* symbol built by the CMake `tessera_rocm_gemm` target — the shipped
half of `hardware_verified`.

Skip-clean: lib not built / no usable GPU (the symbol returns rc=2).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")

REPO_ROOT = Path(__file__).resolve().parents[2]
GEMM_LIB = (REPO_ROOT / "build" / "src" / "compiler" / "codegen"
            / "Tessera_ROCM_Backend" / "runtime" / "hip" / "libtessera_rocm_gemm.so")
ROCM_LIB_DIR = os.environ.get("ROCM_PATH", "/opt/rocm") + "/lib"


def _load_symbol():
    if not GEMM_LIB.is_file():
        pytest.skip(f"build the shipped GEMM lib: ninja -C build tessera_rocm_gemm "
                    f"({GEMM_LIB} missing)")
    # Preload the HIP runtime + HIPRTC globally so the gemm lib resolves them.
    for dep in ("libamdhip64.so", "libhiprtc.so"):
        p = os.path.join(ROCM_LIB_DIR, dep)
        if os.path.isfile(p):
            try:
                ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass
    lib = ctypes.CDLL(str(GEMM_LIB), mode=ctypes.RTLD_GLOBAL)
    fn = lib.tessera_rocm_wmma_gemm_f16
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def test_shipped_rocm_wmma_symbol_matches_numpy():
    fn = _load_symbol()
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((16, 16)) * 0.5).astype(np.float16)
    B = (rng.standard_normal((16, 16)) * 0.5).astype(np.float16)
    D = np.zeros((16, 16), dtype=np.float32)
    rc = fn(A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            D.ctypes.data_as(ctypes.c_void_p), 16, 16, 16)
    if rc == 2:
        pytest.skip("no usable AMD GPU / HIPRTC (shipped symbol returned rc=2)")
    assert rc == 0, f"tessera_rocm_wmma_gemm_f16 returned {rc}"
    ref = A.astype(np.float32) @ B.astype(np.float32)
    maxerr = float(np.max(np.abs(D - ref)))
    assert maxerr < 1e-2, f"WMMA GEMM maxerr={maxerr} vs numpy reference"


def test_shipped_rocm_wmma_symbol_rejects_unsupported_shape():
    fn = _load_symbol()
    A = np.zeros((16, 16), dtype=np.float16)
    B = np.zeros((16, 16), dtype=np.float16)
    D = np.zeros((16, 16), dtype=np.float32)
    # 17x17x17 is not the single supported tile -> rc=1, no device needed.
    rc = fn(A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            D.ctypes.data_as(ctypes.c_void_p), 17, 17, 17)
    assert rc == 1, f"expected rc=1 (unsupported shape), got {rc}"
