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


def _bf16():
    ml = pytest.importorskip("ml_dtypes")
    return ml.bfloat16


def _load_lib():
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
    return ctypes.CDLL(str(GEMM_LIB), mode=ctypes.RTLD_GLOBAL)


def _bind(lib, name):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def _gemm(fn, store, M, N, K):
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.5).astype(store)
    B = (rng.standard_normal((K, N)) * 0.5).astype(store)
    D = np.zeros((M, N), dtype=np.float32)
    rc = fn(A.ctypes.data_as(ctypes.c_void_p),
            B.ctypes.data_as(ctypes.c_void_p),
            D.ctypes.data_as(ctypes.c_void_p), M, N, K)
    return rc, A, B, D


# General tiled/K-looped GEMM over both storage dtypes — including ragged
# (non-multiple-of-16) shapes that exercise the zero-pad load + bounds-checked
# store, and a K-loop (K > 16).
# Small shapes exercise the 2x4 production tile; the last (min-dim >= 1024)
# exercises the size-adaptive 3x4 large-problem tile (the occupancy lever — see
# STRIX Stage H), including a ragged large shape.
@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (17, 17, 17),
                                   (128, 96, 64), (1024, 1037, 1031)])
def test_shipped_rocm_wmma_f16_matches_numpy(shape):
    fn = _bind(_load_lib(), "tessera_rocm_wmma_gemm_f16")
    rc, A, B, D = _gemm(fn, np.float16, *shape)
    if rc == 2:
        pytest.skip("no usable AMD GPU / HIPRTC (shipped symbol returned rc=2)")
    assert rc == 0, f"tessera_rocm_wmma_gemm_f16{shape} returned {rc}"
    ref = A.astype(np.float32) @ B.astype(np.float32)
    maxerr = float(np.max(np.abs(D - ref)))
    # Large-K contractions accumulate more rounding; scale the bound by K.
    tol = 1e-2 if shape[2] <= 256 else 5e-2
    assert maxerr < tol, f"f16 WMMA GEMM{shape} maxerr={maxerr}"
    # execute-compare against the numpy reference (the conformance dashboard's
    # complete-cell gate looks for this idiom).
    np.testing.assert_allclose(D, ref, atol=tol, rtol=0)


@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (100, 33, 80)])
def test_shipped_rocm_wmma_bf16_matches_numpy(shape):
    bf16 = _bf16()
    fn = _bind(_load_lib(), "tessera_rocm_wmma_gemm_bf16")
    rc, A, B, D = _gemm(fn, bf16, *shape)
    if rc == 2:
        pytest.skip("no usable AMD GPU / HIPRTC (shipped symbol returned rc=2)")
    assert rc == 0, f"tessera_rocm_wmma_gemm_bf16{shape} returned {rc}"
    ref = A.astype(np.float32) @ B.astype(np.float32)
    maxerr = float(np.max(np.abs(D - ref)))
    # bf16 has ~8 mantissa bits — looser tolerance, scaled by the K-contraction.
    assert maxerr < 5e-2 * shape[2], f"bf16 WMMA GEMM{shape} maxerr={maxerr}"


def _bind_staged(lib, name):
    # Shared signature for the LDS (rung 2) + pipelined (rung 3) run symbols:
    # (A, B, D, M, N, K, wm, wn, mt, nt).
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def _bind_lds(lib):
    return _bind_staged(lib, "tessera_rocm_wmma_gemm_f16_lds")


# The rung-2 LDS-staged kernel (multi-wave workgroup, cooperative A/B staging).
# It is NOT the production lane (register blocking wins on Strix Halo — see the
# perf ladder in STRIX_HALO_EXECUTION_PLAN.md), but it ships as a symbol so it
# stays correctness-guarded as the substrate for rung-3 software pipelining.
@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (128, 128, 128),
                                   (100, 33, 80), (257, 129, 200)])
def test_shipped_rocm_wmma_lds_matches_numpy(shape):
    fn = _bind_lds(_load_lib())
    M, N, K = shape
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    D = np.zeros((M, N), dtype=np.float32)
    rc = fn(A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
            D.ctypes.data_as(ctypes.c_void_p), M, N, K, 2, 2, 2, 4)
    if rc == 2:
        pytest.skip("no usable AMD GPU / HIPRTC (LDS symbol returned rc=2)")
    assert rc == 0, f"tessera_rocm_wmma_gemm_f16_lds{shape} returned {rc}"
    ref = A.astype(np.float32) @ B.astype(np.float32)
    maxerr = float(np.max(np.abs(D - ref)))
    assert maxerr < 1e-2, f"LDS WMMA GEMM{shape} maxerr={maxerr}"


# The rung-3 2-stage software-pipelined (double-buffered LDS) kernel. Like rung 2
# it is NOT production (register blocking wins on Strix Halo; rung 3 only edges
# +8% in a narrow 1024³–2048³ window — see STRIX_HALO_EXECUTION_PLAN.md), but it
# ships as a correctness-guarded symbol + the rung-3 reference.
@pytest.mark.parametrize("shape", [(16, 16, 16), (64, 48, 32), (128, 128, 128),
                                   (100, 33, 80), (257, 129, 200)])
def test_shipped_rocm_wmma_pipe_matches_numpy(shape):
    fn = _bind_staged(_load_lib(), "tessera_rocm_wmma_gemm_f16_pipe")
    M, N, K = shape
    rng = np.random.default_rng(0)
    A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
    B = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    D = np.zeros((M, N), dtype=np.float32)
    rc = fn(A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
            D.ctypes.data_as(ctypes.c_void_p), M, N, K, 2, 2, 2, 4)
    if rc == 2:
        pytest.skip("no usable AMD GPU / HIPRTC (pipe symbol returned rc=2)")
    assert rc == 0, f"tessera_rocm_wmma_gemm_f16_pipe{shape} returned {rc}"
    ref = A.astype(np.float32) @ B.astype(np.float32)
    maxerr = float(np.max(np.abs(D - ref)))
    assert maxerr < 1e-2, f"pipe WMMA GEMM{shape} maxerr={maxerr}"


def test_zerocopy_path_matches_numpy_subprocess():
    """The opt-in APU zero-copy path (TESSERA_ROCM_ZEROCOPY=1) — hipHostRegister
    device-mapped host buffers, no H2D/D2H copy — must match numpy. Run in a
    subprocess because the flag is read once per process (cached static), so a
    clean env is needed; this also keeps it isolated from the other in-process
    rocm tests. Skip-clean when the lib isn't built / no GPU."""
    import sys
    if not GEMM_LIB.is_file():
        pytest.skip("build the shipped GEMM lib: ninja -C build tessera_rocm_gemm")
    sys.path.insert(0, str(Path(__file__).parent))
    import _subprocess  # repo's timeout/OOM-aware runner

    script = f"""
import ctypes, os, sys
import numpy as np
for dep in ("libamdhip64.so", "libhiprtc.so"):
    p = os.path.join(os.environ.get("ROCM_PATH", "/opt/rocm"), "lib", dep)
    if os.path.isfile(p):
        try: ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
        except OSError: pass
lib = ctypes.CDLL({str(GEMM_LIB)!r}, mode=ctypes.RTLD_GLOBAL)
fn = lib.tessera_rocm_wmma_gemm_f16
fn.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*3
fn.restype = ctypes.c_int
M, N, K = 128, 96, 64
rng = np.random.default_rng(0)
A = (rng.standard_normal((M, K)) * 0.5).astype(np.float16)
B = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
D = np.zeros((M, N), dtype=np.float32)
rc = fn(A.ctypes.data_as(ctypes.c_void_p), B.ctypes.data_as(ctypes.c_void_p),
        D.ctypes.data_as(ctypes.c_void_p), M, N, K)
if rc == 2:
    print("SKIP_NO_GPU"); sys.exit(0)
assert rc == 0, f"rc={{rc}}"
err = float(np.max(np.abs(D - A.astype(np.float32) @ B.astype(np.float32))))
assert err < 1e-2, f"zerocopy maxerr={{err}}"
print("ZEROCOPY_OK")
"""
    env = dict(os.environ, TESSERA_ROCM_ZEROCOPY="1")
    res = _subprocess.run([sys.executable, "-c", script], env=env, timeout=120)
    if "SKIP_NO_GPU" in res.stdout:
        pytest.skip("no usable AMD GPU / HIPRTC (zero-copy symbol returned rc=2)")
    assert res.ok and "ZEROCOPY_OK" in res.stdout, \
        f"stdout={res.stdout!r} stderr={res.stderr!r}"


def test_shipped_rocm_wmma_symbol_rejects_bad_shape():
    fn = _bind(_load_lib(), "tessera_rocm_wmma_gemm_f16")
    a = np.zeros((16, 16), dtype=np.float16)
    d = np.zeros((16, 16), dtype=np.float32)
    # Non-positive dim -> rc=1, no device needed (shape validated before launch).
    rc = fn(a.ctypes.data_as(ctypes.c_void_p),
            a.ctypes.data_as(ctypes.c_void_p),
            d.ctypes.data_as(ctypes.c_void_p), 0, 16, 16)
    assert rc == 1, f"expected rc=1 (bad shape), got {rc}"
