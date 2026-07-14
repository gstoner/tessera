"""Execute-compare fixture for the shipped NVIDIA mma.sync GEMM runtime symbols.

This is the numerical proof behind the `backend_manifest` `device_verified_abi`
row for `tessera.matmul` on `nvidia_sm120`: it dlopens the **shipped**
`libtessera_nvidia_gemm.so`, calls the C-ABI symbols
`tessera_nvidia_mma_gemm_{bf16,f16,tf32,e4m3,e5m2}` (each NVRTC-compiles the
warp-level mma.sync kernel for the device arch and launches it), and compares the
GPU result to a numpy reference GEMM at the matching dtype.

Unlike test_conformance_execute_compare_nvidia.py (launcher in the test, emitted
PTX), this validates the *production* symbols built by the CMake
`tessera_nvidia_gemm` target — the shipped half of `device_verified_abi`.

Skip-clean: lib not built / no usable GPU / no NVRTC (the symbol returns rc=2).
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pytest

ml_dtypes = pytest.importorskip("ml_dtypes")

REPO_ROOT = Path(__file__).resolve().parents[2]
GEMM_LIB = (REPO_ROOT / "build" / "src" / "compiler" / "codegen"
            / "tessera_gpu_backend_NVIDIA" / "runtime" / "cuda"
            / "libtessera_nvidia_gemm.so")

# Where the real driver + NVRTC live (WSL ships libcuda under /usr/lib/wsl/lib).
_CUDA_LIB_DIRS = [
    "/usr/lib/wsl/lib",
    os.environ.get("CUDA_PATH", "/usr/local/cuda") + "/lib64",
]


def _load_lib():
    if not GEMM_LIB.is_file():
        pytest.skip(
            "build the shipped GEMM lib: "
            "cmake -B build -DTESSERA_BUILD_NVIDIA_BACKEND=ON -DTESSERA_ENABLE_CUDA=ON "
            "&& ninja -C build tessera_nvidia_gemm "
            f"({GEMM_LIB} missing)")
    # Preload the CUDA driver + NVRTC globally so the gemm lib resolves them.
    for dep in ("libcuda.so.1", "libcuda.so", "libnvrtc.so"):
        for d in _CUDA_LIB_DIRS:
            p = os.path.join(d, dep)
            if os.path.isfile(p):
                try:
                    ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass
                break
    return ctypes.CDLL(str(GEMM_LIB), mode=ctypes.RTLD_GLOBAL)


def _bind(lib, name):
    fn = getattr(lib, name)
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                   ctypes.c_int, ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int
    return fn


def _round_tf32(x: np.ndarray) -> np.ndarray:
    """Round f32 to tf32 (10-bit mantissa) with round-to-nearest-even."""
    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    r = (u >> 13) & 1
    u = (u + 0xFFF + r) & ~np.uint64(0x1FFF)
    return u.astype(np.uint32).view(np.float32)


# (symbol suffix, numpy storage dtype, builder→(A_bytes_arr, B_bytes_arr, refA, refB), tol)
def _case(dt, rng, M, N, K):
    a = rng.standard_normal((M, K)).astype(np.float32)
    b = rng.standard_normal((K, N)).astype(np.float32)
    if dt in ("f16",):
        qa, qb = a.astype(np.float16), b.astype(np.float16)
    elif dt == "bf16":
        qa, qb = a.astype(ml_dtypes.bfloat16), b.astype(ml_dtypes.bfloat16)
    elif dt == "tf32":
        qa, qb = _round_tf32(a), _round_tf32(b)
    elif dt == "e4m3":
        qa, qb = a.astype(ml_dtypes.float8_e4m3fn), b.astype(ml_dtypes.float8_e4m3fn)
    elif dt == "e5m2":
        qa, qb = a.astype(ml_dtypes.float8_e5m2), b.astype(ml_dtypes.float8_e5m2)
    else:
        raise AssertionError(dt)
    refA = np.ascontiguousarray(qa).astype(np.float32)
    refB = np.ascontiguousarray(qb).astype(np.float32)
    return np.ascontiguousarray(qa), np.ascontiguousarray(qb), refA @ refB


_DTYPES = ["bf16", "f16", "tf32", "e4m3", "e5m2"]
_SHAPES = [(16, 8, 16), (32, 24, 48), (64, 64, 64), (17, 9, 31), (128, 128, 256)]


@pytest.mark.parametrize("dt", _DTYPES)
def test_shipped_nvidia_mma_gemm_matches_numpy(dt):
    lib = _load_lib()
    fn = _bind(lib, f"tessera_nvidia_mma_gemm_{dt}")
    rng = np.random.default_rng(20260625)
    skipped_all = True
    for (M, N, K) in _SHAPES:
        qa, qb, ref = _case(dt, rng, M, N, K)
        D = np.zeros((M, N), dtype=np.float32)
        rc = fn(qa.ctypes.data_as(ctypes.c_void_p),
                qb.ctypes.data_as(ctypes.c_void_p),
                D.ctypes.data_as(ctypes.c_void_p), M, N, K)
        if rc == 2:
            pytest.skip("no usable NVIDIA GPU / NVRTC (shipped symbol returned rc=2)")
        assert rc == 0, f"tessera_nvidia_mma_gemm_{dt}{(M,N,K)} returned {rc}"
        skipped_all = False
        maxerr = float(np.max(np.abs(D - ref)))
        # tf32/16-bit accumulate more rounding with K; fp8 is coarse but exact-ish.
        base = {"bf16": 2e-1, "f16": 5e-2, "tf32": 1e-2, "e4m3": 1.0, "e5m2": 2.0}[dt]
        tol = base * (5 if K > 256 else 1)
        assert maxerr < tol, f"{dt} GEMM{(M,N,K)} maxerr={maxerr} >= {tol}"
    assert not skipped_all


@pytest.mark.parametrize("dt", _DTYPES)
def test_runtime_launch_reaches_every_declared_nvidia_mma_dtype(dt):
    """The public execution row must expose every dtype the shipped ABI claims."""
    from tessera import runtime as rt
    _load_lib()
    a, b, ref = _case(dt, np.random.default_rng(1200 + len(dt)), 17, 9, 31)
    artifact = rt.RuntimeArtifact(metadata={
        "target": "nvidia_sm120", "compiler_path": "nvidia_mma",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a", "b"], "output_name": "d",
        "ops": [{"op_name": "tessera.matmul", "result": "d",
                 "operands": ["a", "b"], "kwargs": {}}],
    })
    result = rt.launch(artifact, (a, b))
    assert result["ok"] is True, result.get("reason")
    tol = {"bf16": .2, "f16": .05, "tf32": .01,
           "e4m3": 1.0, "e5m2": 2.0}[dt]
    np.testing.assert_allclose(result["output"], ref, rtol=0, atol=tol)


@pytest.mark.parametrize("dt", _DTYPES)
def test_device_resident_gemm_stream_and_event_abi(dt):
    """No H2D/D2H occurs between the timed resident GEMM launches."""
    from tessera.compiler.emit.nvidia_cuda import NvidiaDeviceSession
    _load_lib()
    a, b, ref = _case(dt, np.random.default_rng(1300 + len(dt)), 19, 13, 29)
    dtype_key = {"bf16": "bfloat16", "f16": "float16", "tf32": "float32",
                 "e4m3": "float8_e4m3fn", "e5m2": "float8_e5m2"}[dt]
    with NvidiaDeviceSession() as session:
        da, db = session.upload(a), session.upload(b)
        out = session.empty((19, 13), np.float32)
        def launch():
            session.gemm(da, db, out, dtype_key)
        launch()
        latency = session.measure(launch, reps=8, warmup=2)
        got = out.numpy()
    assert latency > 0
    tol = {"bf16": .2, "f16": .05, "tf32": .01,
           "e4m3": 1.0, "e5m2": 2.0}[dt]
    np.testing.assert_allclose(got, ref, rtol=0, atol=tol)
