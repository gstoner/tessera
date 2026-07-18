"""Execute-compare fixture for the shipped NVIDIA mma.sync GEMM runtime symbols.

This is the numerical proof behind the `backend_manifest` `device_verified_abi`
row for `tessera.matmul` on `nvidia_sm120`: it dlopens the **shipped**
`libtessera_nvidia_gemm.so`, calls the C-ABI symbols
`tessera_nvidia_mma_gemm_{bf16,f16,tf32,e4m3,e5m2,nvfp4}` (each NVRTC-compiles the
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
from tests._support.nvidia_numerics import assert_matches

pytestmark = pytest.mark.hardware_nvidia

ml_dtypes = pytest.importorskip("ml_dtypes")

REPO_ROOT = Path(__file__).resolve().parents[3]
_GEMM_LIBS = tuple(
    REPO_ROOT / build / "src" / "compiler" / "codegen"
    / "tessera_gpu_backend_NVIDIA" / "runtime" / "cuda"
    / "libtessera_nvidia_gemm.so"
    for build in ("build-nvidia-cuda", "build")
)

# Where the real driver + NVRTC live (WSL ships libcuda under /usr/lib/wsl/lib).
_CUDA_LIB_DIRS = [
    "/usr/lib/wsl/lib",
    os.environ.get("CUDA_PATH", "/usr/local/cuda") + "/lib64",
]


def _load_lib():
    gemm_lib = next((path for path in _GEMM_LIBS if path.is_file()), _GEMM_LIBS[0])
    if not gemm_lib.is_file():
        pytest.skip(
            "build the shipped GEMM lib: "
            "cmake -B build -DTESSERA_BUILD_NVIDIA_BACKEND=ON -DTESSERA_ENABLE_CUDA=ON "
            "&& ninja -C build tessera_nvidia_gemm "
            f"({gemm_lib} missing)")
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
    return ctypes.CDLL(str(gemm_lib), mode=ctypes.RTLD_GLOBAL)


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
        assert_matches(D, ref, dt, reduction_length=K)
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


def _pack_nvfp4(codes: np.ndarray, *, contraction_axis: int) -> np.ndarray:
    """Pack adjacent K nibbles, preserving every non-contraction dimension."""
    moved = np.moveaxis(np.asarray(codes, np.uint8), contraction_axis, -1)
    k = moved.shape[-1]
    packed = np.zeros(moved.shape[:-1] + ((k + 1) // 2,), np.uint8)
    packed[..., :] = moved[..., 0::2]
    packed[..., : k // 2] |= moved[..., 1::2] << 4
    return np.ascontiguousarray(np.moveaxis(packed, -1, contraction_axis))


def _decode_e2m1(codes: np.ndarray) -> np.ndarray:
    lut = np.asarray([0, .5, 1, 1.5, 2, 3, 4, 6], np.float32)
    return lut[codes & 7] * np.where(codes & 8, -1, 1).astype(np.float32)


def _decode_ue4m3(codes: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes, np.uint8)
    exponent = (codes >> 3) & 15
    mantissa = codes & 7
    normal = np.ldexp(1.0 + mantissa.astype(np.float32) / 8.0,
                      exponent.astype(np.int32) - 7)
    subnormal = np.ldexp(mantissa.astype(np.float32) / 8.0, -6)
    return np.where(exponent == 0, subnormal, normal).astype(np.float32)


@pytest.mark.parametrize("M,N,K", [(16, 8, 64), (33, 19, 129), (7, 5, 31)])
def test_shipped_nvfp4_general_shape_runtime(M, N, K):
    """Multi-tile/ragged ABI proof for packed data and non-uniform scales."""
    lib = _load_lib()
    fn = getattr(lib, "tessera_nvidia_mma_gemm_nvfp4")
    fn.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3
    fn.restype = ctypes.c_int
    rng = np.random.default_rng(120_400 + M + N + K)
    a_codes = rng.integers(0, 16, size=(M, K), dtype=np.uint8)
    b_codes = rng.integers(0, 16, size=(K, N), dtype=np.uint8)
    scale_codes = np.asarray([0x30, 0x38, 0x40], np.uint8)
    sk = (K + 15) // 16
    sa = scale_codes[(np.arange(M)[:, None] + np.arange(sk)[None, :]) % 3]
    sb = scale_codes[(2 * np.arange(sk)[:, None] + np.arange(N)[None, :]) % 3]
    ap = _pack_nvfp4(a_codes, contraction_axis=1)
    bp = _pack_nvfp4(b_codes, contraction_axis=0)
    out = np.zeros((M, N), np.float32)
    rc = fn(ap.ctypes.data_as(ctypes.c_void_p),
            bp.ctypes.data_as(ctypes.c_void_p),
            np.ascontiguousarray(sa).ctypes.data_as(ctypes.c_void_p),
            np.ascontiguousarray(sb).ctypes.data_as(ctypes.c_void_p),
            out.ctypes.data_as(ctypes.c_void_p), M, N, K)
    if rc == 2:
        pytest.skip("NVFP4 runtime requires sm_120a CUDA/NVRTC")
    assert rc == 0
    a = _decode_e2m1(a_codes) * np.repeat(_decode_ue4m3(sa), 16, axis=1)[:, :K]
    b = _decode_e2m1(b_codes) * np.repeat(_decode_ue4m3(sb), 16, axis=0)[:K, :]
    assert_matches(out, a @ b, "nvfp4", reduction_length=K)


def test_python_nvfp4_dispatch_rejects_malformed_scale_views_before_launch():
    from tessera import runtime as rt

    with pytest.raises(ValueError, match="scale_a"):
        rt._nvidia_nvfp4_gemm_2d(
            np.zeros((17, 33), np.uint8), np.zeros((33, 9), np.uint8),
            np.zeros((17, 4), np.uint8), np.zeros((5, 9), np.uint8),
            17, 9, 65)
