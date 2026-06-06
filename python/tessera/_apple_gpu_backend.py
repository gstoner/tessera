"""Apple GPU back-half for the production lane (Phase 3).

There is **no upstream MLIR Metal/AIR backend**, so Apple GPU does NOT go through
the `linalg → LLVM → ORC` path the CPU lane uses (Phases 0–2). Per the production
plan (D2 + the hard-problems register), Apple GPU is a **bespoke back-half**: the
`tessera` graph's structure is shared (and the CPU lane is the oracle), but
execution routes to hand-tuned Metal kernels (MPS / MSL / fused MSL).

This module is the clean dispatch surface. It reuses the existing
`tessera_apple_gpu_*` *kernel* C ABI (the kernels, not `runtime.py`'s op-by-op
dispatch logic) and is loaded via the shared on-the-fly compiler in
`runtime._load_apple_gpu_runtime`. Every kernel has a CPU reference fallback
inside the runtime, so results are correct even when MPS is unavailable.

The contract (D4 across targets): a GPU result must match the **compiled CPU
production lane** (`tessera._jit_boundary`), which matches numpy. f32 only for now.
"""

from __future__ import annotations

import ctypes
import sys
from typing import Any

import numpy as np


class AppleGpuError(RuntimeError):
    """Raised when the Apple GPU back-half is unavailable or misused."""


_LIB: Any = None


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    if not sys.platform.startswith("darwin"):
        raise AppleGpuError("Apple GPU back-half is Darwin-only")
    try:
        from tessera.runtime import _load_apple_gpu_runtime

        lib = _load_apple_gpu_runtime()
    except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
        raise AppleGpuError(f"could not load apple_gpu runtime: {exc}") from exc

    fp = ctypes.POINTER(ctypes.c_float)
    i32 = ctypes.c_int32
    for name, argtypes in (
        ("tessera_apple_gpu_mps_matmul_f32", [fp, fp, fp, i32, i32, i32]),
        ("tessera_apple_gpu_softmax_f32", [fp, fp, i32, i32]),
        ("tessera_apple_gpu_matmul_softmax_f32", [fp, fp, fp, i32, i32, i32]),
        ("tessera_apple_gpu_gelu_f32", [fp, fp, i32]),
    ):
        try:
            sym = getattr(lib, name)
        except AttributeError as exc:
            raise AppleGpuError(f"runtime missing symbol {name}") from exc
        sym.restype = None
        sym.argtypes = argtypes
    _LIB = lib
    return lib


def is_available() -> bool:
    """True when the Apple GPU runtime can be loaded (Darwin + kernels present)."""
    try:
        _load()
        return True
    except AppleGpuError:
        return False


def _f32(a: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.float32:
        raise AppleGpuError(f"{name} must be f32 (got {a.dtype})")
    return np.ascontiguousarray(a)


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ── Bespoke Metal kernels (the back-half) ────────────────────────────────────


def gpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C = A @ B on the Apple GPU (MPS), rank-2 f32."""
    a = _f32(a, "a")
    b = _f32(b, "b")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise AppleGpuError(f"gpu_matmul rank-2, K must match: {a.shape} @ {b.shape}")
    M, K = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    out = np.zeros((M, N), np.float32)
    _load().tessera_apple_gpu_mps_matmul_f32(_ptr(a), _ptr(b), _ptr(out), M, N, K)
    return out


def gpu_softmax(x: np.ndarray) -> np.ndarray:
    """Row-softmax over the last axis on the Apple GPU (MSL), rank-2 f32."""
    x = _f32(x, "x")
    if x.ndim != 2:
        raise AppleGpuError(f"gpu_softmax is rank-2 (last-axis) only: {x.shape}")
    M, K = int(x.shape[0]), int(x.shape[1])
    out = np.zeros((M, K), np.float32)
    _load().tessera_apple_gpu_softmax_f32(_ptr(x), _ptr(out), M, K)
    return out


def gpu_matmul_softmax(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fused O = softmax(A @ B) in a single Metal kernel (the D2 fused-chain
    target override), rank-2 f32."""
    a = _f32(a, "a")
    b = _f32(b, "b")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise AppleGpuError(f"gpu_matmul_softmax shapes: {a.shape} @ {b.shape}")
    M, K = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    out = np.zeros((M, N), np.float32)
    _load().tessera_apple_gpu_matmul_softmax_f32(_ptr(a), _ptr(b), _ptr(out), M, N, K)
    return out


def gpu_gelu(x: np.ndarray) -> np.ndarray:
    """GELU on the Apple GPU (MSL), f32. Flattened elementwise."""
    x = _f32(x, "x")
    flat = np.ascontiguousarray(x.reshape(-1))
    out = np.zeros_like(flat)
    _load().tessera_apple_gpu_gelu_f32(_ptr(flat), _ptr(out), int(flat.size))
    return out.reshape(x.shape)
