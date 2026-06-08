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

try:  # bf16 boundary dtype (RUNTIME_ABI_SPEC §12.5); soft dependency.
    import ml_dtypes as _ml_dtypes

    _BF16: Any = _ml_dtypes.bfloat16
except Exception:  # noqa: BLE001 - ml_dtypes optional
    _BF16 = None


class AppleGpuError(RuntimeError):
    """Raised when the Apple GPU back-half is unavailable or misused."""


_LIB: Any = None


def _load():
    global _LIB
    if _LIB is not None:
        return _LIB
    # NB: alias through ``str(...)`` so mypy does not statically evaluate the
    # platform check (mypy special-cases ``sys.platform`` comparisons *and*
    # ``.startswith``, which would mark the body below unreachable on non-Darwin
    # CI while it stays reachable locally on macOS — a baseline-breaking skew).
    platform = str(sys.platform)
    if not platform.startswith("darwin"):
        raise AppleGpuError("Apple GPU back-half is Darwin-only")
    try:
        from tessera.runtime import _load_apple_gpu_runtime

        lib = _load_apple_gpu_runtime()
    except Exception as exc:  # noqa: BLE001 - surface any load failure uniformly
        raise AppleGpuError(f"could not load apple_gpu runtime: {exc}") from exc

    fp = ctypes.POINTER(ctypes.c_float)
    u16 = ctypes.POINTER(ctypes.c_uint16)
    ip = ctypes.POINTER(ctypes.c_int32)
    i32 = ctypes.c_int32
    i64 = ctypes.c_int64
    flt = ctypes.c_float
    for name, argtypes in (
        ("tessera_apple_gpu_mps_matmul_f32", [fp, fp, fp, i32, i32, i32]),
        ("tessera_apple_gpu_softmax_f32", [fp, fp, i32, i32]),
        ("tessera_apple_gpu_matmul_softmax_f32", [fp, fp, fp, i32, i32, i32]),
        ("tessera_apple_gpu_gelu_f32", [fp, fp, i32]),
        # Sprint 3.2 — norms, activations, attention, fused-MLP chains.
        ("tessera_apple_gpu_rmsnorm_gpu_f32", [fp, fp, fp, i32, i32, flt]),
        ("tessera_apple_gpu_layer_norm_f32", [fp, fp, fp, fp, i32, i32, flt]),
        ("tessera_apple_gpu_mpsgraph_unary_f32", [i32, fp, fp, i64]),
        ("tessera_apple_gpu_mpsgraph_binary_f32", [i32, fp, fp, fp, i64]),
        ("tessera_apple_gpu_matmul_softmax_matmul_f32", [fp, fp, fp, fp, i32, i32, i32, i32]),
        ("tessera_apple_gpu_matmul_gelu_f32", [fp, fp, fp, i32, i32, i32]),
        ("tessera_apple_gpu_matmul_rmsnorm_f32", [fp, fp, fp, i32, i32, i32, flt]),
        # Sprint 3.3 follow-on — fused SwiGLU MLP block.
        ("tessera_apple_gpu_swiglu_f32", [fp, fp, fp, fp, fp, i32, i32, i32, i32]),
        # Sprint 3.3 perf-fusion — fused pre-norm + projection.
        ("tessera_apple_gpu_rmsnorm_matmul_f32", [fp, fp, fp, fp, i32, i32, i32, flt]),
        # Sprint 3.4 — native bf16 kernels (raw 16-bit boundary; f32 accumulate).
        ("tessera_apple_gpu_mps_matmul_bf16", [u16, u16, u16, i32, i32, i32]),
        ("tessera_apple_gpu_softmax_bf16", [u16, u16, i32, i32]),
        ("tessera_apple_gpu_gelu_bf16", [u16, u16, i32]),
        ("tessera_apple_gpu_matmul_softmax_bf16", [u16, u16, u16, i32, i32, i32]),
        ("tessera_apple_gpu_matmul_softmax_matmul_bf16", [u16, u16, u16, u16, i32, i32, i32, i32]),
        ("tessera_apple_gpu_matmul_gelu_bf16", [u16, u16, u16, i32, i32, i32]),
        ("tessera_apple_gpu_matmul_rmsnorm_bf16", [u16, u16, u16, i32, i32, i32, flt]),
        ("tessera_apple_gpu_swiglu_bf16", [u16, u16, u16, u16, u16, i32, i32, i32, i32]),
        # Sprint 3.5 — native bf16 MPSGraph unary/binary/norm kernels.
        ("tessera_apple_gpu_mpsgraph_unary_bf16", [i32, u16, u16, i64]),
        ("tessera_apple_gpu_mpsgraph_binary_bf16", [i32, u16, u16, u16, i64]),
        ("tessera_apple_gpu_rmsnorm_gpu_bf16", [u16, u16, u16, i32, i32, flt]),
        ("tessera_apple_gpu_layer_norm_bf16", [u16, u16, u16, u16, i32, i32, flt]),
        # Thrust #3a — fused ragged grouped-GEMM (X, W, expert_ids, O, T, K, N, E).
        ("tessera_apple_gpu_grouped_gemm_f32", [fp, fp, ip, fp, i32, i32, i32, i32]),
        # LDT candidate-axis ops on Metal.
        ("tessera_apple_gpu_popcount_i32", [ip, ip, i32]),
        ("tessera_apple_gpu_count_nonzero_lastaxis_f32", [fp, ip, i32, i32]),
        # MoE-aux / LDT loss ops (MPSGraph subgraphs → scalar).
        ("tessera_apple_gpu_z_loss_f32", [fp, fp, i32, i32]),
        ("tessera_apple_gpu_asymmetric_bce_f32", [fp, fp, fp, i32, flt, flt]),
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


def _is_bf16(a) -> bool:
    return _BF16 is not None and np.asarray(a).dtype == _BF16


def _arr(a, name: str) -> np.ndarray:
    """Accept an f32 OR bf16 (ml_dtypes) array; reject anything else. The
    dtype-polymorphic kernels branch on the result's dtype."""
    a = np.asarray(a)
    if a.dtype == np.float32 or _is_bf16(a):
        return np.ascontiguousarray(a)
    raise AppleGpuError(f"{name} must be f32 or bf16 (got {a.dtype})")


def _to_f32(a) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def _to_bf16(a) -> np.ndarray:
    if _BF16 is None:
        raise AppleGpuError("bf16 requires ml_dtypes")
    return np.ascontiguousarray(np.asarray(a).astype(_BF16))


def _ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))


# ── Bespoke Metal kernels (the back-half) ────────────────────────────────────


def gpu_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """C = A @ B on the Apple GPU (MPS), rank-2 f32 or bf16 (native bf16 kernel,
    f32 accumulate)."""
    a = _arr(a, "a")
    b = _arr(b, "b")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise AppleGpuError(f"gpu_matmul rank-2, K must match: {a.shape} @ {b.shape}")
    M, K = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    if _is_bf16(a) or _is_bf16(b):
        a, b = _to_bf16(a), _to_bf16(b)
        out = np.zeros((M, N), _BF16)
        _load().tessera_apple_gpu_mps_matmul_bf16(
            _u16ptr(a), _u16ptr(b), _u16ptr(out), M, N, K)
        return out
    out = np.zeros((M, N), np.float32)
    _load().tessera_apple_gpu_mps_matmul_f32(_ptr(a), _ptr(b), _ptr(out), M, N, K)
    return out


def gpu_grouped_gemm(x: np.ndarray, w: np.ndarray,
                     expert_ids: np.ndarray) -> np.ndarray:
    """Fused ragged grouped matmul on the Apple GPU — ONE dispatch over the whole
    (T, N) output. Row ``t`` of the result is ``x[t] @ w[expert_ids[t]]``.

    x: (T, K) f32; w: (E, K, N) f32; expert_ids: (T,) int — the per-token expert
    id (e.g. ``np.repeat(arange(E), group_sizes)``). Returns (T, N) f32.
    Replaces the per-group MPS-matmul loop with a single kernel that folds the
    routing in, removing the per-expert dispatch overhead."""
    x = _arr(x, "x").astype(np.float32, copy=False)
    w = np.ascontiguousarray(w, dtype=np.float32)
    e = np.ascontiguousarray(expert_ids, dtype=np.int32)
    if x.ndim != 2 or w.ndim != 3:
        raise AppleGpuError(
            f"gpu_grouped_gemm expects x (T,K) + w (E,K,N); got {x.shape}, {w.shape}")
    T, K = int(x.shape[0]), int(x.shape[1])
    E, wk, N = int(w.shape[0]), int(w.shape[1]), int(w.shape[2])
    if wk != K:
        raise AppleGpuError(f"gpu_grouped_gemm K mismatch: x K={K}, w K={wk}")
    if e.shape != (T,):
        raise AppleGpuError(f"gpu_grouped_gemm expert_ids must be (T={T},); got {e.shape}")
    out = np.zeros((T, N), np.float32)
    _load().tessera_apple_gpu_grouped_gemm_f32(
        _ptr(x), _ptr(w), e.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _ptr(out), T, K, N, E)
    return out


def _i32ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))


def gpu_popcount(x: np.ndarray) -> np.ndarray:
    """Per-element population count (set bits) of an integer tensor on the Apple
    GPU (MSL ``popcount`` intrinsic). Returns int32, same shape."""
    a = np.ascontiguousarray(x).astype(np.int32, copy=False)
    out = np.zeros(a.shape, np.int32)
    _load().tessera_apple_gpu_popcount_i32(_i32ptr(a), _i32ptr(out), int(a.size))
    return out


def gpu_count_nonzero_lastaxis(x: np.ndarray) -> np.ndarray:
    """Count of non-zero entries along the innermost axis on the Apple GPU.
    ``x`` is (..., axis_len) f32; returns int32 of shape ``x.shape[:-1]``."""
    a = np.ascontiguousarray(x).astype(np.float32, copy=False)
    if a.ndim < 1:
        raise AppleGpuError("gpu_count_nonzero_lastaxis needs a >=1-D tensor")
    axis_len = int(a.shape[-1])
    outer = int(a.size // max(1, axis_len))
    out = np.zeros((outer,), np.int32)
    _load().tessera_apple_gpu_count_nonzero_lastaxis_f32(
        _ptr(a), _i32ptr(out), outer, axis_len)
    return out.reshape(a.shape[:-1])


def gpu_z_loss(logits: np.ndarray) -> float:
    """Router z-loss on the Apple GPU (MPSGraph): mean over rows of
    ``logsumexp(row)²``. ``logits`` is (..., classes); returns a Python float."""
    a = np.ascontiguousarray(logits, dtype=np.float32)
    if a.ndim < 1:
        raise AppleGpuError("gpu_z_loss needs a >=1-D tensor")
    classes = int(a.shape[-1])
    rows = int(a.size // max(1, classes))
    out = np.zeros((1,), np.float32)
    _load().tessera_apple_gpu_z_loss_f32(
        _ptr(a.reshape(rows, classes)), _ptr(out), rows, classes)
    return float(out[0])


def gpu_asymmetric_bce(z: np.ndarray, t: np.ndarray, pos_w: float = 1.0,
                       neg_w: float = 1.0) -> float:
    """Asymmetric BCE-with-logits (mean) on the Apple GPU (MPSGraph):
    ``mean(pos·t·softplus(-z) + neg·(1-t)·softplus(z))``. Returns a float."""
    za = np.ascontiguousarray(z, dtype=np.float32)
    ta = np.ascontiguousarray(np.broadcast_to(np.asarray(t, np.float32), za.shape))
    za = za.ravel(); ta = ta.ravel()
    out = np.zeros((1,), np.float32)
    _load().tessera_apple_gpu_asymmetric_bce_f32(
        _ptr(za), _ptr(ta), _ptr(out), int(za.size), float(pos_w), float(neg_w))
    return float(out[0])


def gpu_softmax(x: np.ndarray) -> np.ndarray:
    """Row-softmax over the last axis on the Apple GPU (MSL), rank-2 f32/bf16."""
    x = _arr(x, "x")
    if x.ndim != 2:
        raise AppleGpuError(f"gpu_softmax is rank-2 (last-axis) only: {x.shape}")
    M, K = int(x.shape[0]), int(x.shape[1])
    if _is_bf16(x):
        out = np.zeros((M, K), _BF16)
        _load().tessera_apple_gpu_softmax_bf16(_u16ptr(x), _u16ptr(out), M, K)
        return out
    out = np.zeros((M, K), np.float32)
    _load().tessera_apple_gpu_softmax_f32(_ptr(x), _ptr(out), M, K)
    return out


def gpu_matmul_softmax(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fused O = softmax(A @ B) in a single Metal kernel (the D2 fused-chain
    target override), rank-2 f32/bf16."""
    a = _arr(a, "a")
    b = _arr(b, "b")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise AppleGpuError(f"gpu_matmul_softmax shapes: {a.shape} @ {b.shape}")
    M, K = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    if _is_bf16(a) or _is_bf16(b):
        a, b = _to_bf16(a), _to_bf16(b)
        out = np.zeros((M, N), _BF16)
        _load().tessera_apple_gpu_matmul_softmax_bf16(
            _u16ptr(a), _u16ptr(b), _u16ptr(out), M, N, K)
        return out
    out = np.zeros((M, N), np.float32)
    _load().tessera_apple_gpu_matmul_softmax_f32(_ptr(a), _ptr(b), _ptr(out), M, N, K)
    return out


def gpu_gelu(x: np.ndarray) -> np.ndarray:
    """GELU on the Apple GPU (MSL), f32/bf16. Flattened elementwise."""
    if _is_bf16(x):
        xb = _to_bf16(x)
        flat = np.ascontiguousarray(xb.reshape(-1))
        out = np.zeros_like(flat)
        _load().tessera_apple_gpu_gelu_bf16(_u16ptr(flat), _u16ptr(out), int(flat.size))
        return out.reshape(xb.shape)
    x = _f32(x, "x")
    flat = np.ascontiguousarray(x.reshape(-1))
    out = np.zeros_like(flat)
    _load().tessera_apple_gpu_gelu_f32(_ptr(flat), _ptr(out), int(flat.size))
    return out.reshape(x.shape)


# ── Sprint 3.2 — norms, activations, attention, fused-MLP chains ─────────────

_MPSG_SILU = 4  # tessera_apple_gpu_mpsgraph_unary opcode: silu = x*sigmoid(x)

# mpsgraph_binary opcodes (elementwise, used by the GraphFn GPU executor).
_BINARY_OPCODE = {"add": 0, "sub": 1, "mul": 2, "div": 3}


def gpu_binary(kind: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise add/sub/mul/div on the Apple GPU (MPSGraph), f32/bf16, equal
    shapes (native bf16 kernel; f32 internal accumulation)."""
    if kind not in _BINARY_OPCODE:
        raise AppleGpuError(f"gpu_binary kind must be one of {sorted(_BINARY_OPCODE)}")
    if _is_bf16(a) or _is_bf16(b):
        a, b = _to_bf16(a), _to_bf16(b)
        if a.shape != b.shape:
            raise AppleGpuError(f"gpu_binary needs equal shapes: {a.shape} vs {b.shape}")
        fa = np.ascontiguousarray(a.reshape(-1))
        fb = np.ascontiguousarray(b.reshape(-1))
        out = np.zeros_like(fa)
        _load().tessera_apple_gpu_mpsgraph_binary_bf16(
            _BINARY_OPCODE[kind], _u16ptr(fa), _u16ptr(fb), _u16ptr(out), int(fa.size))
        return out.reshape(a.shape)
    a = _f32(a, "a")
    b = _f32(b, "b")
    if a.shape != b.shape:
        raise AppleGpuError(f"gpu_binary needs equal shapes: {a.shape} vs {b.shape}")
    fa = np.ascontiguousarray(a.reshape(-1))
    fb = np.ascontiguousarray(b.reshape(-1))
    out = np.zeros_like(fa)
    _load().tessera_apple_gpu_mpsgraph_binary_f32(
        _BINARY_OPCODE[kind], _ptr(fa), _ptr(fb), _ptr(out), int(fa.size))
    return out.reshape(a.shape)


def gpu_rmsnorm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Unweighted RMSNorm over the last axis on the Apple GPU, rank-2 f32.

    Calls the weighted GPU kernel with gamma=1 so it matches the CPU lane's
    unweighted ``jit_rmsnorm`` oracle (``x / sqrt(mean(x²) + eps)``). f32/bf16
    (native bf16 kernel; f32 internal accumulation).
    """
    if _is_bf16(x):
        x = _to_bf16(x)
        if x.ndim != 2:
            raise AppleGpuError(f"gpu_rmsnorm is rank-2 (last-axis) only: {x.shape}")
        rows, cols = int(x.shape[0]), int(x.shape[1])
        gamma = np.ones((cols,), _BF16)
        out = np.zeros((rows, cols), _BF16)
        _load().tessera_apple_gpu_rmsnorm_gpu_bf16(
            _u16ptr(x), _u16ptr(gamma), _u16ptr(out), rows, cols, float(eps))
        return out
    x = _f32(x, "x")
    if x.ndim != 2:
        raise AppleGpuError(f"gpu_rmsnorm is rank-2 (last-axis) only: {x.shape}")
    rows, cols = int(x.shape[0]), int(x.shape[1])
    gamma = np.ones((cols,), np.float32)
    out = np.zeros((rows, cols), np.float32)
    _load().tessera_apple_gpu_rmsnorm_gpu_f32(
        _ptr(x), _ptr(gamma), _ptr(out), rows, cols, float(eps))
    return out


def gpu_layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Unweighted LayerNorm over the last axis (gamma=1, beta=0), rank-2 f32.

    Matches the CPU lane's unweighted ``jit_layer_norm``
    (``(x - mean) / sqrt(var + eps)``). f32/bf16 (native bf16 kernel).
    """
    if _is_bf16(x):
        x = _to_bf16(x)
        if x.ndim != 2:
            raise AppleGpuError(f"gpu_layer_norm is rank-2 (last-axis) only: {x.shape}")
        rows, cols = int(x.shape[0]), int(x.shape[1])
        gamma = np.ones((cols,), _BF16)
        beta = np.zeros((cols,), _BF16)
        out = np.zeros((rows, cols), _BF16)
        _load().tessera_apple_gpu_layer_norm_bf16(
            _u16ptr(x), _u16ptr(gamma), _u16ptr(beta), _u16ptr(out), rows, cols, float(eps))
        return out
    x = _f32(x, "x")
    if x.ndim != 2:
        raise AppleGpuError(f"gpu_layer_norm is rank-2 (last-axis) only: {x.shape}")
    rows, cols = int(x.shape[0]), int(x.shape[1])
    gamma = np.ones((cols,), np.float32)
    beta = np.zeros((cols,), np.float32)
    out = np.zeros((rows, cols), np.float32)
    _load().tessera_apple_gpu_layer_norm_f32(
        _ptr(x), _ptr(gamma), _ptr(beta), _ptr(out), rows, cols, float(eps))
    return out


# mpsgraph_unary opcodes (elementwise activations, used by the GraphFn executor).
_UNARY_OPCODE = {"relu": 0, "sigmoid": 1, "tanh": 2, "silu": 4}


def gpu_unary(kind: str, x: np.ndarray) -> np.ndarray:
    """Elementwise activation on the Apple GPU (MPSGraph), f32. Flattened.

    `kind` ∈ {relu, sigmoid, tanh, silu}. (gelu has its own dedicated MSL kernel
    — use `gpu_gelu`.) f32/bf16 (native bf16 kernel; f32 internal accumulation).
    """
    if kind not in _UNARY_OPCODE:
        raise AppleGpuError(f"gpu_unary kind must be one of {sorted(_UNARY_OPCODE)}")
    if _is_bf16(x):
        xb = _to_bf16(x)
        flat = np.ascontiguousarray(xb.reshape(-1))
        out = np.zeros_like(flat)
        _load().tessera_apple_gpu_mpsgraph_unary_bf16(
            _UNARY_OPCODE[kind], _u16ptr(flat), _u16ptr(out), int(flat.size))
        return out.reshape(xb.shape)
    x = _f32(x, "x")
    flat = np.ascontiguousarray(x.reshape(-1))
    out = np.zeros_like(flat)
    _load().tessera_apple_gpu_mpsgraph_unary_f32(
        _UNARY_OPCODE[kind], _ptr(flat), _ptr(out), int(flat.size))
    return out.reshape(x.shape)


def gpu_silu(x: np.ndarray) -> np.ndarray:
    """SiLU/swish = x*sigmoid(x) on the Apple GPU (MPSGraph), f32. Flattened."""
    return gpu_unary("silu", x)


def gpu_attention(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Fused single-head attention block ``O = softmax(A @ B) @ C`` in ONE Metal
    kernel (the D2 fused-chain target override).

    Matches the CPU lane's un-fused matmul→softmax→matmul composition. No scale
    is applied — pre-scale ``A`` by ``1/sqrt(d)`` if you want scaled-dot-product.

    ``A:(M,K)  B:(K,N)  C:(N,P)  ->  O:(M,P)``. f32/bf16 (native bf16 kernel).
    """
    a = _arr(a, "a")
    b = _arr(b, "b")
    c = _arr(c, "c")
    if a.ndim != 2 or b.ndim != 2 or c.ndim != 2:
        raise AppleGpuError("gpu_attention is rank-2 only")
    M, K = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    P = int(c.shape[1])
    if int(b.shape[0]) != K or int(c.shape[0]) != N:
        raise AppleGpuError(
            f"gpu_attention shape mismatch: {a.shape} @ {b.shape} @ {c.shape}")
    if _is_bf16(a) or _is_bf16(b) or _is_bf16(c):
        a, b, c = _to_bf16(a), _to_bf16(b), _to_bf16(c)
        out = np.zeros((M, P), _BF16)
        _load().tessera_apple_gpu_matmul_softmax_matmul_bf16(
            _u16ptr(a), _u16ptr(b), _u16ptr(c), _u16ptr(out), M, K, N, P)
        return out
    out = np.zeros((M, P), np.float32)
    _load().tessera_apple_gpu_matmul_softmax_matmul_f32(
        _ptr(a), _ptr(b), _ptr(c), _ptr(out), M, K, N, P)
    return out


def gpu_matmul_gelu(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Fused ``O = gelu(A @ B)`` in one Metal kernel (tanh-approx gelu). f32/bf16
    (native bf16 kernel)."""
    a = _arr(a, "a")
    b = _arr(b, "b")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise AppleGpuError(f"gpu_matmul_gelu shapes: {a.shape} @ {b.shape}")
    M, K = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    if _is_bf16(a) or _is_bf16(b):
        a, b = _to_bf16(a), _to_bf16(b)
        out = np.zeros((M, N), _BF16)
        _load().tessera_apple_gpu_matmul_gelu_bf16(
            _u16ptr(a), _u16ptr(b), _u16ptr(out), M, N, K)
        return out
    out = np.zeros((M, N), np.float32)
    _load().tessera_apple_gpu_matmul_gelu_f32(_ptr(a), _ptr(b), _ptr(out), M, N, K)
    return out


def gpu_matmul_rmsnorm(a: np.ndarray, b: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Fused ``O = rmsnorm(A @ B)`` (unweighted, last-axis) in one Metal kernel.
    f32/bf16 (native bf16 kernel)."""
    a = _arr(a, "a")
    b = _arr(b, "b")
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise AppleGpuError(f"gpu_matmul_rmsnorm shapes: {a.shape} @ {b.shape}")
    M, K = int(a.shape[0]), int(a.shape[1])
    N = int(b.shape[1])
    if _is_bf16(a) or _is_bf16(b):
        a, b = _to_bf16(a), _to_bf16(b)
        out = np.zeros((M, N), _BF16)
        _load().tessera_apple_gpu_matmul_rmsnorm_bf16(
            _u16ptr(a), _u16ptr(b), _u16ptr(out), M, N, K, float(eps))
        return out
    out = np.zeros((M, N), np.float32)
    _load().tessera_apple_gpu_matmul_rmsnorm_f32(
        _ptr(a), _ptr(b), _ptr(out), M, N, K, float(eps))
    return out


def gpu_rmsnorm_matmul(x: np.ndarray, w: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Fused pre-norm + projection ``O = rmsnorm(X) @ W`` (unweighted) in ONE
    Metal/MPSGraph dispatch — the hottest chain in a pre-norm transformer.

    Calls the weighted kernel with gamma=1 so it matches the CPU lane's
    ``matmul(rmsnorm(x), W)`` composition. ``X:(M,K)  W:(K,N)  ->  O:(M,N)``.
    f32/bf16 (bf16 via f32 compute + round — no native bf16 variant).
    """
    if _is_bf16(x) or _is_bf16(w):
        return _to_bf16(gpu_rmsnorm_matmul(_to_f32(x), _to_f32(w), eps=eps))
    x = _f32(x, "x")
    w = _f32(w, "w")
    if x.ndim != 2 or w.ndim != 2 or x.shape[1] != w.shape[0]:
        raise AppleGpuError(f"gpu_rmsnorm_matmul shapes: rmsnorm{x.shape} @ {w.shape}")
    M, K = int(x.shape[0]), int(x.shape[1])
    N = int(w.shape[1])
    gamma = np.ones((K,), np.float32)
    out = np.zeros((M, N), np.float32)
    _load().tessera_apple_gpu_rmsnorm_matmul_f32(
        _ptr(x), _ptr(gamma), _ptr(w), _ptr(out), M, K, N, float(eps))
    return out


# ── Metal-4 resident-weight MLP decode session ───────────────────────────────

_ACT_CODE = {"none": 0, "relu": 1, "gelu": 2, "silu": 3}


def _to_half_uint16(a: np.ndarray, bf16: bool) -> np.ndarray:
    """Pack f32 → raw 16-bit halves (the session's X/W boundary dtype). f16 is a
    native cast; bf16 is the upper 16 bits of the f32 pattern (truncate, matching
    the runtime's bf16 convention)."""
    a = np.ascontiguousarray(np.asarray(a, dtype=np.float32))
    if bf16:
        return (a.view(np.uint32) >> 16).astype(np.uint16)
    return a.astype(np.float16).view(np.uint16)


def _u16ptr(a: np.ndarray):
    return a.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16))


class Mtl4MlpSession:
    """Metal-4 resident-weight fused MLP decode session: ``Y = act(X @ W + bias)``
    with ``W[K,N]`` uploaded once and kept resident across runs (the per-call cost
    is just the dispatch — the decode-loop win). ``X`` is f16/bf16, ``Y`` is f32,
    ``act`` ∈ {none, relu, gelu, silu}. Requires macOS 26+ (the MTL4 matrix-unit
    lane); construction raises :class:`AppleGpuError` otherwise."""

    def __init__(self, w, bias=None, act: str = "none", bf16: bool = False):
        if act not in _ACT_CODE:
            raise AppleGpuError(f"act must be one of {sorted(_ACT_CODE)}")
        lib = _load()
        w = np.ascontiguousarray(np.asarray(w, dtype=np.float32))
        if w.ndim != 2:
            raise AppleGpuError("W must be rank-2 [K, N]")
        self._K, self._N = int(w.shape[0]), int(w.shape[1])
        self._bf16 = bool(bf16)
        self._wu = _to_half_uint16(w, self._bf16)  # keep alive across create
        try:
            create = lib.tessera_apple_gpu_mtl4_mlp_session_create
            self._run_fn = lib.tessera_apple_gpu_mtl4_mlp_session_run
            self._destroy_fn = lib.tessera_apple_gpu_mtl4_mlp_session_destroy
        except AttributeError as exc:
            raise AppleGpuError(f"runtime missing MTL4 MLP session symbol: {exc}")
        create.restype = ctypes.c_void_p
        create.argtypes = [ctypes.POINTER(ctypes.c_uint16),
                           ctypes.POINTER(ctypes.c_float), ctypes.c_int32,
                           ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
        self._run_fn.restype = ctypes.c_int32
        self._run_fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint16),
                                ctypes.POINTER(ctypes.c_float), ctypes.c_int32]
        self._destroy_fn.restype = None
        self._destroy_fn.argtypes = [ctypes.c_void_p]
        bptr = None
        self._bias = None
        if bias is not None:
            b = np.ascontiguousarray(np.asarray(bias, dtype=np.float32))
            if b.shape != (self._N,):
                raise AppleGpuError(f"bias must be shape ({self._N},)")
            self._bias = b
            bptr = b.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        h = create(_u16ptr(self._wu), bptr, _ACT_CODE[act],
                   self._K, self._N, int(self._bf16))
        if not h:
            raise AppleGpuError(
                "MTL4 MLP session unavailable (needs macOS 26+ / MTL4 ML lane)")
        self._h: Any = ctypes.c_void_p(h)

    @property
    def shape(self) -> tuple:
        return (self._K, self._N)

    def run(self, x: np.ndarray) -> np.ndarray:
        """One decode step: ``Y[M,N] = act(X[M,K] @ W + bias)``. X is cast to the
        session dtype (f16/bf16); Y is returned f32."""
        if self._h is None:
            raise AppleGpuError("session is closed")
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
        if x.ndim != 2 or int(x.shape[1]) != self._K:
            raise AppleGpuError(f"X must be [M, {self._K}], got {x.shape}")
        M = int(x.shape[0])
        xu = _to_half_uint16(x, self._bf16)
        y = np.zeros((M, self._N), np.float32)
        if not self._run_fn(self._h, _u16ptr(xu), _ptr(y), M):
            raise AppleGpuError("MTL4 MLP session run failed")
        return y

    def close(self) -> None:
        if getattr(self, "_h", None) is not None:
            self._destroy_fn(self._h)
            self._h = None

    def __enter__(self) -> "Mtl4MlpSession":
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def mtl4_mlp_available() -> bool:
    """True when a Metal-4 resident-weight MLP session can be created (Darwin +
    macOS 26+ + MTL4 ML lane). Probes by building a tiny session."""
    try:
        Mtl4MlpSession(np.ones((2, 2), np.float32)).close()
        return True
    except AppleGpuError:
        return False


def gpu_swiglu(
    x: np.ndarray, wg: np.ndarray, wu: np.ndarray, wd: np.ndarray
) -> np.ndarray:
    """Fused SwiGLU MLP block ``O = (silu(X @ Wg) ⊙ (X @ Wu)) @ Wd`` in ONE Metal
    kernel (gate/up projections + silu + elementwise gate + down projection).

    ``X:(M,K)  Wg,Wu:(K,H)  Wd:(H,Kout)  ->  O:(M,Kout)``. Matches the CPU lane's
    `(silu(matmul(x,wg)) * matmul(x,wu)) @ wd` composition. f32/bf16 (native bf16
    kernel).
    """
    x = _arr(x, "x")
    wg = _arr(wg, "wg")
    wu = _arr(wu, "wu")
    wd = _arr(wd, "wd")
    if x.ndim != 2 or wg.ndim != 2 or wu.ndim != 2 or wd.ndim != 2:
        raise AppleGpuError("gpu_swiglu operands are all rank-2")
    M, K = int(x.shape[0]), int(x.shape[1])
    H = int(wg.shape[1])
    Kout = int(wd.shape[1])
    if wg.shape != (K, H) or wu.shape != (K, H) or wd.shape != (H, Kout):
        raise AppleGpuError(
            f"gpu_swiglu shape mismatch: X{x.shape} Wg{wg.shape} "
            f"Wu{wu.shape} Wd{wd.shape}")
    if any(_is_bf16(t) for t in (x, wg, wu, wd)):
        x, wg, wu, wd = (_to_bf16(x), _to_bf16(wg), _to_bf16(wu), _to_bf16(wd))
        out = np.zeros((M, Kout), _BF16)
        _load().tessera_apple_gpu_swiglu_bf16(
            _u16ptr(x), _u16ptr(wg), _u16ptr(wu), _u16ptr(wd), _u16ptr(out),
            M, K, H, Kout)
        return out
    out = np.zeros((M, Kout), np.float32)
    _load().tessera_apple_gpu_swiglu_f32(
        _ptr(x), _ptr(wg), _ptr(wu), _ptr(wd), _ptr(out), M, K, H, Kout)
    return out
