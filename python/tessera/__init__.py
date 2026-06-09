"""
Tessera: Next-Generation Deep Learning Programming Model

A revolutionary deep learning framework that treats numerical precision,
data movement, parallelism, and correctness as first-class semantic objects.

Phase 1 exports (Python frontend + Graph IR):
    tessera.domain.Rect                  — logical iteration space
    tessera.dist.Block / Cyclic / Replicated — placement strategies
    tessera.array.from_domain(...)       — create a DistributedArray
    tessera.Region["read"/"write"/...]   — region privilege annotations
    tessera.index_launch(axis=...)       — fan kernel over mesh partitions
    tessera.kernel                       — tile kernel decorator
    tessera.jit                          — JIT decorator (constraint + effect + Graph IR)
    tessera.require(constraint)          — register a structural constraint
    tessera.constraint.Divisible/Range/Equal — constraint predicates
    tessera.ops.*                        — op name namespace (for IR emission)
"""

__version__ = "0.1.0"
__author__ = "Tessera Team"

import math
import types
import builtins
from dataclasses import dataclass, field
import typing
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Legacy core (Tensor, Module, NumericalPolicy)
# ─────────────────────────────────────────────────────────────────────────────
from . import core
from . import arch
from . import shape
from . import debug
from . import dtype  # Canonical dtype + alias normalization — Sprint A0
from . import telemetry
from . import profiler
from . import collectives
from . import autotune as _autotune_module
from . import fault
from . import elastic
from . import checkpoint
from . import server
# Apple-GPU encode-session surface (single-cb decode chain — see
# docs/audit/backend/apple/APPLE_AUDIT.md). Importing
# unconditionally; the module degrades gracefully off-Darwin (the
# session_available() check returns False).
from . import apple_gpu_ops  # noqa: F401
from .core import Tensor, Module
from .shape import (
    Dim,
    Layout,
    RuntimeShapeWitness,
    Shape,
    ShapeConstraintGraph,
    ShapeShard,
    ShapeSystemError,
    broadcast_shape,
    check_schedule_tile,
    check_shapes,
    check_shard,
    dim,
    matmul_shape,
    reshape_shape,
    sym,
)

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: distributed API
# ─────────────────────────────────────────────────────────────────────────────
from . import distributed

from .distributed.region import Region
from .distributed.domain import Rect, Block, Cyclic, Replicated
from .distributed.array import DistributedArray
from .distributed.launch import index_launch, kernel
from .distributed.shard import ShardSpec, MeshSpec

# Namespace objects for the tessera.domain.X, tessera.dist.X, tessera.array.X API
domain = types.SimpleNamespace(
    Rect=Rect,
)

dist = types.SimpleNamespace(
    Block=Block,
    Cyclic=Cyclic,
    Replicated=Replicated,
    config=lambda **kwargs: kwargs,
    elastic=elastic.elastic,
    reshard=elastic.reshard,
    current_mesh=elastic.current_mesh,
    world_size=elastic.world_size,
)

array = types.SimpleNamespace(
    from_domain=DistributedArray.from_domain,
)

graph = types.SimpleNamespace(
    trace=debug.trace_graph,
    debug_trace=debug.debug_trace,
    debug_value=debug.debug_value,
    export_graphviz=debug.export_graphviz,
    replay_capture=debug.replay_capture,
)

# ``autotune`` is a function used as a namespace for its companion
# helpers (``.load`` / ``.cache_key`` / ``.schedule_artifact`` /
# ``.RooflineCostModel``).  mypy treats assignments to function
# attributes as ``[attr-defined]`` errors; type-ignore the
# deliberately-dynamic namespace pattern.
autotune = _autotune_module.autotune
autotune.load = _autotune_module.load                     # type: ignore[attr-defined]
autotune.cache_key = _autotune_module.cache_key           # type: ignore[attr-defined]
autotune.schedule_artifact = _autotune_module.schedule_artifact  # type: ignore[attr-defined]
autotune.RooflineCostModel = _autotune_module.RooflineCostModel  # type: ignore[attr-defined]

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: compiler API
# ─────────────────────────────────────────────────────────────────────────────
from . import compiler

from .compiler.jit import jit, require, TesseraJitError
from .compiler.from_text import from_text
from .compiler.constraints import (
    ConstraintSolver,
    Divisible,
    Range,
    Equal,
    TesseraConstraintError,
)
from .compiler.effects import Effect, EffectLattice, TesseraEffectError

# tessera.constraint.Divisible etc.
constraint = types.SimpleNamespace(
    Divisible=Divisible,
    Range=Range,
    Equal=Equal,
    ConstraintSolver=ConstraintSolver,
)

# ─────────────────────────────────────────────────────────────────────────────
# tessera.ops — op name namespace
#
# In Phase 1 these are plain functions that call through to numpy or return
# a sentinel. The Graph IR builder recognises calls to these names and emits
# the corresponding tessera.* op.
#
# In Phase 3 these will dispatch to compiled MLIR kernels.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _OperatorEntry:
    name: str
    reference: Callable[..., Any] | None = None
    lowering: Callable[..., Any] | None = None
    runtime_kernel: Callable[..., Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class _OperatorRegistry:
    def __init__(self) -> None:
        self._entries: dict[str, _OperatorEntry] = {}

    def register_reference(self, name: str, fn: Callable[..., Any], **metadata: Any) -> _OperatorEntry:
        entry = self._entries.setdefault(name, _OperatorEntry(name=name))
        entry.reference = fn
        entry.metadata.update(metadata)
        return entry

    def register_lowering(self, name: str, fn: Callable[..., Any], **metadata: Any) -> _OperatorEntry:
        entry = self._entries.setdefault(name, _OperatorEntry(name=name))
        entry.lowering = fn
        entry.metadata.update(metadata)
        return entry

    def register_runtime_kernel(self, name: str, fn: Callable[..., Any], **metadata: Any) -> _OperatorEntry:
        entry = self._entries.setdefault(name, _OperatorEntry(name=name))
        entry.runtime_kernel = fn
        entry.metadata.update(metadata)
        return entry

    def get(self, name: str) -> _OperatorEntry | None:
        return self._entries.get(name)

    def list(self) -> list[str]:
        return sorted(self._entries)

    def dispatch(self, name: str, *args: Any, prefer_runtime: bool = True, **kwargs: Any) -> Any:
        entry = self._entries.get(name)
        if entry is None:
            raise KeyError(f"unknown tessera op: {name}")
        if prefer_runtime and entry.runtime_kernel is not None:
            return entry.runtime_kernel(*args, **kwargs)
        if not prefer_runtime and entry.reference is not None:
            return entry.reference(*args, **kwargs)
        if entry.lowering is not None:
            return entry.lowering(*args, **kwargs)
        if entry.reference is not None:
            return entry.reference(*args, **kwargs)
        raise NotImplementedError(f"tessera op {name!r} has no registered implementation")


_ops_registry = _OperatorRegistry()


def _register_reference(name: str, fn: Callable[..., Any], **metadata: Any) -> _OperatorEntry:
    return _ops_registry.register_reference(name, fn, **metadata)


def _register_lowering(name: str, fn: Callable[..., Any], **metadata: Any) -> _OperatorEntry:
    return _ops_registry.register_lowering(name, fn, **metadata)


def _register_runtime_kernel(name: str, fn: Callable[..., Any], **metadata: Any) -> _OperatorEntry:
    return _ops_registry.register_runtime_kernel(name, fn, **metadata)

def _make_ops_namespace() -> types.SimpleNamespace:
    """Build the tessera.ops namespace with Phase 1 numpy-backed stubs."""
    import numpy as np

    def gemm(A, B, epilogue=None):
        """Matrix multiply A @ B."""
        if hasattr(A, "_data"):
            A = A._data
        if hasattr(B, "_data"):
            B = B._data
        out = np.matmul(A, B)
        if epilogue:
            out = fused_epilogue(out, **epilogue)
        return out

    def matmul(A, B, epilogue=None):
        return gemm(A, B, epilogue=epilogue)

    def batched_gemm(A, B, epilogue=None):
        return gemm(A, B, epilogue=epilogue)

    def einsum(spec: str, *tensors):
        tensors = tuple(t._data if hasattr(t, "_data") else t for t in tensors)
        return np.einsum(spec, *tensors)

    def factorized_matmul(A, B, rank: int):
        out = gemm(A, B)
        u, s, vh = np.linalg.svd(out, full_matrices=False)
        r = max(1, min(int(rank), s.shape[-1]))
        return (u[..., :r] * s[..., :r]) @ vh[..., :r, :]

    def grouped_gemm(x, weights, group_sizes):
        """Ragged grouped matmul — the MoE expert-FFN compute core.

        ``x`` shape ``(T, K)`` holds tokens **sorted by expert assignment**;
        ``weights`` shape ``(E, K, N)`` holds one weight matrix per expert;
        ``group_sizes`` shape ``(E,)`` (summing to ``T``) gives the token count
        per expert. Row block ``e`` of the output is ``x[block e] @ weights[e]``,
        i.e. each contiguous group is multiplied by *its own* expert weight.
        Returns ``(T, N)``. Distinct from ``batched_gemm`` (equal-size batches):
        groups are ragged, so the per-expert blocks vary in length.
        """
        xa = np.asarray(x._data if hasattr(x, "_data") else x)
        w = np.asarray(weights._data if hasattr(weights, "_data") else weights)
        gs = np.asarray(group_sizes).astype(np.int64).reshape(-1)
        if xa.ndim != 2:
            raise ValueError(f"grouped_gemm: x must be (T, K); got {xa.shape}")
        if w.ndim != 3 or w.shape[0] != gs.shape[0]:
            raise ValueError(
                f"grouped_gemm: weights {w.shape} must be (E, K, N) with "
                f"E == len(group_sizes)={gs.shape[0]}")
        if w.shape[1] != xa.shape[1]:
            raise ValueError(
                f"grouped_gemm: K mismatch — x K={xa.shape[1]}, weights K={w.shape[1]}")
        T = xa.shape[0]
        if int(gs.sum()) != T:
            raise ValueError(
                f"grouped_gemm: group_sizes sum {int(gs.sum())} != T={T}")
        out = np.zeros((T, w.shape[2]), dtype=xa.dtype)
        off = 0
        for e in range(w.shape[0]):
            n = int(gs[e])
            if n:
                out[off:off + n] = xa[off:off + n] @ w[e]
            off += n
        return out

    def tri_solve(A, b, lower: bool = True):
        if hasattr(A, "_data"):
            A = A._data
        if hasattr(b, "_data"):
            b = b._data
        tri = np.tril(A) if lower else np.triu(A)
        return np.linalg.solve(tri, b)

    def cholesky(A):
        if hasattr(A, "_data"):
            A = A._data
        return np.linalg.cholesky(A)

    def qr(A):
        if hasattr(A, "_data"):
            A = A._data
        return np.linalg.qr(A)

    def svd(A):
        if hasattr(A, "_data"):
            A = A._data
        return np.linalg.svd(A, full_matrices=False)

    def cholesky_solve(L, b):
        # L is the lower Cholesky factor of an SPD matrix; solve (L Lᵀ) x = b.
        if hasattr(L, "_data"):
            L = L._data
        if hasattr(b, "_data"):
            b = b._data
        L = np.asarray(L)
        y = np.linalg.solve(np.tril(L), b)
        return np.linalg.solve(np.swapaxes(np.tril(L), -1, -2), y)

    def lu(A):
        # LAPACK getrf-style factorization: returns (packed_lu, pivots).
        if hasattr(A, "_data"):
            A = A._data
        from scipy.linalg import lu_factor
        lu_packed, piv = lu_factor(np.asarray(A))
        return lu_packed, piv

    def layer_norm(x, eps: float = 1e-5):
        if hasattr(x, "_data"):
            x = x._data
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def _rmsnorm(x, eps: float):
        if hasattr(x, "_data"):
            x = x._data
        return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)

    def rmsnorm(x, eps: float = 1e-5):
        return _rmsnorm(x, eps=eps)

    def rmsnorm_safe(x, eps: float = 1e-6):
        return _rmsnorm(x, eps=eps)

    def softmax(x, axis: int = -1):
        if hasattr(x, "_data"):
            x = x._data
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    def softmax_safe(x, axis: int = -1):
        return softmax(x, axis=axis)

    def reduce(x, op: str = "sum", axis=None, keepdims: bool = False):
        if hasattr(x, "_data"):
            x = x._data
        if op != "sum":
            raise ValueError("only reduce op='sum' is implemented in the CPU reference path")
        return np.sum(x, axis=axis, keepdims=keepdims)

    def sum(x, axis=None, keepdims: bool = False):
        return reduce(x, op="sum", axis=axis, keepdims=keepdims)

    def sigmoid(x):
        if hasattr(x, "_data"):
            x = x._data
        return 1.0 / (1.0 + np.exp(-x))

    def gelu(x):
        if hasattr(x, "_data"):
            x = x._data
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def tanh(x):
        if hasattr(x, "_data"):
            x = x._data
        return np.tanh(x)

    def add(x, y=None, *, scalar=None):
        if hasattr(x, "_data"):
            x = x._data
        if y is not None and hasattr(y, "_data"):
            y = y._data
        rhs = scalar if y is None else y
        return np.asarray(x) + rhs

    def mul(x, y=None, *, scalar=None):
        if hasattr(x, "_data"):
            x = x._data
        if y is not None and hasattr(y, "_data"):
            y = y._data
        rhs = scalar if y is None else y
        return np.asarray(x) * rhs

    def relu(x):
        if hasattr(x, "_data"):
            x = x._data
        return np.maximum(0, x)

    def silu(x):
        if hasattr(x, "_data"):
            x = x._data
        return x / (1.0 + np.exp(-x))

    def silu_mul(a, b):
        """Fused SiLU-and-multiply primitive: silu(a) * b.

        Single op (rather than `mul(silu(a), b)`) so the Schedule IR fusion
        recognizer can match the SwiGLU 3-op chain `matmul → silu_mul → matmul`
        and emit a fused `tessera.swiglu_fused` op for backends that ship a
        fused MLP-block kernel (Apple GPU MSL, NVIDIA WGMMA epilogue, etc.).
        """
        if hasattr(a, "_data"):
            a = a._data
        if hasattr(b, "_data"):
            b = b._data
        a = np.asarray(a)
        b = np.asarray(b)
        s = a / (1.0 + np.exp(-a))
        return s * b

    def swiglu(x, W_gate, W_up, W_down):
        """SwiGLU MLP block decomposed as `gemm → gemm → silu_mul → gemm`.

        Calls into the wrapped ops (`ops.gemm`, `ops.silu_mul`) so an active
        autodiff tape records the four primitives. The Schedule IR fusion
        recognizer matches this chain and emits `tessera.swiglu_fused` for
        backends with a fused MLP-block kernel (Apple GPU MSL, NVIDIA WGMMA
        epilogue, etc.).
        """
        # Resolve via the namespace at call time so the wrapped versions
        # (installed by autodiff.install_op_wrappers) are picked up.
        gate = ops.gemm(x, W_gate)
        up = ops.gemm(x, W_up)
        hidden = ops.silu_mul(gate, up)
        return ops.gemm(hidden, W_down)

    # ── Theme 9 utility tensor ops (Tier 3 #8) ──────────────────────────────
    # Small numpy-reference primitives that several advanced examples want
    # for shape munging / masking / range generation. Each ships with a VJP
    # in `python/tessera/autodiff/vjp.py` so they compose with the tape.

    def arange(start, stop=None, step=1, dtype="fp32"):
        """`numpy.arange` over a ``[start, stop)`` range with the given step.

        Single-arg form (``arange(stop)``) starts at 0, mirroring numpy. The
        result is a 1-D array. Non-differentiable — produces no `.grad`.
        """
        if stop is None:
            stop = start
            start = 0
        np_dtype = {"fp16": np.float16, "fp32": np.float32, "fp64": np.float64,
                    "i32": np.int32, "i64": np.int64}.get(dtype, np.float32)
        return np.arange(start, stop, step, dtype=np_dtype)

    def gather(x, indices, *, axis=0):
        """Gather slices of ``x`` along ``axis`` per integer ``indices``.

        Mirrors ``numpy.take(x, indices, axis=axis)``. ``indices`` may be any
        int dtype. Differentiable through ``x``; ``indices`` is treated as a
        non-tensor argument by the tape.
        """
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(indices, "_data"):
            indices = indices._data
        x = np.asarray(x)
        idx = np.asarray(indices, dtype=np.int64)
        return np.take(x, idx, axis=axis)

    def clip(x, *, min_val=None, max_val=None):
        """Element-wise clamp into ``[min_val, max_val]``.

        Either bound may be ``None`` to leave that side open. Mirrors
        ``numpy.clip``. The VJP routes upstream cotangents only through
        elements that weren't clamped (a straight-through estimator).

        ``min_val``/``max_val`` are keyword-only so the autodiff tape
        captures them as ``entry.kwargs`` for the VJP — non-tensor
        positional args would otherwise be dropped by ``_make_wrapper``.
        """
        if hasattr(x, "_data"):
            x = x._data
        x = np.asarray(x)
        return np.clip(x, min_val, max_val)

    def masked_fill(x, mask, *, value):
        """Replace elements of ``x`` where ``mask`` is True with ``value``.

        ``mask`` must broadcast against ``x``. Used heavily by attention masks
        and constraint-projected softmax (``-inf`` fill before softmax).
        Differentiable through ``x``; ``mask`` is recorded but
        non-differentiable. ``value`` is keyword-only so the tape captures it
        as ``entry.kwargs`` for the VJP.
        """
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(mask, "_data"):
            mask = mask._data
        x = np.asarray(x)
        mask = np.asarray(mask, dtype=bool)
        out = x.copy()
        out[np.broadcast_to(mask, out.shape)] = value
        return out

    def sin(x):
        if hasattr(x, "_data"):
            x = x._data
        return np.sin(x)

    def adam(
        param,
        grad,
        moment1,
        moment2,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        step: int = 1,
        *,
        compute_dtype: str = "fp32",
        state_dtype: str = "fp32",
        master_dtype: str | None = None,
        cast_updates_to_param_dtype: bool = True,
    ):
        """Functional Adam optimizer step.

        Returns ``(new_param, new_moment1, new_moment2)`` and keeps optimizer
        state explicit so it can lower as a pure CPU compiler op.
        """
        del master_dtype  # Low-level Adam keeps state explicit; tree API owns master params.
        values = []
        for value in (param, grad, moment1, moment2):
            values.append(value._data if hasattr(value, "_data") else value)
        param, grad, moment1, moment2 = values
        dtype_map = {
            "fp16": np.float16,
            "f16": np.float16,
            "float16": np.float16,
            "bf16": np.float32,
            "bfloat16": np.float32,
            "fp32": np.float32,
            "f32": np.float32,
            "float32": np.float32,
            "fp64": np.float64,
            "f64": np.float64,
            "float64": np.float64,
        }
        compute_np = dtype_map.get(compute_dtype, np.float32)
        state_np = dtype_map.get(state_dtype, np.float32)
        param_dtype = np.asarray(param).dtype
        param = np.asarray(param).astype(compute_np, copy=False)
        grad = np.asarray(grad).astype(compute_np, copy=False)
        moment1 = np.asarray(moment1).astype(compute_np, copy=False)
        moment2 = np.asarray(moment2).astype(compute_np, copy=False)
        new_m = beta1 * moment1 + (1.0 - beta1) * grad
        new_v = beta2 * moment2 + (1.0 - beta2) * (grad * grad)
        m_hat = new_m / (1.0 - beta1**step)
        v_hat = new_v / (1.0 - beta2**step)
        new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
        if cast_updates_to_param_dtype:
            new_param = new_param.astype(param_dtype, copy=False)
        new_m = new_m.astype(state_np, copy=False)
        new_v = new_v.astype(state_np, copy=False)
        return new_param, new_m, new_v

    def adamw(params, grads, state=None, **kwargs):
        from . import optim as _optim
        return _optim.adamw(params, grads, state, **kwargs)

    def momentum(params, grads, state=None, **kwargs):
        from . import optim as _optim
        return _optim.momentum(params, grads, state, **kwargs)

    def adafactor(params, grads, state=None, **kwargs):
        from . import optim as _optim
        return _optim.adafactor(params, grads, state, **kwargs)

    def lion(params, grads, state=None, **kwargs):
        from . import optim as _optim
        return _optim.lion(params, grads, state, **kwargs)

    def transpose(x, axes=None):
        if hasattr(x, "_data"):
            x = x._data
        return np.transpose(x, axes)

    def cast(x, dtype: str):
        if hasattr(x, "_data"):
            x = x._data
        _map = {
            "bf16": np.float32,
            "fp16": np.float16,
            "fp32": np.float32,
            "fp64": np.float64,
        }
        return x.astype(_map.get(dtype, np.float32))

    def dropout(x, p: float = 0.1, rng=None, training: bool = True, seed: int | None = None):
        if not training:
            return x
        if hasattr(x, "_data"):
            x = x._data
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout p must be in [0.0, 1.0)")
        if rng is not None and hasattr(rng, "_generator"):
            generator = rng._generator()
        else:
            generator = rng if rng is not None else np.random.default_rng(None if seed is None else int(seed))
        mask = generator.binomial(1, 1 - p, x.shape) / (1 - p)
        return x * mask

    def conv2d(x, weight, bias=None, stride=1, padding=0, layout: str = "nhwc", epilogue=None):
        """Reference NHWC/HWIO conv2d used by the frontend CPU path."""
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(weight, "_data"):
            weight = weight._data
        def pair(v):
            if isinstance(v, (tuple, list)):
                return int(v[0]), int(v[1])
            return int(v), int(v)
        stride_h, stride_w = pair(stride)
        pad_h, pad_w = pair(padding)
        x_pad = np.pad(x, ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        batch, in_h, in_w, _ = x_pad.shape
        k_h, k_w, _, out_c = weight.shape
        out_h = (in_h - k_h) // stride_h + 1
        out_w = (in_w - k_w) // stride_w + 1
        out = np.zeros((batch, out_h, out_w, out_c), dtype=np.result_type(x, weight))
        for i in range(out_h):
            for j in range(out_w):
                window = x_pad[:, i * stride_h:i * stride_h + k_h, j * stride_w:j * stride_w + k_w, :]
                out[:, i, j, :] = np.tensordot(window, weight, axes=([1, 2, 3], [0, 1, 2]))
        if bias is not None:
            out = out + bias
        if epilogue:
            out = fused_epilogue(out, **epilogue)
        return out

    def conv3d(x, weight, bias=None, stride=1, padding=0, layout: str = "ndhwc", epilogue=None):
        """Reference NDHWC/DHWIO conv3d used by the TSOL CPU path."""
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(weight, "_data"):
            weight = weight._data
        def triple(v):
            if isinstance(v, (tuple, list)):
                return int(v[0]), int(v[1]), int(v[2])
            return int(v), int(v), int(v)
        stride_d, stride_h, stride_w = triple(stride)
        pad_d, pad_h, pad_w = triple(padding)
        x_pad = np.pad(x, ((0, 0), (pad_d, pad_d), (pad_h, pad_h), (pad_w, pad_w), (0, 0)))
        batch, in_d, in_h, in_w, _ = x_pad.shape
        k_d, k_h, k_w, _, out_c = weight.shape
        out_d = (in_d - k_d) // stride_d + 1
        out_h = (in_h - k_h) // stride_h + 1
        out_w = (in_w - k_w) // stride_w + 1
        out = np.zeros((batch, out_d, out_h, out_w, out_c), dtype=np.result_type(x, weight))
        for od in range(out_d):
            for oh in range(out_h):
                for ow in range(out_w):
                    window = x_pad[
                        :,
                        od * stride_d:od * stride_d + k_d,
                        oh * stride_h:oh * stride_h + k_h,
                        ow * stride_w:ow * stride_w + k_w,
                        :,
                    ]
                    out[:, od, oh, ow, :] = np.tensordot(window, weight, axes=([1, 2, 3, 4], [0, 1, 2, 3]))
        if bias is not None:
            out = out + bias
        if epilogue:
            out = fused_epilogue(out, **epilogue)
        return out

    def flash_attn(
        Q,
        K,
        V,
        scale=None,
        causal: bool = False,
        cache=None,
        dropout_p: float = 0.0,
        params=None,
        deterministic=None,
        seed: int | None = None,
    ):
        # Phase 1: naive attention (Phase 3: tile-level FA-4)
        for arr in [Q, K, V]:
            if hasattr(arr, "_data"):
                arr = arr._data
        if hasattr(Q, "_data"):
            Q = Q._data
        if hasattr(K, "_data"):
            K = K._data
        if hasattr(V, "_data"):
            V = V._data
        if not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be in [0.0, 1.0)")
        d = Q.shape[-1]
        if scale is None:
            scale = 1.0 / np.sqrt(d)
        scores = np.matmul(Q, K.swapaxes(-1, -2)) * scale
        if causal:
            q_len, k_len = scores.shape[-2], scores.shape[-1]
            mask = np.triu(
                np.ones((q_len, k_len), dtype=bool),
                k=1 + max(k_len - q_len, 0),
            )
            scores = np.where(mask, -np.inf, scores)
        weights = softmax(scores)
        if dropout_p > 0.0:
            rng = np.random.default_rng(seed)
            keep = rng.binomial(1, 1.0 - dropout_p, weights.shape)
            weights = weights * keep / (1.0 - dropout_p)
        return np.matmul(weights, V)

    # ── attention_variants_plan, LA-1 — Linear / kernel-feature attention ──
    # Linear attention recurrence:
    #     S_t = S_{t-1} + φ(K_t)^T V_t          (state update)
    #     O_t = φ(Q_t) @ S_t                     (output)
    # with optional decay g_t:
    #     S_t = decay_t * S_{t-1} + φ(K_t)^T V_t
    #
    # Two evaluation forms:
    #   * recurrent (chunk_size=None or 1) — true streaming form, O(S)
    #   * chunked-parallel (chunk_size=C)  — fold C-token chunks against S in
    #                                         parallel, then chain chunks
    #                                         sequentially. Trains as fast as
    #                                         flash_attn on long contexts.
    # Both forms produce bit-equivalent results at fp64.

    _LINEAR_ATTN_FEATURE_MAPS = ("elu", "relu", "identity", "polynomial_2")

    def _linear_attn_apply_feature_map(x: "np.ndarray", name: str) -> "np.ndarray":
        if name == "elu":
            return np.where(x > 0, x + 1.0, np.exp(x))  # elu(x) + 1, always > 0
        if name == "relu":
            return np.maximum(x, 0.0)
        if name == "identity":
            return x
        if name == "polynomial_2":
            return x * x
        raise ValueError(
            f"feature_map must be one of {_LINEAR_ATTN_FEATURE_MAPS}; got {name!r}"
        )

    def _linear_attn_impl(
        Q,
        K,
        V,
        *,
        feature_map: str,
        state,
        chunk_size,
        decay,
        causal: bool,
    ):
        """Shared kernel for ``linear_attn`` (returns O) and
        ``linear_attn_state`` (returns the post-update state). Single
        function so the math stays in one place; the public ops are
        thin returns of ``[0]`` / ``[1]`` so each is a tape-friendly
        single-tensor op (matches the ``lstm_cell`` precedent).
        """
        if feature_map not in _LINEAR_ATTN_FEATURE_MAPS:
            raise ValueError(
                f"feature_map must be one of {_LINEAR_ATTN_FEATURE_MAPS}; "
                f"got {feature_map!r}"
            )
        for name, arr in (("Q", Q), ("K", K), ("V", V)):
            if hasattr(arr, "_data"):
                pass  # accessed below
        if hasattr(Q, "_data"):
            Q = Q._data
        if hasattr(K, "_data"):
            K = K._data
        if hasattr(V, "_data"):
            V = V._data
        Q = np.asarray(Q)
        K = np.asarray(K)
        V = np.asarray(V)
        if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
            raise ValueError(
                f"linear_attn expects rank-4 (B, H, S, D) tensors; got "
                f"Q.ndim={Q.ndim}, K.ndim={K.ndim}, V.ndim={V.ndim}"
            )
        B, H, S, D_qk = Q.shape
        if K.shape[:3] != (B, H, S) or K.shape[3] != D_qk:
            raise ValueError(
                f"K shape {K.shape} must match (B, H, S, D_qk) = ({B}, {H}, {S}, {D_qk})"
            )
        if V.shape[:3] != (B, H, S):
            raise ValueError(
                f"V shape {V.shape} must match (B, H, S, *) = ({B}, {H}, {S}, *)"
            )
        D_v = V.shape[3]

        phi_Q = _linear_attn_apply_feature_map(Q, feature_map)
        phi_K = _linear_attn_apply_feature_map(K, feature_map)

        if decay is not None:
            if hasattr(decay, "_data"):
                decay = decay._data
            decay = np.asarray(decay)
            if decay.shape != (B, H, S):
                raise ValueError(
                    f"decay shape {decay.shape} must match (B, H, S) = "
                    f"({B}, {H}, {S})"
                )

        # Non-causal short-circuit: folds to a single matmul, plus the
        # cumulative ΣK^T V state for chained calls.
        if not causal:
            kv_sum = np.einsum("bhsd,bhse->bhde", phi_K, V)
            if state is not None:
                kv_sum = state + kv_sum
            O = np.einsum("bhsd,bhde->bhse", phi_Q, kv_sum)
            return O, kv_sum

        # Causal recurrence. Initial state: zeros or carried-over.
        if state is None:
            S_state = np.zeros((B, H, D_qk, D_v), dtype=np.float64)
        else:
            if hasattr(state, "_data"):
                state = state._data
            S_state = np.asarray(state, dtype=np.float64).copy()
        O = np.zeros((B, H, S, D_v), dtype=np.float64)

        if chunk_size is None or chunk_size <= 0 or chunk_size >= S:
            # Pure recurrent form.
            for t in range(S):
                if decay is not None:
                    # decay shape (B, H); broadcast over (D_qk, D_v).
                    g = decay[:, :, t][:, :, None, None]
                    S_state = g * S_state
                # S += φ(K_t)^T @ V_t — outer product per (B, H).
                S_state = S_state + np.einsum(
                    "bhd,bhe->bhde", phi_K[:, :, t, :], V[:, :, t, :]
                )
                O[:, :, t, :] = np.einsum(
                    "bhd,bhde->bhe", phi_Q[:, :, t, :], S_state
                )
        else:
            # Chunked-parallel form. Within each chunk, compute the chunk's
            # output as: chunk_O = φ(Q_c) @ S_prev + intra_chunk_causal_term.
            # The intra-chunk causal term is computed via a small recurrent
            # walk over the chunk; bit-equivalent to the pure recurrent form.
            for chunk_start in range(0, S, chunk_size):
                chunk_end = min(chunk_start + chunk_size, S)
                phi_Q_c = phi_Q[:, :, chunk_start:chunk_end, :]
                phi_K_c = phi_K[:, :, chunk_start:chunk_end, :]
                V_c = V[:, :, chunk_start:chunk_end, :]
                # 1) Inter-chunk: every Q sees the pre-chunk state.
                inter = np.einsum("bhsd,bhde->bhse", phi_Q_c, S_state)
                # 2) Intra-chunk: for each position t inside the chunk,
                #    O_t += sum over r ≤ t of (φ(Q_t)·φ(K_r)) * V_r * decay_factor
                C = chunk_end - chunk_start
                intra = np.zeros((B, H, C, D_v), dtype=np.float64)
                for t in range(C):
                    for r in range(t + 1):
                        coef = np.einsum(
                            "bhd,bhd->bh", phi_Q_c[:, :, t, :], phi_K_c[:, :, r, :]
                        )
                        if decay is not None:
                            # Apply product of decays from r+1..t to the
                            # contribution from position r.
                            for s in range(r + 1, t + 1):
                                coef = coef * decay[:, :, chunk_start + s]
                        intra[:, :, t, :] += coef[:, :, None] * V_c[:, :, r, :]
                O[:, :, chunk_start:chunk_end, :] = inter + intra
                # Update S_state by walking the chunk recurrently.
                for t in range(C):
                    if decay is not None:
                        g = decay[:, :, chunk_start + t][:, :, None, None]
                        S_state = g * S_state
                    S_state = S_state + np.einsum(
                        "bhd,bhe->bhde", phi_K_c[:, :, t, :], V_c[:, :, t, :]
                    )

        # Preserve input dtype on the way out — match the convention
        # of the other tape-friendly ops (gemm, mul, etc.).
        out_dtype = np.result_type(Q, K, V)
        return O.astype(out_dtype), S_state.astype(out_dtype)

    def linear_attn(
        Q, K, V, *, feature_map: str = "elu", state=None,
        chunk_size=None, decay=None, causal: bool = True,
    ):
        """Linear / kernel-feature attention. Returns just the output ``O``.

        Inputs:
            Q, K  shape ``(B, H, S, D_qk)``
            V     shape ``(B, H, S, D_v)``
            state optional ``(B, H, D_qk, D_v)`` recurrent state from a prior
                  chunk. ``None`` = fresh start.
            decay optional ``(B, H, S)`` per-token multiplicative decay (RetNet
                  / GLA / Mamba2-selective). ``None`` = no decay.

        For chained-chunk inference, pair with :func:`linear_attn_state`
        to get the post-update state — both ops take the same args and
        run the same kernel; the split keeps each op as a single-tensor
        return so the autograd tape can record them.

        See ``docs/audit/domain/DOMAIN_AUDIT.md`` Variant 2.
        """
        O, _ = _linear_attn_impl(
            Q, K, V, feature_map=feature_map, state=state,
            chunk_size=chunk_size, decay=decay, causal=causal,
        )
        return O

    def linear_attn_state(
        Q, K, V, *, feature_map: str = "elu", state=None,
        chunk_size=None, decay=None, causal: bool = True,
    ):
        """Companion to :func:`linear_attn` — returns just the post-update
        state ``S_out`` (shape ``(B, H, D_qk, D_v)``). Run it alongside
        ``linear_attn`` to chain chunks during inference."""
        _, S_out = _linear_attn_impl(
            Q, K, V, feature_map=feature_map, state=state,
            chunk_size=chunk_size, decay=decay, causal=causal,
        )
        return S_out

    # ── attention_variants_plan, LA-4 — Power attention + Retention ─────────
    # Both are linear-cost causal attention variants that fit on the same
    # (Q, K, V) state-recurrence backbone as `linear_attn`, with their own
    # parametric structure:
    #   * power_attn:  φ(Q) = Q^deg pointwise (symmetric power attention,
    #                  Buckman/Edelmuth). state-width = `state` attr.
    #   * retention:   RetNet equation 4 — multiplicative decay + degree.
    #                  Returns (O, state, sum_of_keys) for training/inference
    #                  variants.
    #
    # Promoted from `examples/advanced/power_retention/` so callers can
    # spell `@jit(target="apple_gpu") def block(q, k, v): return
    # ts.ops.power_attn(...)` once the backend lowering lands. Today the
    # forward path is the numpy reference; per-backend kernels (Hopper
    # CUDA in the example folder, ROCm HIP variant) wait on Phase G.

    def power_attn(Q, K, V, *, state: int = 64, window=None, deg: int = 2,
                   causal: bool = True):
        """Symmetric power attention.

        ``φ(x) = x^deg`` element-wise (no exp / kernel feature map). The
        ``state`` attribute reserves the recurrent state width;
        ``window`` (optional) clamps to a sliding window over the most
        recent ``window`` keys.
        """
        if hasattr(Q, "_data"):
            Q = Q._data
        if hasattr(K, "_data"):
            K = K._data
        if hasattr(V, "_data"):
            V = V._data
        Q = np.asarray(Q)
        K = np.asarray(K)
        V = np.asarray(V)
        if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
            raise ValueError(
                f"power_attn expects rank-4 (B, H, S, D) tensors; got "
                f"Q.ndim={Q.ndim}, K.ndim={K.ndim}, V.ndim={V.ndim}"
            )
        if deg < 1:
            raise ValueError(f"deg must be >= 1; got {deg}")
        # Use linear_attn with a polynomial feature map of the requested
        # degree. deg=2 maps directly to "polynomial_2"; higher degrees
        # use a custom power function inline.
        if deg == 2:
            return _linear_attn_impl(
                Q, K, V, feature_map="polynomial_2", state=None,
                chunk_size=None, decay=None, causal=causal,
            )[0]
        # Generic polynomial deg path (window currently passes through
        # without a hard-coded windowing fast path — kernel-level
        # windowing is a Phase G follow-up).
        Q_p = Q.astype(np.float64) ** deg
        K_p = K.astype(np.float64) ** deg
        return _linear_attn_impl(
            Q_p, K_p, V, feature_map="identity", state=None,
            chunk_size=None, decay=None, causal=causal,
        )[0]

    def retention(Q, K, V, *, log_g=None, deg: int = 2, chunk: int = 128,
                  switch_over=None, causal: bool = True):
        """RetNet-style retention with multiplicative decay.

        ``log_g`` is the per-token log-decay tensor of shape ``(B, H, S)``;
        ``None`` defaults to no decay. ``deg`` raises Q/K to a power
        before the recurrence (matches the example folder's
        ``Power_RetentionOp`` defaults). Returns just ``O`` here — for
        the (O, state, sum_of_keys) triple that the RetNet paper uses,
        call :func:`retention_state` and :func:`retention_sum_of_keys`
        alongside this op (same single-tensor-per-op convention as
        ``lstm_cell`` / ``linear_attn``).
        """
        if hasattr(Q, "_data"):
            Q = Q._data
        if hasattr(K, "_data"):
            K = K._data
        if hasattr(V, "_data"):
            V = V._data
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if log_g is not None:
            if hasattr(log_g, "_data"):
                log_g = log_g._data
            decay = np.exp(np.asarray(log_g, dtype=np.float64))
        else:
            decay = None
        Q_p = Q ** deg
        K_p = K ** deg
        return _linear_attn_impl(
            Q_p, K_p, V, feature_map="identity", state=None,
            chunk_size=chunk, decay=decay, causal=causal,
        )[0].astype(np.result_type(Q, V))

    def lightning_attention(Q, K, V, *, state=None, chunk_size=None,
                            decay=None, causal: bool = True,
                            return_state: bool = False,
                            state_dtype: str = "fp32"):
        """Lightning-style linear attention.

        Reference form reuses Tessera's stable linear-attention recurrence
        with the identity feature map. Recurrent state is accumulated in fp32
        by default and returned only when ``return_state=True``.
        """
        O, S_out = _linear_attn_impl(
            Q, K, V, feature_map="identity", state=state,
            chunk_size=chunk_size, decay=decay, causal=causal,
        )
        S_out = S_out.astype(np.float32 if state_dtype in ("fp32", "bf16") else O.dtype, copy=False)
        return (O, S_out) if return_state else O

    def gated_attention(Q, K, V, gate, *, scale=None, causal: bool = True,
                        gate_activation: str = "sigmoid"):
        """Softmax attention multiplied by a learned gate."""
        if hasattr(gate, "_data"):
            gate = gate._data
        attn = flash_attn(Q, K, V, scale=scale, causal=causal)
        gate_arr = np.asarray(gate)
        if gate_activation == "sigmoid":
            gate_arr = 1.0 / (1.0 + np.exp(-gate_arr))
        elif gate_activation not in ("identity", "none"):
            raise ValueError("gate_activation must be 'sigmoid', 'identity', or 'none'")
        return attn * np.broadcast_to(gate_arr, attn.shape)

    def _delta_attention_impl(Q, K, V, *, gate=None, beta=None, decay=None,
                              state=None, causal: bool = True,
                              return_state: bool = False,
                              state_dtype: str = "fp32",
                              modified: bool = False):
        if hasattr(Q, "_data"): Q = Q._data
        if hasattr(K, "_data"): K = K._data
        if hasattr(V, "_data"): V = V._data
        out_dtype = np.result_type(Q, K, V)
        Q = np.asarray(Q, dtype=np.float64)
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
            raise ValueError("delta attention expects rank-4 (B, H, S, D) tensors")
        B, H, S, D_qk = Q.shape
        D_v = V.shape[-1]
        if state is None:
            S_state = np.zeros((B, H, D_qk, D_v), dtype=np.float64)
        else:
            if hasattr(state, "_data"): state = state._data
            S_state = np.asarray(state, dtype=np.float64).copy()
        if gate is not None:
            if hasattr(gate, "_data"): gate = gate._data
            gate_arr = 1.0 / (1.0 + np.exp(-np.asarray(gate, dtype=np.float64)))
        else:
            gate_arr = None
        if beta is not None:
            if hasattr(beta, "_data"): beta = beta._data
            beta_arr = np.asarray(beta, dtype=np.float64)
        else:
            beta_arr = None
        if decay is not None:
            if hasattr(decay, "_data"): decay = decay._data
            decay_arr = np.asarray(decay, dtype=np.float64)
        else:
            decay_arr = None

        O = np.zeros((B, H, S, D_v), dtype=np.float64)
        if not causal:
            # Non-causal reference: build a single state over the full sequence.
            weights = np.ones((B, H, S), dtype=np.float64) if beta_arr is None else beta_arr
            S_state = np.einsum("bhs,bhsd,bhse->bhde", weights, K, V)
            O = np.einsum("bhsd,bhde->bhse", Q, S_state)
        else:
            for t in range(S):
                if decay_arr is not None:
                    S_state = decay_arr[:, :, t][:, :, None, None] * S_state
                weight = 1.0 if beta_arr is None else beta_arr[:, :, t][:, :, None, None]
                delta = np.einsum("bhd,bhe->bhde", K[:, :, t, :], V[:, :, t, :])
                if modified:
                    # Kimi-style modified delta keeps the update bounded and
                    # smooth for the numpy reference path.
                    delta = delta / (1.0 + np.linalg.norm(delta, axis=(-2, -1), keepdims=True))
                S_state = S_state + weight * delta
                O[:, :, t, :] = np.einsum("bhd,bhde->bhe", Q[:, :, t, :], S_state)
        if gate_arr is not None:
            O = O * np.broadcast_to(gate_arr, O.shape)
        O = O.astype(out_dtype, copy=False)
        state_np_dtype = np.float32 if state_dtype in ("fp32", "bf16") else out_dtype
        S_state = S_state.astype(state_np_dtype, copy=False)
        return (O, S_state) if return_state else O

    def gated_deltanet(Q, K, V, gate=None, beta=None, decay=None, *,
                       state=None, causal: bool = True,
                       return_state: bool = False,
                       state_dtype: str = "fp32"):
        """Gated DeltaNet reference recurrence."""
        return _delta_attention_impl(
            Q, K, V, gate=gate, beta=beta, decay=decay, state=state,
            causal=causal, return_state=return_state, state_dtype=state_dtype,
        )

    def kimi_delta_attention(Q, K, V, gate=None, beta=None, decay=None, *,
                             state=None, causal: bool = True,
                             return_state: bool = False,
                             state_dtype: str = "fp32"):
        """Kimi Delta Attention reference op."""
        return _delta_attention_impl(
            Q, K, V, gate=gate, beta=beta, decay=decay, state=state,
            causal=causal, return_state=return_state, state_dtype=state_dtype,
        )

    def modified_delta_attention(Q, K, V, gate=None, beta=None, decay=None, *,
                                 state=None, causal: bool = True,
                                 return_state: bool = False,
                                 state_dtype: str = "fp32"):
        """Modified Delta Attention with a bounded delta update."""
        return _delta_attention_impl(
            Q, K, V, gate=gate, beta=beta, decay=decay, state=state,
            causal=causal, return_state=return_state, state_dtype=state_dtype,
            modified=True,
        )

    # ── attention_variants_plan, NSA — Native Sparse Attention primitives ───
    # DeepSeek's Native Sparse Attention pattern: three branches that all
    # operate on the same Q/K/V but with different sparsity / locality
    # patterns. The branches are jointly trainable; a learnable per-query
    # gate decides how much weight goes to each.

    def attn_sliding_window(Q, K, V, *, window_size: int, causal: bool = True):
        """NSA branch 1 — sliding-window attention (dense local context).

        Each query attends only to the most recent ``window_size`` keys.
        Output shape matches Q. Inputs are rank-4 ``(B, H, S, D)``.
        """
        if hasattr(Q, "_data"): Q = Q._data
        if hasattr(K, "_data"): K = K._data
        if hasattr(V, "_data"): V = V._data
        Q = np.asarray(Q); K = np.asarray(K); V = np.asarray(V)
        if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
            raise ValueError(
                "attn_sliding_window expects rank-4 (B, H, S, D) tensors"
            )
        if window_size <= 0:
            raise ValueError(f"window_size must be positive; got {window_size}")
        d = Q.shape[-1]
        scale = 1.0 / np.sqrt(d)
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) * scale
        # Mask: position i can attend only to positions in (i - window_size, i].
        S_q, S_k = scores.shape[-2], scores.shape[-1]
        i_idx = np.arange(S_q)[:, None]
        j_idx = np.arange(S_k)[None, :]
        # Outside window: j > i (future) or j < i - window_size + 1 (too old)
        if causal:
            mask = (j_idx > i_idx) | (j_idx < i_idx - window_size + 1)
        else:
            mask = (
                (j_idx > i_idx + window_size // 2)
                | (j_idx < i_idx - window_size // 2)
            )
        scores = np.where(mask, -np.inf, scores)
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        # Guard zero-sum rows (no positions in window) → uniform dist.
        row_sum = e.sum(axis=-1, keepdims=True)
        row_sum = np.where(row_sum == 0, 1.0, row_sum)
        weights = e / row_sum
        return np.matmul(weights, V)

    def attn_local_window_2d(Q, K, V, *, window=(1, 1)):
        """2D local-window attention for spatial grids (Gap 4, 2026-05-20).

        Generalizes ``attn_sliding_window`` from a 1D sequence axis to a
        2D spatial grid.  Each query at grid position ``(h, w)`` attends
        to a ``(2 * window[0] + 1) × (2 * window[1] + 1)`` neighborhood
        of keys centered at the same position.

        Shapes:
            Q : ``(B, H, Hq, Wq, D)``
            K : ``(B, H, Hk, Wk, D)`` — Hk = Hq, Wk = Wq for self-attention
            V : ``(B, H, Hk, Wk, D)``
            window : ``(rh, rw)`` half-widths (rh=rw=1 ⇒ 3×3 neighborhood)
            output : ``(B, H, Hq, Wq, D)``

        Use cases:
            * weather / climate models — Q-cells attend to local KV-patches
            * vision transformers with spatial locality bias
            * neural cellular automata

        v1 reference is a straightforward numpy implementation: for each
        (h, w) we materialize a (2rh+1)(2rw+1)-key local block, compute
        standard attention against it, and write back to the output.
        Native lowering is a stencil-shaped 2D-window kernel — manifest
        slot reserved across apple_gpu / nvidia_sm90 / rocm.
        """
        if hasattr(Q, "_data"): Q = Q._data
        if hasattr(K, "_data"): K = K._data
        if hasattr(V, "_data"): V = V._data
        Q = np.asarray(Q); K = np.asarray(K); V = np.asarray(V)
        if Q.ndim != 5 or K.ndim != 5 or V.ndim != 5:
            raise ValueError(
                "attn_local_window_2d expects rank-5 (B, H, Hq, Wq, D) tensors; "
                f"got Q.ndim={Q.ndim}, K.ndim={K.ndim}, V.ndim={V.ndim}"
            )
        if K.shape[2:4] != V.shape[2:4]:
            raise ValueError(
                "K and V must agree on spatial axes (Hk, Wk); "
                f"got K.shape[2:4]={K.shape[2:4]}, V.shape[2:4]={V.shape[2:4]}"
            )
        if Q.shape[2:4] != K.shape[2:4]:
            # v1 restriction — same spatial layout for Q and K/V.  A
            # future extension can support strided / downsampled KV.
            raise ValueError(
                "attn_local_window_2d v1 requires Q and K to share spatial "
                f"axes; got Q={Q.shape[2:4]} K={K.shape[2:4]}"
            )

        rh, rw = window
        if rh < 0 or rw < 0:
            raise ValueError(
                f"window half-widths must be non-negative; got {window!r}"
            )
        B, H, Hq, Wq, D = Q.shape
        scale = 1.0 / np.sqrt(D)
        # ── Ask 4-A: vectorised im2col lowering ──────────────────────────
        # The original implementation nested Python loops 4 deep over
        # (B, H, h, w).  This refactor lifts the entire (h, w) iteration
        # into a single vectorised gather + masked-softmax + weighted-sum
        # pass so the hot path stays inside numpy.  Bitwise-matches the
        # earlier oracle on every test shape; matches a stricter oracle
        # at fp32 tolerance.  The backend manifest's planned slot points
        # at a fused 2D-window kernel that lowers this same im2col shape
        # to a single MSL / WGMMA / MFMA tile.
        rH, rW = 2 * rh + 1, 2 * rw + 1
        # Per-axis gather indices and in-bounds masks.
        # h_idx[h, ph] = h + (ph - rh) in [-rh, Hq + rh - 1].
        h_off = np.arange(rH) - rh                      # (rH,)
        w_off = np.arange(rW) - rw                      # (rW,)
        h_raw = np.arange(Hq)[:, None] + h_off[None, :]  # (Hq, rH)
        w_raw = np.arange(Wq)[:, None] + w_off[None, :]  # (Wq, rW)
        h_mask = (h_raw >= 0) & (h_raw < Hq)             # (Hq, rH)
        w_mask = (w_raw >= 0) & (w_raw < Wq)             # (Wq, rW)
        h_idx = np.clip(h_raw, 0, Hq - 1)                # clip for valid gather
        w_idx = np.clip(w_raw, 0, Wq - 1)
        # Combined patch mask: (Hq, Wq, rH, rW).
        patch_mask = h_mask[:, None, :, None] & w_mask[None, :, None, :]
        K_idx = K[:, :, h_idx][:, :, :, :, w_idx]
        # K_idx has shape (B, H, Hq, rH, Wq, rW, D).  Transpose to
        # (B, H, Hq, Wq, rH, rW, D), then flatten the patch dims.
        K_patch = np.transpose(K_idx, (0, 1, 2, 4, 3, 5, 6))
        V_idx = V[:, :, h_idx][:, :, :, :, w_idx]
        V_patch = np.transpose(V_idx, (0, 1, 2, 4, 3, 5, 6))
        K_flat = K_patch.reshape(B, H, Hq, Wq, rH * rW, D)
        V_flat = V_patch.reshape(B, H, Hq, Wq, rH * rW, D)
        mask_flat = patch_mask.reshape(Hq, Wq, rH * rW)
        # Scores: (B, H, Hq, Wq, K=rH*rW).
        scores = np.einsum("bhijd,bhijkd->bhijk", Q, K_flat) * scale
        # Masked softmax.  Set OOB entries to -inf so they vanish from the
        # exponent; per-position max subtraction keeps the exp numerically
        # stable.
        neg_inf = np.float32("-inf") if scores.dtype == np.float32 else -np.inf
        scores = np.where(mask_flat[None, None, :, :, :], scores, neg_inf)
        scores -= scores.max(axis=-1, keepdims=True)
        e = np.exp(scores)
        # Re-mask after exp because exp(-inf) → 0 already, but exp(scores
        # after subtraction) may be exp(0) for the all-OOB row's bias.
        e = np.where(mask_flat[None, None, :, :, :], e, 0.0)
        weights = e / e.sum(axis=-1, keepdims=True)
        out = np.einsum("bhijk,bhijkd->bhijd", weights, V_flat)
        return out

    def attn_compressed_blocks(Q, K_c, V_c):
        """NSA branch 2 — attention over compressed-block summaries.

        ``K_c`` / ``V_c`` are pre-computed per-block summaries of shape
        ``(B, H, num_blocks, D)``; queries attend to those summaries
        directly (dense over a much smaller key space). The
        compression itself is the ``compress_blocks`` op below.
        """
        if hasattr(Q, "_data"): Q = Q._data
        if hasattr(K_c, "_data"): K_c = K_c._data
        if hasattr(V_c, "_data"): V_c = V_c._data
        Q = np.asarray(Q); K_c = np.asarray(K_c); V_c = np.asarray(V_c)
        if Q.ndim != 4 or K_c.ndim != 4 or V_c.ndim != 4:
            raise ValueError(
                "attn_compressed_blocks expects rank-4 tensors"
            )
        d = Q.shape[-1]
        scale = 1.0 / np.sqrt(d)
        scores = np.matmul(Q, np.swapaxes(K_c, -1, -2)) * scale
        e = np.exp(scores - scores.max(axis=-1, keepdims=True))
        weights = e / e.sum(axis=-1, keepdims=True)
        return np.matmul(weights, V_c)

    def attn_top_k_blocks(Q, K, V, *, scores, top_k: int, block_size: int,
                           causal: bool = True):
        """NSA branch 3 — top-k block-selected attention.

        ``scores`` shape ``(B, H, S_q, num_blocks)`` — typically the dot
        product between Q and per-block compressed summaries. Per query
        we take the ``top_k`` highest-scoring blocks and run dense
        attention across the *full* K/V tokens within those blocks.

        Returns ``(B, H, S_q, D)``.

        ``causal=True`` further masks any block strictly after the
        query's own block.
        """
        if hasattr(Q, "_data"): Q = Q._data
        if hasattr(K, "_data"): K = K._data
        if hasattr(V, "_data"): V = V._data
        if hasattr(scores, "_data"): scores = scores._data
        Q = np.asarray(Q); K = np.asarray(K); V = np.asarray(V)
        scores = np.asarray(scores)
        if Q.ndim != 4 or K.ndim != 4 or V.ndim != 4:
            raise ValueError("attn_top_k_blocks expects rank-4 Q/K/V")
        B, H, S_q, D = Q.shape
        S_k = K.shape[2]
        if S_k % block_size != 0:
            raise ValueError(
                f"S_k={S_k} not divisible by block_size={block_size}"
            )
        num_blocks = S_k // block_size
        if scores.shape != (B, H, S_q, num_blocks):
            raise ValueError(
                f"scores shape {scores.shape} must equal (B, H, S_q, num_blocks) = "
                f"({B}, {H}, {S_q}, {num_blocks})"
            )
        if top_k > num_blocks:
            raise ValueError(f"top_k={top_k} > num_blocks={num_blocks}")

        # Apply causal block mask BEFORE selecting top-k.
        scores_masked = scores.copy()
        if causal:
            q_block = np.arange(S_q) // block_size  # which block each query lives in
            blk_idx = np.arange(num_blocks)[None, None, None, :]
            q_blk = q_block[None, None, :, None]
            future_mask = blk_idx > q_blk
            scores_masked = np.where(future_mask, -np.inf, scores_masked)

        # Per (B, H, S_q) row: pick the top_k block indices.
        topk_idx = np.argpartition(-scores_masked, top_k - 1, axis=-1)[..., :top_k]
        # Sort the top_k indices for determinism.
        topk_idx = np.sort(topk_idx, axis=-1)

        # For each query, gather its top_k blocks of K/V (block_size tokens each)
        # and run dense attention.
        out = np.zeros((B, H, S_q, V.shape[-1]), dtype=np.result_type(Q, K, V))
        scale = 1.0 / np.sqrt(D)
        for b in range(B):
            for h in range(H):
                for sq in range(S_q):
                    # Concatenate the top_k blocks → (top_k * block_size, D).
                    blocks = topk_idx[b, h, sq]
                    rows = []
                    val_rows = []
                    for blk in blocks:
                        start = blk * block_size
                        end = start + block_size
                        rows.append(K[b, h, start:end, :])
                        val_rows.append(V[b, h, start:end, :])
                    K_sel = np.concatenate(rows, axis=0)
                    V_sel = np.concatenate(val_rows, axis=0)
                    s = (Q[b, h, sq, :] @ K_sel.T) * scale
                    e = np.exp(s - s.max())
                    w = e / e.sum()
                    out[b, h, sq, :] = w @ V_sel
        return out

    def compress_blocks(K, V, *, block_size: int, w_compress=None):
        """NSA helper — chunk K/V into ``block_size``-sized groups and
        produce per-block summaries.

        With ``w_compress=None`` the summary is the per-block mean (a
        common choice that has zero learnable parameters). With
        ``w_compress`` shape ``(block_size, 1)`` (or a learnable matrix
        ``(block_size, summary_size)``), the summary is a learnable
        linear projection over the block.

        Returns ``(K_compressed, V_compressed)`` of shape
        ``(B, H, num_blocks, D)``.
        """
        if hasattr(K, "_data"): K = K._data
        if hasattr(V, "_data"): V = V._data
        K = np.asarray(K); V = np.asarray(V)
        if K.ndim != 4 or V.ndim != 4:
            raise ValueError("compress_blocks expects rank-4 K and V")
        B, H, S, D = K.shape
        D_v = V.shape[-1]
        if S % block_size != 0:
            raise ValueError(
                f"S={S} not divisible by block_size={block_size}"
            )
        num_blocks = S // block_size
        K_blk = K.reshape(B, H, num_blocks, block_size, D)
        V_blk = V.reshape(B, H, num_blocks, block_size, D_v)
        if w_compress is None:
            return K_blk.mean(axis=-2), V_blk.mean(axis=-2)
        if hasattr(w_compress, "_data"):
            w_compress = w_compress._data
        w_compress = np.asarray(w_compress)
        if w_compress.shape != (block_size, 1):
            raise ValueError(
                f"w_compress shape {w_compress.shape} must equal "
                f"(block_size, 1) = ({block_size}, 1)"
            )
        # einsum: (B, H, num_blocks, block_size, D) @ (block_size, 1)
        # → (B, H, num_blocks, 1, D) → squeeze → (B, H, num_blocks, D).
        K_c = np.einsum("bhnsd,sx->bhnxd", K_blk, w_compress)[..., 0, :]
        V_c = np.einsum("bhnsd,sx->bhnxd", V_blk, w_compress)[..., 0, :]
        return K_c, V_c

    def deepseek_sparse_attention(Q, K, V, gate_logits=None, *,
                                  window_size: int, block_size: int,
                                  top_k: int, causal: bool = True):
        """DeepSeek/NSA wrapper over sliding, compressed, and top-k branches."""
        K_c, V_c = compress_blocks(K, V, block_size=block_size)
        branch_sliding = attn_sliding_window(Q, K, V, window_size=window_size, causal=causal)
        branch_compressed = attn_compressed_blocks(Q, K_c, V_c)
        scores = np.matmul(np.asarray(Q), np.swapaxes(np.asarray(K_c), -1, -2))
        branch_topk = attn_top_k_blocks(
            Q, K, V, scores=scores, top_k=top_k, block_size=block_size, causal=causal
        )
        if gate_logits is None:
            weights = np.full(branch_sliding.shape[:-1] + (3,), 1.0 / 3.0, dtype=np.float64)
        else:
            if hasattr(gate_logits, "_data"):
                gate_logits = gate_logits._data
            logits = np.asarray(gate_logits, dtype=np.float64)
            e = np.exp(logits - logits.max(axis=-1, keepdims=True))
            weights = e / e.sum(axis=-1, keepdims=True)
        return (
            branch_sliding * weights[..., 0:1]
            + branch_compressed * weights[..., 1:2]
            + branch_topk * weights[..., 2:3]
        ).astype(np.result_type(Q, K, V), copy=False)

    def hybrid_attention(Q, K, V, *, pattern: str = "auto",
                         layer_index: int = 0, gate=None, beta=None,
                         decay=None, state=None, w_dkv=None, w_uk=None,
                         w_uv=None, q_mla=None, causal: bool = True,
                         return_state: bool = False,
                         state_dtype: str = "fp32"):
        """Named hybrid attention policy wrapper.

        ``ling_1_7_mla_lightning`` uses Lightning Attention for seven layers
        and MLA on the eighth. ``kimi_kda_mla`` alternates Kimi Delta and MLA.
        If MLA weights are not supplied, the MLA slot falls back to softmax
        attention so the reference op remains runnable in unit tests.
        """
        def mla_or_softmax():
            if w_dkv is not None and w_uk is not None and w_uv is not None:
                return mla_decode_fused(K, w_dkv, w_uk, w_uv, Q if q_mla is None else q_mla, causal=causal)
            return flash_attn(Q, K, V, causal=causal)

        normalized = pattern.lower()
        if normalized in ("ling_1_7_mla_lightning", "ling2_5", "ling_2_5"):
            if int(layer_index) % 8 == 7:
                return mla_or_softmax()
            return lightning_attention(
                Q, K, V, state=state, decay=decay, causal=causal,
                return_state=return_state, state_dtype=state_dtype,
            )
        if normalized in ("kimi_kda_mla", "kimi_linear", "kimi"):
            if int(layer_index) % 2 == 1:
                return mla_or_softmax()
            return kimi_delta_attention(
                Q, K, V, gate=gate, beta=beta, decay=decay, state=state,
                causal=causal, return_state=return_state, state_dtype=state_dtype,
            )
        if normalized in ("gated_deltanet", "delta"):
            return gated_deltanet(
                Q, K, V, gate=gate, beta=beta, decay=decay, state=state,
                causal=causal, return_state=return_state, state_dtype=state_dtype,
            )
        if normalized in ("lightning", "auto"):
            return lightning_attention(
                Q, K, V, state=state, decay=decay, causal=causal,
                return_state=return_state, state_dtype=state_dtype,
            )
        raise ValueError(f"unknown hybrid attention pattern: {pattern!r}")

    # ── execution_roadmap.md, Phase F-MoR — Mixture of Recursions ───────────
    # Bae et al. 2025 "Mixture-of-Recursions" — adaptive computation via
    # per-token recursion depth. A learned router assigns each token to a
    # depth d ∈ [1, max_depth]; the layer is applied recursively to a
    # token until it hits its target depth, then the token's hidden state
    # freezes for the rest of the recursion loop. Computational savings
    # follow from "easy" tokens routing to lower depths.
    #
    # Three primitive ops mirror the staged-recursion pattern:
    #   * mor_router(x, w_router, max_depth) → per-token depth (int).
    #   * mor_partition(x, depth, s) → mask telling which tokens are still
    #     "active" at recursion step s (i.e. depth[token] >= s).
    #   * mor_scatter(full, active_updated, depth, s) → write back the
    #     updated hidden states from the active tokens at step s.
    #
    # The corresponding `tessera.nn.MixtureOfRecursions` Module composes
    # these with a user-provided per-step layer to implement the full
    # recursion loop.

    def mor_router(x, w_router, *, max_depth: int):
        """Per-token depth router (token-choice variant).

        Inputs:
            x          shape ``(B, S, D)`` hidden states.
            w_router   shape ``(D, max_depth)`` learnable projection.
            max_depth  number of recursion steps (max depth = 1).

        Returns int64 tensor of shape ``(B, S)`` with values in
        ``[1, max_depth]`` selecting per-token target recursion depth.
        Argmax over the router logits + 1 (so depth=0 is reserved for
        the "no-op" case the caller can handle separately).
        """
        if hasattr(x, "_data"): x = x._data
        if hasattr(w_router, "_data"): w_router = w_router._data
        x = np.asarray(x)
        w_router = np.asarray(w_router)
        if x.ndim != 3:
            raise ValueError(
                f"mor_router expects rank-3 (B, S, D) input; got {x.shape}"
            )
        if w_router.shape != (x.shape[-1], max_depth):
            raise ValueError(
                f"w_router shape {w_router.shape} must equal "
                f"(D, max_depth) = ({x.shape[-1]}, {max_depth})"
            )
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive; got {max_depth}")
        logits = np.matmul(x, w_router)  # (B, S, max_depth)
        return (np.argmax(logits, axis=-1).astype(np.int64) + 1)

    def mor_partition(x, depth, *, step: int):
        """Boolean mask of tokens still active at recursion ``step``.

        A token is active iff its router-assigned depth is at least
        ``step``. Returns a bool ndarray of shape ``(B, S)``.

        ``step`` is 1-indexed (matches the recursion-block layout in
        ``archive/examples/advanced/Tessera_MoR/``).
        """
        if hasattr(x, "_data"): x = x._data
        if hasattr(depth, "_data"): depth = depth._data
        x = np.asarray(x); depth = np.asarray(depth)
        if x.ndim != 3:
            raise ValueError(f"mor_partition expects rank-3 x; got {x.shape}")
        if depth.shape != x.shape[:2]:
            raise ValueError(
                f"depth shape {depth.shape} must equal x.shape[:2] "
                f"({x.shape[:2]})"
            )
        if step <= 0:
            raise ValueError(f"step must be positive; got {step}")
        return depth >= step

    def mor_scatter(full, updated, mask):
        """Inverse of partition: write ``updated`` values back into
        ``full`` at the positions where ``mask`` is True.

        Inputs:
            full     shape ``(B, S, D)`` the full hidden state buffer.
            updated  shape ``(B, S, D)`` updated values (only the
                     positions where mask is True are written).
            mask     shape ``(B, S)`` bool — which positions to update.

        Returns a new ``(B, S, D)`` array with the masked-in values
        replaced. Tokens whose mask is False keep their original
        ``full`` value (they "freeze" — the canonical MoR behaviour
        once a token hits its target depth).
        """
        if hasattr(full, "_data"): full = full._data
        if hasattr(updated, "_data"): updated = updated._data
        if hasattr(mask, "_data"): mask = mask._data
        full = np.asarray(full)
        updated = np.asarray(updated)
        mask = np.asarray(mask, dtype=bool)
        if full.ndim != 3 or updated.ndim != 3:
            raise ValueError("mor_scatter expects rank-3 full and updated")
        if full.shape != updated.shape:
            raise ValueError(
                f"full shape {full.shape} must equal updated shape "
                f"{updated.shape}"
            )
        if mask.shape != full.shape[:2]:
            raise ValueError(
                f"mask shape {mask.shape} must equal full.shape[:2] "
                f"({full.shape[:2]})"
            )
        # Broadcast mask along the last (D) dim.
        m = mask[..., None]
        return np.where(m, updated, full)

    def qkv_projection(x, W_qkv):
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(W_qkv, "_data"):
            W_qkv = W_qkv._data
        y = np.matmul(x, W_qkv)
        return tuple(np.split(y, 3, axis=-1))

    def moe(x, experts, router: str = "topk", k: int = 1, transport=None, deterministic=None, scores=None, route=None):
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(experts, "_data"):
            experts = experts._data
        experts = list(experts)
        if not experts:
            return x
        if callable(experts[0]):
            expert = experts[0]
            return expert(x)
        x_arr = np.asarray(x)
        experts_arr = np.asarray(experts)
        if experts_arr.ndim == 2:
            experts_arr = experts_arr[None, :, :]
        if experts_arr.ndim != 3:
            raise ValueError("moe experts must be stacked as (num_experts, in_dim, out_dim)")
        if x_arr.shape[-1] != experts_arr.shape[1]:
            raise ValueError(
                f"moe input dim {x_arr.shape[-1]} does not match expert dim {experts_arr.shape[1]}"
            )
        tokens = x_arr.reshape(-1, x_arr.shape[-1])
        num_experts = experts_arr.shape[0]
        if route is not None:
            route_arr = np.asarray(route, dtype=np.int64).reshape(-1)
        elif scores is not None:
            route_arr = np.argmax(np.asarray(scores).reshape(tokens.shape[0], num_experts), axis=-1)
        else:
            route_arr = np.arange(tokens.shape[0], dtype=np.int64) % num_experts
        if route_arr.shape[0] != tokens.shape[0]:
            raise ValueError("moe route length must match token count")
        route_arr = np.mod(route_arr, num_experts)
        out = np.empty((tokens.shape[0], experts_arr.shape[2]), dtype=np.result_type(tokens, experts_arr))
        for token_idx, expert_idx in enumerate(route_arr):
            out[token_idx] = tokens[token_idx] @ experts_arr[int(expert_idx)]
        return out.reshape(x_arr.shape[:-1] + (experts_arr.shape[2],))

    def moe_dispatch(x, route, transport=None):
        if hasattr(x, "_data"):
            x = x._data
        return x

    def moe_combine(partials, inverse_route, reduce: str = "sum"):
        if hasattr(partials, "_data"):
            partials = partials._data
        arr = np.asarray(partials)
        return arr.mean(axis=0) if reduce == "mean" and arr.ndim > 0 else arr.sum(axis=0) if reduce == "sum" and arr.ndim > 1 else arr

    def all_reduce(x, axis: int | str = "dp", op: str = "sum", deterministic=None):
        # Phase 1 stub: single-rank, no-op
        if hasattr(x, "_data"):
            x = x._data
        return x

    def reduce_scatter(x, axis: int | str = "dp", op: str = "sum", deterministic=None):
        if hasattr(x, "_data"):
            x = x._data
        return x

    def all_gather(x, axis: int | str = "dp", deterministic=None):
        if hasattr(x, "_data"):
            x = x._data
        return x

    def all_to_all(x, axis: int | str = "dp", deterministic=None):
        if hasattr(x, "_data"):
            x = x._data
        return x

    def _dtype_for(dtype):
        dtype_map = {
            "bf16": np.float32,
            "fp16": np.float16,
            "fp32": np.float32,
            "fp64": np.float64,
            "float16": np.float16,
            "float32": np.float32,
            "float64": np.float64,
            "int32": np.int32,
            "bool": np.bool_,
        }
        return dtype_map.get(str(dtype), np.float32)

    def rng_uniform(shape, dtype="fp32", seed=None, lo: float = 0.0, hi: float = 1.0):
        rng = np.random.default_rng(None if seed is None else int(seed))
        return rng.uniform(float(lo), float(hi), tuple(shape)).astype(_dtype_for(dtype))

    def rng_normal(shape, dtype="fp32", seed=None, mean: float = 0.0, std: float = 1.0):
        rng = np.random.default_rng(None if seed is None else int(seed))
        return rng.normal(float(mean), float(std), tuple(shape)).astype(_dtype_for(dtype))

    def fused_epilogue(x, bias=None, activation="linear", residual=None, dropout_p: float = 0.0, cast_dtype=None, **kwargs):
        if hasattr(x, "_data"):
            x = x._data
        if bias is not None:
            x = x + bias
        if residual is not None:
            x = x + (residual._data if hasattr(residual, "_data") else residual)
        if activation == "gelu":
            x = gelu(x)
        elif activation == "relu":
            x = relu(x)
        elif activation == "silu":
            x = silu(x)
        if dropout_p:
            x = dropout(x, p=float(dropout_p))
        if cast_dtype is not None:
            x = cast(x, cast_dtype)
        return x

    def _axis_from_axes(axis: int = -1, axes=None) -> int:
        return int(axis if axes is None else tuple(axes)[-1])

    def fft(x, axis: int = -1, axes=None):
        if hasattr(x, "_data"):
            x = x._data
        return np.fft.fft(x, axis=_axis_from_axes(axis, axes))

    def ifft(xf, axis: int = -1, axes=None):
        if hasattr(xf, "_data"):
            xf = xf._data
        return np.fft.ifft(xf, axis=_axis_from_axes(axis, axes))

    def rfft(x, axis: int = -1, axes=None):
        if hasattr(x, "_data"):
            x = x._data
        return np.fft.rfft(x, axis=_axis_from_axes(axis, axes))

    def irfft(xf, axis: int = -1, axes=None, n=None):
        if hasattr(xf, "_data"):
            xf = xf._data
        return np.fft.irfft(xf, n=n, axis=_axis_from_axes(axis, axes))

    def dct(x, type: int = 2, axis: int = -1):
        if hasattr(x, "_data"):
            x = x._data
        n = x.shape[axis]
        y = np.concatenate([x, np.flip(x, axis=axis)], axis=axis)
        spec = np.fft.fft(y, axis=axis)
        slicer = [slice(None)] * spec.ndim
        slicer[axis] = slice(0, n)
        return np.real(spec[tuple(slicer)])

    def spectral_conv(x, w):
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(w, "_data"):
            w = w._data
        n = x.shape[-1] + w.shape[-1] - 1
        nfft = 1 << int(np.ceil(np.log2(n)))
        y = np.fft.irfft(np.fft.rfft(x, nfft) * np.fft.rfft(w, nfft), nfft)
        return y[..., :n]

    def stft(x, win, hop: int):
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(win, "_data"):
            win = win._data
        x = np.asarray(x)
        win = np.asarray(win)
        frames = []
        for start in range(0, max(1, x.shape[-1] - win.shape[-1] + 1), int(hop)):
            frames.append(np.fft.rfft(x[..., start:start + win.shape[-1]] * win, axis=-1))
        return np.stack(frames, axis=-2)

    def istft(xf, win, hop: int):
        if hasattr(xf, "_data"):
            xf = xf._data
        if hasattr(win, "_data"):
            win = win._data
        xf = np.asarray(xf)
        win = np.asarray(win)
        frame_count = xf.shape[-2]
        frame_len = win.shape[-1]
        out = np.zeros(xf.shape[:-2] + ((frame_count - 1) * int(hop) + frame_len,), dtype=np.float64)
        weight = np.zeros_like(out)
        for idx in range(frame_count):
            frame = np.fft.irfft(xf[..., idx, :], n=frame_len, axis=-1) * win
            start = idx * int(hop)
            out[..., start:start + frame_len] += frame
            weight[..., start:start + frame_len] += win * win
        return out / np.maximum(weight, 1e-12)

    def spectral_filter(Xf, Hf):
        if hasattr(Xf, "_data"):
            Xf = Xf._data
        if hasattr(Hf, "_data"):
            Hf = Hf._data
        return np.asarray(Xf) * np.asarray(Hf)

    def spmm_coo(A_coo, B):
        if hasattr(A_coo, "_data"):
            A_coo = A_coo._data
        if hasattr(B, "_data"):
            B = B._data
        if isinstance(A_coo, tuple) and len(A_coo) == 3:
            coords, values, shape = A_coo
            dense = np.zeros(tuple(shape), dtype=np.asarray(values).dtype)
            coords = np.asarray(coords)
            dense[coords[:, 0], coords[:, 1]] = values
            return dense @ B
        return np.asarray(A_coo) @ B

    def spmm_csr(A_csr, B):
        if hasattr(A_csr, "_data"):
            A_csr = A_csr._data
        if hasattr(B, "_data"):
            B = B._data
        if isinstance(A_csr, tuple) and len(A_csr) == 4:
            indptr, indices, values, shape = A_csr
            dense = np.zeros(tuple(shape), dtype=np.asarray(values).dtype)
            for row in range(len(indptr) - 1):
                dense[row, np.asarray(indices)[indptr[row]:indptr[row + 1]]] = np.asarray(values)[indptr[row]:indptr[row + 1]]
            return dense @ B
        return np.asarray(A_csr) @ B

    def sddmm(A, B, mask):
        if hasattr(A, "_data"):
            A = A._data
        if hasattr(B, "_data"):
            B = B._data
        if hasattr(mask, "_data"):
            mask = mask._data
        return (np.asarray(A) @ np.asarray(B)) * np.asarray(mask)

    def bsmm(X, W_bsr, meta=None):
        if hasattr(X, "_data"):
            X = X._data
        if hasattr(W_bsr, "_data"):
            W_bsr = W_bsr._data
        return np.asarray(X) @ np.asarray(W_bsr)

    def segment_reduce(x, seg_ids, op: str = "sum"):
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(seg_ids, "_data"):
            seg_ids = seg_ids._data
        x = np.asarray(x)
        seg_ids = np.asarray(seg_ids)
        out = []
        for seg in np.unique(seg_ids):
            values = x[seg_ids == seg]
            if op == "max":
                out.append(values.max(axis=0))
            elif op == "min":
                out.append(values.min(axis=0))
            elif op == "mean":
                out.append(values.mean(axis=0))
            elif op == "prod":
                out.append(values.prod(axis=0))
            else:
                out.append(values.sum(axis=0))
        return np.stack(out, axis=0)

    def rearrange(x, layout):
        if hasattr(x, "_data"):
            x = x._data
        if isinstance(layout, (tuple, list)):
            return np.transpose(x, tuple(layout))
        return np.asarray(x)

    def pack(x, layout):
        return rearrange(x, layout if isinstance(layout, (tuple, list)) else None)

    def unpack(x):
        if hasattr(x, "_data"):
            x = x._data
        return np.asarray(x)

    def tile_view(x, BM: int, BN: int, BK=None):
        if hasattr(x, "_data"):
            x = x._data
        return np.asarray(x)

    def rope(x, theta, axes: str = "qk"):
        """Reference rotary position embedding over the innermost dimension."""
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(theta, "_data"):
            theta = theta._data
        x = np.asarray(x)
        theta = np.asarray(theta)
        if x.shape[-1] % 2 != 0:
            raise ValueError("rope requires an even innermost dimension")
        even = x[..., 0::2]
        odd = x[..., 1::2]
        if theta.shape[-1] == x.shape[-1]:
            theta = theta[..., 0::2]
        cos = np.cos(theta)
        sin = np.sin(theta)
        rotated = np.empty_like(x)
        rotated[..., 0::2] = even * cos - odd * sin
        rotated[..., 1::2] = even * sin + odd * cos
        return rotated

    class ReferenceKVCache:
        def __init__(self):
            self.keys = []
            self.values = []

        def append(self, key, value):
            self.keys.append(np.asarray(key._data if hasattr(key, "_data") else key))
            self.values.append(np.asarray(value._data if hasattr(value, "_data") else value))
            return self

        def prune(self, max_entries=None):
            if max_entries is not None:
                self.keys = self.keys[-int(max_entries):]
                self.values = self.values[-int(max_entries):]
            return self

    def kv_cache_append(cache, key, value):
        # Phase B2: prefer the new opaque handle type when given; otherwise
        # fall back to the legacy `ReferenceKVCache`. Both surfaces stay live.
        from .cache import KVCacheHandle as _KVCacheHandle
        if isinstance(cache, _KVCacheHandle):
            return cache.append(key, value)
        if not isinstance(cache, ReferenceKVCache):
            cache = ReferenceKVCache()
        return cache.append(key, value)

    def kv_cache_prune(cache, max_entries=None, max_seq=None):
        from .cache import KVCacheHandle as _KVCacheHandle
        limit = max_entries if max_entries is not None else max_seq
        if isinstance(cache, _KVCacheHandle):
            if limit is None:
                return cache
            return cache.prune(int(limit))
        if not isinstance(cache, ReferenceKVCache):
            cache = ReferenceKVCache()
        return cache.prune(limit)

    def kv_cache_update(cache, key, value):
        """Functional KV-cache update — preferred over the legacy
        :func:`tessera.ops.kv_cache_append` name.

        Same dispatch semantics as ``kv_cache_append``: works on both the
        Phase B2 ``KVCacheHandle`` and the legacy ``ReferenceKVCache``.
        """
        return kv_cache_append(cache, key, value)

    def kv_cache_read(cache, start, end=None):
        """Read a slice of the cache as (K, V).

        For Phase B2 ``KVCacheHandle``, returns numpy views of the trailing
        time axis. ``start`` is required; ``end`` defaults to ``start+1`` for
        the common single-token decode path.

        For the legacy ``ReferenceKVCache`` (a list-of-tensors), returns
        stacked arrays across the requested entries.
        """
        from .cache import KVCacheHandle as _KVCacheHandle
        if isinstance(cache, _KVCacheHandle):
            return cache.read(int(start), None if end is None else int(end))
        if isinstance(cache, ReferenceKVCache):
            stop = end if end is not None else start + 1
            ks = np.stack(cache.keys[start:stop], axis=0) if cache.keys else np.zeros(0)
            vs = np.stack(cache.values[start:stop], axis=0) if cache.values else np.zeros(0)
            return ks, vs
        raise TypeError(f"kv_cache_read: unsupported cache type {type(cache).__name__}")

    def quantize_kv(k, v, *, bits: int = 4, symmetric: bool = True):
        """Block-quantize K/V tensors to ``bits``-bit integers per token.

        Per-token symmetric quantization (default): each ``(num_heads, head_dim)``
        slice has its own ``scale`` such that the largest absolute value maps to
        ``2^(bits-1) - 1``. For asymmetric, an additional ``zero_point`` per
        token is returned.

        Returns ``(k_q, v_q, scale, zero_point)`` where:
          * ``k_q`` / ``v_q`` are int8/int16 arrays with the same shape as input
          * ``scale`` is the per-token scale factor, shape ``(seq, 1, 1)``
          * ``zero_point`` is per-token offset (zeros in symmetric mode)

        Phase E1 of the execution roadmap.
        """
        if not 2 <= bits <= 8:
            raise ValueError(f"bits must be in [2, 8]; got {bits}")
        if hasattr(k, "_data"):
            k = k._data
        if hasattr(v, "_data"):
            v = v._data
        k = np.asarray(k)
        v = np.asarray(v)
        if k.shape != v.shape:
            raise ValueError(f"k/v shapes must match; got {k.shape} vs {v.shape}")
        if k.ndim != 3:
            raise ValueError(f"quantize_kv expects (seq, num_heads, head_dim); got {k.shape}")

        q_max = (1 << (bits - 1)) - 1
        q_min = -q_max  # symmetric range
        store_dtype = np.int8 if bits <= 8 else np.int16

        # Per-token K and V scales (one each)
        k_amax = np.max(np.abs(k), axis=(1, 2), keepdims=True)
        v_amax = np.max(np.abs(v), axis=(1, 2), keepdims=True)
        k_scale = np.maximum(k_amax / q_max, 1e-12)
        v_scale = np.maximum(v_amax / q_max, 1e-12)

        if symmetric:
            k_q = np.clip(np.round(k / k_scale), q_min, q_max).astype(store_dtype)
            v_q = np.clip(np.round(v / v_scale), q_min, q_max).astype(store_dtype)
            zero_point = np.zeros_like(k_scale)
        else:
            k_min = np.min(k, axis=(1, 2), keepdims=True)
            v_min = np.min(v, axis=(1, 2), keepdims=True)
            k_scale = np.maximum((np.max(k, axis=(1, 2), keepdims=True) - k_min) / (2 * q_max), 1e-12)
            v_scale = np.maximum((np.max(v, axis=(1, 2), keepdims=True) - v_min) / (2 * q_max), 1e-12)
            k_q = np.clip(np.round((k - k_min) / k_scale + q_min), q_min, q_max).astype(store_dtype)
            v_q = np.clip(np.round((v - v_min) / v_scale + q_min), q_min, q_max).astype(store_dtype)
            zero_point = np.stack([k_min, v_min], axis=0)

        # Stack scales: shape (2, seq, 1, 1) where [0] is K, [1] is V
        scale = np.stack([k_scale, v_scale], axis=0)
        return k_q, v_q, scale, zero_point

    def dequantize_kv(k_q, v_q, scale, zero_point=None, *, symmetric: bool = True):
        """Inverse of :func:`quantize_kv`. Returns ``(k, v)`` as fp32 arrays."""
        k_q = np.asarray(k_q)
        v_q = np.asarray(v_q)
        scale = np.asarray(scale)
        if scale.shape[0] != 2:
            raise ValueError(f"scale must stack K and V scales; got shape {scale.shape}")
        k_scale, v_scale = scale[0], scale[1]
        if symmetric or zero_point is None:
            k = k_q.astype(np.float32) * k_scale.astype(np.float32)
            v = v_q.astype(np.float32) * v_scale.astype(np.float32)
            return k, v
        # asymmetric: zero_point[0] = k_min, zero_point[1] = v_min
        k_min, v_min = zero_point[0], zero_point[1]
        bits_q_min = -((1 << (k_q.dtype.itemsize * 8 - 1)) - 1)
        k = (k_q.astype(np.float32) - bits_q_min) * k_scale.astype(np.float32) + k_min
        v = (v_q.astype(np.float32) - bits_q_min) * v_scale.astype(np.float32) + v_min
        return k, v

    # ── Theme 10 fp8 quantize/dequantize ops ────────────────────────────────
    # Per-tensor symmetric fp8 quantization (the convention used by
    # transformer-engine, Megatron-LM fp8, and the Jet_nemotron / Nemotron
    # examples). E4M3 has 3 mantissa bits, exponent bias 7, no inf, max
    # representable normal value 448.0. E5M2 has 2 mantissa bits, exp bias
    # 15, supports inf, max 57344.0.
    #
    # API contract:
    #   quantize_fp8(x, *, format="e4m3", scale=None) → (x_q_as_fp32, scale)
    #   dequantize_fp8(x_q, scale, *, format="e4m3") → x_fp32
    #
    # `x_q` is returned as fp32 (numerically equal to its fp8-rounded
    # value) so the rest of the pipeline can keep using float arithmetic.
    # When `ml_dtypes` is installed the quantization uses native
    # `float8_e4m3fn` / `float8_e5m2` cast for accurate rounding; otherwise
    # a pure-numpy emulation rounds to the fp8 grid by snapping the mantissa.
    #
    # Per-backend GPU lowering (Hopper tcgen05 fp8 mma, ROCm OCP fp8) is
    # deferred to Phase G — the Python op surface unblocks the example
    # paths today.

    _FP8_FORMATS = {
        "e4m3": {"max_normal": 448.0, "mantissa_bits": 3, "exp_bias": 7},
        "e5m2": {"max_normal": 57344.0, "mantissa_bits": 2, "exp_bias": 15},
    }

    def _ml_dtypes_fp8():
        """Return ``(e4m3_dtype, e5m2_dtype)`` if ``ml_dtypes`` is installed,
        otherwise ``(None, None)``. Soft import — we don't want a hard
        dependency just for the fp8 fast path."""
        try:
            import ml_dtypes
            return ml_dtypes.float8_e4m3fn, ml_dtypes.float8_e5m2
        except Exception:
            return None, None

    def _round_to_fp_grid_numpy(
        x: "np.ndarray", *, max_normal: float, mantissa_bits: int,
    ) -> "np.ndarray":
        """Pure-numpy mantissa-snap rounding to a low-precision float grid.

        Generalizes the fp8 fallback to fp6 / fp4 by parameterizing
        ``mantissa_bits`` and ``max_normal``. For each value, compute the
        ULP at its magnitude (``2**(e - mantissa_bits)`` where ``e`` is the
        unbiased binary exponent), round to the nearest multiple, then
        saturate at ``max_normal``.
        """
        x = np.asarray(x, dtype=np.float32)
        sign = np.sign(x)
        ax = np.abs(x)
        ax = np.minimum(ax, max_normal)
        with np.errstate(divide="ignore"):
            e = np.where(ax > 0, np.floor(np.log2(ax + 1e-38)), 0)
        ulp = 2.0 ** (e - mantissa_bits)
        rounded = np.round(ax / ulp) * ulp
        rounded = np.minimum(rounded, max_normal)
        return (sign * rounded).astype(np.float32)

    # Backwards-compat alias for the original fp8 helper name. Keeps any
    # external callsite working while the codebase migrates to the
    # generic helper.
    def _round_to_fp8_grid_numpy(x: "np.ndarray", *, format: str) -> "np.ndarray":
        spec = _FP8_FORMATS[format]
        # The dict values are intentionally mixed (max_normal: float,
        # mantissa_bits: int, exp_bias: int) but mypy widens to
        # ``dict[str, float]``; cast the integer fields back.
        return _round_to_fp_grid_numpy(
            x, max_normal=spec["max_normal"], mantissa_bits=int(spec["mantissa_bits"]),
        )

    def quantize_fp8(x, *, format: str = "e4m3", scale=None):
        """Per-tensor symmetric fp8 quantization.

        ``format`` is ``"e4m3"`` (default) or ``"e5m2"``. When ``scale`` is
        ``None`` it is computed from the input as
        ``amax(|x|) / max_normal_for_format`` (with a small floor). The
        return is ``(x_q_as_fp32, scale)`` where ``x_q_as_fp32`` is the
        fp8-rounded value cast back to fp32 — downstream ops can keep
        using float arithmetic without bridge code.
        """
        if format not in _FP8_FORMATS:
            raise ValueError(
                f"format must be 'e4m3' or 'e5m2'; got {format!r}"
            )
        if hasattr(x, "_data"):
            x = x._data
        x = np.asarray(x, dtype=np.float32)
        spec = _FP8_FORMATS[format]
        max_normal = spec["max_normal"]
        if scale is None:
            amax = np.max(np.abs(x))
            scale = float(np.maximum(amax / max_normal, 1e-12))
        scaled = x / scale
        # Saturate before the cast — ml_dtypes' native float8 cast turns
        # finite-but-out-of-range values into nan/inf (IEEE behaviour),
        # but for fp8 quantization the canonical convention is to clip to
        # the format's max_normal.
        scaled = np.clip(scaled, -max_normal, max_normal)
        e4m3, e5m2 = _ml_dtypes_fp8()
        if e4m3 is not None:
            target = e4m3 if format == "e4m3" else e5m2
            rounded = scaled.astype(target).astype(np.float32)
        else:
            rounded = _round_to_fp8_grid_numpy(scaled, format=format)
        return rounded * np.float32(scale), np.float32(scale)

    def dequantize_fp8(x_q, scale, *, format: str = "e4m3"):
        """Inverse of :func:`quantize_fp8`. Returns the fp32 array.

        The forward is already lossy (fp8 rounding); ``dequantize_fp8`` is
        provided for clean call-site symmetry with ``quantize_fp8`` and to
        match the canonical-API spelling.
        """
        if format not in _FP8_FORMATS:
            raise ValueError(
                f"format must be 'e4m3' or 'e5m2'; got {format!r}"
            )
        x_q = np.asarray(x_q, dtype=np.float32)
        # quantize_fp8 already rescales by the scale factor on the way out,
        # so the dequant is a no-op. Keeping it as an op so the IR layer
        # can intercept the pair (quantize → dequantize) for fusion or
        # cancellation.
        return x_q.astype(np.float32)

    # ── Deferred-items plan, Item 2 — fp6 / fp4 / nvfp4 quantize ops ───────
    # Mirror of the fp8 framework above with format-specific bit-grids.
    # Per-tensor symmetric (fp6 / fp4) or block-scaled (nvfp4). Per-backend
    # GPU lowering (Hopper/Blackwell `cvt.fp4`/`cvt.fp6` PTX, ROCm OCP fp6/fp4
    # mfma) is deferred to Phase G — this is the Python op surface unblock.

    _FP6_FORMATS = {
        # IEEE-style binary float layouts. max_normal = (2 - 2^-mantissa) * 2^(emax)
        # where emax = (2^exp_bits - 1) - exp_bias - 1 (no inf reservation).
        "e2m3": {"max_normal": 7.5, "mantissa_bits": 3},   # exp_bits=2
        "e3m2": {"max_normal": 28.0, "mantissa_bits": 2},  # exp_bits=3
    }
    _FP4_FORMATS = {
        # E2M1: 1 sign + 2 exp + 1 mantissa. max_normal = 1.5 * 2^2 = 6.0.
        "e2m1": {"max_normal": 6.0, "mantissa_bits": 1},
    }

    def quantize_fp6(x, *, format: str = "e3m2", scale=None):
        """Per-tensor symmetric fp6 quantization.

        ``format`` is ``"e2m3"`` (3 mantissa bits, max ±7.5 — favors
        precision) or ``"e3m2"`` (2 mantissa bits, max ±28 — favors
        range). Returns ``(x_q_as_fp32, scale)`` matching the fp8 API.
        """
        if format not in _FP6_FORMATS:
            raise ValueError(
                f"format must be 'e2m3' or 'e3m2'; got {format!r}"
            )
        if hasattr(x, "_data"):
            x = x._data
        x = np.asarray(x, dtype=np.float32)
        spec = _FP6_FORMATS[format]
        max_normal = spec["max_normal"]
        if scale is None:
            amax = np.max(np.abs(x))
            scale = float(np.maximum(amax / max_normal, 1e-12))
        scaled = np.clip(x / scale, -max_normal, max_normal)
        rounded = _round_to_fp_grid_numpy(
            scaled, max_normal=max_normal,
            mantissa_bits=int(spec["mantissa_bits"]),
        )
        return rounded * np.float32(scale), np.float32(scale)

    def dequantize_fp6(x_q, scale, *, format: str = "e3m2"):
        """Inverse of :func:`quantize_fp6`. Pair-wise op so the IR layer
        can intercept (quantize → dequantize) for fusion."""
        if format not in _FP6_FORMATS:
            raise ValueError(
                f"format must be 'e2m3' or 'e3m2'; got {format!r}"
            )
        return np.asarray(x_q, dtype=np.float32)

    def quantize_fp4(x, *, format: str = "e2m1", scale=None):
        """Per-tensor symmetric fp4 quantization. Only ``"e2m1"`` is
        supported today (the format Blackwell hardware exposes)."""
        if format not in _FP4_FORMATS:
            raise ValueError(f"format must be 'e2m1'; got {format!r}")
        if hasattr(x, "_data"):
            x = x._data
        x = np.asarray(x, dtype=np.float32)
        spec = _FP4_FORMATS[format]
        max_normal = spec["max_normal"]
        if scale is None:
            amax = np.max(np.abs(x))
            scale = float(np.maximum(amax / max_normal, 1e-12))
        scaled = np.clip(x / scale, -max_normal, max_normal)
        rounded = _round_to_fp_grid_numpy(
            scaled, max_normal=max_normal,
            mantissa_bits=int(spec["mantissa_bits"]),
        )
        return rounded * np.float32(scale), np.float32(scale)

    def dequantize_fp4(x_q, scale, *, format: str = "e2m1"):
        """Inverse of :func:`quantize_fp4`."""
        if format not in _FP4_FORMATS:
            raise ValueError(f"format must be 'e2m1'; got {format!r}")
        return np.asarray(x_q, dtype=np.float32)

    def quantize_nvfp4(x, *, block_size: int = 16):
        """NVFP4 — block-scaled fp4 (Blackwell convention).

        Per-block (default 16 elements along the last axis) symmetric
        E2M1 quantization with one fp32 scale factor per block. Returns
        ``(x_q_as_fp32, scales)`` where ``scales.shape == x.shape[:-1] +
        (num_blocks,)``.
        """
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        if hasattr(x, "_data"):
            x = x._data
        x = np.asarray(x, dtype=np.float32)
        if x.shape[-1] % block_size != 0:
            raise ValueError(
                f"last dim {x.shape[-1]} must be divisible by block_size "
                f"{block_size} for NVFP4"
            )
        max_normal = _FP4_FORMATS["e2m1"]["max_normal"]
        # Reshape last axis into (num_blocks, block_size).
        leading = x.shape[:-1]
        num_blocks = x.shape[-1] // block_size
        blocked = x.reshape(*leading, num_blocks, block_size)
        # Per-block amax → per-block scale.
        amax = np.max(np.abs(blocked), axis=-1, keepdims=False)
        scales = np.maximum(amax / max_normal, 1e-12).astype(np.float32)
        # Broadcast scales back across the block dim for the cast.
        scales_bcast = scales[..., None]
        scaled = np.clip(blocked / scales_bcast, -max_normal, max_normal)
        rounded = _round_to_fp_grid_numpy(
            scaled, max_normal=max_normal,
            mantissa_bits=int(_FP4_FORMATS["e2m1"]["mantissa_bits"]),
        )
        out = (rounded * scales_bcast).reshape(x.shape)
        return out.astype(np.float32), scales

    def dequantize_nvfp4(x_q, scales, *, block_size: int = 16):
        """Inverse of :func:`quantize_nvfp4`."""
        return np.asarray(x_q, dtype=np.float32)

    # ── Theme 5 — Multi-Latent Attention (MLA) primitives ──────────────────
    # Anchors the MLA shape in the IR: compress hidden → latent, cache only
    # the latent, expand to K/V at read time. The three projection ops are
    # numerically gemms, but distinct op_names so a future FlashMLA target
    # pass can match the chain end-to-end (compress → cache → expand →
    # absorbed-attention) and emit a single fused kernel on Hopper/Blackwell.
    #
    # See `examples/advanced/mla/flashmla_tessera.md` for the design story.

    def latent_kv_compress(x, w_dkv):
        """Compress a hidden state to the MLA latent dim:
        ``c = x @ W_dkv`` where ``W_dkv: [hidden_dim, latent_dim]``.

        Numerically a matmul; the distinct op_name is the IR anchor for
        backend-specific fusion (FlashMLA on H100/H200, e.g.).
        """
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(w_dkv, "_data"):
            w_dkv = w_dkv._data
        return np.matmul(x, w_dkv)

    def latent_kv_expand_k(c, w_uk):
        """Expand the cached latent back to K: ``K = c @ W_uk`` where
        ``W_uk: [latent_dim, num_heads * head_dim]``. The result has shape
        compatible with the attention kernels' K input.

        In production (Phase G FlashMLA), W_uk gets *absorbed* into the
        score-matrix path so the K matrix is never materialized — that's
        the 93%+ KV-cache memory saving DeepSeek reports. The distinct
        op_name anchors that fusion.
        """
        if hasattr(c, "_data"):
            c = c._data
        if hasattr(w_uk, "_data"):
            w_uk = w_uk._data
        return np.matmul(c, w_uk)

    def latent_kv_expand_v(c, w_uv):
        """Expand the cached latent back to V: ``V = c @ W_uv``. Mirror
        of :func:`latent_kv_expand_k`."""
        if hasattr(c, "_data"):
            c = c._data
        if hasattr(w_uv, "_data"):
            w_uv = w_uv._data
        return np.matmul(c, w_uv)

    # attention_variants_plan, MLA-1 — fused decode op (numpy reference).
    # Result of the Schedule IR MLAFusionPass collapsing
    #   c = x @ W_dkv  → K = c @ W_uk  → V = c @ W_uv  → flash_attn(Q, K, V)
    # into a single op carrying (x, W_dkv, W_uk, W_uv, Q). On backends
    # without a fused absorb-K kernel this just expands back to the chain.
    def mla_decode_fused(x, w_dkv, w_uk, w_uv, q, *, scale=None,
                          causal: bool = False):
        """Fused MLA decode block (numpy reference).

        ``q`` shape: ``(B, S_q, D_q)`` (or rank-3 with H = 1 implicit).
        Returns ``O`` of the same shape as ``q``.
        """
        if hasattr(x, "_data"): x = x._data
        if hasattr(w_dkv, "_data"): w_dkv = w_dkv._data
        if hasattr(w_uk, "_data"): w_uk = w_uk._data
        if hasattr(w_uv, "_data"): w_uv = w_uv._data
        if hasattr(q, "_data"): q = q._data
        c = np.matmul(x, w_dkv)
        K = np.matmul(c, w_uk)
        V = np.matmul(c, w_uv)
        return flash_attn(q, K, V, scale=scale, causal=causal)

    def rope_split(x, *, rope_dim: int):
        """Split a tensor's last dim into ``(rope_part, no_rope_part)``.

        The first ``rope_dim`` channels along the last axis are returned
        as ``rope_part`` (these get RoPE applied); the rest are returned
        unchanged. Used by MLA's decoupled-RoPE design — the positional
        encoding only touches the rope_dim slice, so the compressed
        latent doesn't have to carry positional information.

        ``rope_dim`` must be ≤ ``x.shape[-1]``. Returns a 2-tuple.
        """
        if hasattr(x, "_data"):
            x = x._data
        x = np.asarray(x)
        if rope_dim < 0 or rope_dim > x.shape[-1]:
            raise ValueError(
                f"rope_dim must be in [0, {x.shape[-1]}]; got {rope_dim}"
            )
        return x[..., :rope_dim], x[..., rope_dim:]

    def rope_merge(rope_part, no_rope_part):
        """Inverse of :func:`rope_split` — concatenate the two parts along
        the last axis."""
        if hasattr(rope_part, "_data"):
            rope_part = rope_part._data
        if hasattr(no_rope_part, "_data"):
            no_rope_part = no_rope_part._data
        return np.concatenate(
            [np.asarray(rope_part), np.asarray(no_rope_part)], axis=-1
        )

    def depthwise_conv1d(x, w, *, kernel_size: int, padding: int = 0, causal: bool = False, state=None):
        """Depthwise 1-D convolution (one filter per channel).

        Inputs:
          * `x` shape `(N, C, L)`
          * `w` shape `(C, K)` — one filter per channel
          * `state` shape `(N, C, K-1)` or None — prepended to `x` for the conv,
            enabling chunked / streaming inference. Phase D1 of the
            execution roadmap.

        Output: `(N, C, L_out)` where `L_out = L_full - K + 1`,
        `L_full = L + (state.shape[-1] if state else 0) + (K-1 if causal else 2*padding)`.

        For streaming, the next-call state is
        `np.concatenate([state, x], axis=-1)[..., -(K-1):]` if `K > 1`.
        """
        if hasattr(x, "_data"):
            x = x._data
        if hasattr(w, "_data"):
            w = w._data
        x = np.asarray(x)
        w = np.asarray(w)
        if x.ndim != 3:
            raise ValueError(f"depthwise_conv1d expects (N, C, L) input; got shape {x.shape}")
        N, C, L = x.shape
        K = int(kernel_size)
        if w.shape != (C, K):
            raise ValueError(f"depthwise_conv1d weight shape {w.shape} does not match (C={C}, K={K})")

        if state is not None:
            if hasattr(state, "_data"):
                state = state._data
            state = np.asarray(state)
            if state.shape != (N, C, K - 1):
                raise ValueError(f"state shape {state.shape} does not match (N, C, K-1)=({N}, {C}, {K - 1})")
            x_full = np.concatenate([state, x], axis=-1)
        elif causal:
            x_full = np.pad(x, ((0, 0), (0, 0), (K - 1, 0)))
        else:
            x_full = np.pad(x, ((0, 0), (0, 0), (int(padding), int(padding))))

        L_out = x_full.shape[-1] - K + 1
        if L_out <= 0:
            raise ValueError(f"depthwise_conv1d: non-positive output length {L_out}")
        out = np.zeros((N, C, L_out), dtype=x.dtype)
        for k in range(K):
            out += x_full[..., k:k + L_out] * w[None, :, k:k + 1]
        return out

    def lstm_cell(x_t, h_prev, c_prev, W_ih, W_hh, b_ih=None, b_hh=None):
        """One-step LSTM cell. Returns packed ``(B, 2*hidden_size)`` of ``concat([h_t, c_t])``.

        Packing the two outputs in the last dim is the v1 workaround for the
        single-output autodiff tape. Use ``lstm_state_h`` / ``lstm_state_c`` to
        extract the parts under tape (slicing into the packed value isn't
        traced because numpy slicing returns a fresh ``ndarray`` object).

        Phase H2 of the execution roadmap — the state-propagation primitive
        for RNN cells.
        """
        if hasattr(x_t, "_data"):
            x_t = x_t._data
        if hasattr(h_prev, "_data"):
            h_prev = h_prev._data
        if hasattr(c_prev, "_data"):
            c_prev = c_prev._data
        if hasattr(W_ih, "_data"):
            W_ih = W_ih._data
        if hasattr(W_hh, "_data"):
            W_hh = W_hh._data
        x_t = np.asarray(x_t)
        h_prev = np.asarray(h_prev)
        c_prev = np.asarray(c_prev)
        W_ih = np.asarray(W_ih)
        W_hh = np.asarray(W_hh)
        H = h_prev.shape[-1]
        gates = x_t @ W_ih.T + h_prev @ W_hh.T
        if b_ih is not None:
            if hasattr(b_ih, "_data"):
                b_ih = b_ih._data
            gates = gates + np.asarray(b_ih)
        if b_hh is not None:
            if hasattr(b_hh, "_data"):
                b_hh = b_hh._data
            gates = gates + np.asarray(b_hh)
        i_g, f_g, g_g, o_g = (
            gates[..., :H], gates[..., H:2*H], gates[..., 2*H:3*H], gates[..., 3*H:4*H]
        )
        i = 1.0 / (1.0 + np.exp(-i_g))
        f = 1.0 / (1.0 + np.exp(-f_g))
        g = np.tanh(g_g)
        o = 1.0 / (1.0 + np.exp(-o_g))
        c_t = f * c_prev + i * g
        h_t = o * np.tanh(c_t)
        return np.concatenate([h_t, c_t], axis=-1)

    def lstm_state_h(packed):
        """Extract ``h_t`` from a packed lstm_cell output. Traced for autodiff."""
        if hasattr(packed, "_data"):
            packed = packed._data
        packed = np.asarray(packed)
        H = packed.shape[-1] // 2
        # Return a contiguous copy so id() is distinct from any view; the
        # autodiff wrapper records this op's output as the tape entry.
        return packed[..., :H].copy()

    def lstm_state_c(packed):
        """Extract ``c_t`` from a packed lstm_cell output. Traced for autodiff."""
        if hasattr(packed, "_data"):
            packed = packed._data
        packed = np.asarray(packed)
        H = packed.shape[-1] // 2
        return packed[..., H:].copy()

    def depthwise_conv2d(
        x, w, *, kernel_size, stride=(1, 1), padding=(0, 0), causal=False
    ):
        """Depthwise 2-D convolution (NHWC; one filter per channel).

        Inputs:
          * ``x`` shape ``(N, H, W, C)``
          * ``w`` shape ``(kH, kW, C)`` — one filter per channel
          * ``kernel_size`` int or ``(kH, kW)`` tuple
          * ``stride`` int or ``(sH, sW)`` tuple
          * ``padding`` int or ``(pH, pW)`` tuple — symmetric pad on H/W
          * ``causal`` — if True, pad only the top + left (kernel_size-1) edges
            so output[n, h, w, c] depends only on inputs at ``h' <= h`` and
            ``w' <= w``.

        Output: ``(N, H_out, W_out, C)``.

        D3 follow-up of the execution roadmap. Streaming-state variant
        (matching the D1 ``state=`` kwarg) is left for a follow-on; this v1
        is single-shot.
        """
        def _pair(v):
            if isinstance(v, (tuple, list)):
                return (int(v[0]), int(v[1]))
            return (int(v), int(v))

        kH, kW = _pair(kernel_size)
        sH, sW = _pair(stride)
        pH, pW = _pair(padding)

        if hasattr(x, "_data"):
            x = x._data
        if hasattr(w, "_data"):
            w = w._data
        x = np.asarray(x)
        w = np.asarray(w)
        if x.ndim != 4:
            raise ValueError(f"depthwise_conv2d expects (N, H, W, C); got {x.shape}")
        N, H, W, C = x.shape
        if w.shape != (kH, kW, C):
            raise ValueError(
                f"depthwise_conv2d weight shape {w.shape} must be (kH={kH}, kW={kW}, C={C})"
            )

        if causal:
            x_pad = np.pad(x, ((0, 0), (kH - 1, 0), (kW - 1, 0), (0, 0)))
        else:
            x_pad = np.pad(x, ((0, 0), (pH, pH), (pW, pW), (0, 0)))

        H_full = x_pad.shape[1]
        W_full = x_pad.shape[2]
        H_out = (H_full - kH) // sH + 1
        W_out = (W_full - kW) // sW + 1
        if H_out <= 0 or W_out <= 0:
            raise ValueError(
                f"depthwise_conv2d: non-positive output (H_out={H_out}, W_out={W_out})"
            )

        out = np.zeros((N, H_out, W_out, C), dtype=x.dtype)
        for kh in range(kH):
            for kw in range(kW):
                # x_pad[:, kh:kh+H_out*sH:sH, kw:kw+W_out*sW:sW, :] would do strides;
                # for clarity keep the simple case (stride=1) explicit.
                if sH == 1 and sW == 1:
                    patch = x_pad[:, kh:kh + H_out, kw:kw + W_out, :]
                else:
                    patch = x_pad[:, kh:kh + H_out * sH:sH, kw:kw + W_out * sW:sW, :]
                out += patch * w[None, kh:kh + 1, kw:kw + 1, :]
        return out

    def online_softmax(x, *, axis: int = -1, state=None):
        """Numerically stable softmax with optional streaming state.

        Two call patterns:

        1. **Single chunk (no state)** — equivalent to `tessera.ops.softmax`.
        2. **Streaming** — pass `state=(running_max, running_sum)` carried over
           from the previous chunk's :func:`tessera.ops.online_softmax_state`
           call. Returns the per-chunk softmax of `x` against the cumulative
           running stats.

        State protocol is `(running_max, running_sum)` numpy tensors over
        every dim except ``axis`` (which has size 1, kept dims). The narrow
        protocol matches v1; a future revision may switch to an opaque handle.

        Phase D2 of the execution roadmap.
        """
        if hasattr(x, "_data"):
            x = x._data
        x = np.asarray(x)
        if state is None:
            m = x.max(axis=axis, keepdims=True)
            e = np.exp(x - m)
            s = e.sum(axis=axis, keepdims=True)
            return e / s
        prev_m, prev_s = state
        prev_m = np.asarray(prev_m)
        prev_s = np.asarray(prev_s)
        chunk_m = x.max(axis=axis, keepdims=True)
        new_m = np.maximum(prev_m, chunk_m)
        new_s = prev_s * np.exp(prev_m - new_m) + np.exp(x - new_m).sum(axis=axis, keepdims=True)
        return np.exp(x - new_m) / new_s

    def selective_ssm(x, A, B, C, delta, *, gate=None, state=None, chunk_size: int = 128):
        """Mamba2-style selective state-space model (numpy reference path).

        Inputs:
          * ``x`` shape ``(B, S, D)`` — input sequence
          * ``A`` shape ``(D, N)`` or ``(D,)`` — state-matrix diagonal. If
            1-d, broadcast over ``N`` (taken from B/C). Typically negative.
          * ``B`` shape ``(B, S, N)`` — input projection
          * ``C`` shape ``(B, S, N)`` — output projection
          * ``delta`` shape ``(B, S, D)`` — selective time-step (controls
            per-token dynamics)
          * ``gate`` shape ``(B, S, D)`` or None — optional output gate
          * ``state`` shape ``(B, D, N)`` or None — optional initial state
            (for chunked / streaming inference)
          * ``chunk_size`` — internal chunked-scan size; affects numerical
            layout, not correctness

        Returns ``y`` shape ``(B, S, D)``.

        Algorithm (per token t, channel d, state dim n):
            A_bar = exp(delta[b,t,d] * A[d,n])
            B_bar = delta[b,t,d] * B[b,t,n]
            h[t,d,n] = A_bar * h[t-1,d,n] + B_bar * x[b,t,d]
            y[t,d] = sum_n C[b,t,n] * h[t,d,n]

        Phase D3 of the execution roadmap. **Forward-only in v1** — calling
        inside a tape and backpropping through it raises the standard
        ``TesseraAutodiffError`` pointing to
        ``tessera.autodiff.custom_rule("selective_ssm")``. The Mamba2 adjoint
        is on the Phase D3 follow-up list.
        """
        for arr_name, arr in (("x", x), ("A", A), ("B", B), ("C", C), ("delta", delta)):
            if hasattr(arr, "_data"):
                pass  # `np.asarray` below handles the unwrap via __array__
        x = np.asarray(x)
        A = np.asarray(A)
        B_arr = np.asarray(B)
        C_arr = np.asarray(C)
        delta = np.asarray(delta)

        if x.ndim != 3:
            raise ValueError(f"selective_ssm: x must be (B, S, D); got {x.shape}")
        Bsz, S, D = x.shape
        if delta.shape != (Bsz, S, D):
            raise ValueError(f"selective_ssm: delta {delta.shape} != x {x.shape}")
        if B_arr.ndim != 3 or B_arr.shape[:2] != (Bsz, S):
            raise ValueError(f"selective_ssm: B must be (B, S, N); got {B_arr.shape}")
        if C_arr.shape != B_arr.shape:
            raise ValueError(
                f"selective_ssm: C shape {C_arr.shape} must match B shape {B_arr.shape}"
            )
        N = B_arr.shape[2]

        if A.ndim == 1:
            if A.shape[0] != D:
                raise ValueError(f"selective_ssm: A 1-d shape {A.shape} != (D={D},)")
            A2d = np.broadcast_to(A[:, None], (D, N))
        elif A.ndim == 2:
            if A.shape != (D, N):
                raise ValueError(f"selective_ssm: A 2-d shape {A.shape} != (D={D}, N={N})")
            A2d = A
        else:
            raise ValueError(f"selective_ssm: A must be (D,) or (D, N); got {A.shape}")

        # Initial state h: (B, D, N)
        if state is not None:
            state = np.asarray(state)
            if state.shape != (Bsz, D, N):
                raise ValueError(
                    f"selective_ssm: state shape {state.shape} must be (B, D, N)=({Bsz}, {D}, {N})"
                )
            h = state.astype(x.dtype, copy=True)
        else:
            h = np.zeros((Bsz, D, N), dtype=x.dtype)

        y = np.zeros((Bsz, S, D), dtype=x.dtype)

        # Chunked scan — purely a layout choice, doesn't change semantics.
        for chunk_start in range(0, S, max(1, int(chunk_size))):
            chunk_end = min(S, chunk_start + max(1, int(chunk_size)))
            for t in range(chunk_start, chunk_end):
                # delta_t: (B, D) ; A2d: (D, N) → A_bar: (B, D, N)
                A_bar = np.exp(delta[:, t, :, None] * A2d[None, :, :])
                # B_bar: (B, D, N) — delta * B
                B_bar = delta[:, t, :, None] * B_arr[:, t, None, :]
                # x_t: (B, D) → broadcast to (B, D, N)
                h = A_bar * h + B_bar * x[:, t, :, None]
                # y_t = sum over n of C[b,t,n] * h[b,d,n]
                y[:, t, :] = np.einsum("bdn,bn->bd", h, C_arr[:, t, :])

        if gate is not None:
            gate = np.asarray(gate)
            if gate.shape != y.shape:
                raise ValueError(
                    f"selective_ssm: gate shape {gate.shape} must match output {y.shape}"
                )
            y = y * gate

        return y

    def online_softmax_state(x, *, axis: int = -1, state=None):
        """Compute the new ``(running_max, running_sum)`` state for streaming softmax.

        Pure helper — non-differentiable on purpose. Pair with
        :func:`tessera.ops.online_softmax` for chunked decoding:

        ::

            y_a = ops.online_softmax(x_a)
            state = ops.online_softmax_state(x_a)
            y_b = ops.online_softmax(x_b, state=state)
            state = ops.online_softmax_state(x_b, state=state)
            ...
        """
        if hasattr(x, "_data"):
            x = x._data
        x = np.asarray(x)
        if state is None:
            m = x.max(axis=axis, keepdims=True)
            e = np.exp(x - m)
            return (m, e.sum(axis=axis, keepdims=True))
        prev_m, prev_s = np.asarray(state[0]), np.asarray(state[1])
        chunk_m = x.max(axis=axis, keepdims=True)
        new_m = np.maximum(prev_m, chunk_m)
        new_s = prev_s * np.exp(prev_m - new_m) + np.exp(x - new_m).sum(axis=axis, keepdims=True)
        return (new_m, new_s)

    # ─────────────────────────────────────────────────────────────────────
    # S-series sprint S2 — reductions, stability primitives, numeric helpers,
    # and comparisons. Each is a numpy-reference op exposed via tessera.ops.*
    # so autodiff/jit can pick them up. VJPs land in autodiff/vjp.py.
    # ─────────────────────────────────────────────────────────────────────

    def _unwrap(x):
        return np.asarray(x._data if hasattr(x, "_data") else x)

    # Reductions ----------------------------------------------------------
    def mean(x, axis=None, keepdims: bool = False):
        return np.mean(_unwrap(x), axis=axis, keepdims=keepdims)

    def prod(x, axis=None, keepdims: bool = False):
        return np.prod(_unwrap(x), axis=axis, keepdims=keepdims)

    def amax(x, axis=None, keepdims: bool = False):
        return np.max(_unwrap(x), axis=axis, keepdims=keepdims)

    def amin(x, axis=None, keepdims: bool = False):
        return np.min(_unwrap(x), axis=axis, keepdims=keepdims)

    def max_reduce(x, axis=None, keepdims: bool = False):
        return amax(x, axis=axis, keepdims=keepdims)

    def min_reduce(x, axis=None, keepdims: bool = False):
        return amin(x, axis=axis, keepdims=keepdims)

    def var(x, axis=None, keepdims: bool = False, ddof: int = 0):
        return np.var(_unwrap(x), axis=axis, keepdims=keepdims, ddof=ddof)

    def std(x, axis=None, keepdims: bool = False, ddof: int = 0):
        return np.std(_unwrap(x), axis=axis, keepdims=keepdims, ddof=ddof)

    def argmax(x, axis: int | None = None, keepdims: bool = False):
        return np.argmax(_unwrap(x), axis=axis, keepdims=keepdims)

    def argmin(x, axis: int | None = None, keepdims: bool = False):
        return np.argmin(_unwrap(x), axis=axis, keepdims=keepdims)

    def cumsum(x, axis: int = -1):
        return np.cumsum(_unwrap(x), axis=axis)

    def cumprod(x, axis: int = -1):
        return np.cumprod(_unwrap(x), axis=axis)

    def cummax(x, axis: int = -1):
        return np.maximum.accumulate(_unwrap(x), axis=axis)

    def cummin(x, axis: int = -1):
        return np.minimum.accumulate(_unwrap(x), axis=axis)

    # Numerical-stability primitives -------------------------------------
    def logsumexp(x, axis=None, keepdims: bool = False):
        a = _unwrap(x)
        m = np.max(a, axis=axis, keepdims=True)
        # `m` may be -inf for an all-(-inf) row; clamp to avoid nan in exp(a-m).
        m_safe = np.where(np.isfinite(m), m, np.zeros_like(m))
        out = np.log(np.sum(np.exp(a - m_safe), axis=axis, keepdims=True)) + m_safe
        if not keepdims:
            out = np.squeeze(out, axis=axis) if axis is not None else out.reshape(())
        return out

    def log_softmax(x, axis: int = -1):
        a = _unwrap(x)
        m = np.max(a, axis=axis, keepdims=True)
        shifted = a - m
        return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True))

    def log1p(x):
        return np.log1p(_unwrap(x))

    def expm1(x):
        return np.expm1(_unwrap(x))

    def softplus(x):
        # log(1 + exp(x)) computed as max(x,0) + log1p(exp(-|x|)) for stability.
        a = _unwrap(x)
        return np.maximum(a, 0) + np.log1p(np.exp(-np.abs(a)))

    def sigmoid_safe(x):
        # 1/(1+exp(-x)) computed branch-wise so neither branch overflows.
        # For x >= 0: 1 / (1 + exp(-x))     (uses exp of negative arg).
        # For x <  0: exp(x) / (1 + exp(x)) (uses exp of negative arg).
        a = _unwrap(x)
        return np.where(
            a >= 0,
            1.0 / (1.0 + np.exp(-np.abs(a))),
            np.exp(-np.abs(a)) / (1.0 + np.exp(-np.abs(a))),
        )

    # Scalar math breadth -------------------------------------------------
    def sub(x, y):
        return np.subtract(_unwrap(x), _unwrap(y))

    def div(x, y):
        return np.divide(_unwrap(x), _unwrap(y))

    def floor_div(x, y):
        return np.floor_divide(_unwrap(x), _unwrap(y))

    def mod(x, y):
        return np.mod(_unwrap(x), _unwrap(y))

    def exp(x):
        return np.exp(_unwrap(x))

    def log(x):
        return np.log(_unwrap(x))

    def sqrt(x):
        return np.sqrt(_unwrap(x))

    def rsqrt(x):
        return np.reciprocal(np.sqrt(_unwrap(x)))

    def pow(x, y):
        return np.power(_unwrap(x), _unwrap(y))

    def cos(x):
        return np.cos(_unwrap(x))

    def tan(x):
        return np.tan(_unwrap(x))

    def sinh(x):
        return np.sinh(_unwrap(x))

    def cosh(x):
        return np.cosh(_unwrap(x))

    def asin(x):
        return np.arcsin(_unwrap(x))

    def acos(x):
        return np.arccos(_unwrap(x))

    def atan(x):
        return np.arctan(_unwrap(x))

    def atan2(y, x):
        return np.arctan2(_unwrap(y), _unwrap(x))

    def erf(x):
        # NumPy 2.0+ no longer guarantees np.erf; vectorize math.erf to keep
        # the reference path dependency-light.
        a = _unwrap(x)
        return np.vectorize(math.erf, otypes=[np.float64])(a).astype(a.dtype, copy=False)

    def erfc(x):
        a = _unwrap(x)
        return np.vectorize(math.erfc, otypes=[np.float64])(a).astype(a.dtype, copy=False)

    def lgamma(x):
        a = _unwrap(x)
        return np.vectorize(math.lgamma, otypes=[np.float64])(a).astype(a.dtype, copy=False)

    def _digamma_scalar(x):
        # Reflection formula handles negative non-integers; poles return nan.
        x = float(x)
        if x <= 0.0:
            if abs(x - round(x)) < 1e-12:
                return float("nan")
            return _digamma_scalar(1.0 - x) - math.pi / math.tan(math.pi * x)
        result = 0.0
        while x < 8.0:
            result -= 1.0 / x
            x += 1.0
        inv = 1.0 / x
        inv2 = inv * inv
        # Asymptotic expansion: psi(x) = log(x) - 1/(2x) - 1/(12x^2)
        # + 1/(120x^4) - 1/(252x^6) + 1/(240x^8) ...
        return (
            result
            + math.log(x)
            - 0.5 * inv
            - inv2 / 12.0
            + inv2 * inv2 / 120.0
            - inv2 * inv2 * inv2 / 252.0
            + inv2 * inv2 * inv2 * inv2 / 240.0
        )

    def digamma(x):
        a = _unwrap(x)
        return np.vectorize(_digamma_scalar, otypes=[np.float64])(a).astype(a.dtype, copy=False)

    # Numeric helpers + comparisons --------------------------------------
    def clamp(x, min=None, max=None):
        a = _unwrap(x)
        if min is None and max is None:
            return a
        return np.clip(a, a_min=min, a_max=max)

    def where(cond, x, y):
        return np.where(_unwrap(cond), _unwrap(x), _unwrap(y))

    def absolute(x):
        return np.abs(_unwrap(x))

    def abs(x):
        return absolute(x)

    def sign(x):
        return np.sign(_unwrap(x))

    def reciprocal(x):
        return np.reciprocal(_unwrap(x))

    def floor(x):
        return np.floor(_unwrap(x))

    def ceil(x):
        return np.ceil(_unwrap(x))

    def round(x):
        return np.round(_unwrap(x))

    def trunc(x):
        return np.trunc(_unwrap(x))

    def minimum(x, y):
        return np.minimum(_unwrap(x), _unwrap(y))

    def maximum(x, y):
        return np.maximum(_unwrap(x), _unwrap(y))

    def isnan(x):
        return np.isnan(_unwrap(x))

    def isinf(x):
        return np.isinf(_unwrap(x))

    def isfinite(x):
        return np.isfinite(_unwrap(x))

    def eq(x, y):
        return np.equal(_unwrap(x), _unwrap(y))

    def ne(x, y):
        return np.not_equal(_unwrap(x), _unwrap(y))

    def lt(x, y):
        return np.less(_unwrap(x), _unwrap(y))

    def le(x, y):
        return np.less_equal(_unwrap(x), _unwrap(y))

    def gt(x, y):
        return np.greater(_unwrap(x), _unwrap(y))

    def ge(x, y):
        return np.greater_equal(_unwrap(x), _unwrap(y))

    def logical_and(x, y):
        return np.logical_and(_unwrap(x), _unwrap(y))

    def logical_or(x, y):
        return np.logical_or(_unwrap(x), _unwrap(y))

    def logical_not(x):
        return np.logical_not(_unwrap(x))

    def logical_xor(x, y):
        return np.logical_xor(_unwrap(x), _unwrap(y))

    def bitwise_and(x, y):
        return np.bitwise_and(_unwrap(x), _unwrap(y))

    def bitwise_or(x, y):
        return np.bitwise_or(_unwrap(x), _unwrap(y))

    def bitwise_xor(x, y):
        return np.bitwise_xor(_unwrap(x), _unwrap(y))

    def bitwise_not(x):
        return np.bitwise_not(_unwrap(x))

    # Tensor algebra + indexing -----------------------------------------
    def reshape(x, shape):
        return np.reshape(_unwrap(x), tuple(shape))

    def view(x, shape):
        return reshape(x, shape)

    def flatten(x, start_axis: int = 0, end_axis: int = -1):
        a = _unwrap(x)
        ndim = a.ndim
        start = start_axis if start_axis >= 0 else ndim + start_axis
        end = end_axis if end_axis >= 0 else ndim + end_axis
        if start < 0 or end >= ndim or start > end:
            raise ValueError(f"invalid flatten axes {start_axis}, {end_axis} for rank {ndim}")
        flat = int(np.prod(a.shape[start:end + 1]))
        return np.reshape(a, a.shape[:start] + (flat,) + a.shape[end + 1:])

    def squeeze(x, axis=None):
        return np.squeeze(_unwrap(x), axis=axis)

    def unsqueeze(x, axis: int):
        return np.expand_dims(_unwrap(x), axis=axis)

    def permute(x, axes):
        return np.transpose(_unwrap(x), axes=tuple(axes))

    def broadcast(x, shape):
        return np.broadcast_to(_unwrap(x), tuple(shape))

    def expand(x, shape):
        return broadcast(x, shape)

    def cat(xs, axis: int = 0):
        return np.concatenate([_unwrap(x) for x in xs], axis=axis)

    def stack(xs, axis: int = 0):
        return np.stack([_unwrap(x) for x in xs], axis=axis)

    def split(x, indices_or_sections, axis: int = 0):
        return tuple(np.array_split(_unwrap(x), indices_or_sections, axis=axis)
                     if isinstance(indices_or_sections, int)
                     else np.split(_unwrap(x), indices_or_sections, axis=axis))

    def chunk(x, chunks: int, axis: int = 0):
        return tuple(np.array_split(_unwrap(x), chunks, axis=axis))

    def pad(x, pad_width, mode: str = "constant", constant_values=0):
        a = _unwrap(x)
        # numpy's `pad` is heavily overloaded on the ``mode`` literal;
        # we accept a free-form string at this layer (validated by
        # numpy at runtime).  Cast through ``Any`` for mypy.
        np_pad: Any = np.pad
        if mode == "constant":
            return np_pad(a, pad_width, mode=mode, constant_values=constant_values)
        return np_pad(a, pad_width, mode=mode)

    def tile(x, reps):
        return np.tile(_unwrap(x), reps)

    def repeat(x, repeats, axis=None):
        return np.repeat(_unwrap(x), repeats, axis=axis)

    def roll(x, shift, axis=None):
        return np.roll(_unwrap(x), shift=shift, axis=axis)

    def flip(x, axis=None):
        return np.flip(_unwrap(x), axis=axis)

    def dynamic_slice(x, start_indices, slice_sizes):
        a = _unwrap(x)
        if len(start_indices) != a.ndim or len(slice_sizes) != a.ndim:
            raise ValueError("start_indices and slice_sizes must match input rank")
        slices = tuple(
            builtins.slice(int(start), int(start) + int(size))
            for start, size in zip(start_indices, slice_sizes)
        )
        return a[slices]

    def slice_op(x, start_indices, slice_sizes):
        return dynamic_slice(x, start_indices, slice_sizes)

    def select(x, index: int, axis: int = 0):
        return np.take(_unwrap(x), int(index), axis=axis)

    def dynamic_update_slice(x, update, start_indices):
        a = np.array(_unwrap(x), copy=True)
        u = _unwrap(update)
        if len(start_indices) != a.ndim:
            raise ValueError("start_indices must match input rank")
        slices = tuple(
            builtins.slice(int(start), int(start) + int(size))
            for start, size in zip(start_indices, u.shape)
        )
        a[slices] = u
        return a

    def take(x, indices, axis: int | None = None):
        return np.take(_unwrap(x), _unwrap(indices).astype(np.int64), axis=axis)

    def index_select(x, indices, axis: int = 0):
        return take(x, indices, axis=axis)

    def _scatter_base(x, indices, updates, *, axis: int = 0, mode: str = "set", reduce: str = "sum"):
        out = np.array(_unwrap(x), copy=True)
        idx = _unwrap(indices).astype(np.int64)
        upd = _unwrap(updates)
        ax = axis if axis >= 0 else out.ndim + axis
        out_m = np.moveaxis(out, ax, 0)
        upd_m = np.moveaxis(upd, ax, 0) if upd.ndim == out.ndim else upd
        if idx.ndim == 1:
            if mode == "set":
                out_m[idx] = upd_m
            elif reduce == "sum":
                np.add.at(out_m, idx, upd_m)
            elif reduce == "min":
                np.minimum.at(out_m, idx, upd_m)
            elif reduce == "max":
                np.maximum.at(out_m, idx, upd_m)
            else:
                raise ValueError(f"unsupported scatter_reduce reduce={reduce!r}")
        else:
            if mode == "set":
                np.put_along_axis(out, idx, upd, axis=ax)
                return out
            if reduce != "sum":
                raise ValueError("non-1D scatter_reduce only supports reduce='sum' today")
            np.add.at(out_m, idx, upd_m)
        return np.moveaxis(out_m, 0, ax)

    def scatter(x, indices, updates, axis: int = 0):
        return _scatter_base(x, indices, updates, axis=axis, mode="set")

    def scatter_add(x, indices, updates, axis: int = 0):
        return _scatter_base(x, indices, updates, axis=axis, mode="add", reduce="sum")

    def scatter_reduce(x, indices, updates, axis: int = 0, reduce: str = "sum"):
        return _scatter_base(x, indices, updates, axis=axis, mode="add", reduce=reduce)

    def index_update(x, indices, updates, axis: int = 0):
        return scatter(x, indices, updates, axis=axis)

    def nonzero(x):
        return np.nonzero(_unwrap(x))

    def count_nonzero(x, axis=None, keepdims: bool = False):
        """Count of non-zero / truthy elements along ``axis`` (LDT candidate
        cardinality). Reduction over a boolean predicate; version-independent."""
        a = np.asarray(_unwrap(x))
        return (a != 0).sum(axis=axis, keepdims=keepdims)

    def popcount(x):
        """Per-element population count (number of set bits) of an integer
        tensor — e.g. candidates-remaining for a bitmask-encoded lattice cell.
        Elementwise, shape-preserving, non-negative integers."""
        a = np.asarray(_unwrap(x))
        if hasattr(np, "bitwise_count"):          # numpy >= 2.0 fast path
            return np.asarray(np.bitwise_count(a))
        v = a.astype(np.uint64, copy=True)        # numpy < 2.0 masking fallback
        out = np.zeros(v.shape, dtype=np.int64)
        while np.any(v):
            out += (v & np.uint64(1)).astype(np.int64)
            v >>= np.uint64(1)
        return out

    def masked_categorical(logits, mask, key=None, axis: int = -1):
        """Categorical decision over ``logits`` restricted to a candidate
        ``mask`` (masked-out positions get ``-inf`` so they're never chosen).
        ``key=None`` → deterministic greedy ``argmax`` (testable); a seed/key →
        a Gumbel-max sample over the surviving candidates. Returns indices
        (non-differentiable)."""
        z = np.asarray(_unwrap(logits), dtype=np.float64)
        m = np.asarray(_unwrap(mask)).astype(bool)
        masked = np.where(m, z, -np.inf)
        if key is None:
            return np.argmax(masked, axis=axis)
        seed = key if isinstance(key, (int, np.integer)) else (abs(hash(key)) % (2**32))
        rng = np.random.default_rng(int(seed))
        g = -np.log(-np.log(rng.random(size=masked.shape) + 1e-20) + 1e-20)
        return np.argmax(np.where(np.isfinite(masked), masked + g, -np.inf), axis=axis)

    def top_k(x, k: int, axis: int = -1):
        a = _unwrap(x)
        idx = np.argsort(a, axis=axis)
        idx = np.take(idx, np.arange(a.shape[axis] - k, a.shape[axis]), axis=axis)
        idx = np.flip(idx, axis=axis)
        values = np.take_along_axis(a, idx, axis=axis)
        return values, idx

    def sort(x, axis: int = -1, descending: bool = False):
        out = np.sort(_unwrap(x), axis=axis)
        return np.flip(out, axis=axis) if descending else out

    def argsort(x, axis: int = -1, descending: bool = False):
        out = np.argsort(_unwrap(x), axis=axis)
        return np.flip(out, axis=axis) if descending else out

    def _lazy_module_fn(module_name: str, fn_name: str):
        def _wrapper(*args, **kwargs):
            module = __import__(f"tessera.{module_name}", fromlist=[fn_name])
            return getattr(module, fn_name)(*args, **kwargs)
        return _wrapper

    linear_general_ref = _lazy_module_fn("nn.functional", "linear_general")
    alibi_ref = _lazy_module_fn("nn.functional", "alibi")
    ntk_rope_ref = _lazy_module_fn("nn.functional", "ntk_rope")
    multi_head_attention_ref = _lazy_module_fn("nn.functional", "multi_head_attention")
    gqa_attention_ref = _lazy_module_fn("nn.functional", "gqa_attention")
    mqa_attention_ref = _lazy_module_fn("nn.functional", "mqa_attention")
    mla_decode_ref = _lazy_module_fn("nn.functional", "mla_decode")
    fake_quantize_ref = _lazy_module_fn("quantization", "fake_quantize")
    sgd_ref = _lazy_module_fn("optim", "sgd")
    mse_loss_ref = _lazy_module_fn("losses", "mse_loss")
    mae_loss_ref = _lazy_module_fn("losses", "mae_loss")
    huber_loss_ref = _lazy_module_fn("losses", "huber_loss")
    smooth_l1_loss_ref = _lazy_module_fn("losses", "smooth_l1_loss")
    log_cosh_loss_ref = _lazy_module_fn("losses", "log_cosh_loss")
    cross_entropy_loss_ref = _lazy_module_fn("losses", "cross_entropy_loss")
    binary_cross_entropy_loss_ref = _lazy_module_fn("losses", "binary_cross_entropy_loss")
    asymmetric_bce_ref = _lazy_module_fn("losses", "asymmetric_bce")
    z_loss_ref = _lazy_module_fn("losses", "z_loss")
    load_balance_loss_ref = _lazy_module_fn("losses", "load_balance_loss")
    ddpm_noise_pred_loss_ref = _lazy_module_fn("losses", "ddpm_noise_pred_loss")
    score_matching_loss_ref = _lazy_module_fn("losses", "score_matching_loss")
    # EBM training losses (#5) — CD / PCD / ISM / DSM.
    contrastive_divergence_loss_ref = _lazy_module_fn("losses", "contrastive_divergence_loss")
    persistent_cd_loss_ref = _lazy_module_fn("losses", "persistent_cd_loss")
    implicit_score_matching_loss_ref = _lazy_module_fn("losses", "implicit_score_matching_loss")
    denoising_score_matching_loss_ref = _lazy_module_fn("losses", "denoising_score_matching_loss")
    vlb_loss_ref = _lazy_module_fn("losses", "vlb_loss")
    kl_divergence_ref = _lazy_module_fn("losses", "kl_divergence")
    js_divergence_ref = _lazy_module_fn("losses", "js_divergence")
    normalize_group_advantages_ref = _lazy_module_fn("rl", "normalize_group_advantages")
    ppo_policy_loss_ref = _lazy_module_fn("rl", "ppo_policy_loss")
    grpo_policy_loss_ref = _lazy_module_fn("rl", "grpo_policy_loss")
    cispo_policy_loss_ref = _lazy_module_fn("rl", "cispo_policy_loss")

    references = {
        "gemm": gemm,
        "matmul": matmul,
        "batched_gemm": batched_gemm,
        "einsum": einsum,
        "factorized_matmul": factorized_matmul,
        "grouped_gemm": grouped_gemm,
        "tri_solve": tri_solve,
        "cholesky": cholesky,
        "cholesky_solve": cholesky_solve,
        "lu": lu,
        "qr": qr,
        "svd": svd,
        "conv2d": conv2d,
        "conv3d": conv3d,
        "layer_norm": layer_norm,
        "softmax": softmax,
        "softmax_safe": softmax_safe,
        "reduce": reduce,
        "sum": sum,
        "gelu": gelu,
        "tanh": tanh,
        "add": add,
        "mul": mul,
        "relu": relu,
        "sigmoid": sigmoid,
        "sin": sin,
        "silu": silu,
        "silu_mul": silu_mul,
        "arange": arange,
        "gather": gather,
        "clip": clip,
        "masked_fill": masked_fill,
        "adam": adam,
        "transpose": transpose,
        "cast": cast,
        "dropout": dropout,
        "qkv_projection": qkv_projection,
        "flash_attn": flash_attn,
        "linear_attn": linear_attn,
        "linear_attn_state": linear_attn_state,
        "power_attn": power_attn,
        "retention": retention,
        "attn_sliding_window": attn_sliding_window,
        # Gap 4 (2026-05-20): 2D spatial-grid local-window attention.
        "attn_local_window_2d": attn_local_window_2d,
        "attn_compressed_blocks": attn_compressed_blocks,
        "attn_top_k_blocks": attn_top_k_blocks,
        # Phase F-MoR — Mixture of Recursions primitives.
        "mor_router": mor_router,
        "mor_partition": mor_partition,
        "mor_scatter": mor_scatter,
        "moe": moe,
        "moe_dispatch": moe_dispatch,
        "moe_combine": moe_combine,
        "all_reduce": all_reduce,
        "reduce_scatter": reduce_scatter,
        "all_gather": all_gather,
        "all_to_all": all_to_all,
        "rng_uniform": rng_uniform,
        "rng_normal": rng_normal,
        "fused_epilogue": fused_epilogue,
        "fft": fft,
        "ifft": ifft,
        "rfft": rfft,
        "irfft": irfft,
        "stft": stft,
        "istft": istft,
        "spectral_filter": spectral_filter,
        "dct": dct,
        "spectral_conv": spectral_conv,
        "spmm_coo": spmm_coo,
        "spmm_csr": spmm_csr,
        "sddmm": sddmm,
        "bsmm": bsmm,
        "segment_reduce": segment_reduce,
        "rearrange": rearrange,
        "pack": pack,
        "unpack": unpack,
        "tile_view": tile_view,
        "rmsnorm": rmsnorm,
        "rmsnorm_safe": rmsnorm_safe,
        "rope": rope,
        "kv_cache_append": kv_cache_append,
        "kv_cache_update": kv_cache_update,
        "kv_cache_prune": kv_cache_prune,
        "kv_cache_read": kv_cache_read,
        "depthwise_conv1d": depthwise_conv1d,
        "depthwise_conv2d": depthwise_conv2d,
        "lstm_cell": lstm_cell,
        "lstm_state_h": lstm_state_h,
        "lstm_state_c": lstm_state_c,
        "online_softmax": online_softmax,
        "online_softmax_state": online_softmax_state,
        "quantize_kv": quantize_kv,
        "dequantize_kv": dequantize_kv,
        "quantize_fp8": quantize_fp8,
        "dequantize_fp8": dequantize_fp8,
        "quantize_fp6": quantize_fp6,
        "dequantize_fp6": dequantize_fp6,
        "quantize_fp4": quantize_fp4,
        "dequantize_fp4": dequantize_fp4,
        "quantize_nvfp4": quantize_nvfp4,
        "dequantize_nvfp4": dequantize_nvfp4,
        "latent_kv_compress": latent_kv_compress,
        "latent_kv_expand_k": latent_kv_expand_k,
        "latent_kv_expand_v": latent_kv_expand_v,
        "mla_decode_fused": mla_decode_fused,
        "rope_split": rope_split,
        "rope_merge": rope_merge,
        "alibi": alibi_ref,
        "ntk_rope": ntk_rope_ref,
        "multi_head_attention": multi_head_attention_ref,
        "gqa_attention": gqa_attention_ref,
        "mqa_attention": mqa_attention_ref,
        "mla_decode": mla_decode_ref,
        "fake_quantize": fake_quantize_ref,
        "selective_ssm": selective_ssm,
        # S-series sprint S2 — reductions
        "mean": mean,
        "prod": prod,
        "amax": amax,
        "amin": amin,
        "max": max_reduce,
        "min": min_reduce,
        "var": var,
        "std": std,
        "argmax": argmax,
        "argmin": argmin,
        "cumsum": cumsum,
        "cumprod": cumprod,
        "cummax": cummax,
        "cummin": cummin,
        # S2 — numerical-stability primitives
        "logsumexp": logsumexp,
        "log_softmax": log_softmax,
        "log1p": log1p,
        "expm1": expm1,
        "softplus": softplus,
        "sigmoid_safe": sigmoid_safe,
        "sub": sub,
        "div": div,
        "floor_div": floor_div,
        "mod": mod,
        "exp": exp,
        "log": log,
        "sqrt": sqrt,
        "rsqrt": rsqrt,
        "pow": pow,
        "cos": cos,
        "tan": tan,
        "sinh": sinh,
        "cosh": cosh,
        "asin": asin,
        "acos": acos,
        "atan": atan,
        "atan2": atan2,
        "erf": erf,
        "erfc": erfc,
        "lgamma": lgamma,
        "digamma": digamma,
        # S2 — numeric helpers
        "clamp": clamp,
        "where": where,
        "absolute": absolute,
        "abs": abs,
        "sign": sign,
        "reciprocal": reciprocal,
        "floor": floor,
        "ceil": ceil,
        "round": round,
        "trunc": trunc,
        "minimum": minimum,
        "maximum": maximum,
        "isnan": isnan,
        "isinf": isinf,
        "isfinite": isfinite,
        # S2 — comparisons
        "eq": eq,
        "ne": ne,
        "lt": lt,
        "le": le,
        "gt": gt,
        "ge": ge,
        # S2 — logical / bitwise
        "logical_and": logical_and,
        "logical_or": logical_or,
        "logical_not": logical_not,
        "logical_xor": logical_xor,
        "bitwise_and": bitwise_and,
        "bitwise_or": bitwise_or,
        "bitwise_xor": bitwise_xor,
        "bitwise_not": bitwise_not,
        # S2 — tensor algebra + indexing
        "reshape": reshape,
        "view": view,
        "flatten": flatten,
        "squeeze": squeeze,
        "unsqueeze": unsqueeze,
        "permute": permute,
        "broadcast": broadcast,
        "expand": expand,
        "cat": cat,
        "stack": stack,
        "split": split,
        "chunk": chunk,
        "pad": pad,
        "tile": tile,
        "repeat": repeat,
        "roll": roll,
        "flip": flip,
        "slice": slice_op,
        "select": select,
        "dynamic_slice": dynamic_slice,
        "dynamic_update_slice": dynamic_update_slice,
        "take": take,
        "index_select": index_select,
        "scatter": scatter,
        "scatter_add": scatter_add,
        "scatter_reduce": scatter_reduce,
        "index_update": index_update,
        "nonzero": nonzero,
        "count_nonzero": count_nonzero,
        "popcount": popcount,
        "masked_categorical": masked_categorical,
        "top_k": top_k,
        "sort": sort,
        "argsort": argsort,
        "linear_general": linear_general_ref,
        "sgd": sgd_ref,
        "mse_loss": mse_loss_ref,
        "mae_loss": mae_loss_ref,
        "huber_loss": huber_loss_ref,
        "smooth_l1_loss": smooth_l1_loss_ref,
        "log_cosh_loss": log_cosh_loss_ref,
        "cross_entropy_loss": cross_entropy_loss_ref,
        "binary_cross_entropy_loss": binary_cross_entropy_loss_ref,
        "kl_divergence": kl_divergence_ref,
        "js_divergence": js_divergence_ref,
        "asymmetric_bce": asymmetric_bce_ref,
        "z_loss": z_loss_ref,
        "load_balance_loss": load_balance_loss_ref,
        "ddpm_noise_pred_loss": ddpm_noise_pred_loss_ref,
        "score_matching_loss": score_matching_loss_ref,
        "contrastive_divergence_loss": contrastive_divergence_loss_ref,
        "persistent_cd_loss": persistent_cd_loss_ref,
        "implicit_score_matching_loss": implicit_score_matching_loss_ref,
        "denoising_score_matching_loss": denoising_score_matching_loss_ref,
        "vlb_loss": vlb_loss_ref,
        "normalize_group_advantages": normalize_group_advantages_ref,
        "ppo_policy_loss": ppo_policy_loss_ref,
        "grpo_policy_loss": grpo_policy_loss_ref,
        "cispo_policy_loss": cispo_policy_loss_ref,
        "adamw": adamw,
        "momentum": momentum,
        "adafactor": adafactor,
        "lion": lion,
        "gated_attention": gated_attention,
        "hybrid_attention": hybrid_attention,
        "deepseek_sparse_attention": deepseek_sparse_attention,
        "lightning_attention": lightning_attention,
        "gated_deltanet": gated_deltanet,
        "kimi_delta_attention": kimi_delta_attention,
        "modified_delta_attention": modified_delta_attention,
    }
    # E3 (2026-05-20) — M7 Visual Complex Analysis numpy reference
    # surface.  All 20 M7 ops live in ``tessera.complex.*`` and have
    # closed-form numpy implementations today; wire them into the
    # ops registry so ``tessera.ops.registry.list()`` reports them
    # alongside the rest of the OP_SPECS catalog.
    from tessera import complex as _ts_complex  # noqa: E402
    m7_references = {
        "complex_mul":                _ts_complex.complex_mul,
        "complex_div":                _ts_complex.complex_div,
        "complex_exp":                _ts_complex.complex_exp,
        "complex_log":                _ts_complex.complex_log,
        "complex_sqrt":               _ts_complex.complex_sqrt,
        "complex_pow":                _ts_complex.complex_pow,
        "complex_conjugate":          _ts_complex.complex_conjugate,
        "complex_abs":                _ts_complex.complex_abs,
        "complex_arg":                _ts_complex.complex_arg,
        "mobius":                     _ts_complex.mobius,
        "mobius_from_three_points":   _ts_complex.mobius_from_three_points,
        "stereographic":              _ts_complex.stereographic,
        "cross_ratio":                _ts_complex.cross_ratio,
        "is_concyclic":               _ts_complex.is_concyclic,
        "check_cauchy_riemann":       _ts_complex.check_cauchy_riemann,
        "dz":                         _ts_complex.dz,
        "dbar":                       _ts_complex.dbar,
        "laplacian_2d":               _ts_complex.laplacian_2d,
        "conformal_jacobian":         _ts_complex.conformal_jacobian,
        "conformal_energy_on_sphere": _ts_complex.conformal_energy_on_sphere,
    }
    references.update(m7_references)
    # ``ebm_energy_quadratic`` / ``ebm_langevin_step`` were added to
    # ``OP_SPECS`` (Graph IR ODS + verifiers, 2026) but their numpy
    # references were never wired into the ops registry, leaving them in
    # the catalog yet absent from ``tessera.ops.registry.list()`` and
    # ``PYTHON_API_SPEC.md``.  Register the real EBM reference impls so
    # the OP_SPECS ⊆ registry invariant holds.
    from tessera import ebm as _ts_ebm  # noqa: E402
    references.update({
        "ebm_energy_quadratic": _ts_ebm.energy_quadratic,
        "ebm_langevin_step": _ts_ebm.langevin_step,
    })
    # Canonical GA shim: tessera.ops.clifford_* flat-coefficient wrappers over the
    # tessera.ga.* Multivector lane (which already GPU-dispatches to the cl30
    # kernels). Registering here puts them on the tessera.ops surface AND through
    # the autodiff tape (so their VJP/JVP in autodiff/{vjp,jvp}.py are honored).
    from . import _clifford_ops as _clifford_ops_mod
    references.update(_clifford_ops_mod.CLIFFORD_OPS)
    # Canonical EBM shim: tessera.ops.ebm_* flat-array wrappers over the
    # tessera.ebm.* lane (tensor-clean subset; several GPU-dispatch to dedicated
    # MSL kernels). Same unification as the clifford shim — onto tessera.ops +
    # through the tape so their VJP/JVP in autodiff/{vjp,jvp}.py are honored.
    from . import _ebm_ops as _ebm_ops_mod
    references.update(_ebm_ops_mod.EBM_OPS)
    for op_name, fn in references.items():
        _register_reference(op_name, fn, backend="numpy")
        _register_lowering(op_name, lambda *args, _op=op_name, **kwargs: {"op": _op, "status": "artifact_only"}, backend="graph_ir")

    _ns = types.SimpleNamespace(
        clifford_geometric_product=_clifford_ops_mod.clifford_geometric_product,
        clifford_wedge=_clifford_ops_mod.clifford_wedge,
        clifford_left_contraction=_clifford_ops_mod.clifford_left_contraction,
        clifford_inner=_clifford_ops_mod.clifford_inner,
        clifford_rotor_sandwich=_clifford_ops_mod.clifford_rotor_sandwich,
        clifford_reverse=_clifford_ops_mod.clifford_reverse,
        clifford_grade_involution=_clifford_ops_mod.clifford_grade_involution,
        clifford_conjugate=_clifford_ops_mod.clifford_conjugate,
        clifford_grade_projection=_clifford_ops_mod.clifford_grade_projection,
        clifford_hodge_star=_clifford_ops_mod.clifford_hodge_star,
        clifford_ext_deriv=_clifford_ops_mod.clifford_ext_deriv,
        clifford_vec_deriv=_clifford_ops_mod.clifford_vec_deriv,
        clifford_codiff=_clifford_ops_mod.clifford_codiff,
        clifford_exp=_clifford_ops_mod.clifford_exp,
        clifford_log=_clifford_ops_mod.clifford_log,
        clifford_norm=_clifford_ops_mod.clifford_norm,
        clifford_norm_squared=_clifford_ops_mod.clifford_norm_squared,
        ebm_energy_quadratic=_ebm_ops_mod.ebm_energy_quadratic,
        ebm_self_verify=_ebm_ops_mod.ebm_self_verify,
        ebm_refinement=_ebm_ops_mod.ebm_refinement,
        ebm_inner_step=_ebm_ops_mod.ebm_inner_step,
        registry=_ops_registry,
        register_reference=_register_reference,
        register_lowering=_register_lowering,
        register_runtime_kernel=_register_runtime_kernel,
        gemm=gemm,
        matmul=matmul,
        batched_gemm=batched_gemm,
        einsum=einsum,
        factorized_matmul=factorized_matmul,
        grouped_gemm=grouped_gemm,
        tri_solve=tri_solve,
        cholesky=cholesky,
        cholesky_solve=cholesky_solve,
        lu=lu,
        qr=qr,
        svd=svd,
        layer_norm=layer_norm,
        softmax=softmax,
        softmax_safe=softmax_safe,
        reduce=reduce,
        sum=sum,
        sigmoid=sigmoid,
        gelu=gelu,
        tanh=tanh,
        add=add,
        mul=mul,
        relu=relu,
        silu=silu,
        silu_mul=silu_mul,
        swiglu=swiglu,
        arange=arange,
        gather=gather,
        clip=clip,
        masked_fill=masked_fill,
        rmsnorm=rmsnorm,
        rmsnorm_safe=rmsnorm_safe,
        sin=sin,
        adam=adam,
        adamw=adamw,
        momentum=momentum,
        adafactor=adafactor,
        lion=lion,
        transpose=transpose,
        cast=cast,
        dropout=dropout,
        conv2d=conv2d,
        conv3d=conv3d,
        qkv_projection=qkv_projection,
        flash_attn=flash_attn,
        linear_attn=linear_attn,
        linear_attn_state=linear_attn_state,
        power_attn=power_attn,
        retention=retention,
        gated_attention=gated_attention,
        hybrid_attention=hybrid_attention,
        deepseek_sparse_attention=deepseek_sparse_attention,
        lightning_attention=lightning_attention,
        gated_deltanet=gated_deltanet,
        kimi_delta_attention=kimi_delta_attention,
        modified_delta_attention=modified_delta_attention,
        attn_sliding_window=attn_sliding_window,
        attn_local_window_2d=attn_local_window_2d,
        attn_compressed_blocks=attn_compressed_blocks,
        attn_top_k_blocks=attn_top_k_blocks,
        compress_blocks=compress_blocks,
        mor_router=mor_router,
        mor_partition=mor_partition,
        mor_scatter=mor_scatter,
        moe=moe,
        moe_dispatch=moe_dispatch,
        moe_combine=moe_combine,
        all_reduce=all_reduce,
        reduce_scatter=reduce_scatter,
        all_gather=all_gather,
        all_to_all=all_to_all,
        rng_uniform=rng_uniform,
        rng_normal=rng_normal,
        fused_epilogue=fused_epilogue,
        fft=fft,
        ifft=ifft,
        rfft=rfft,
        irfft=irfft,
        stft=stft,
        istft=istft,
        spectral_filter=spectral_filter,
        dct=dct,
        spectral_conv=spectral_conv,
        spmm_coo=spmm_coo,
        spmm_csr=spmm_csr,
        sddmm=sddmm,
        bsmm=bsmm,
        segment_reduce=segment_reduce,
        rearrange=rearrange,
        pack=pack,
        unpack=unpack,
        tile_view=tile_view,
        rope=rope,
        ReferenceKVCache=ReferenceKVCache,
        kv_cache_append=kv_cache_append,
        kv_cache_update=kv_cache_update,
        kv_cache_prune=kv_cache_prune,
        kv_cache_read=kv_cache_read,
        depthwise_conv1d=depthwise_conv1d,
        depthwise_conv2d=depthwise_conv2d,
        lstm_cell=lstm_cell,
        lstm_state_h=lstm_state_h,
        lstm_state_c=lstm_state_c,
        online_softmax=online_softmax,
        online_softmax_state=online_softmax_state,
        quantize_kv=quantize_kv,
        dequantize_kv=dequantize_kv,
        quantize_fp8=quantize_fp8,
        dequantize_fp8=dequantize_fp8,
        quantize_fp6=quantize_fp6,
        dequantize_fp6=dequantize_fp6,
        quantize_fp4=quantize_fp4,
        dequantize_fp4=dequantize_fp4,
        quantize_nvfp4=quantize_nvfp4,
        dequantize_nvfp4=dequantize_nvfp4,
        latent_kv_compress=latent_kv_compress,
        latent_kv_expand_k=latent_kv_expand_k,
        latent_kv_expand_v=latent_kv_expand_v,
        mla_decode_fused=mla_decode_fused,
        rope_split=rope_split,
        rope_merge=rope_merge,
        alibi=alibi_ref,
        ntk_rope=ntk_rope_ref,
        multi_head_attention=multi_head_attention_ref,
        gqa_attention=gqa_attention_ref,
        mqa_attention=mqa_attention_ref,
        mla_decode=mla_decode_ref,
        fake_quantize=fake_quantize_ref,
        selective_ssm=selective_ssm,
        # S-series sprint S2 — reductions, stability, numeric helpers,
        # comparisons. See `_make_ops_namespace`'s S2 block for definitions.
        mean=mean,
        prod=prod,
        amax=amax,
        amin=amin,
        max=max_reduce,
        min=min_reduce,
        var=var,
        std=std,
        argmax=argmax,
        argmin=argmin,
        cumsum=cumsum,
        cumprod=cumprod,
        cummax=cummax,
        cummin=cummin,
        logsumexp=logsumexp,
        log_softmax=log_softmax,
        log1p=log1p,
        expm1=expm1,
        softplus=softplus,
        sigmoid_safe=sigmoid_safe,
        sub=sub,
        div=div,
        floor_div=floor_div,
        mod=mod,
        exp=exp,
        log=log,
        sqrt=sqrt,
        rsqrt=rsqrt,
        pow=pow,
        cos=cos,
        tan=tan,
        sinh=sinh,
        cosh=cosh,
        asin=asin,
        acos=acos,
        atan=atan,
        atan2=atan2,
        erf=erf,
        erfc=erfc,
        lgamma=lgamma,
        digamma=digamma,
        clamp=clamp,
        where=where,
        absolute=absolute,
        abs=abs,
        sign=sign,
        reciprocal=reciprocal,
        floor=floor,
        ceil=ceil,
        round=round,
        trunc=trunc,
        minimum=minimum,
        maximum=maximum,
        isnan=isnan,
        isinf=isinf,
        isfinite=isfinite,
        eq=eq,
        ne=ne,
        lt=lt,
        le=le,
        gt=gt,
        ge=ge,
        logical_and=logical_and,
        logical_or=logical_or,
        logical_not=logical_not,
        logical_xor=logical_xor,
        bitwise_and=bitwise_and,
        bitwise_or=bitwise_or,
        bitwise_xor=bitwise_xor,
        bitwise_not=bitwise_not,
        reshape=reshape,
        view=view,
        flatten=flatten,
        squeeze=squeeze,
        unsqueeze=unsqueeze,
        permute=permute,
        broadcast=broadcast,
        expand=expand,
        cat=cat,
        stack=stack,
        split=split,
        chunk=chunk,
        pad=pad,
        tile=tile,
        repeat=repeat,
        roll=roll,
        flip=flip,
        slice=slice_op,
        select=select,
        dynamic_slice=dynamic_slice,
        dynamic_update_slice=dynamic_update_slice,
        take=take,
        index_select=index_select,
        scatter=scatter,
        scatter_add=scatter_add,
        scatter_reduce=scatter_reduce,
        index_update=index_update,
        nonzero=nonzero,
        count_nonzero=count_nonzero,
        popcount=popcount,
        masked_categorical=masked_categorical,
        top_k=top_k,
        sort=sort,
        argsort=argsort,
        linear_general=linear_general_ref,
        sgd=sgd_ref,
        mse_loss=mse_loss_ref,
        mae_loss=mae_loss_ref,
        huber_loss=huber_loss_ref,
        smooth_l1_loss=smooth_l1_loss_ref,
        log_cosh_loss=log_cosh_loss_ref,
        cross_entropy_loss=cross_entropy_loss_ref,
        binary_cross_entropy_loss=binary_cross_entropy_loss_ref,
        kl_divergence=kl_divergence_ref,
        js_divergence=js_divergence_ref,
        asymmetric_bce=asymmetric_bce_ref,
        z_loss=z_loss_ref,
        load_balance_loss=load_balance_loss_ref,
        ddpm_noise_pred_loss=ddpm_noise_pred_loss_ref,
        score_matching_loss=score_matching_loss_ref,
        contrastive_divergence_loss=contrastive_divergence_loss_ref,
        persistent_cd_loss=persistent_cd_loss_ref,
        implicit_score_matching_loss=implicit_score_matching_loss_ref,
        denoising_score_matching_loss=denoising_score_matching_loss_ref,
        vlb_loss=vlb_loss_ref,
        normalize_group_advantages=normalize_group_advantages_ref,
        ppo_policy_loss=ppo_policy_loss_ref,
        grpo_policy_loss=grpo_policy_loss_ref,
        cispo_policy_loss=cispo_policy_loss_ref,
    )
    return _ns


ops = _make_ops_namespace()

# Phase 2.1c (2026-06-01) — wrap the 8 encode-eligible ops with the
# apple_gpu trace-capture interceptor. Backward-compatible: when no
# @auto_batch trace is active, the wrappers call straight through to
# the existing numpy reference. Inside @auto_batch, calls with the
# encode-required kwargs (gamma, rows, cols, …) route to
# apple_gpu_ops.* automatically.
from . import apple_gpu_ops_interception as _agpu_intercept
_agpu_intercept.install_apple_gpu_interception(ops)

# Common op aliases kept at the top level for older advanced examples. The
# canonical compiler-visible spelling remains ``tessera.ops.<name>``.
arange = ops.arange
gather = ops.gather
clip = ops.clip
einsum = ops.einsum
masked_fill = ops.masked_fill

# nn module depends on `ops`, so import after the ops namespace is built.
from . import nn  # noqa: E402
from . import aot  # noqa: E402
from . import custom  # noqa: E402
from . import data  # noqa: E402
from . import losses  # noqa: E402
from . import memory  # noqa: E402
from . import optim  # noqa: E402
optimizers = types.SimpleNamespace(
    Adam=optim.Adam,
    AdamW=optim.AdamW,
)
from . import rng  # noqa: E402
from . import quantization  # noqa: E402
from . import rl  # noqa: E402
from .quantization import (  # noqa: E402
    CalibrationObserver,
    calibration_observer,
    dequantize_int4,
    dequantize_int8,
    fake_quantize,
    grad_scaler_step,
    quantize_int4,
    quantize_int8,
)
from .custom import (  # noqa: E402
    CustomPrimitive,
    custom_batching,
    custom_call,
    custom_jvp,
    custom_primitive,
    custom_vjp,
)
from .memory import (  # noqa: E402
    MemoryReadResult,
    MemoryTable,
    memory_evict,
    memory_read,
    memory_write,
)

# autodiff installs tape-aware wrappers on `ops.<name>`; load after `ops` and `nn`.
from . import autodiff  # noqa: E402
autocast = autodiff.autocast
rematerialize = autodiff.rematerialize
activation_checkpoint = autodiff.checkpoint
# Refresh top-level op aliases after autodiff installs tape-aware wrappers.
arange = ops.arange
gather = ops.gather
clip = ops.clip
einsum = ops.einsum
masked_fill = ops.masked_fill

# S5/S6 standalone compiler semantics.
from . import control  # noqa: E402
from . import sharding  # noqa: E402
from .control import (  # noqa: E402
    associative_scan,
    axis_index,
    axis_name,
    axis_size,
    cond,
    fori_loop,
    map,
    pmap,
    scan,
    switch,
    value_and_grad,
    vjp,
    while_loop,
)
from .sharding import (  # noqa: E402
    NamedMesh,
    NamedSharding,
    PartitionSpec,
    broadcast_to_axis,
    collective_permute,
    named_sharding,
    partition_spec,
    pmax,
    pmean,
    pmin,
    psum,
    shard_map,
)

# KV-cache handle abstraction (Phase B2 of execution_roadmap.md).
from . import cache  # noqa: E402

# Speculative decoding scheduler primitives (Theme 6).
from . import speculative  # noqa: E402

# Probability distributions (deferred-items plan, Item 1).
from . import distributions  # noqa: E402

from .runtime import (  # noqa: E402
    RuntimeArtifact,
    RuntimeProfile,
    available_backends,
    backend_capabilities,
    compile as compile_artifact,
    get_last_profile,
    launch,
    load_artifact,
    query_backend,
)

# ─────────────────────────────────────────────────────────────────────────────
# Dtype annotation shorthands
#
# These allow writing:  fn(A: tessera.f16[..., ...])
# as type annotations. In Phase 1 they are plain sentinel objects.
# ─────────────────────────────────────────────────────────────────────────────

class _DtypeAnnotation:
    """Type annotation sentinel for Tessera dtype-annotated tensors."""

    # Each subclass sets ``_dtype`` to a canonical dtype string
    # (``fp16``, ``bf16``, ``fp32``, etc.).  The base default is
    # ``fp32`` so the type-checker can see the attribute on the base
    # class; subclass values always win.
    _dtype: typing.ClassVar[str] = "fp32"

    def __init__(self, dtype: str, shape=None, layout: str | None = None) -> None:
        self.dtype = dtype
        self.shape = shape if isinstance(shape, tuple) else (() if shape is None else (shape,))
        self.layout = layout

    def __class_getitem__(cls, shape):
        return cls(cls._dtype, shape)

    def __getitem__(self, shape):
        return type(self)(shape)

    def __repr__(self) -> str:
        shape = ", ".join("..." if dim is Ellipsis else str(dim) for dim in self.shape) or "..."
        return f"tessera.{self.dtype}[{shape}]"


class f16(_DtypeAnnotation):
    _dtype = "fp16"
    def __init__(self, shape=None):
        super().__init__("fp16", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class bf16(_DtypeAnnotation):
    _dtype = "bf16"
    def __init__(self, shape=None):
        super().__init__("bf16", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class f32(_DtypeAnnotation):
    _dtype = "fp32"
    def __init__(self, shape=None):
        super().__init__("fp32", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class mut_f32(_DtypeAnnotation):
    """Mutable fp32 — write-privileged tensor annotation."""
    _dtype = "fp32"
    def __init__(self, shape=None):
        super().__init__("fp32", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class fp8_e4m3(_DtypeAnnotation):
    _dtype = "fp8_e4m3"
    def __init__(self, shape=None):
        super().__init__("fp8_e4m3", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class fp8_e5m2(_DtypeAnnotation):
    _dtype = "fp8_e5m2"
    def __init__(self, shape=None):
        super().__init__("fp8_e5m2", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class fp6_e2m3(_DtypeAnnotation):
    _dtype = "fp6_e2m3"
    def __init__(self, shape=None):
        super().__init__("fp6_e2m3", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class fp6_e3m2(_DtypeAnnotation):
    _dtype = "fp6_e3m2"
    def __init__(self, shape=None):
        super().__init__("fp6_e3m2", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class fp6(fp6_e3m2):
    _dtype = "fp6_e3m2"


class fp4_e2m1(_DtypeAnnotation):
    _dtype = "fp4_e2m1"
    def __init__(self, shape=None):
        super().__init__("fp4_e2m1", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


class fp4(fp4_e2m1):
    _dtype = "fp4_e2m1"


class nvfp4(_DtypeAnnotation):
    _dtype = "nvfp4"
    def __init__(self, shape=None):
        super().__init__("nvfp4", shape)
    def __class_getitem__(cls, shape):
        return cls(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level tensor factories
#
# Thin ergonomic wrappers over `tessera.array.from_domain(...)` with
# `Replicated()` distribution. For sharded construction, use
# `tessera.array.from_domain` directly with an explicit `dist.*`.
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_shape(shape):
    if isinstance(shape, (tuple, list)):
        return tuple(int(d) for d in shape)
    return (int(shape),)


def _make_replicated(shape, dtype, fill):
    return DistributedArray.from_domain(
        Rect(_normalize_shape(shape)),
        dtype=dtype,
        distribution=Replicated(),
        fill=fill,
    )


def zeros(shape, dtype: str = "fp32"):
    """Create a Replicated DistributedArray filled with zeros."""
    return _make_replicated(shape, dtype, "zeros")


def ones(shape, dtype: str = "fp32"):
    """Create a Replicated DistributedArray filled with ones."""
    return _make_replicated(shape, dtype, "ones")


def randn(shape, dtype: str = "fp32"):
    """Create a Replicated DistributedArray filled with N(0,1) samples."""
    return _make_replicated(shape, dtype, "randn")


def empty(shape, dtype: str = "fp32"):
    """Create a Replicated DistributedArray with uninitialized storage."""
    return _make_replicated(shape, dtype, "empty")


def full(shape, fill_value, dtype: str = "fp32"):
    """Create a Replicated DistributedArray filled with `fill_value`."""
    arr = _make_replicated(shape, dtype, "zeros")
    arr._data[...] = fill_value
    return arr


# ─────────────────────────────────────────────────────────────────────────────
# __all__
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Core types
    "core", "nn", "Tensor", "Module",
    # Distributed API
    "distributed", "Region", "index_launch", "kernel",
    "DistributedArray", "ShardSpec", "MeshSpec",
    "domain", "dist", "array",
    # Compiler
    "compiler", "jit", "require",
    "constraint", "ConstraintSolver", "Divisible", "Range", "Equal",
    "Effect", "EffectLattice",
    # Error types
    "TesseraJitError", "TesseraConstraintError", "TesseraEffectError",
    # Ops namespace + runtime
    "ops", "RuntimeArtifact", "RuntimeProfile", "available_backends",
    "backend_capabilities", "compile_artifact", "get_last_profile", "launch",
    "load_artifact", "query_backend",
    # S15 data, S14 AOT, S13 custom primitives, S11 losses, S10 optimizers /
    # schedules, and S9 quantization / numerics
    "data",
    "aot", "custom", "CustomPrimitive", "custom_primitive", "custom_call",
    "custom_vjp", "custom_jvp", "custom_batching",
    "losses", "memory", "optim", "optimizers", "rng", "rl",
    "MemoryReadResult", "MemoryTable", "memory_read", "memory_write",
    "memory_evict",
    "quantization", "CalibrationObserver", "calibration_observer",
    "quantize_int8", "dequantize_int8", "quantize_int4", "dequantize_int4",
    "fake_quantize", "grad_scaler_step",
    # Top-level op compatibility aliases
    "arange", "gather", "clip", "einsum", "masked_fill",
    # Dtype annotations
    "f16", "bf16", "f32", "mut_f32",
    "fp8_e4m3", "fp8_e5m2", "fp6_e2m3", "fp6_e3m2", "fp6",
    "fp4_e2m1", "fp4", "nvfp4",
    # Tensor factories (Replicated)
    "zeros", "ones", "randn", "empty", "full",
    # Auxiliary submodules (debug / profile / autotune / fault-tolerance / observability)
    "shape", "debug", "graph", "telemetry", "profiler", "autotune",
    "collectives", "fault", "elastic", "checkpoint", "server", "arch",
    # Autodiff (Tier 2 v1)
    "autodiff", "autocast", "rematerialize", "activation_checkpoint",
    # KV-cache handle (Phase B2)
    "cache", "distributions",
]
