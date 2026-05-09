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

import types
from dataclasses import dataclass, field
from typing import Any, Callable

# ─────────────────────────────────────────────────────────────────────────────
# Legacy core (Tensor, Module, NumericalPolicy)
# ─────────────────────────────────────────────────────────────────────────────
from . import core
from . import arch
from . import shape
from . import debug
from . import telemetry
from . import profiler
from . import collectives
from . import autotune as _autotune_module
from . import fault
from . import elastic
from . import checkpoint
from . import server
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

autotune = _autotune_module.autotune
autotune.load = _autotune_module.load
autotune.cache_key = _autotune_module.cache_key
autotune.schedule_artifact = _autotune_module.schedule_artifact
autotune.RooflineCostModel = _autotune_module.RooflineCostModel

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1: compiler API
# ─────────────────────────────────────────────────────────────────────────────
from . import compiler

from .compiler.jit import jit, require, TesseraJitError
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

    def sin(x):
        if hasattr(x, "_data"):
            x = x._data
        return np.sin(x)

    def adam(param, grad, moment1, moment2, lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, step: int = 1):
        """Functional Adam optimizer step.

        Returns ``(new_param, new_moment1, new_moment2)`` and keeps optimizer
        state explicit so it can lower as a pure CPU compiler op.
        """
        values = []
        for value in (param, grad, moment1, moment2):
            values.append(value._data if hasattr(value, "_data") else value)
        param, grad, moment1, moment2 = values
        new_m = beta1 * moment1 + (1.0 - beta1) * grad
        new_v = beta2 * moment2 + (1.0 - beta2) * (grad * grad)
        m_hat = new_m / (1.0 - beta1**step)
        v_hat = new_v / (1.0 - beta2**step)
        new_param = param - lr * m_hat / (np.sqrt(v_hat) + eps)
        return new_param, new_m, new_v

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

    references = {
        "gemm": gemm,
        "matmul": matmul,
        "batched_gemm": batched_gemm,
        "einsum": einsum,
        "factorized_matmul": factorized_matmul,
        "tri_solve": tri_solve,
        "cholesky": cholesky,
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
        "adam": adam,
        "transpose": transpose,
        "cast": cast,
        "dropout": dropout,
        "qkv_projection": qkv_projection,
        "flash_attn": flash_attn,
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
        "selective_ssm": selective_ssm,
    }
    for op_name, fn in references.items():
        _register_reference(op_name, fn, backend="numpy")
        _register_lowering(op_name, lambda *args, _op=op_name, **kwargs: {"op": _op, "status": "artifact_only"}, backend="graph_ir")

    return types.SimpleNamespace(
        registry=_ops_registry,
        register_reference=_register_reference,
        register_lowering=_register_lowering,
        register_runtime_kernel=_register_runtime_kernel,
        gemm=gemm,
        matmul=matmul,
        batched_gemm=batched_gemm,
        einsum=einsum,
        factorized_matmul=factorized_matmul,
        tri_solve=tri_solve,
        cholesky=cholesky,
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
        rmsnorm=rmsnorm,
        rmsnorm_safe=rmsnorm_safe,
        sin=sin,
        adam=adam,
        transpose=transpose,
        cast=cast,
        dropout=dropout,
        conv2d=conv2d,
        conv3d=conv3d,
        qkv_projection=qkv_projection,
        flash_attn=flash_attn,
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
        selective_ssm=selective_ssm,
    )


ops = _make_ops_namespace()

# nn module depends on `ops`, so import after the ops namespace is built.
from . import nn  # noqa: E402

# autodiff installs tape-aware wrappers on `ops.<name>`; load after `ops` and `nn`.
from . import autodiff  # noqa: E402

# KV-cache handle abstraction (Phase B2 of execution_roadmap.md).
from . import cache  # noqa: E402

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

    def __init__(self, dtype: str, shape=None, layout: str | None = None) -> None:
        self.dtype = dtype
        self.shape = shape if isinstance(shape, tuple) else (() if shape is None else (shape,))
        self.layout = layout

    def __class_getitem__(cls, shape):
        return cls(cls._dtype, shape)  # type: ignore[attr-defined]

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
    # Dtype annotations
    "f16", "bf16", "f32", "mut_f32",
    # Tensor factories (Replicated)
    "zeros", "ones", "randn", "empty", "full",
    # Auxiliary submodules (debug / profile / autotune / fault-tolerance / observability)
    "shape", "debug", "graph", "telemetry", "profiler", "autotune",
    "collectives", "fault", "elastic", "checkpoint", "server", "arch",
    # Autodiff (Tier 2 v1)
    "autodiff",
    # KV-cache handle (Phase B2)
    "cache",
]
