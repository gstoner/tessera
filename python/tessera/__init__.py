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
from . import nn
from . import arch
from . import shape
from . import debug
from . import profiler
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
    export_graphviz=debug.export_graphviz,
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

    def gemm(A, B):
        """Matrix multiply A @ B."""
        if hasattr(A, "_data"):
            A = A._data
        if hasattr(B, "_data"):
            B = B._data
        return np.matmul(A, B)

    def matmul(A, B):
        return gemm(A, B)

    def layer_norm(x, eps: float = 1e-5):
        if hasattr(x, "_data"):
            x = x._data
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps)

    def softmax(x, axis: int = -1):
        if hasattr(x, "_data"):
            x = x._data
        e = np.exp(x - x.max(axis=axis, keepdims=True))
        return e / e.sum(axis=axis, keepdims=True)

    def sigmoid(x):
        if hasattr(x, "_data"):
            x = x._data
        return 1.0 / (1.0 + np.exp(-x))

    def gelu(x):
        if hasattr(x, "_data"):
            x = x._data
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def relu(x):
        if hasattr(x, "_data"):
            x = x._data
        return np.maximum(0, x)

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

    def dropout(x, p: float = 0.1, training: bool = True):
        if not training:
            return x
        if hasattr(x, "_data"):
            x = x._data
        mask = np.random.binomial(1, 1 - p, x.shape) / (1 - p)
        return x * mask

    def conv2d(x, weight, bias=None, stride=1, padding=0):
        # Phase 1 stub — returns zeros of expected shape
        if hasattr(x, "_data"):
            x = x._data
        return np.zeros_like(x)

    def flash_attn(
        Q,
        K,
        V,
        scale=None,
        causal: bool = False,
        dropout_p: float = 0.0,
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

    def all_reduce(x, op: str = "sum"):
        # Phase 1 stub: single-rank, no-op
        if hasattr(x, "_data"):
            x = x._data
        return x

    def reduce_scatter(x, op: str = "sum", axis: int = 0):
        if hasattr(x, "_data"):
            x = x._data
        return x

    def all_gather(x, axis: int = 0):
        if hasattr(x, "_data"):
            x = x._data
        return x

    def fused_epilogue(x, bias=None, activation="linear"):
        if hasattr(x, "_data"):
            x = x._data
        if bias is not None:
            x = x + bias
        if activation == "gelu":
            return gelu(x)
        elif activation == "relu":
            return relu(x)
        return x

    def fft(x, axis: int = -1):
        if hasattr(x, "_data"):
            x = x._data
        return np.fft.fft(x, axis=axis)

    def ifft(xf, axis: int = -1):
        if hasattr(xf, "_data"):
            xf = xf._data
        return np.fft.ifft(xf, axis=axis)

    def rfft(x, axis: int = -1):
        if hasattr(x, "_data"):
            x = x._data
        return np.fft.rfft(x, axis=axis)

    def irfft(xf, axis: int = -1, n=None):
        if hasattr(xf, "_data"):
            xf = xf._data
        return np.fft.irfft(xf, n=n, axis=axis)

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

    references = {
        "gemm": gemm,
        "matmul": matmul,
        "conv2d": conv2d,
        "layer_norm": layer_norm,
        "softmax": softmax,
        "gelu": gelu,
        "relu": relu,
        "sigmoid": sigmoid,
        "sin": sin,
        "adam": adam,
        "transpose": transpose,
        "cast": cast,
        "dropout": dropout,
        "flash_attn": flash_attn,
        "all_reduce": all_reduce,
        "reduce_scatter": reduce_scatter,
        "all_gather": all_gather,
        "fused_epilogue": fused_epilogue,
        "fft": fft,
        "ifft": ifft,
        "rfft": rfft,
        "irfft": irfft,
        "dct": dct,
        "spectral_conv": spectral_conv,
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
        layer_norm=layer_norm,
        softmax=softmax,
        sigmoid=sigmoid,
        gelu=gelu,
        relu=relu,
        sin=sin,
        adam=adam,
        transpose=transpose,
        cast=cast,
        dropout=dropout,
        conv2d=conv2d,
        flash_attn=flash_attn,
        all_reduce=all_reduce,
        reduce_scatter=reduce_scatter,
        all_gather=all_gather,
        fused_epilogue=fused_epilogue,
        fft=fft,
        ifft=ifft,
        rfft=rfft,
        irfft=irfft,
        dct=dct,
        spectral_conv=spectral_conv,
    )


ops = _make_ops_namespace()

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

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype

    def __class_getitem__(cls, shape):
        return cls(cls._dtype)  # type: ignore[attr-defined]

    def __getitem__(self, shape):
        return self

    def __repr__(self) -> str:
        return f"tessera.{self.dtype}[...]"


class f16(_DtypeAnnotation):
    _dtype = "fp16"
    def __init__(self):
        super().__init__("fp16")
    def __class_getitem__(cls, shape):
        return cls()


class bf16(_DtypeAnnotation):
    _dtype = "bf16"
    def __init__(self):
        super().__init__("bf16")
    def __class_getitem__(cls, shape):
        return cls()


class f32(_DtypeAnnotation):
    _dtype = "fp32"
    def __init__(self):
        super().__init__("fp32")
    def __class_getitem__(cls, shape):
        return cls()


class mut_f32(_DtypeAnnotation):
    """Mutable fp32 — write-privileged tensor annotation."""
    _dtype = "fp32"
    def __init__(self):
        super().__init__("fp32")
    def __class_getitem__(cls, shape):
        return cls()


# ─────────────────────────────────────────────────────────────────────────────
# __all__
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Legacy
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
    # Ops namespace
    "ops", "RuntimeArtifact", "RuntimeProfile", "available_backends",
    "backend_capabilities", "compile_artifact", "get_last_profile", "launch",
    "load_artifact", "query_backend",
    # Dtype annotations
    "f16", "bf16", "f32", "mut_f32",
]
