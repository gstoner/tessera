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

    def gelu(x):
        if hasattr(x, "_data"):
            x = x._data
        return x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

    def relu(x):
        if hasattr(x, "_data"):
            x = x._data
        return np.maximum(0, x)

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

    def flash_attn(Q, K, V, scale=None):
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
        d = Q.shape[-1]
        if scale is None:
            scale = 1.0 / np.sqrt(d)
        scores = np.matmul(Q, K.swapaxes(-1, -2)) * scale
        weights = softmax(scores)
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

    return types.SimpleNamespace(
        gemm=gemm,
        matmul=matmul,
        layer_norm=layer_norm,
        softmax=softmax,
        gelu=gelu,
        relu=relu,
        transpose=transpose,
        cast=cast,
        dropout=dropout,
        conv2d=conv2d,
        flash_attn=flash_attn,
        all_reduce=all_reduce,
        reduce_scatter=reduce_scatter,
        all_gather=all_gather,
        fused_epilogue=fused_epilogue,
    )


ops = _make_ops_namespace()

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
    "ops",
    # Dtype annotations
    "f16", "bf16", "f32", "mut_f32",
]
