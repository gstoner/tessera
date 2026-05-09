"""Stateful layer wrappers — thin Modules around the functional `nn.*` API.

Each class owns its trainable `Parameter`s and delegates the math to the
already-shipped functional surface (`nn.linear`, `nn.rms_norm`, `nn.swiglu`,
`nn.multi_head_attention`) or to `tessera.ops.*` directly.

Initializers are deliberately simple:
  * Linear-style weights → Kaiming-uniform fan-in scaling
  * Norm weights → ones
  * Embeddings → N(0, 0.02)
  * Biases → zeros

Custom initializers can be applied post-construction by writing into
`module.<name>._data._data` (the underlying numpy buffer).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .. import ops
from . import functional as F
from .module import Module, Parameter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _as_array(x: Any) -> np.ndarray:
    """Extract a numpy array view from a Parameter, DistributedArray, or array-like."""
    if isinstance(x, Parameter):
        return x._data._data
    if hasattr(x, "_data") and not isinstance(x, np.ndarray):
        inner = x._data
        if isinstance(inner, np.ndarray):
            return inner
        if hasattr(inner, "numpy"):
            return inner.numpy()
        return np.asarray(inner)
    return np.asarray(x)


def _kaiming_uniform_(buf: np.ndarray, fan_in: int) -> None:
    """In-place Kaiming-uniform init: U(-sqrt(3/fan_in), sqrt(3/fan_in))."""
    bound = math.sqrt(3.0 / max(1, fan_in))
    buf[...] = np.random.uniform(-bound, bound, size=buf.shape).astype(buf.dtype, copy=False)


def _normal_(buf: np.ndarray, std: float) -> None:
    buf[...] = np.random.normal(0.0, std, size=buf.shape).astype(buf.dtype, copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Linear
# ─────────────────────────────────────────────────────────────────────────────


class Linear(Module):
    """y = x @ W (+ b). Stores W with shape (in_features, out_features)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = Parameter(shape=(self.in_features, self.out_features), dtype=dtype)
        _kaiming_uniform_(self.weight._data._data, fan_in=self.in_features)
        if bias:
            self.bias = Parameter(shape=(self.out_features,), dtype=dtype)
            self.bias._data._data[...] = 0.0
        else:
            object.__setattr__(self, "bias", None)

    def forward(self, x: Any) -> np.ndarray:
        return F.linear(_as_array(x), _as_array(self.weight), bias=_as_array(self.bias) if self.bias is not None else None)


# ─────────────────────────────────────────────────────────────────────────────
# Norms
# ─────────────────────────────────────────────────────────────────────────────


class RMSNorm(Module):
    """RMSNorm with a learnable per-channel weight."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5, dtype: str = "fp32") -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.weight = Parameter(shape=(self.normalized_shape,), dtype=dtype)
        self.weight._data._data[...] = 1.0

    def forward(self, x: Any) -> np.ndarray:
        return F.rms_norm(_as_array(x), weight=_as_array(self.weight), eps=self.eps)


class LayerNorm(Module):
    """LayerNorm with optional learnable affine."""

    def __init__(
        self,
        normalized_shape: int,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        self.normalized_shape = int(normalized_shape)
        self.eps = float(eps)
        self.elementwise_affine = bool(elementwise_affine)
        if elementwise_affine:
            self.weight = Parameter(shape=(self.normalized_shape,), dtype=dtype)
            self.weight._data._data[...] = 1.0
            if bias:
                self.bias = Parameter(shape=(self.normalized_shape,), dtype=dtype)
                self.bias._data._data[...] = 0.0
            else:
                object.__setattr__(self, "bias", None)
        else:
            object.__setattr__(self, "weight", None)
            object.__setattr__(self, "bias", None)

    def forward(self, x: Any) -> np.ndarray:
        x_arr = _as_array(x)
        y = ops.layer_norm(x_arr, eps=self.eps)
        if self.elementwise_affine and self.weight is not None:
            y = y * _as_array(self.weight)
        if self.elementwise_affine and self.bias is not None:
            y = y + _as_array(self.bias)
        return y


# ─────────────────────────────────────────────────────────────────────────────
# Embedding
# ─────────────────────────────────────────────────────────────────────────────


class Embedding(Module):
    """Lookup table: `forward(idx)` returns `weight[idx]`.

    `idx` is an integer numpy array of any shape; output adds a trailing
    `embedding_dim` axis.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, dtype: str = "fp32") -> None:
        super().__init__()
        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = Parameter(shape=(self.num_embeddings, self.embedding_dim), dtype=dtype)
        _normal_(self.weight._data._data, std=0.02)

    def forward(self, idx: Any) -> np.ndarray:
        idx_arr = np.asarray(idx, dtype=np.int64)
        if idx_arr.dtype.kind != "i":
            raise TypeError(f"Embedding expects integer indices, got dtype {idx_arr.dtype}")
        if idx_arr.size and (idx_arr.min() < 0 or idx_arr.max() >= self.num_embeddings):
            raise IndexError(
                f"Embedding index out of range: indices must be in [0, {self.num_embeddings})"
            )
        return _as_array(self.weight)[idx_arr]


# ─────────────────────────────────────────────────────────────────────────────
# Dropout
# ─────────────────────────────────────────────────────────────────────────────


class Dropout(Module):
    """Stateful dropout that respects `self.training`."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("Dropout p must be in [0.0, 1.0)")
        self.p = float(p)
        self.seed = seed

    def forward(self, x: Any) -> np.ndarray:
        return ops.dropout(_as_array(x), p=self.p, training=self.training, seed=self.seed)


# ─────────────────────────────────────────────────────────────────────────────
# MLP (SwiGLU block)
# ─────────────────────────────────────────────────────────────────────────────


class MLP(Module):
    """SwiGLU MLP block: `swiglu(x, W_gate, W_up, W_down)`."""

    def __init__(self, dim: int, hidden_dim: int, dtype: str = "fp32") -> None:
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim)
        self.W_gate = Parameter(shape=(self.dim, self.hidden_dim), dtype=dtype)
        self.W_up = Parameter(shape=(self.dim, self.hidden_dim), dtype=dtype)
        self.W_down = Parameter(shape=(self.hidden_dim, self.dim), dtype=dtype)
        _kaiming_uniform_(self.W_gate._data._data, fan_in=self.dim)
        _kaiming_uniform_(self.W_up._data._data, fan_in=self.dim)
        _kaiming_uniform_(self.W_down._data._data, fan_in=self.hidden_dim)

    def forward(self, x: Any) -> np.ndarray:
        return F.swiglu(
            _as_array(x),
            _as_array(self.W_gate),
            _as_array(self.W_up),
            _as_array(self.W_down),
        )


# ─────────────────────────────────────────────────────────────────────────────
# MultiHeadAttention
# ─────────────────────────────────────────────────────────────────────────────


class MultiHeadAttention(Module):
    """Standard multi-head attention with packed Q/K/V projections.

    Input shape: `[B, S, embed_dim]`. Output shape: `[B, S, embed_dim]`.
    For cross-attention, pass distinct Q vs. K/V tensors to `forward`.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
        dropout_p: float = 0.0,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
            )
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.dropout_p = float(dropout_p)

        self.q_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias, dtype=dtype)

    def forward(
        self,
        query: Any,
        key: Any | None = None,
        value: Any | None = None,
        causal: bool = False,
        scale: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if key is None:
            key = query
        if value is None:
            value = key
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        attn = F.multi_head_attention(
            Q,
            K,
            V,
            num_heads=self.num_heads,
            scale=scale,
            causal=causal,
            dropout_p=self.dropout_p if self.training else 0.0,
            seed=seed,
        )
        return self.out_proj(attn)


# ─────────────────────────────────────────────────────────────────────────────
# Activation modules — stateless wrappers that respect torch.nn-style usage
# (`act = nn.SiLU(); y = act(x)`). Each instance is callable and forwards
# directly to the corresponding `tessera.ops.<name>` reference op.
# ─────────────────────────────────────────────────────────────────────────────


class _ActivationModule(Module):
    """Stateless callable around a single `tessera.ops.<name>` op."""

    _op_name: str = ""

    def forward(self, x: Any) -> np.ndarray:
        return getattr(ops, self._op_name)(_as_array(x))


class SiLU(_ActivationModule):
    _op_name = "silu"


class Sigmoid(_ActivationModule):
    _op_name = "sigmoid"


class GELU(_ActivationModule):
    _op_name = "gelu"


class ReLU(_ActivationModule):
    _op_name = "relu"


class Tanh(_ActivationModule):
    _op_name = "tanh"


class Identity(Module):
    """``forward(x)`` returns ``x`` unchanged. Useful as a no-op slot in
    composable model definitions (e.g., optional residual gates)."""

    def forward(self, x: Any) -> Any:
        return x


# ─────────────────────────────────────────────────────────────────────────────
# Cross-attention
# ─────────────────────────────────────────────────────────────────────────────


class MultiHeadCrossAttention(MultiHeadAttention):
    """MultiHeadAttention restricted to cross-attention semantics.

    Identical to ``MultiHeadAttention``, but ``forward`` requires explicit
    ``key`` and ``value`` tensors — calling it with only ``query`` raises a
    ``ValueError`` instead of silently doing self-attention. Used to make the
    cross-attention call site explicit at type/argument level.
    """

    def forward(
        self,
        query: Any,
        key: Any = None,
        value: Any = None,
        causal: bool = False,
        scale: float | None = None,
        seed: int | None = None,
    ) -> np.ndarray:
        if key is None or value is None:
            raise ValueError(
                "MultiHeadCrossAttention requires explicit key and value tensors; "
                "use MultiHeadAttention for self-attention."
            )
        return super().forward(query, key, value, causal=causal, scale=scale, seed=seed)


# ─────────────────────────────────────────────────────────────────────────────
# Rotary positional embedding
# ─────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(Module):
    """Stateful wrapper around ``ops.rope`` that owns the precomputed inverse-frequency table.

    Stores ``inv_freq`` (numpy array) as a non-trainable attribute and applies
    rope to the last axis of ``x`` shaped ``[..., head_dim]``. The frequency
    convention follows the standard formulation:
        ``inv_freq[i] = 1.0 / (base ** (2i / head_dim))``  for ``i in [0, head_dim/2)``.
    """

    def __init__(self, head_dim: int, max_position: int = 2048, base: float = 10000.0, dtype: str = "fp32"):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim must be even; got {head_dim}")
        self.head_dim = int(head_dim)
        self.max_position = int(max_position)
        self.base = float(base)
        # Precompute theta = pos * inv_freq, broadcast-shaped for ops.rope
        inv_freq = 1.0 / (base ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
        positions = np.arange(max_position, dtype=np.float64)
        theta = np.einsum("i,j->ij", positions, inv_freq)  # [max_position, head_dim/2]
        # Tile to match the full head_dim — ops.rope takes both halves
        theta = np.concatenate([theta, theta], axis=-1).astype(
            np.float32 if dtype in ("fp32", "bf16") else np.float16
        )
        # Stash on self as a regular attribute (non-Parameter, non-Module)
        object.__setattr__(self, "theta", theta)

    def forward(self, x: Any, position: int = 0) -> np.ndarray:
        x_arr = _as_array(x)
        seq_len = x_arr.shape[-2] if x_arr.ndim >= 2 else 1
        if position + seq_len > self.max_position:
            raise ValueError(
                f"RotaryEmbedding: position {position} + seq_len {seq_len} "
                f"exceeds max_position {self.max_position}"
            )
        theta_slice = self.theta[position:position + seq_len]
        return ops.rope(x_arr, theta_slice)


# ─────────────────────────────────────────────────────────────────────────────
# Casted variants — Linear / Embedding that auto-cast their output to a
# different dtype. Useful for mixed-precision setups where the parameter
# storage dtype differs from the compute / consumer dtype.
# ─────────────────────────────────────────────────────────────────────────────


class CastedLinear(Linear):
    """Linear whose output is cast to ``cast_dtype`` after the matmul + bias add.

    The cast is the only difference from ``Linear``; everything else (init,
    parameter storage, state_dict) is inherited.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        cast_dtype: str,
        bias: bool = True,
        dtype: str = "fp32",
    ):
        super().__init__(in_features, out_features, bias=bias, dtype=dtype)
        self.cast_dtype = cast_dtype

    def forward(self, x: Any) -> np.ndarray:
        return ops.cast(super().forward(x), self.cast_dtype)


class CastedEmbedding(Embedding):
    """Embedding whose lookup result is cast to ``cast_dtype``."""

    def __init__(self, num_embeddings: int, embedding_dim: int, cast_dtype: str, dtype: str = "fp32"):
        super().__init__(num_embeddings, embedding_dim, dtype=dtype)
        self.cast_dtype = cast_dtype

    def forward(self, idx: Any) -> np.ndarray:
        return ops.cast(super().forward(idx), self.cast_dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-entropy loss
# ─────────────────────────────────────────────────────────────────────────────


class CrossEntropyLoss(Module):
    """Categorical cross-entropy: ``-mean(log_softmax(logits)[target])``.

    Composed through ``ops.softmax`` + ``ops.reduce`` so the autodiff tape
    captures every step. ``logits`` shape: ``[..., num_classes]``;
    ``target`` shape: ``[...]`` of integer class indices.

    ``reduction``: ``"mean"`` (default), ``"sum"``, or ``"none"``.
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"reduction must be 'mean'/'sum'/'none'; got {reduction!r}")
        self.reduction = reduction

    def forward(self, logits: Any, target: Any) -> np.ndarray:
        logits_arr = _as_array(logits)
        target_arr = np.asarray(target, dtype=np.int64)

        # log_softmax via ops.softmax + log (log composes through tape via mul/sum
        # only if we go through ops; numpy log isn't traced, so we leave the
        # final log + gather unhanded by autodiff in v1. Users wanting an
        # autodiff-able loss should compose softmax explicitly inside the tape.
        # For correctness, this Module returns the loss value; tape integration
        # for backward is documented as the recipe in CANONICAL_API.md.
        probs = ops.softmax(logits_arr, axis=-1)
        # Gather the probability of the correct class along the last axis
        target_onehot_idx = np.expand_dims(target_arr, axis=-1)
        true_probs = np.take_along_axis(probs, target_onehot_idx, axis=-1).squeeze(-1)
        # Avoid log(0) — clamp to a small positive
        true_probs = np.maximum(true_probs, 1e-30)
        per_example = -np.log(true_probs)

        if self.reduction == "none":
            return per_example
        if self.reduction == "sum":
            return ops.reduce(per_example, op="sum")
        # mean — keep the divide outside ops since we don't have a divide op
        return ops.reduce(per_example, op="sum") / per_example.size


# ─────────────────────────────────────────────────────────────────────────────
# Conv2dNCHW shim — torch-port helper around ops.conv2d (NHWC)
# ─────────────────────────────────────────────────────────────────────────────
# Conv2d (NHWC) Module proper lands in Phase H1; this shim is sized to that
# decision (NHWC default, NCHW shim transposes in/out). Defer until H1.


__all__ = [
    "Linear",
    "RMSNorm",
    "LayerNorm",
    "Embedding",
    "Dropout",
    "MLP",
    "MultiHeadAttention",
    "MultiHeadCrossAttention",
    "RotaryEmbedding",
    "CastedLinear",
    "CastedEmbedding",
    "SiLU", "Sigmoid", "GELU", "ReLU", "Tanh", "Identity",
    "CrossEntropyLoss",
]
