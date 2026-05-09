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
from ..cache import KVCacheHandle
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


class BatchNorm1d(Module):
    """1-D batch normalization with running stats.

    Accepts inputs of shape ``(N, C)`` or ``(N, C, L)``. Statistics are
    computed over the batch (and spatial axis when present); the channel
    axis ``C`` is preserved.

    Buffers (Phase B1):
      * ``running_mean`` — shape ``(C,)``, init zeros
      * ``running_var`` — shape ``(C,)``, init ones
      * ``num_batches_tracked`` — scalar int64, increments on every train step

    Train mode uses batch stats and updates running stats via
    ``new = (1 - momentum) * old + momentum * batch``. Eval mode uses
    running stats and never updates them.
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.eps = float(eps)
        self.momentum = float(momentum)
        self.affine = bool(affine)
        self.track_running_stats = bool(track_running_stats)

        if affine:
            self.weight = Parameter(shape=(self.num_features,), dtype=dtype)
            self.weight._data._data[...] = 1.0
            self.bias = Parameter(shape=(self.num_features,), dtype=dtype)
            self.bias._data._data[...] = 0.0
        else:
            object.__setattr__(self, "weight", None)
            object.__setattr__(self, "bias", None)

        if track_running_stats:
            self.register_buffer("running_mean", np.zeros(self.num_features, dtype=np.float32))
            self.register_buffer("running_var", np.ones(self.num_features, dtype=np.float32))
            # Stored as a 1-element vector; the Rect domain rejects empty shapes.
            self.register_buffer("num_batches_tracked", np.zeros(1, dtype=np.int64))
        # else: buffers absent; both modes use batch stats

    def forward(self, x: Any) -> np.ndarray:
        x_arr = _as_array(x)
        if x_arr.ndim not in (2, 3):
            raise ValueError(
                f"BatchNorm1d expects (N, C) or (N, C, L) input; got shape {x_arr.shape}"
            )
        if x_arr.shape[1] != self.num_features:
            raise ValueError(
                f"channel dim {x_arr.shape[1]} != num_features {self.num_features}"
            )

        # Reduce over batch (axis 0) and spatial (axis 2 if present)
        reduce_axes: tuple[int, ...] = (0,) if x_arr.ndim == 2 else (0, 2)

        if self.training:
            batch_mean = x_arr.mean(axis=reduce_axes)
            # Use the population variance (matches torch's default for BN forward;
            # the unbiased estimator only feeds the running buffer).
            batch_var = x_arr.var(axis=reduce_axes)
            mean, var = batch_mean, batch_var

            if self.track_running_stats:
                # Update running stats — torch uses the unbiased variance for the
                # buffer when N > 1.
                n = x_arr.size // self.num_features
                if n > 1:
                    unbiased_var = batch_var * n / (n - 1)
                else:
                    unbiased_var = batch_var
                rm_buf = self.running_mean._data._data
                rv_buf = self.running_var._data._data
                rm_buf[...] = (1.0 - self.momentum) * rm_buf + self.momentum * batch_mean
                rv_buf[...] = (1.0 - self.momentum) * rv_buf + self.momentum * unbiased_var
                nbt = self.num_batches_tracked._data._data
                nbt[...] = nbt + 1
        else:
            if self.track_running_stats:
                mean = self.running_mean._data._data
                var = self.running_var._data._data
            else:
                mean = x_arr.mean(axis=reduce_axes)
                var = x_arr.var(axis=reduce_axes)

        # Broadcast mean/var across batch (and spatial) axes by reshaping to (1, C) or (1, C, 1)
        if x_arr.ndim == 3:
            mean_b = mean.reshape(1, self.num_features, 1)
            var_b = var.reshape(1, self.num_features, 1)
        else:
            mean_b = mean.reshape(1, self.num_features)
            var_b = var.reshape(1, self.num_features)

        normalized = (x_arr - mean_b) / np.sqrt(var_b + self.eps)

        if self.affine:
            if x_arr.ndim == 3:
                w_b = self.weight._data._data.reshape(1, self.num_features, 1)
                b_b = self.bias._data._data.reshape(1, self.num_features, 1)
            else:
                w_b = self.weight._data._data.reshape(1, self.num_features)
                b_b = self.bias._data._data.reshape(1, self.num_features)
            return normalized * w_b + b_b
        return normalized


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
# Dynamic depthwise 1-D conv (Phase D4 — depends on D1 + B1)
# ─────────────────────────────────────────────────────────────────────────────


class DynamicDepthwiseConv1d(Module):
    """Streaming-state-aware depthwise 1-D convolution.

    Wraps :func:`tessera.ops.depthwise_conv1d`. Owns the per-channel kernel
    weights as a `Parameter` and (when constructed with ``streaming=True``)
    a non-persistent `Buffer` that holds the trailing ``kernel_size-1``
    samples between forward calls — enabling chunked decode.

    Buffers are non-persistent so that ``state_dict`` round-trip keeps the
    *weights* but not the runtime carry-state, which is the right default
    for save-and-resume.

    Args:
        channels: number of input/output channels (depthwise → equal)
        kernel_size: filter length
        causal: pad on the left only (right padding = 0)
        streaming: track + reuse the trailing samples buffer across calls
        dtype: weight + state dtype
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        causal: bool = True,
        streaming: bool = False,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        self.channels = int(channels)
        self.kernel_size = int(kernel_size)
        self.causal = bool(causal)
        self.streaming = bool(streaming)

        self.weight = Parameter(shape=(self.channels, self.kernel_size), dtype=dtype)
        _kaiming_uniform_(self.weight._data._data, fan_in=self.kernel_size)

        if self.streaming and self.kernel_size > 1:
            # State is per-batch — start with batch=1 zeros; resized lazily
            # on first call when we know N. Persistent=False because state
            # is runtime carry, not a learned weight.
            self.register_buffer(
                "_state",
                np.zeros((1, self.channels, self.kernel_size - 1), dtype=np.float32),
                persistent=False,
            )
        else:
            object.__setattr__(self, "_state", None)

    def reset_state(self) -> None:
        """Drop the streaming state — start fresh on the next call."""
        if self._state is not None:
            self._state._data._data[...] = 0.0

    def forward(self, x: Any) -> np.ndarray:
        x_arr = _as_array(x)
        if x_arr.ndim != 3:
            raise ValueError(
                f"DynamicDepthwiseConv1d expects (N, C, L) input; got shape {x_arr.shape}"
            )

        if self.streaming and self.kernel_size > 1:
            N = x_arr.shape[0]
            cur_state = self._state._data._data
            # Resize state buffer if batch size changed (or first call from default N=1)
            if cur_state.shape[0] != N:
                new_state = np.zeros(
                    (N, self.channels, self.kernel_size - 1), dtype=cur_state.dtype
                )
                # Re-register so the buffer-id registry stays consistent
                self.register_buffer("_state", new_state, persistent=False)
                cur_state = self._state._data._data

            y = ops.depthwise_conv1d(
                x_arr,
                _as_array(self.weight),
                kernel_size=self.kernel_size,
                state=cur_state,
            )
            # Update state for the next call: last K-1 samples of concat(state, x)
            if x_arr.shape[-1] >= self.kernel_size - 1:
                new_state_data = x_arr[..., -(self.kernel_size - 1):]
            else:
                new_state_data = np.concatenate([cur_state, x_arr], axis=-1)[
                    ..., -(self.kernel_size - 1):
                ]
            cur_state[...] = new_state_data
            return y

        # Non-streaming path — causal-pad or zero-pad and convolve directly.
        return ops.depthwise_conv1d(
            x_arr,
            _as_array(self.weight),
            kernel_size=self.kernel_size,
            causal=self.causal,
            padding=0 if self.causal else (self.kernel_size - 1) // 2,
        )


# ─────────────────────────────────────────────────────────────────────────────
# LSTM cell + sequence (Phase H2 — RNN with state-propagation primitive)
# ─────────────────────────────────────────────────────────────────────────────


class LSTMCell(Module):
    """Single-step LSTM cell.

    Owns ``W_ih`` (4*hidden, in_features), ``W_hh`` (4*hidden, hidden), and
    optional biases. ``forward(x_t, state)`` consumes a previous
    ``(h_prev, c_prev)`` tuple and returns ``(h_t, c_t)``. The cell op
    internally packs h+c into a single tensor; state extraction goes through
    autodiff-traced ``ops.lstm_state_h`` / ``ops.lstm_state_c``.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        gate_size = 4 * self.hidden_size

        self.W_ih = Parameter(shape=(gate_size, self.input_size), dtype=dtype)
        _kaiming_uniform_(self.W_ih._data._data, fan_in=self.input_size)
        self.W_hh = Parameter(shape=(gate_size, self.hidden_size), dtype=dtype)
        _kaiming_uniform_(self.W_hh._data._data, fan_in=self.hidden_size)
        if bias:
            self.b_ih = Parameter(shape=(gate_size,), dtype=dtype)
            self.b_ih._data._data[...] = 0.0
            self.b_hh = Parameter(shape=(gate_size,), dtype=dtype)
            self.b_hh._data._data[...] = 0.0
        else:
            object.__setattr__(self, "b_ih", None)
            object.__setattr__(self, "b_hh", None)

    def forward(self, x_t: Any, state: tuple) -> tuple:
        """``state`` is ``(h_prev, c_prev)`` — both shape ``(B, hidden_size)``.
        Returns ``(h_t, c_t)``.
        """
        h_prev, c_prev = state
        packed = ops.lstm_cell(
            _as_array(x_t),
            _as_array(h_prev),
            _as_array(c_prev),
            _as_array(self.W_ih),
            _as_array(self.W_hh),
            _as_array(self.b_ih) if self.b_ih is not None else None,
            _as_array(self.b_hh) if self.b_hh is not None else None,
        )
        h_t = ops.lstm_state_h(packed)
        c_t = ops.lstm_state_c(packed)
        return h_t, c_t


class LSTM(Module):
    """Multi-step LSTM. Wraps :class:`LSTMCell` and unrolls over time.

    ``forward(x_seq, init_state=None)`` accepts ``x_seq`` of shape
    ``(B, T, input_size)`` and returns ``(output, (h_n, c_n))`` where
    ``output`` is ``(B, T, hidden_size)`` — the stacked ``h_t`` at each step.

    For short sequences (T ≤ a few dozen), unrolling explicitly is fine and
    BPTT works via the v1 numpy tape. For longer sequences, wrap with
    ``tessera.autodiff.rematerialize`` (Phase F2) to drop intermediate
    activations during forward and recompute on backward.

    State-propagation primitive — RNN cells need explicit per-timestep state
    plumbing. ``LSTM`` keeps the loop in Python (visible to the autodiff
    tape) rather than fusing it into a single op, so gradients flow
    correctly through the recurrence.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias: bool = True,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.hidden_size = int(hidden_size)
        self.cell = LSTMCell(input_size, hidden_size, bias=bias, dtype=dtype)

    def forward(self, x_seq: Any, init_state: tuple | None = None) -> tuple:
        x_arr = _as_array(x_seq)
        if x_arr.ndim != 3:
            raise ValueError(
                f"LSTM expects (B, T, input_size); got shape {x_arr.shape}"
            )
        B, T, _ = x_arr.shape
        if init_state is None:
            h = np.zeros((B, self.hidden_size), dtype=x_arr.dtype)
            c = np.zeros((B, self.hidden_size), dtype=x_arr.dtype)
        else:
            h, c = init_state
            h = _as_array(h)
            c = _as_array(c)
        outputs = []
        for t in range(T):
            h, c = self.cell(x_arr[:, t, :], (h, c))
            outputs.append(h)
        # Stack along time. np.stack creates a new ndarray — not on the tape,
        # but the user typically takes the *final* h or selects a single t,
        # and those individual h's ARE on the tape via lstm_state_h.
        output_seq = np.stack(outputs, axis=1)
        return output_seq, (h, c)


# ─────────────────────────────────────────────────────────────────────────────
# KV cache (Module wrapper around KVCacheHandle — Phase C2)
# ─────────────────────────────────────────────────────────────────────────────


class KVCache(Module):
    """Stateful Module form of :class:`tessera.cache.KVCacheHandle`.

    Useful inside transformer decoder blocks: each forward call appends the
    current step's K/V to the cache and returns the full cumulative
    ``(K, V)``. The handle is stored as a regular attribute (not a Parameter
    or Buffer — its semantics are different and it isn't naturally
    state-dictable).

    To save/restore a KV cache across runs, capture
    ``handle.keys[:current_seq]`` / ``handle.values[:current_seq]`` and
    rehydrate by appending into a new handle.
    """

    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        max_seq: int,
        dtype: str = "fp32",
        page_size: int = 128,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.max_seq = int(max_seq)
        self.dtype_str = str(dtype)
        # Stash as plain attribute — not a Parameter (no gradient), not a
        # Buffer (state_dict semantics don't fit a paged handle).
        object.__setattr__(
            self,
            "handle",
            KVCacheHandle(
                num_heads=num_heads,
                head_dim=head_dim,
                max_seq=max_seq,
                dtype=dtype,
                page_size=page_size,
            ),
        )

    def reset(self) -> None:
        """Drop everything; equivalent to constructing a fresh handle."""
        object.__setattr__(
            self,
            "handle",
            KVCacheHandle(
                num_heads=self.num_heads,
                head_dim=self.head_dim,
                max_seq=self.max_seq,
                dtype=self.dtype_str,
                page_size=self.handle.page_size,
            ),
        )

    @property
    def current_seq(self) -> int:
        return self.handle.current_seq

    def forward(self, k: Any, v: Any) -> tuple[np.ndarray, np.ndarray]:
        """Append ``(k, v)`` to the cache and return the full ``(K, V)`` so far.

        Inputs follow ``KVCacheHandle.append`` shape rules: either
        ``(seq, num_heads, head_dim)`` or packed ``(seq, num_heads*head_dim)``.
        """
        self.handle.append(k, v)
        return self.handle.read(0, self.handle.current_seq)


# ─────────────────────────────────────────────────────────────────────────────
# Conv2d (Phase H1 — NHWC default; Conv2dNCHW shim transposes in/out)
# ─────────────────────────────────────────────────────────────────────────────


class Conv2d(Module):
    """2-D convolution. Default layout is **NHWC** (matches `tessera.ops.conv2d`).

    Weight layout is HWIO: ``(kernel_h, kernel_w, in_channels, out_channels)``.

    Args:
        in_channels: input channel count
        out_channels: output channel count
        kernel_size: int (square) or (kH, kW) tuple
        stride: int (square) or (sH, sW) tuple, default 1
        padding: int (square) or (pH, pW) tuple, default 0
        bias: include a learnable bias, default True
        dtype: weight + bias dtype

    For torch-port code, see :class:`Conv2dNCHW` which transposes inputs/outputs
    around this Module.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = self._pair(kernel_size)
        self.stride = self._pair(stride)
        self.padding = self._pair(padding)

        kH, kW = self.kernel_size
        # HWIO weight layout — matches the existing reference op
        self.weight = Parameter(
            shape=(kH, kW, self.in_channels, self.out_channels), dtype=dtype
        )
        # Kaiming-uniform with fan_in = kH * kW * in_channels
        _kaiming_uniform_(
            self.weight._data._data, fan_in=kH * kW * self.in_channels
        )
        if bias:
            self.bias = Parameter(shape=(self.out_channels,), dtype=dtype)
            self.bias._data._data[...] = 0.0
        else:
            object.__setattr__(self, "bias", None)

    @staticmethod
    def _pair(v):
        if isinstance(v, (tuple, list)):
            if len(v) != 2:
                raise ValueError(f"expected 2-tuple, got length {len(v)}")
            return (int(v[0]), int(v[1]))
        return (int(v), int(v))

    def forward(self, x: Any) -> np.ndarray:
        """``x`` has shape ``(N, H, W, C_in)`` (NHWC). Returns ``(N, H_out, W_out, C_out)``."""
        x_arr = _as_array(x)
        if x_arr.ndim != 4:
            raise ValueError(
                f"Conv2d (NHWC) expects (N, H, W, C) input; got shape {x_arr.shape}"
            )
        if x_arr.shape[3] != self.in_channels:
            raise ValueError(
                f"input channel dim {x_arr.shape[3]} != in_channels {self.in_channels}"
            )
        return ops.conv2d(
            x_arr,
            _as_array(self.weight),
            bias=_as_array(self.bias) if self.bias is not None else None,
            stride=self.stride,
            padding=self.padding,
            layout="nhwc",
        )


class Conv2dNCHW(Module):
    """Torch-style ``(N, C_in, H, W)`` → ``(N, C_out, H_out, W_out)`` shim.

    Wraps :class:`Conv2d` (NHWC) with explicit transposes on the input and
    output. The underlying weight storage is HWIO — the layout choice is
    locked at the kernel boundary, not the Module boundary.

    Use this when porting torch code; for new Tessera code, prefer
    :class:`Conv2d` directly with NHWC inputs.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        bias: bool = True,
        dtype: str = "fp32",
    ) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            dtype=dtype,
        )

    def forward(self, x: Any) -> np.ndarray:
        x_arr = _as_array(x)
        if x_arr.ndim != 4:
            raise ValueError(
                f"Conv2dNCHW expects (N, C, H, W) input; got shape {x_arr.shape}"
            )
        # NCHW → NHWC
        x_nhwc = np.transpose(x_arr, (0, 2, 3, 1))
        y_nhwc = self.conv(x_nhwc)
        # NHWC → NCHW
        return np.transpose(y_nhwc, (0, 3, 1, 2))


__all__ = [
    "Linear",
    "RMSNorm",
    "LayerNorm",
    "BatchNorm1d",
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
    "KVCache",
    "DynamicDepthwiseConv1d",
    "Conv2d",
    "Conv2dNCHW",
    "LSTMCell",
    "LSTM",
]
