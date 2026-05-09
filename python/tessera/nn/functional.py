"""Functional `tessera.nn` surface — stateless layer wrappers over `tessera.ops`.

Use these when you don't want to allocate `Parameter` storage; pass weights in
explicitly as numpy arrays / `DistributedArray`s. Stateful versions live in
`tessera.nn` (`Linear`, `RMSNorm`, etc.) and just compose these calls.

Both `tessera.nn.linear` and `tessera.nn.functional.linear` resolve here.
"""

from __future__ import annotations

import numpy as np

from .. import ops


def linear(x, W, bias=None):
    """y = x @ W (+ bias). Bias is optional and broadcast over leading dims.

    Composed through `ops.gemm` (+ `ops.add` when bias is non-None) so an
    active autodiff tape sees every primitive.
    """
    y = ops.gemm(x, W)
    if bias is not None:
        if hasattr(bias, "_data"):
            bias = bias._data
        y = ops.add(y, bias)
    return y


def rms_norm(x, weight=None, eps: float = 1e-5):
    """RMSNorm: x / sqrt(mean(x*x) + eps), optionally scaled by `weight`.

    Composed through `ops.rmsnorm` and `ops.mul` so the autodiff tape captures
    both the normalization and the affine.
    """
    y = ops.rmsnorm(x, eps=eps)
    if weight is not None:
        if hasattr(weight, "_data"):
            weight = weight._data
        y = ops.mul(y, weight)
    return y


def swiglu(x, W_gate, W_up, W_down):
    """SwiGLU MLP block: (silu(x @ W_gate) * (x @ W_up)) @ W_down.

    Composed through primitive ops (`gemm` / `silu_mul` / `gemm`) so the
    autodiff tape sees every step AND the Schedule IR fusion recognizer
    matches the `matmul → silu_mul → matmul` chain. Functionally equivalent
    to `ops.swiglu(...)`; either spelling works.
    """
    gate = ops.gemm(x, W_gate)
    up = ops.gemm(x, W_up)
    hidden = ops.silu_mul(gate, up)
    return ops.gemm(hidden, W_down)


def multi_head_attention(
    Q,
    K,
    V,
    num_heads: int,
    scale: float | None = None,
    causal: bool = False,
    dropout_p: float = 0.0,
    seed: int | None = None,
):
    """Multi-head attention via `ops.flash_attn`.

    Inputs are `[B, S, H*D]` tensors. They are reshaped to `[B, num_heads, S, D]`,
    passed through `ops.flash_attn`, then reshaped back to `[B, S, H*D]`.

    For raw `[B, H, S, D]` inputs, call `ops.flash_attn` directly.
    """
    if hasattr(Q, "_data"):
        Q = Q._data
    if hasattr(K, "_data"):
        K = K._data
    if hasattr(V, "_data"):
        V = V._data
    Q, K, V = np.asarray(Q), np.asarray(K), np.asarray(V)
    if Q.ndim != 3 or K.ndim != 3 or V.ndim != 3:
        raise ValueError(
            "multi_head_attention expects [B, S, H*D] inputs; "
            "for raw [B, H, S, D] tensors, call tessera.ops.flash_attn directly"
        )
    B, Sq, hd = Q.shape
    Sk = K.shape[1]
    Sv = V.shape[1]
    if K.shape[0] != B or V.shape[0] != B:
        raise ValueError("Q/K/V must share batch dim")
    if K.shape[2] != hd or V.shape[2] != hd:
        raise ValueError("Q/K/V must share hidden dim")
    if Sk != Sv:
        raise ValueError(f"K and V must share sequence length; got {Sk} vs {Sv}")
    if hd % num_heads != 0:
        raise ValueError(f"hidden dim {hd} not divisible by num_heads {num_heads}")
    D = hd // num_heads

    def split(t):
        S_t = t.shape[1]
        return t.reshape(B, S_t, num_heads, D).transpose(0, 2, 1, 3)

    out = ops.flash_attn(
        split(Q),
        split(K),
        split(V),
        scale=scale,
        causal=causal,
        dropout_p=dropout_p,
        seed=seed,
    )
    return out.transpose(0, 2, 1, 3).reshape(B, Sq, hd)


# Alias for torch-style `nn.flash_attention` callsites (same signature as ops.flash_attn).
flash_attention = ops.flash_attn


__all__ = [
    "linear",
    "rms_norm",
    "swiglu",
    "multi_head_attention",
    "flash_attention",
]
