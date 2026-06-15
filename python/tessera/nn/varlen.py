"""Variable-length (varlen) scaled-dot-product attention — the packed-sequence
attention primitive.

Motivation (NVIDIA Cosmos 3, *Omnimodal World Models for Physical AI*, 2026-06,
§5.2.2 "Attention Implementation"): a Mixture-of-Transformers forward pass must
co-express two heterogeneous masks in one layer — a **causal** mask over the
Reasoner tokens and a **bidirectional** mask over the concatenation of Reasoner
and Generator tokens. Cosmos found that expressing this with a general-purpose
masked attention (FlexAttention) is *correct but slow*: "the masking structure
is opaque to the kernel, and padding-equivalent work is performed inside
otherwise-skipped attention blocks." Their fix is **two-way flat attention** —
two variable-length SDPA launches per layer keyed on ``cu_seqlens``, which
"yields 22% improvement in end-to-end training throughput compared to a
FlexAttention-based baseline for the Cosmos3-Nano model."

This module makes ``cu_seqlens`` a first-class Tessera contract. ``varlen_sdpa``
is a *single* op that consumes a packed query stream, a packed key/value stream,
and **separate** ``cu_seqlens_q`` / ``cu_seqlens_k`` cumulative-length vectors —
so the rectangular block-diagonal case (each query block sees its own, longer,
key block) is expressible directly, not approximated.

Two formulations are provided and are provably equal (the metamorphic /
"derive validates declare" oracle in ``tests/unit/test_varlen_sdpa.py``):

* :func:`varlen_sdpa` — packs each sample's ``(query_block, kv_block)`` and runs
  it through the rank-3 ``ops.flash_attn`` lane (Apple GPU ``metal_runtime`` /
  numpy reference / ``@jit``-traced). This is the structure Cosmos lowers to two
  varlen kernel launches; on Apple Silicon it is one dispatch per block today.
* :func:`block_diagonal_bias` — materializes the dense additive ``(Lq, Lk)`` mask
  that makes a *single* masked ``ops.flash_attn(attn_bias=...)`` reproduce the
  varlen result. This is the FlexAttention-equivalent reference path; it is the
  numeric oracle the varlen path is checked against.

The FA-3 (Hopper) / NATTEN-CUTLASS (Blackwell) varlen *kernels* themselves are
NVIDIA-frontier and land on Tessera's hardware-gated ``backend_kernel`` wall
(Phase G/H). What is portable today — and what carries the win — is the *op
contract*: packed streams + separate ``cu_seqlens`` + the causal/bidirectional
block semantics.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Union

import numpy as np

from .. import ops

# cu_seqlens may arrive as a plain list or as the int32 prefix-sum array returned
# by cu_seqlens_from_lengths — accept both at the contract boundary.
CuSeqlens = Union[Sequence[int], "np.ndarray"]

__all__ = [
    "cu_seqlens_from_lengths",
    "lengths_from_cu_seqlens",
    "block_diagonal_bias",
    "varlen_sdpa",
]

_NEG_INF = -1e30  # additive-mask "forbidden" sentinel (matches the attn_bias substrate)


def _asarray(x) -> np.ndarray:
    if hasattr(x, "_data"):
        x = x._data
    if hasattr(x, "_data"):
        x = x._data
    return np.asarray(x)


# ---------------------------------------------------------------------------
# cu_seqlens contract
# ---------------------------------------------------------------------------

def cu_seqlens_from_lengths(lengths: Sequence[int]) -> np.ndarray:
    """Cumulative sequence-length offsets — the varlen packing index.

    Given per-sample block lengths ``[l0, l1, ..., l_{n-1}]`` returns the
    ``(n + 1,)`` int32 prefix-sum ``[0, l0, l0+l1, ..., sum(l)]`` so sample ``i``
    owns the packed rows ``[cu[i], cu[i+1])``. This is the exact layout in the
    Cosmos 3 report (Fig. 14: ``cu_seqlens: [0,3,5,9]`` for blocks ``(3,2,4)``).
    """
    lens = np.asarray(list(lengths), dtype=np.int64)
    if lens.ndim != 1:
        raise ValueError("lengths must be a 1-D sequence of per-sample block lengths")
    if np.any(lens < 0):
        raise ValueError("lengths must be non-negative")
    cu = np.zeros(lens.shape[0] + 1, dtype=np.int32)
    cu[1:] = np.cumsum(lens)
    return cu


def lengths_from_cu_seqlens(cu_seqlens: CuSeqlens) -> np.ndarray:
    """Inverse of :func:`cu_seqlens_from_lengths` — per-sample block lengths."""
    cu = np.asarray(list(cu_seqlens), dtype=np.int64)
    if cu.ndim != 1 or cu.shape[0] < 1:
        raise ValueError("cu_seqlens must be a 1-D vector of length n+1")
    if cu[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0; got {cu[0]}")
    d = np.diff(cu)
    if np.any(d < 0):
        raise ValueError("cu_seqlens must be non-decreasing")
    return d.astype(np.int32)


def _validate_pair(cu_q: np.ndarray, cu_k: np.ndarray, total_q: int, total_k: int):
    if cu_q.shape[0] != cu_k.shape[0]:
        raise ValueError(
            f"cu_seqlens_q and cu_seqlens_k must describe the same number of samples; "
            f"got {cu_q.shape[0] - 1} vs {cu_k.shape[0] - 1}"
        )
    if int(cu_q[-1]) != total_q:
        raise ValueError(f"cu_seqlens_q[-1]={int(cu_q[-1])} != packed query rows {total_q}")
    if int(cu_k[-1]) != total_k:
        raise ValueError(f"cu_seqlens_k[-1]={int(cu_k[-1])} != packed key rows {total_k}")


# ---------------------------------------------------------------------------
# Dense reference: block-diagonal additive mask (the oracle bridge)
# ---------------------------------------------------------------------------

def block_diagonal_bias(
    cu_seqlens_q: CuSeqlens,
    cu_seqlens_k: CuSeqlens,
    *,
    causal: bool = False,
) -> np.ndarray:
    """Additive ``(total_q, total_k)`` mask: 0 inside a sample's block, -inf out.

    This makes a *single* dense ``ops.flash_attn(Q, K, V, attn_bias=bias)`` over
    the fully-packed streams reproduce :func:`varlen_sdpa` exactly — the
    FlexAttention-equivalent path Cosmos 3 contrasts against. Query row ``q`` and
    key row ``k`` interact iff they belong to the same sample (block-diagonal);
    when ``causal`` they additionally obey a **bottom-right-aligned** causal rule
    within the block (query offset ``i`` sees key offsets ``0 .. (Lk - Lq + i)``),
    which is the standard varlen-causal convention and reduces to ordinary
    triangular causal when the block is square (``Lq == Lk``).
    """
    cu_q = np.asarray(list(cu_seqlens_q), dtype=np.int64)
    cu_k = np.asarray(list(cu_seqlens_k), dtype=np.int64)
    total_q, total_k = int(cu_q[-1]), int(cu_k[-1])
    _validate_pair(cu_q, cu_k, total_q, total_k)

    bias = np.full((total_q, total_k), _NEG_INF, dtype=np.float32)
    for i in range(cu_q.shape[0] - 1):
        q0, q1 = int(cu_q[i]), int(cu_q[i + 1])
        k0, k1 = int(cu_k[i]), int(cu_k[i + 1])
        Lq, Lk = q1 - q0, k1 - k0
        if Lq == 0 or Lk == 0:
            continue
        block = np.zeros((Lq, Lk), dtype=np.float32)
        if causal:
            qi = np.arange(Lq)[:, None]
            ki = np.arange(Lk)[None, :]
            # bottom-right alignment: last query attends to the whole key block.
            allow = ki <= (Lk - Lq + qi)
            block = np.where(allow, 0.0, _NEG_INF).astype(np.float32)
        bias[q0:q1, k0:k1] = block
    return bias


# ---------------------------------------------------------------------------
# The varlen primitive
# ---------------------------------------------------------------------------

def varlen_sdpa(
    Q,
    K,
    V,
    *,
    cu_seqlens_q: CuSeqlens,
    cu_seqlens_k: CuSeqlens,
    causal: bool = False,
    scale: Optional[float] = None,
    attention_fn: Optional[Callable] = None,
):
    """Variable-length scaled-dot-product attention over packed streams.

    Shapes (heads folded into the batch axis, the rank-3 ``ops.flash_attn`` lane):

        Q : (H, total_q, Dh)   packed query rows for all samples
        K : (H, total_k, Dh)   packed key rows
        V : (H, total_k, Dh)   packed value rows
        →  (H, total_q, Dh)    packed output rows

    ``cu_seqlens_q`` / ``cu_seqlens_k`` are the ``(n+1,)`` prefix-sum offsets from
    :func:`cu_seqlens_from_lengths`; sample ``i`` owns query rows
    ``[cu_q[i], cu_q[i+1])`` and key rows ``[cu_k[i], cu_k[i+1])``. The two vectors
    are **independent** — the rectangular case (Generator pathway: each query
    block attends over a longer ``[Reasoner; Generator]`` key block) is the point
    of this primitive, not a special case.

    Each sample's ``(query_block, kv_block)`` is dispatched through the rank-3
    attention core (``attention_fn`` or ``ops.flash_attn``), so the whole op rides
    the Apple GPU ``metal_runtime`` lane today and is exactly the structure Cosmos
    lowers to one varlen kernel launch per pathway. ``causal`` uses the same
    bottom-right alignment as :func:`block_diagonal_bias`.
    """
    Qa, Ka, Va = _asarray(Q), _asarray(K), _asarray(V)
    if Qa.ndim != 3 or Ka.ndim != 3 or Va.ndim != 3:
        raise ValueError("varlen_sdpa expects rank-3 (H, total, Dh) packed streams")
    H, total_q, Dh = Qa.shape
    if Ka.shape[0] != H or Va.shape[0] != H:
        raise ValueError("Q/K/V must share the (folded) head axis")
    total_k = Ka.shape[1]
    if Va.shape[1] != total_k:
        raise ValueError("K and V must share the packed key length")
    if Ka.shape[2] != Dh or Va.shape[2] != Dh:
        raise ValueError("Q/K/V must share head_dim")

    cu_q = np.asarray(list(cu_seqlens_q), dtype=np.int64)
    cu_k = np.asarray(list(cu_seqlens_k), dtype=np.int64)
    _validate_pair(cu_q, cu_k, total_q, total_k)

    if scale is None:
        scale = float(Dh) ** -0.5
    core = attention_fn if attention_fn is not None else ops.flash_attn

    out = np.empty((H, total_q, Dh), dtype=np.result_type(Qa.dtype, np.float32))
    n = cu_q.shape[0] - 1
    for i in range(n):
        q0, q1 = int(cu_q[i]), int(cu_q[i + 1])
        k0, k1 = int(cu_k[i]), int(cu_k[i + 1])
        Lq, Lk = q1 - q0, k1 - k0
        if Lq == 0:
            continue
        if Lk == 0:
            out[:, q0:q1, :] = 0.0
            continue
        qb = Qa[:, q0:q1, :]
        kb = Ka[:, k0:k1, :]
        vb = Va[:, k0:k1, :]
        bias = None
        if causal:
            # Fold the bottom-right causal rule into an additive bias so the
            # rectangular (Lk != Lq) case is handled uniformly by the core.
            qi = np.arange(Lq)[:, None]
            ki = np.arange(Lk)[None, :]
            allow = ki <= (Lk - Lq + qi)
            b2 = np.where(allow, 0.0, _NEG_INF).astype(np.float32)
            bias = np.broadcast_to(b2, (H, Lq, Lk)).astype(np.float32)
        ob = core(qb, kb, vb, scale=scale, causal=False, attn_bias=bias)
        out[:, q0:q1, :] = _asarray(ob)
    return out
