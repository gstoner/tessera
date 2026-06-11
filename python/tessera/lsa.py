"""Reference oracle for Lookahead Sparse Attention (LSA) — experimental.

Pure-NumPy contract for two new standalone primitives:

  * ``memory_index_select`` — sigmoid-threshold selection over compressed
    historical block keys.  Selection is the genuinely new piece: unlike
    ``tessera.memory.memory_read`` (top-k similarity + softmax weighting) this
    is a ``sigmoid(score) >= threshold`` boolean retrieval with union across
    indexer layers and an empty-selection fallback to the query's local block.

  * ``lookahead_sparse_attention`` — composite attention *policy*: each query
    attends over the union of its causal local window and the tokens of the
    historical blocks selected by ``memory_index_select``.  This is an explicit
    composition of local-window attention + selected-block sparse attention, not
    a new kernel.

Scope (see ``docs/audit/domain/archive/lsa_scope.md``, decisions D1–D5):

  D1  Naming.  This is ``lookahead_sparse_attention``; there is deliberately no
      "FlashMemory" branding until the CPU↔GPU KV-tiering substrate exists.
  D2  Statefulness.  The op is **pure per call** — ``tau`` / ``threshold`` /
      ``window_size`` / ``block_size`` are attributes.  The every-``tau``
      lookahead *cadence* is owned by the caller's decode loop, not the op; a
      single forward call performs exactly one selection.
  D3  Selector semantics.  ``memory_index_select`` does **not** reuse
      ``memory_read`` — it is a sigmoid-threshold boolean selector.
  D4  Memory tiering.  v1 keeps selection host-mediated + data-dependent
      (matching the existing ``deepseek_sparse_attention`` Apple-GPU lane).  No
      CPU cold-pool ↔ GPU-resident KV staging — that is explicitly deferred.
  D5  Conformance honesty.  ``tau=64`` / ``threshold=0.5`` are *chosen test
      fixtures*, not a reproduced paper result.  This module makes no
      equivalence claim against any external implementation.

The functions are intentionally functional + deterministic so the Graph IR /
runtime lanes can be validated against them bit-for-bit at fp32 tolerance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _unwrap(value):
    return value._data if hasattr(value, "_data") else value


def _asarray(value, dtype=None):
    arr = np.asarray(_unwrap(value))
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable logistic sigmoid.
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


@dataclass(frozen=True)
class SelectionResult:
    """Result of :func:`memory_index_select`.

    ``mask`` (bool, shape ``(..., S_q, num_blocks)``) is the canonical,
    deterministic, dense contract — a block is ``True`` for a query iff it was
    retained after thresholding, causal masking, union, and fallback.

    ``scores`` carries the post-union sigmoid probabilities (max over layers)
    for introspection / debugging; it is not part of the equivalence contract.
    """

    mask: np.ndarray
    scores: np.ndarray

    def selected_blocks(self, *index) -> np.ndarray:
        """Ascending block indices selected for the addressed query row.

        ``index`` addresses the leading ``(..., S_q)`` axes (e.g.
        ``result.selected_blocks(b, h, sq)``).  Ties are broken by lowest block
        index first (``np.flatnonzero`` is already ascending).
        """
        return np.flatnonzero(self.mask[index])


def compress_block_keys(K: np.ndarray, *, block_size: int) -> np.ndarray:
    """Mean-pool ``K`` into per-block summary keys.

    ``K`` is rank-4 ``(B, H, S, D)`` and ``S`` must be divisible by
    ``block_size``.  Returns ``(B, H, num_blocks, D)``.  This is the zero-param
    default indexer-key construction (matching ``compress_blocks`` in the NSA
    reference); callers may instead pass learned indexer keys directly.
    """
    arr = _asarray(K, np.float64)
    if arr.ndim != 4:
        raise ValueError("compress_block_keys expects rank-4 (B, H, S, D) K")
    B, H, S, D = arr.shape
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if S % block_size != 0:
        raise ValueError(f"S={S} not divisible by block_size={block_size}")
    nb = S // block_size
    return arr.reshape(B, H, nb, block_size, D).mean(axis=-2)


def memory_index_select(
    indexer_keys,
    query,
    *,
    block_size: int,
    threshold: float = 0.5,
    causal: bool = True,
    scale: float | None = None,
    fallback_local: bool = True,
) -> SelectionResult:
    """Sigmoid-threshold selection of historical blocks (D3).

    Parameters
    ----------
    indexer_keys
        Either one ``(B, H, num_blocks, Dk)`` array, or a list/tuple of such
        arrays (one per indexer layer).  When a list is supplied the per-layer
        boolean masks are **unioned** (a block is selected if *any* layer
        selects it).
    query
        ``(B, H, S_q, Dk)`` per-token selection query.
    block_size
        Tokens per historical block; used to map each query token to its own
        block for causal masking and fallback.
    threshold
        Selection cutoff.  A block is selected when ``sigmoid(score) >=
        threshold``.  The ``>=`` makes the exact-tie case (``score`` such that
        ``sigmoid(score) == threshold``) deterministically *selected*.
    causal
        When ``True``, a query in block ``qb`` may only select blocks
        ``blk <= qb``.
    scale
        Score scale; defaults to ``1/sqrt(Dk)``.
    fallback_local
        When ``True`` (default), any query row left with an empty selection
        after thresholding + causal masking selects its own block ``qb``.  This
        guarantees a non-empty footprint for the standalone selector.

    Returns
    -------
    SelectionResult
        ``mask`` of shape ``(B, H, S_q, num_blocks)`` and the unioned scores.
    """

    layers = list(indexer_keys) if isinstance(indexer_keys, (list, tuple)) else [indexer_keys]
    if not layers:
        raise ValueError("indexer_keys must contain at least one layer")
    q = _asarray(query, np.float64)
    if q.ndim != 4:
        raise ValueError("query must be rank-4 (B, H, S_q, Dk)")
    B, H, S_q, Dk = q.shape
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")

    union_mask: np.ndarray | None = None
    union_scores: np.ndarray | None = None
    num_blocks = None
    for keys in layers:
        k = _asarray(keys, np.float64)
        if k.ndim != 4:
            raise ValueError("each indexer_keys layer must be rank-4 (B, H, num_blocks, Dk)")
        if k.shape[0] != B or k.shape[1] != H or k.shape[-1] != Dk:
            raise ValueError("indexer_keys must match query (B, H, *, Dk)")
        nb = int(k.shape[2])
        if num_blocks is None:
            num_blocks = nb
        elif nb != num_blocks:
            raise ValueError("all indexer_keys layers must share num_blocks")
        sc = float(scale) if scale is not None else 1.0 / np.sqrt(Dk)
        scores = np.matmul(q, np.swapaxes(k, -1, -2)) * sc  # (B, H, S_q, num_blocks)
        probs = _sigmoid(scores)
        layer_mask = probs >= float(threshold)
        if union_mask is None:
            union_mask = layer_mask
            union_scores = probs
        else:
            assert union_scores is not None  # set together with union_mask
            union_mask = union_mask | layer_mask
            union_scores = np.maximum(union_scores, probs)

    assert union_mask is not None and union_scores is not None and num_blocks is not None

    # Causal block mask: query token sq lives in block sq // block_size.
    qb = np.arange(S_q) // block_size  # (S_q,)
    if causal:
        future = np.arange(num_blocks)[None, None, None, :] > qb[None, None, :, None]
        union_mask = union_mask & ~future
        union_scores = np.where(future, 0.0, union_scores)

    if fallback_local:
        empty = np.asarray(~union_mask.any(axis=-1))  # (B, H, S_q)
        if empty.any():
            own = np.clip(qb, 0, num_blocks - 1)  # (S_q,)
            own_onehot = (np.arange(num_blocks)[None, :] == own[:, None])  # (S_q, nb)
            own_onehot = np.broadcast_to(own_onehot[None, None], union_mask.shape)
            union_mask = np.where(empty[..., None], own_onehot, union_mask)

    return SelectionResult(mask=np.ascontiguousarray(union_mask), scores=np.ascontiguousarray(union_scores))


def _local_window_indices(sq: int, S: int, window_size: int, causal: bool) -> np.ndarray:
    if causal:
        start = max(0, sq - window_size + 1)
        return np.arange(start, sq + 1)
    half = window_size // 2
    start = max(0, sq - half)
    end = min(S - 1, sq + half)
    return np.arange(start, end + 1)


def lookahead_sparse_attention(
    Q,
    K,
    V,
    *,
    window_size: int,
    block_size: int,
    tau: int = 64,
    threshold: float = 0.5,
    causal: bool = True,
    indexer_keys=None,
    scale: float | None = None,
):
    """Composite lookahead-sparse-attention policy (D2 — pure per call).

    For each query token, the active footprint is the **union** of

      1. its causal local window of ``window_size`` tokens, and
      2. the tokens of every historical block selected by
         :func:`memory_index_select` (sigmoid-threshold, causal).

    A single softmax attention is then computed over that footprint.  ``tau`` is
    accepted and validated (it is the caller-owned re-selection cadence; see
    D2) but does not change a single forward call's math.

    Q/K/V are rank-4 ``(B, H, S, D)``.  ``S`` must be divisible by
    ``block_size``.  Returns ``(B, H, S, Dv)``.
    """
    q = _asarray(Q, np.float64)
    k = _asarray(K, np.float64)
    v = _asarray(V, np.float64)
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("lookahead_sparse_attention expects rank-4 (B, H, S, D) Q/K/V")
    B, H, S, D = q.shape
    if int(tau) <= 0:
        raise ValueError("tau must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if block_size <= 0 or S % block_size != 0:
        raise ValueError(f"S={S} not divisible by block_size={block_size}")
    Dv = int(v.shape[-1])
    sc = float(scale) if scale is not None else 1.0 / np.sqrt(D)

    keys = indexer_keys if indexer_keys is not None else compress_block_keys(k, block_size=block_size)
    selection = memory_index_select(
        keys, q, block_size=block_size, threshold=threshold, causal=causal, fallback_local=True,
    )
    mask = selection.mask  # (B, H, S, num_blocks)

    out = np.zeros((B, H, S, Dv), dtype=np.float64)
    for b in range(B):
        for h in range(H):
            for sq in range(S):
                local = _local_window_indices(sq, S, window_size, causal)
                blocks = np.flatnonzero(mask[b, h, sq])
                block_tokens = [
                    np.arange(blk * block_size, blk * block_size + block_size) for blk in blocks
                ]
                footprint = local if not block_tokens else np.concatenate([local, *block_tokens])
                if causal:
                    footprint = footprint[footprint <= sq]
                footprint = np.unique(footprint)  # sorted + de-duplicated
                ks = k[b, h, footprint]            # (T, D)
                vs = v[b, h, footprint]            # (T, Dv)
                s = (q[b, h, sq] @ ks.T) * sc
                s = s - s.max()
                w = np.exp(s)
                w = w / w.sum()
                out[b, h, sq] = w @ vs
    return out.astype(np.result_type(_asarray(Q), _asarray(K), _asarray(V)), copy=False)


# ─────────────────────────────────────────────────────────────────────────────
# Indexer training (Gap 4 in the scope doc, "indexer-key learning loop").
#
# memory_index_select is non-differentiable (hard sigmoid threshold → bool mask),
# so the indexer keys cannot be learned through it directly. The two helpers
# below are the differentiable scoring surface a *user* training loop drives (the
# loop itself stays outside the compiler, per the scope decision):
#
#   * memory_index_score — the indexer's scoring head: sigmoid(q·kᵀ·scale),
#     fully differentiable in both the indexer keys and the query. Put a
#     supervision / auxiliary loss on these probabilities to train the keys.
#   * memory_index_select_ste — hard selection in the forward pass, straight-
#     through (sigmoid) gradient in the backward pass, so a downstream loss on
#     the *hard* selection still trains the indexer keys.
# ─────────────────────────────────────────────────────────────────────────────


def memory_index_score(indexer_keys, query, *, scale: float | None = None):
    """Differentiable indexer scoring head — ``sigmoid(query·keysᵀ·scale)``.

    ``indexer_keys`` is ``(B, H, num_blocks, Dk)`` and ``query`` is
    ``(B, H, S_q, Dk)``; returns per-block selection probabilities
    ``(B, H, S_q, num_blocks)``. Differentiable in both inputs (closed-form VJP
    + JVP registered), so the indexer keys are trainable.
    """
    k = _asarray(indexer_keys, np.float64)
    q = _asarray(query, np.float64)
    if k.ndim != 4 or q.ndim != 4 or k.shape[-1] != q.shape[-1]:
        raise ValueError("indexer_keys (B,H,nb,Dk) and query (B,H,S_q,Dk) required")
    sc = float(scale) if scale is not None else 1.0 / np.sqrt(q.shape[-1])
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * sc
    return _sigmoid(scores)


def memory_index_select_ste(indexer_keys, query, *, threshold: float = 0.5,
                            scale: float | None = None):
    """Straight-through hard block selection for indexer training.

    Forward: ``(memory_index_score >= threshold)`` as a float 0/1 mask. Backward
    (registered VJP): the straight-through estimator routes the upstream
    gradient through the smooth ``sigmoid`` score, so a loss on the hard
    selection still trains the indexer keys. Forward-mode (JVP) is not
    applicable (the hard step has no meaningful directional derivative).
    """
    probs = memory_index_score(indexer_keys, query, scale=scale)
    return (probs >= float(threshold)).astype(np.float64)


__all__ = [
    "SelectionResult",
    "compress_block_keys",
    "lookahead_sparse_attention",
    "memory_index_select",
    "memory_index_score",
    "memory_index_select_ste",
]
