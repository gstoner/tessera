"""``tessera.stdlib.attention`` — production attention pillars (M3 + M4).

Two model-class attention primitives the frontier models need, built as
compiler-lowerable reference algorithms with oracle gates (the pattern M1/M2
established: a real algorithm + a vertical/metamorphic oracle + an Apple GPU
compose hook, with the fused MSL kernel as a documented follow-up):

**M3 — MLA (multi-head latent attention) as a production decode primitive.**
:class:`MLAWeights` + :func:`mla_attention` implement DeepSeek-style MLA with a
**decoupled (partial) RoPE** split and the **weight-absorption** trick: the
nope-key up-projection ``W_uk`` is folded into the query and ``W_uv`` into the
output, so the per-head K/V are *never materialized* — only the compressed
latent ``c`` (and a tiny shared RoPE key) are read from cache.  The headline
oracle is ``absorb ≡ no-absorb`` (numerically identical).  :func:`mla_prefill`
fills the latent cache from a prompt; :func:`mla_decode_step` advances it one
chunk via :class:`tessera.cache.LatentKVCacheHandle` (paged latent + rope).

**M4 — DSA (DeepSeek sparse attention) native block-sparse lowering.**
:func:`dsa_block_index` (lightning-indexer-style per-GQA-group block scores) →
:func:`dsa_select_blocks` (top-k block selection, causal + forced-local) →
:func:`dsa_block_sparse_attention` (exact attention over only the selected
blocks).  The headline oracle is ``select-all ≡ dense`` (a DESIL cross-path:
when top-k covers every block the sparse path must equal dense causal
attention), plus a metamorphic invariant (perturbing unselected blocks does not
change the output).

Honesty: the heavy matmuls compose on the Metal lane (``backend="apple_gpu"``)
where batchable; a single fused MLA-absorb / block-sparse MSL kernel is the
M3.1 / M4.1 follow-up.  What is real today: the absorption algorithm + decoupled
RoPE + latent-cache paging (M3), and the indexer + exact block-sparse algorithm
+ oracles (M4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _arr(x) -> np.ndarray:
    return np.asarray(x._data if hasattr(x, "_data") else x)


def _softmax_last(x: np.ndarray) -> np.ndarray:
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


def _matmul(a: np.ndarray, b: np.ndarray, backend: str) -> np.ndarray:
    """Backend-selected 2-D matmul (M3.1/M4.1 composed Apple GPU lane).

    ``backend="apple_gpu"`` runs on the Metal matmul lane (honest compose — the
    fused MLA-absorb / block-sparse MSL kernels are the perf follow-up); falls
    back to numpy on any Metal miss.  ``"reference"`` is numpy.
    """
    if backend == "apple_gpu":
        try:
            from .. import _apple_gpu_backend as agb
            return np.asarray(agb.gpu_matmul(
                np.ascontiguousarray(a.astype(np.float32)),
                np.ascontiguousarray(b.astype(np.float32))), dtype=np.float64)
        except Exception:                              # noqa: BLE001 — honest fallback
            return a.astype(np.float64) @ b.astype(np.float64)
    return a.astype(np.float64) @ b.astype(np.float64)


def _bmm(a: np.ndarray, b: np.ndarray, backend: str) -> np.ndarray:
    """``(L, m, k) @ (L, k, n) -> (L, m, n)`` over leading dim ``L`` via the
    backend matmul (loops the leading dim on the Apple GPU lane)."""
    if backend == "reference":
        return a.astype(np.float64) @ b.astype(np.float64)
    return np.stack([_matmul(a[i], b[i], backend) for i in range(a.shape[0])])


# ── shared RoPE helper (decoupled / partial) ─────────────────────────────────
def apply_rope(x, positions, *, base: float = 10000.0) -> np.ndarray:
    """Rotate-half RoPE over the last axis at ``positions`` (axis -2).

    ``x`` ``(..., S, d)`` with even ``d``; ``positions`` ``(S,)``.  cos/sin
    ``(S, d/2)`` broadcast against any leading dims (heads), so this applies to
    both a shared ``(S, d)`` rope key and per-head ``(H, S, d)`` rope queries.
    """
    x = _arr(x).astype(np.float64)
    d = x.shape[-1]
    if d % 2 != 0:
        raise ValueError(f"RoPE needs an even rope dim; got {d}")
    half = d // 2
    pos = np.asarray(positions, dtype=np.float64).reshape(-1)
    inv_freq = base ** (-np.arange(half, dtype=np.float64) / half)
    ang = pos[:, None] * inv_freq[None, :]              # (S, half)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


# ════════════════════════════════════════════════════════════════════════════
# M3 — MLA
# ════════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class MLAWeights:
    """MLA projection weights (single batch, ``num_heads`` heads).

    ``w_dkv`` ``(H, d_c)`` compress; ``w_uk`` ``(d_c, num_heads·d_nope)`` /
    ``w_uv`` ``(d_c, num_heads·d_v)`` expand the latent to per-head nope-K / V;
    ``w_q`` ``(H, num_heads·(d_nope+d_rope))`` query (nope ++ rope per head);
    ``w_kr`` ``(H, d_rope)`` the decoupled (shared-across-heads) RoPE key.
    """

    w_dkv: np.ndarray
    w_uk: np.ndarray
    w_uv: np.ndarray
    w_q: np.ndarray
    w_kr: np.ndarray
    num_heads: int
    d_nope: int
    d_rope: int
    d_v: int

    @property
    def d_c(self) -> int:
        return int(self.w_dkv.shape[1])


def compress_latent(x, weights: MLAWeights):
    """The cacheable MLA state for tokens ``x`` ``(S, H)``:
    ``(c = x·W_dkv  (S, d_c),  k_rope = RoPE(x·W_kr)  (S, d_rope))``."""
    xa = _arr(x).astype(np.float64)
    c = xa @ _arr(weights.w_dkv).astype(np.float64)
    return c, xa @ _arr(weights.w_kr).astype(np.float64)


def _mla_scores_values(q_nope, q_rope, c_kv, k_rope_kv, weights: MLAWeights,
                       *, absorb: bool):
    """Return per-head ``(scores (Hh,Sq,T), values getter)`` for MLA.

    ``absorb`` folds ``W_uk`` into the query (score reads only the latent ``c``,
    never the materialized nope-K) and ``W_uv`` into the output."""
    Hh, d_nope, d_c = weights.num_heads, weights.d_nope, weights.d_c
    c = np.asarray(c_kv, dtype=np.float64)              # (T, d_c)
    wuk = _arr(weights.w_uk).astype(np.float64).reshape(d_c, Hh, d_nope)
    # rope score term (shared rope key across heads): (Hh,Sq,T)
    score_rope = np.einsum("hsd,td->hst", q_rope, np.asarray(k_rope_kv))
    if absorb:
        # q_absorbed[h,s,:d_c] = q_nope[h,s] @ w_uk_h^T  → score = q_abs · c
        q_abs = np.einsum("hsd,chd->hsc", q_nope, wuk)  # (Hh,Sq,d_c)
        score_nope = np.einsum("hsc,tc->hst", q_abs, c)
    else:
        k_nope = np.einsum("tc,chd->thd", c, wuk)       # (T,Hh,d_nope) MATERIALIZED
        score_nope = np.einsum("hsd,thd->hst", q_nope, k_nope)
    return score_nope + score_rope


def mla_attention(x_q, c_kv, k_rope_kv, weights: MLAWeights, *,
                  q_positions=None, kv_start: int = 0,
                  absorb: bool = True, causal: bool = True, scale=None,
                  backend: str = "reference"):
    """MLA attention for query tokens ``x_q`` ``(Sq, H)`` against a cached
    context ``(c_kv (T,d_c), k_rope_kv (T,d_rope))``.

    Returns ``O`` ``(Sq, num_heads·d_v)``.  ``q_positions`` default to the
    decode positions ``kv_start + arange(Sq)`` (so a decode chunk's queries sit
    at the end of the context).  ``absorb`` selects the production absorb path
    (latent-only reads) vs the explicit-K/V reference — they must agree.
    ``backend="apple_gpu"`` runs the per-head score/output matmuls on Metal
    (requires ``absorb``; M3.1 composed lane).
    """
    xa = _arr(x_q).astype(np.float64)
    Sq = xa.shape[0]
    Hh, d_nope, d_rope, d_v, d_c = (weights.num_heads, weights.d_nope,
                                    weights.d_rope, weights.d_v, weights.d_c)
    T = int(np.asarray(c_kv).shape[0])
    if scale is None:
        scale = 1.0 / np.sqrt(d_nope + d_rope)
    if q_positions is None:
        q_positions = kv_start + np.arange(Sq)
    if backend != "reference" and not absorb:
        raise ValueError("apple_gpu MLA lane requires absorb=True (latent-only path)")

    q = (xa @ _arr(weights.w_q).astype(np.float64)).reshape(Sq, Hh, d_nope + d_rope)
    q_nope = q[..., :d_nope].transpose(1, 0, 2)         # (Hh,Sq,d_nope)
    q_rope = apply_rope(q[..., d_nope:].transpose(1, 0, 2), q_positions)  # (Hh,Sq,d_rope)
    c = np.asarray(c_kv, dtype=np.float64)
    kr = np.asarray(k_rope_kv, dtype=np.float64)
    wuk = _arr(weights.w_uk).astype(np.float64).reshape(d_c, Hh, d_nope)
    wuv = _arr(weights.w_uv).astype(np.float64).reshape(d_c, Hh, d_v)

    if backend != "reference":
        # absorbed per-head 2-D matmuls on the Metal lane
        q_abs = _bmm(q_nope, np.transpose(wuk, (1, 0, 2)).transpose(0, 2, 1),
                     "reference")                       # (Hh,Sq,d_c) = q_nope @ wuk_h^T
        cT = np.broadcast_to(c.T, (Hh, d_c, T))
        krT = np.broadcast_to(kr.T, (Hh, d_rope, T))
        scores = (_bmm(q_abs, cT, backend) + _bmm(q_rope, krT, backend)) * scale
    else:
        scores = _mla_scores_values(q_nope, q_rope, c_kv, k_rope_kv, weights,
                                    absorb=absorb) * scale
    if causal:
        qpos = np.asarray(q_positions).reshape(Sq, 1)
        kpos = np.arange(T).reshape(1, T)
        scores = np.where((kpos > qpos)[None], -np.inf, scores)
    w = _softmax_last(scores)                           # (Hh,Sq,T)

    if absorb:
        if backend != "reference":
            cv = _bmm(w, np.broadcast_to(c, (Hh, T, d_c)), backend)   # (Hh,Sq,d_c)
            o = _bmm(cv, np.transpose(wuv, (1, 0, 2)), backend)       # (Hh,Sq,d_v)
        else:
            cv = np.einsum("hst,tc->hsc", w, c)         # V never materialized
            o = np.einsum("hsc,chd->hsd", cv, wuv)
    else:
        v = np.einsum("tc,chd->thd", c, wuv)            # (T,Hh,d_v) MATERIALIZED
        o = np.einsum("hst,thd->hsd", w, v)
    return o.transpose(1, 0, 2).reshape(Sq, Hh * d_v)


def mla_prefill(x, weights: MLAWeights, *, scale=None, causal: bool = True):
    """Prefill: build the latent cache from a prompt ``x`` ``(S, H)`` and run
    full (causal) MLA over it.  Returns ``(O, c_cache (S,d_c), k_rope_cache
    (S,d_rope))`` — the caches seed :func:`mla_decode_step`."""
    c, k_rope = compress_latent(x, weights)
    o = mla_attention(x, c, k_rope, weights, q_positions=np.arange(c.shape[0]),
                      kv_start=0, absorb=True, causal=causal, scale=scale)
    return o, c, k_rope


def mla_decode_step(x_t, latent_cache, rope_cache, weights: MLAWeights, *,
                    absorb: bool = True, scale=None):
    """Advance MLA decode by a chunk ``x_t`` ``(n_new, H)`` using paged
    :class:`tessera.cache.LatentKVCacheHandle`s for the latent and rope key.

    Appends the new tokens' compressed state to both caches, then attends the
    new queries against the *full* cached context.  Returns ``O`` ``(n_new,
    num_heads·d_v)``.  The cache holds only ``d_c + d_rope`` per token — the MLA
    memory win.
    """
    kv_start = int(latent_cache.current_seq)
    c_new, kr_new = compress_latent(x_t, weights)
    latent_cache.append(c_new)
    rope_cache.append(kr_new)
    c_all = latent_cache.read(0, latent_cache.current_seq)
    kr_all = rope_cache.read(0, rope_cache.current_seq)
    return mla_attention(x_t, c_all, kr_all, weights,
                         q_positions=kv_start + np.arange(_arr(x_t).shape[0]),
                         kv_start=kv_start, absorb=absorb, causal=True, scale=scale)


# ════════════════════════════════════════════════════════════════════════════
# M4 — DSA (block-sparse)
# ════════════════════════════════════════════════════════════════════════════
def dsa_block_index(Q, K, *, block_size: int, scale=None):
    """Lightning-indexer-style per-GQA-group KV-block scores.

    ``Q`` ``(B, Hq, Sq, D)``, ``K`` ``(B, Hkv, Sk, D)`` with ``Hq % Hkv == 0``.
    Query heads in a group are mean-pooled; each KV block is summarized by its
    mean key.  Returns raw scores ``(B, Hkv, Sq, num_blocks)`` for top-k
    selection (exp-free — selection consumes the dot products directly).
    """
    Qa, Ka = _arr(Q).astype(np.float64), _arr(K).astype(np.float64)
    if Qa.ndim != 4 or Ka.ndim != 4:
        raise ValueError("dsa_block_index expects rank-4 Q/K")
    B, Hq, Sq, D = Qa.shape
    Bk, Hkv, Sk, Dk = Ka.shape
    if Hkv == 0 or Hq % Hkv != 0 or Sk % block_size != 0:
        raise ValueError("require Hq % Hkv == 0 and Sk % block_size == 0")
    g, nb = Hq // Hkv, Sk // block_size
    if scale is None:
        scale = 1.0 / np.sqrt(D)
    k_c = Ka.reshape(B, Hkv, nb, block_size, D).mean(axis=3)        # (B,Hkv,nb,D)
    q_grp = Qa.reshape(B, Hkv, g, Sq, D).mean(axis=2)              # (B,Hkv,Sq,D)
    return np.matmul(q_grp, np.swapaxes(k_c, -1, -2)) * scale       # (B,Hkv,Sq,nb)


def dsa_select_blocks(scores, *, top_k: int, block_size: int,
                      causal: bool = True, force_local: bool = True,
                      q_positions=None):
    """Top-k block selection from :func:`dsa_block_index` scores.

    Returns a boolean keep-mask ``(B, Hkv, Sq, num_blocks)``.  ``q_positions``
    (default ``arange(Sq)``) gives each query's *global* position so the same
    rule applies to a prefill (positions ``0..S-1``) and an incremental decode
    step (a single query at a large offset) — the property that makes DSA
    decode-loop-consistent.

    Under ``causal`` the ranking pool is **strictly-past blocks only** (the
    query's own and future blocks are excluded from ranking); ``force_local``
    then always re-adds the own block.  This is deliberate: the own/partial
    block's indexer summary depends on how many of its tokens exist (different
    in prefill vs. decode), so letting it compete in the ranking would make the
    selection of *other* blocks differ between a full forward and an incremental
    decode.  Ranking only fully-past blocks (whose summaries are identical in
    both) keeps KV-cached decode ≡ recompute.  Deterministic (ties → lowest id).
    """
    s = _arr(scores).astype(np.float64).copy()
    B, Hkv, Sq, nb = s.shape
    if top_k > nb:
        raise ValueError(f"top_k={top_k} > num_blocks={nb}")
    qpos = np.arange(Sq) if q_positions is None else np.asarray(q_positions).reshape(-1)
    q_block = (qpos // block_size).astype(np.int64)
    blk = np.arange(nb)
    future = blk[None, None, None, :] > q_block[None, None, :, None]
    own = blk[None, None, None, :] == q_block[None, None, :, None]
    rank = s
    if causal:
        rank = np.where(future | own, -np.inf, s)      # rank strictly-past only
    order = np.argsort(np.where(np.isneginf(rank), np.inf, -rank), axis=-1, kind="stable")
    keep = np.zeros((B, Hkv, Sq, nb), dtype=bool)
    np.put_along_axis(keep, order[..., :min(top_k, nb)], True, axis=-1)
    if causal:
        keep &= ~(future | own)                        # ranked keeps are past-only
    if force_local:
        keep |= own
    if causal:
        keep &= ~future                                # safety: never a future block
    return keep


def dsa_block_sparse_attention(Q, K, V, *, top_k_blocks: int, block_size: int,
                               causal: bool = True, scale=None,
                               index_scale=None, backend: str = "reference",
                               q_positions=None):
    """Exact attention over only the top-k selected KV blocks (per GQA group,
    per query).  ``Q`` ``(B,Hq,Sq,D)``, ``K``/``V`` ``(B,Hkv,Sk,Dk/Dv)``.

    The indexer (:func:`dsa_block_index`) scores blocks; selection
    (:func:`dsa_select_blocks`) keeps top-k strictly-past blocks (+ causal +
    forced-local); then dense attention runs over exactly the tokens in the kept
    blocks.  Returns ``(B, Hq, Sq, Dv)``.  With ``top_k_blocks == num_blocks``
    this equals dense causal attention (the select-all oracle).

    ``q_positions`` (default ``arange(Sq)``) gives the queries' global positions
    so this works for an incremental decode step (single query at a large
    offset) as well as a prefill.  ``Sk`` need not be a multiple of
    ``block_size`` — the keys are zero-padded internally and the pad positions
    are causally masked out (so decode at a growing context just works).
    """
    Qa, Ka, Va = (_arr(Q).astype(np.float64), _arr(K).astype(np.float64),
                  _arr(V).astype(np.float64))
    B, Hq, Sq, D = Qa.shape
    Hkv, Sk_real = Ka.shape[1], Ka.shape[2]
    Dv = Va.shape[-1]
    g = Hq // Hkv
    if scale is None:
        scale = 1.0 / np.sqrt(D)
    qpos = np.arange(Sq) if q_positions is None else np.asarray(q_positions).reshape(-1)
    # pad K/V up to a multiple of block_size so blocks tile cleanly; pad keys
    # sit at the largest indices (future) → causally masked, never contribute.
    pad = (-Sk_real) % block_size
    if pad:
        Ka = np.concatenate([Ka, np.zeros((B, Hkv, pad, D))], axis=2)
        Va = np.concatenate([Va, np.zeros((B, Hkv, pad, Dv))], axis=2)
    Sk = Ka.shape[2]
    nb = Sk // block_size
    idx = dsa_block_index(Qa, Ka, block_size=block_size, scale=index_scale)
    keep = dsa_select_blocks(idx, top_k=min(top_k_blocks, nb), block_size=block_size,
                             causal=causal, q_positions=qpos)         # (B,Hkv,Sq,nb)
    tok_keep = np.repeat(keep, block_size, axis=-1)                  # (B,Hkv,Sq,Sk)
    kpos = np.arange(Sk)[None, None, None, :]
    tok_keep = tok_keep & (kpos < Sk_real)                          # drop pad keys
    if causal:
        tok_keep = tok_keep & (kpos <= qpos[None, None, :, None])
    out = np.zeros((B, Hq, Sq, Dv), dtype=np.float64)
    for h in range(Hq):
        kv_h = h // g
        scores = _bmm(Qa[:, h], np.swapaxes(Ka[:, kv_h], -1, -2), backend) * scale  # (B,Sq,Sk)
        scores = np.where(tok_keep[:, kv_h], scores, -np.inf)
        w = _softmax_last(scores)
        out[:, h] = _bmm(w, Va[:, kv_h], backend)
    return out


def dense_causal_attention(Q, K, V, *, scale=None):
    """Dense causal reference (the DSA select-all oracle).  GQA-aware."""
    Qa, Ka, Va = (_arr(Q).astype(np.float64), _arr(K).astype(np.float64),
                  _arr(V).astype(np.float64))
    B, Hq, Sq, D = Qa.shape
    Hkv, Sk = Ka.shape[1], Ka.shape[2]
    g = Hq // Hkv
    if scale is None:
        scale = 1.0 / np.sqrt(D)
    kpos = np.arange(Sk)[None, :]
    qpos = np.arange(Sq)[:, None]
    mask = kpos > qpos
    out = np.zeros((B, Hq, Sq, Va.shape[-1]), dtype=np.float64)
    for h in range(Hq):
        s = np.matmul(Qa[:, h], np.swapaxes(Ka[:, h // g], -1, -2)) * scale
        s = np.where(mask[None], -np.inf, s)
        out[:, h] = np.matmul(_softmax_last(s), Va[:, h // g])
    return out


__all__ = [
    # shared
    "apply_rope",
    # M3 MLA
    "MLAWeights",
    "compress_latent",
    "mla_attention",
    "mla_prefill",
    "mla_decode_step",
    # M4 DSA
    "dsa_block_index",
    "dsa_select_blocks",
    "dsa_block_sparse_attention",
    "dense_causal_attention",
]
