"""LSA conformance slice (S8-style) — tiny deterministic end-to-end check.

A frozen, hand-checkable instance: fixed compressed indexer keys, thresholded
sigmoid selection, a causal local window, and selected historical blocks. The
expected output is computed by an independent full-attention-with-key-mask
reference (a different code path than ``lsa.lookahead_sparse_attention``), so
this is a genuine cross-check rather than a tautology.

Per ``docs/audit/domain/archive/lsa_scope.md`` (D5): the constants below
(``threshold=0.5``, ``tau=64``, ``window_size``, ``block_size``) are chosen test
fixtures, not a reproduced external result.
"""

from __future__ import annotations

import numpy as np

import tessera as ts
from tessera import lsa


def _masked_full_attention(Q, K, V, keep, scale):
    """Independent oracle: per-query softmax over the boolean key-set `keep`."""
    B, H, S, _ = Q.shape
    out = np.zeros_like(V, dtype=np.float64)
    for b in range(B):
        for h in range(H):
            for sq in range(S):
                idx = np.flatnonzero(keep[b, h, sq])
                s = (Q[b, h, sq] @ K[b, h, idx].T) * scale
                s -= s.max()
                w = np.exp(s)
                w /= w.sum()
                out[b, h, sq] = w @ V[b, h, idx]
    return out


def test_tiny_lsa_slice_matches_independent_reference():
    # Deterministic small instance: B=1, H=1, S=8, D=4, block_size=2 → 4 blocks.
    rng = np.random.default_rng(2024)
    B, H, S, D, block_size, window_size = 1, 1, 8, 4, 2, 2
    Q = rng.standard_normal((B, H, S, D))
    K = rng.standard_normal((B, H, S, D))
    V = rng.standard_normal((B, H, S, D))
    scale = 1.0 / np.sqrt(D)

    # Frozen compressed indexer keys (mean-pool) + the selector's own mask.
    keys = lsa.compress_block_keys(K, block_size=block_size)
    mask = lsa.memory_index_select(
        keys, Q, block_size=block_size, threshold=0.5, causal=True, fallback_local=True).mask

    # Build the boolean key-set independently from the mask + local window.
    keep = np.zeros((B, H, S, S), dtype=bool)
    for sq in range(S):
        keep[0, 0, sq, max(0, sq - window_size + 1) : sq + 1] = True
        for blk in np.flatnonzero(mask[0, 0, sq]):
            keep[0, 0, sq, blk * block_size : blk * block_size + block_size] = True
        keep[0, 0, sq, sq + 1 :] = False  # causal

    expected = _masked_full_attention(Q, K, V, keep, scale)
    out = ts.ops.lookahead_sparse_attention(
        Q, K, V, window_size=window_size, block_size=block_size,
        tau=64, threshold=0.5, causal=True)
    np.testing.assert_allclose(out, expected, atol=1e-12)


def test_selection_mask_is_hand_checkable():
    # Craft a case where block 0 is strongly aligned with every query and the
    # others are orthogonal, so only block 0 (plus the causal own-block) selects.
    B, H, S, D, block_size = 1, 1, 4, 2, 1  # 4 blocks of 1 token
    K = np.zeros((B, H, S, D))
    K[0, 0, 0] = [10.0, 0.0]   # block 0: large positive alignment
    K[0, 0, 1] = [0.0, 0.0]
    K[0, 0, 2] = [0.0, 0.0]
    K[0, 0, 3] = [0.0, 0.0]
    Q = np.tile(np.array([1.0, 0.0]), (B, H, S, 1))  # aligned with block 0
    keys = lsa.compress_block_keys(K, block_size=block_size)
    mask = lsa.memory_index_select(
        keys, Q, block_size=block_size, threshold=0.5, causal=True, fallback_local=True).mask
    # Query 0: only block 0 (its own + only positive-score block).
    assert mask[0, 0, 0].tolist() == [True, False, False, False]
    # Later queries: block 0 (sigmoid(10·…)>0.5) selected; zero-score blocks are
    # at exactly 0.5 (sigmoid(0)) and so are also retained under `>=`. Future
    # blocks must stay unselected.
    qb = np.arange(S) // block_size
    nb = keys.shape[2]
    future = np.arange(nb)[None, :] > qb[:, None]
    assert int((mask[0, 0] & future).sum()) == 0
