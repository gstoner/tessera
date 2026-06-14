"""Smoke + mechanism tests for the MoBA model (added via the add-moe-model skill)."""

from __future__ import annotations

import numpy as np

from tessera.train.models.moba import MoBAConfig, MoBAModel, moba_attention


def test_moba_forward_shapes_and_finite_aux():
    cfg = MoBAConfig()
    model = MoBAModel(cfg)
    ids = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 8))
    logits, aux = model.forward(ids)

    assert np.asarray(logits).shape == (2, 8, cfg.vocab_size)
    assert np.all(np.isfinite(np.asarray(logits)))
    assert all(np.isfinite(float(v)) for v in aux.values())


def test_moba_attention_is_block_sparse():
    """Restricting top_k_blocks must change the output vs. full causal attention."""
    rng = np.random.default_rng(0)
    S, hd = 12, 8
    Q = rng.standard_normal((S, hd))
    K = rng.standard_normal((S, hd))
    V = rng.standard_normal((S, hd))
    sparse = moba_attention(Q, K, V, block_size=4, top_k_blocks=1)
    full = moba_attention(Q, K, V, block_size=4, top_k_blocks=99)  # all past blocks
    assert sparse.shape == (S, hd)
    assert np.linalg.norm(sparse - full) > 1e-6


def test_moba_attention_is_causal():
    """A query never attends to future positions (output independent of them)."""
    rng = np.random.default_rng(1)
    S, hd = 8, 4
    Q = rng.standard_normal((S, hd))
    K = rng.standard_normal((S, hd))
    V = rng.standard_normal((S, hd))
    out1 = moba_attention(Q, K, V, block_size=2, top_k_blocks=4)
    # Perturb the LAST position's K/V; positions before it must be unchanged.
    K2, V2 = K.copy(), V.copy()
    K2[-1] += 5.0
    V2[-1] += 5.0
    out2 = moba_attention(Q, K2, V2, block_size=2, top_k_blocks=4)
    np.testing.assert_allclose(out1[:-1], out2[:-1], rtol=1e-9, atol=1e-9)
