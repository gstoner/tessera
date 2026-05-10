"""Phase F-MoR — Mixture of Recursions primitives + nn.MixtureOfRecursions.

Forward correctness of the three MoR ops + the Module wrapper. The
router is non-differentiable (argmax) so VJP coverage is limited to
shape contracts; real router-training relies on auxiliary losses the
user adds explicitly.

Bae et al. 2025 "Mixture-of-Recursions": adaptive computation by
routing tokens through different numbers of recursive layer
applications.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# mor_router
# ─────────────────────────────────────────────────────────────────────────────


class TestMorRouter:
    def test_output_shape_and_dtype(self):
        np.random.seed(0)
        B, S, D = 2, 4, 8
        max_depth = 3
        x = np.random.randn(B, S, D).astype(np.float32)
        w = np.random.randn(D, max_depth).astype(np.float32) * 0.1
        depth = ts.ops.mor_router(x, w, max_depth=max_depth)
        assert depth.shape == (B, S)
        assert depth.dtype == np.int64

    def test_depth_in_valid_range(self):
        np.random.seed(0)
        max_depth = 5
        x = np.random.randn(1, 32, 16).astype(np.float32)
        w = np.random.randn(16, max_depth).astype(np.float32)
        depth = ts.ops.mor_router(x, w, max_depth=max_depth)
        assert depth.min() >= 1
        assert depth.max() <= max_depth

    def test_argmax_picks_highest_logit(self):
        """Construct w so that the argmax is deterministic per token —
        depth column 1 always wins. depth + 1 = 2."""
        x = np.ones((1, 4, 4), dtype=np.float32)
        # w_router shape (4, 3); column 1 has all-3 weights, others zero.
        w = np.zeros((4, 3), dtype=np.float32)
        w[:, 1] = 3.0
        depth = ts.ops.mor_router(x, w, max_depth=3)
        # All tokens map to argmax index 1 → depth = 1 + 1 = 2.
        np.testing.assert_array_equal(depth, np.full((1, 4), 2, dtype=np.int64))

    def test_invalid_w_shape_rejected(self):
        x = np.zeros((1, 4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="w_router shape"):
            ts.ops.mor_router(x, np.zeros((9, 3)), max_depth=3)

    def test_max_depth_must_be_positive(self):
        x = np.zeros((1, 4, 8), dtype=np.float32)
        with pytest.raises(ValueError, match="max_depth"):
            ts.ops.mor_router(x, np.zeros((8, 0)), max_depth=0)


# ─────────────────────────────────────────────────────────────────────────────
# mor_partition
# ─────────────────────────────────────────────────────────────────────────────


class TestMorPartition:
    def test_mask_is_depth_geq_step(self):
        """The bool mask is True iff the token's target depth >= step."""
        x = np.zeros((1, 5, 4), dtype=np.float32)
        depth = np.array([[1, 2, 3, 2, 1]], dtype=np.int64)
        for step, expected in [
            (1, [True, True, True, True, True]),
            (2, [False, True, True, True, False]),
            (3, [False, False, True, False, False]),
        ]:
            mask = ts.ops.mor_partition(x, depth, step=step)
            np.testing.assert_array_equal(mask, [expected])

    def test_step_must_be_positive(self):
        x = np.zeros((1, 4, 4), dtype=np.float32)
        depth = np.zeros((1, 4), dtype=np.int64)
        with pytest.raises(ValueError, match="step"):
            ts.ops.mor_partition(x, depth, step=0)

    def test_depth_shape_validated(self):
        x = np.zeros((1, 4, 4), dtype=np.float32)
        depth = np.zeros((2, 4), dtype=np.int64)
        with pytest.raises(ValueError, match="depth shape"):
            ts.ops.mor_partition(x, depth, step=1)


# ─────────────────────────────────────────────────────────────────────────────
# mor_scatter
# ─────────────────────────────────────────────────────────────────────────────


class TestMorScatter:
    def test_scatter_writes_only_at_mask_true(self):
        full = np.zeros((1, 4, 2), dtype=np.float32)
        updated = np.ones((1, 4, 2), dtype=np.float32)
        mask = np.array([[True, False, True, False]])
        out = ts.ops.mor_scatter(full, updated, mask)
        np.testing.assert_array_equal(out[0, 0], [1, 1])
        np.testing.assert_array_equal(out[0, 1], [0, 0])
        np.testing.assert_array_equal(out[0, 2], [1, 1])
        np.testing.assert_array_equal(out[0, 3], [0, 0])

    def test_shape_validation(self):
        full = np.zeros((1, 4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="updated shape"):
            ts.ops.mor_scatter(full, np.zeros((1, 4, 3)), np.zeros((1, 4)))

    def test_mask_shape_validation(self):
        full = np.zeros((1, 4, 2), dtype=np.float32)
        with pytest.raises(ValueError, match="mask shape"):
            ts.ops.mor_scatter(full, np.zeros((1, 4, 2)), np.zeros((1, 5)))


# ─────────────────────────────────────────────────────────────────────────────
# nn.MixtureOfRecursions Module
# ─────────────────────────────────────────────────────────────────────────────


class _IdentityPlusOne(ts.nn.Module):
    """Trivial layer: returns input + 1. Used in tests to verify the
    recursion loop applies the layer the right number of times per
    token."""

    def forward(self, x):
        if hasattr(x, "_data"):
            x = x._data
        return np.asarray(x) + 1.0


class TestMixtureOfRecursionsModule:
    def test_module_forward_runs(self):
        np.random.seed(0)
        B, S, D = 1, 8, 4
        max_depth = 3
        layer = _IdentityPlusOne()
        m = ts.nn.MixtureOfRecursions(
            layer, embed_dim=D, max_depth=max_depth,
        )
        # Initialize the router with mostly-uniform logits so different
        # tokens go to different depths (exercising all branches).
        m.W_router._data._data[:] = np.random.randn(D, max_depth) * 0.5
        x = np.random.randn(B, S, D).astype(np.float32) * 0.1
        out = m(x)
        assert out.shape == x.shape

    def test_per_token_recursion_depth_applied_correctly(self):
        """With layer = identity + 1 and depth d for token i, output
        for that token should be x[i] + d."""
        D = 4
        max_depth = 3
        layer = _IdentityPlusOne()
        m = ts.nn.MixtureOfRecursions(
            layer, embed_dim=D, max_depth=max_depth,
        )
        # Force depths via hand-tuned router weights: column-aligned
        # logits picking depth 1 / 2 / 3 for the three test tokens.
        x = np.array([
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0]]
        ], dtype=np.float32)  # (1, 3, 4)
        # W_router = identity-like over the first 3 dims so token i's
        # argmax falls on column i (depth = i + 1).
        W = np.zeros((D, max_depth), dtype=np.float32)
        W[0, 0] = 10.0  # token 0: argmax on col 0 → depth 1
        W[1, 1] = 10.0  # token 1: argmax on col 1 → depth 2
        W[2, 2] = 10.0  # token 2: argmax on col 2 → depth 3
        m.W_router._data._data[:] = W
        out = m(x)
        # Expected: token i hidden state increased by depth_i (= i+1).
        np.testing.assert_allclose(out[0, 0], x[0, 0] + 1.0)
        np.testing.assert_allclose(out[0, 1], x[0, 1] + 2.0)
        np.testing.assert_allclose(out[0, 2], x[0, 2] + 3.0)

    def test_invalid_max_depth_rejected(self):
        layer = _IdentityPlusOne()
        with pytest.raises(ValueError, match="max_depth"):
            ts.nn.MixtureOfRecursions(layer, embed_dim=4, max_depth=0)

    def test_module_rejects_rank_2_input(self):
        layer = _IdentityPlusOne()
        m = ts.nn.MixtureOfRecursions(layer, embed_dim=4, max_depth=2)
        m.W_router._data._data[:] = np.random.randn(4, 2)
        x = np.zeros((4, 4), dtype=np.float32)  # rank-2
        with pytest.raises(ValueError, match="rank-3"):
            m(x)


# ─────────────────────────────────────────────────────────────────────────────
# VJP shape contracts (router argmax is non-differentiable; ensure the
# VJPs return zero gradients without crashing)
# ─────────────────────────────────────────────────────────────────────────────


class TestMorVJPs:
    def test_router_vjp_returns_zero_grads(self):
        from tessera.autodiff.vjp import get_vjp
        vjp = get_vjp("mor_router")
        x = np.ones((1, 4, 8), dtype=np.float32)
        w = np.ones((8, 3), dtype=np.float32)
        dout = np.ones((1, 4), dtype=np.int64)
        dx, dw = vjp(dout, x, w, max_depth=3)
        np.testing.assert_array_equal(dx, np.zeros_like(x))
        np.testing.assert_array_equal(dw, np.zeros_like(w))

    def test_scatter_vjp_routes_grad_correctly(self):
        from tessera.autodiff.vjp import get_vjp
        vjp = get_vjp("mor_scatter")
        full = np.zeros((1, 3, 2), dtype=np.float32)
        updated = np.zeros((1, 3, 2), dtype=np.float32)
        mask = np.array([[True, False, True]])
        dout = np.array([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]], dtype=np.float32)
        d_full, d_updated, d_mask = vjp(dout, full, updated, mask)
        # full receives gradient on False positions; updated on True.
        np.testing.assert_array_equal(
            d_full, [[[0, 0], [2, 2], [0, 0]]]
        )
        np.testing.assert_array_equal(
            d_updated, [[[1, 1], [0, 0], [3, 3]]]
        )
        assert d_mask is None
