"""Gap 4 — 2D local-window attention.

Forward correctness, validation, registry hygiene, and VJP/JVP round-trips
for ``tessera.ops.attn_local_window_2d`` — a 2D generalization of
``attn_sliding_window`` for spatial grids (weather/climate, ViT-style
local bias, neural cellular automata).

The v1 reference is a numpy nested loop; the fused kernel is reserved as
a planned backend-manifest slot on apple_gpu / nvidia_sm90 / rocm.
"""

from __future__ import annotations

import numpy as np
import pytest

import importlib

import tessera as ts
from tessera.autodiff import vjp as _vjp_mod
from tessera.compiler import op_catalog
from tessera.compiler import primitive_coverage as _pc

# ``tessera.autodiff.jvp`` the *attribute* is the user-facing function; the
# module that holds the ``_JVPS`` registry has to be imported by full path.
_jvp_mod = importlib.import_module("tessera.autodiff.jvp")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_qkv_2d(B=2, H=2, Hq=4, Wq=4, D=8, seed=0):
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal((B, H, Hq, Wq, D)).astype(np.float32) * 0.5
    K = rng.standard_normal((B, H, Hq, Wq, D)).astype(np.float32) * 0.5
    V = rng.standard_normal((B, H, Hq, Wq, D)).astype(np.float32) * 0.5
    return Q, K, V


def _numpy_oracle(Q, K, V, window):
    """Reference implementation — kept independent of the library so the
    library can be re-implemented without the test silently following."""
    rh, rw = window
    B, H, Hq, Wq, D = Q.shape
    scale = 1.0 / np.sqrt(D)
    out = np.zeros_like(Q)
    for b in range(B):
        for h in range(H):
            for i in range(Hq):
                for j in range(Wq):
                    i0, i1 = max(0, i - rh), min(Hq, i + rh + 1)
                    j0, j1 = max(0, j - rw), min(Wq, j + rw + 1)
                    k = K[b, h, i0:i1, j0:j1, :].reshape(-1, D)
                    v = V[b, h, i0:i1, j0:j1, :].reshape(-1, D)
                    q = Q[b, h, i, j, :]
                    s = (k @ q) * scale
                    s -= s.max()
                    e = np.exp(s)
                    w = e / e.sum()
                    out[b, h, i, j, :] = w @ v
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Forward correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestForward:
    def test_output_shape_matches_q(self):
        Q, K, V = _make_qkv_2d()
        out = ts.ops.attn_local_window_2d(Q, K, V, window=(1, 1))
        assert out.shape == Q.shape

    def test_matches_numpy_oracle_3x3(self):
        Q, K, V = _make_qkv_2d(B=2, H=2, Hq=4, Wq=4, D=8, seed=42)
        got = ts.ops.attn_local_window_2d(Q, K, V, window=(1, 1))
        expected = _numpy_oracle(Q, K, V, (1, 1))
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_matches_numpy_oracle_5x5(self):
        Q, K, V = _make_qkv_2d(B=1, H=1, Hq=6, Wq=6, D=4, seed=7)
        got = ts.ops.attn_local_window_2d(Q, K, V, window=(2, 2))
        expected = _numpy_oracle(Q, K, V, (2, 2))
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-6)

    def test_window_zero_is_identity_in_value(self):
        """window=(0, 0) ⇒ each query sees only its own key, so the output
        at every (h, w) equals V[h, w] regardless of Q and K."""
        Q, K, V = _make_qkv_2d(B=1, H=1, Hq=3, Wq=3, D=4, seed=1)
        out = ts.ops.attn_local_window_2d(Q, K, V, window=(0, 0))
        np.testing.assert_allclose(out, V, rtol=1e-5)

    def test_asymmetric_window(self):
        """(rh, rw) = (1, 0) → vertical strip; (0, 1) → horizontal strip."""
        Q, K, V = _make_qkv_2d(B=1, H=1, Hq=4, Wq=4, D=4, seed=2)
        got_v = ts.ops.attn_local_window_2d(Q, K, V, window=(1, 0))
        got_h = ts.ops.attn_local_window_2d(Q, K, V, window=(0, 1))
        exp_v = _numpy_oracle(Q, K, V, (1, 0))
        exp_h = _numpy_oracle(Q, K, V, (0, 1))
        np.testing.assert_allclose(got_v, exp_v, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(got_h, exp_h, rtol=1e-5, atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────


class TestValidation:
    def test_rejects_rank4_input(self):
        with pytest.raises(ValueError, match="rank-5"):
            ts.ops.attn_local_window_2d(
                np.zeros((1, 2, 4, 4)),
                np.zeros((1, 2, 4, 4)),
                np.zeros((1, 2, 4, 4)),
                window=(1, 1),
            )

    def test_rejects_mismatched_kv_spatial(self):
        Q = np.zeros((1, 1, 4, 4, 4))
        K = np.zeros((1, 1, 4, 4, 4))
        V = np.zeros((1, 1, 4, 3, 4))  # Wk mismatch
        with pytest.raises(ValueError, match="K and V must agree"):
            ts.ops.attn_local_window_2d(Q, K, V, window=(1, 1))

    def test_rejects_mismatched_q_kv_spatial(self):
        Q = np.zeros((1, 1, 4, 4, 4))
        K = np.zeros((1, 1, 3, 3, 4))
        V = np.zeros((1, 1, 3, 3, 4))
        with pytest.raises(ValueError, match="share spatial"):
            ts.ops.attn_local_window_2d(Q, K, V, window=(1, 1))

    def test_rejects_negative_window(self):
        Q, K, V = _make_qkv_2d(B=1, H=1, Hq=2, Wq=2, D=2)
        with pytest.raises(ValueError, match="non-negative"):
            ts.ops.attn_local_window_2d(Q, K, V, window=(-1, 1))


# ─────────────────────────────────────────────────────────────────────────────
# Registry hygiene
# ─────────────────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_op_catalog_entry(self):
        spec = op_catalog.get_op_spec("attn_local_window_2d")
        assert spec is not None
        # OpSpec carries the canonical MLIR symbol and lowering family.
        assert spec.graph_name == "tessera.attn_local_window_2d"
        assert spec.public_name == "attn_local_window_2d"
        assert spec.lowering == "attention"

    def test_primitive_coverage_entry(self):
        # ``primitive_coverage.OP_SPECS`` is a dict keyed by primitive name.
        assert "attn_local_window_2d" in _pc.OP_SPECS

    def test_vjp_registered(self):
        assert "attn_local_window_2d" in _vjp_mod._VJPS

    def test_jvp_registered(self):
        assert "attn_local_window_2d" in _jvp_mod._JVPS

    def test_namespace_export(self):
        assert hasattr(ts.ops, "attn_local_window_2d")
        assert callable(ts.ops.attn_local_window_2d)


# ─────────────────────────────────────────────────────────────────────────────
# Autodiff round-trips (numeric fallback by design — locks correctness, not perf)
# ─────────────────────────────────────────────────────────────────────────────


class TestAutodiff:
    def test_vjp_dQ_matches_finite_diff(self):
        Q, K, V = _make_qkv_2d(B=1, H=1, Hq=3, Wq=3, D=4, seed=11)
        dout = np.ones_like(Q) * 0.1
        vjp_fn = _vjp_mod._VJPS["attn_local_window_2d"]
        dQ, dK, dV = vjp_fn(dout, Q, K, V, window=(1, 1))
        assert dQ.shape == Q.shape
        assert dK.shape == K.shape
        assert dV.shape == V.shape
        # Spot-check one Q entry against a forward finite difference.
        eps = 1e-3
        base = ts.ops.attn_local_window_2d(Q, K, V, window=(1, 1))
        Q2 = Q.copy()
        Q2[0, 0, 0, 0, 0] += eps
        pert = ts.ops.attn_local_window_2d(Q2, K, V, window=(1, 1))
        grad_est = ((pert - base) * dout).sum() / eps
        np.testing.assert_allclose(dQ[0, 0, 0, 0, 0], grad_est, rtol=5e-2, atol=5e-3)

    def test_jvp_tangent_shape(self):
        Q, K, V = _make_qkv_2d(B=1, H=1, Hq=3, Wq=3, D=4, seed=12)
        dQ = np.ones_like(Q) * 0.01
        dK = np.zeros_like(K)
        dV = np.zeros_like(V)
        jvp_fn = _jvp_mod._JVPS["attn_local_window_2d"]
        primal, tangent = jvp_fn((Q, K, V), (dQ, dK, dV), window=(1, 1))
        assert primal.shape == Q.shape
        assert tangent.shape == Q.shape
