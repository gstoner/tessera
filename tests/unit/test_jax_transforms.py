"""Item 5 — JAX-style transforms (vmap, jacrev, jacfwd) + the JVP
engine that backs jacfwd.

vmap is a pure program transformation (scan-then-stack). jacrev runs
N reverse-mode backward passes, one per output dim. jacfwd runs M
forward-mode JVPs, one per input dim. Both Jacobian transforms agree
element-wise for the same function (validated below).
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# vmap
# ─────────────────────────────────────────────────────────────────────────────


class TestVmap:
    def test_vmap_int_in_axes_default(self):
        """vmap with default in_axes=0 stacks per-row outputs along
        out_axes=0."""
        f = lambda x: ts.ops.reduce(ts.ops.mul(x, x), op="sum")
        batch = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = ts.autodiff.vmap(f)(batch)
        np.testing.assert_array_equal(out, [5.0, 25.0, 61.0])

    def test_vmap_per_arg_in_axes(self):
        """Per-arg in_axes — first arg batched on axis 0, second arg
        not batched (passed through to every call)."""
        f = lambda x, w: ts.ops.gemm(x, w)
        batch_x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # (3, 2)
        weight = np.array([[1.0], [2.0]])  # (2, 1)
        out = ts.autodiff.vmap(f, in_axes=(0, None))(batch_x, weight)
        # Each row x_i @ weight = scalar; stacked = (3, 1).
        np.testing.assert_array_equal(out, [[1.0], [2.0], [3.0]])

    def test_vmap_no_batched_args_short_circuits(self):
        """When all in_axes are None, vmap reduces to a single fn call."""
        f = lambda x: ts.ops.reduce(ts.ops.mul(x, x), op="sum")
        x = np.array([1.0, 2.0])
        out = ts.autodiff.vmap(f, in_axes=None)(x)
        # Same as raw fn — scalar result.
        np.testing.assert_allclose(out, 5.0)

    def test_vmap_inconsistent_batch_size_raises(self):
        f = lambda a, b: ts.ops.add(a, b)
        a = np.zeros((3, 2))
        b = np.zeros((4, 2))  # mismatched batch dim
        with pytest.raises(ValueError, match="inconsistent batch sizes"):
            ts.autodiff.vmap(f, in_axes=0)(a, b)

    def test_vmap_in_axes_length_must_match_args(self):
        f = lambda a, b: a
        with pytest.raises(ValueError, match="length"):
            ts.autodiff.vmap(f, in_axes=(0,))(np.zeros(3), np.zeros(3))

    def test_vmap_out_axes_none_returns_list(self):
        """out_axes=None lets the caller see the per-element outputs as
        a list (no stacking)."""
        f = lambda x: ts.ops.reduce(x, op="sum")
        batch = np.arange(6, dtype=np.float64).reshape(3, 2)
        out = ts.autodiff.vmap(f, out_axes=None)(batch)
        assert isinstance(out, list)
        assert len(out) == 3
        np.testing.assert_array_equal(
            [float(o) for o in out], [1.0, 5.0, 9.0]
        )


# ─────────────────────────────────────────────────────────────────────────────
# jacrev
# ─────────────────────────────────────────────────────────────────────────────


class TestJacrev:
    def test_diag_for_pointwise_square(self):
        g = lambda x: ts.ops.mul(x, x)
        J = ts.autodiff.jacrev(g)(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(J, np.diag([2.0, 4.0, 6.0]), rtol=1e-9)

    def test_jacobian_of_linear_map_is_the_matrix(self):
        """For f(x) = A @ x, the Jacobian is A."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f = lambda x: ts.ops.gemm(A, x)
        x = np.array([[0.0], [0.0], [0.0]])  # (3, 1)
        J = ts.autodiff.jacrev(f)(x)
        # J shape: (2, 1, 3, 1); flatten the in-/out- singleton dims to
        # match A.
        np.testing.assert_allclose(J.reshape(2, 3), A, rtol=1e-9)

    def test_jacrev_of_silu_is_diag_of_silu_derivative(self):
        x = np.array([-1.0, 0.5, 2.0])
        J = ts.autodiff.jacrev(ts.ops.silu)(x)
        # Reference: per-element derivative on the diagonal.
        s = 1.0 / (1.0 + np.exp(-x))
        expected_diag = s + x * s * (1.0 - s)
        np.testing.assert_allclose(J, np.diag(expected_diag), rtol=1e-5)


# ─────────────────────────────────────────────────────────────────────────────
# jacfwd
# ─────────────────────────────────────────────────────────────────────────────


class TestJacfwd:
    def test_diag_for_pointwise_square(self):
        g = lambda x: ts.ops.mul(x, x)
        J = ts.autodiff.jacfwd(g)(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(J, np.diag([2.0, 4.0, 6.0]), atol=1e-5)

    def test_jacfwd_matches_jacrev_on_linear_map(self):
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        f = lambda x: ts.ops.gemm(A, x)
        x = np.array([[1.0], [1.0], [1.0]])
        J_rev = ts.autodiff.jacrev(f)(x)
        J_fwd = ts.autodiff.jacfwd(f)(x)
        np.testing.assert_allclose(J_fwd, J_rev, atol=1e-5)

    def test_jacfwd_matches_jacrev_on_nonlinear(self):
        """Same Jacobian whether computed by reverse or forward mode."""
        f = lambda x: ts.ops.silu(x)
        np.random.seed(0)
        x = np.random.randn(5).astype(np.float64) * 2
        J_rev = ts.autodiff.jacrev(f)(x)
        J_fwd = ts.autodiff.jacfwd(f)(x)
        np.testing.assert_allclose(J_fwd, J_rev, atol=1e-4)


# ─────────────────────────────────────────────────────────────────────────────
# JVP engine
# ─────────────────────────────────────────────────────────────────────────────


class TestJVP:
    def test_jvp_engine_for_silu(self):
        """The JVP API: returns (primal_out, tangent_out) where
        tangent_out = f'(x) * v."""
        f = lambda x: ts.ops.silu(x)
        x = np.array([0.5, 1.5])
        v = np.array([1.0, 0.0])
        primal, tangent = ts.autodiff.jvp(f, x, v)
        np.testing.assert_allclose(primal, x / (1 + np.exp(-x)), rtol=1e-5)
        # Tangent picks up the derivative at x[0] only since v[1] = 0.
        s0 = 1.0 / (1.0 + np.exp(-x[0]))
        d0 = s0 + x[0] * s0 * (1 - s0)
        np.testing.assert_allclose(tangent, [d0, 0.0], atol=1e-4)

    def test_register_jvp_returns_named_callable(self):
        from tessera.autodiff.jvp import register_jvp, get_jvp

        def my_jvp(primals, tangents, **_):
            return primals[0], tangents[0]

        register_jvp("__test_my_op__", my_jvp)
        assert get_jvp("__test_my_op__") is my_jvp


# ─────────────────────────────────────────────────────────────────────────────
# Composability — vmap-of-grad pattern
# ─────────────────────────────────────────────────────────────────────────────


class TestCompositions:
    def test_vmap_of_grad_per_sample_gradient(self):
        """``vmap(grad(f))`` is the canonical way to get per-sample
        gradients in JAX-style code. Verified against the analytical
        per-sample gradient of (sum_i x_i^2)."""
        f = lambda x: ts.ops.reduce(ts.ops.mul(x, x), op="sum")
        per_sample_grad = ts.autodiff.vmap(ts.autodiff.grad(f))
        batch = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        out = per_sample_grad(batch)
        np.testing.assert_array_equal(out, 2 * batch)
