"""Item 4 — Higher-order autodiff helpers (grad, hvp, elementwise_grad)
+ retain_graph re-runnable backward.

Phase F4 / F5 already wire IR-level reverse-mode autodiff. This file
covers the JAX-style convenience surface on top of the tape engine:

  * ``tessera.autodiff.grad(fn, argnums)`` — callable returning gradients
  * ``tessera.autodiff.hvp(fn, primals, tangents)`` — finite-difference HVP
  * ``tessera.autodiff.elementwise_grad(fn)`` — per-element derivative
  * ``tape.backward(retain_graph=True)`` — repeated backward passes for
    jacrev (Item 5b) and similar.

True forward-over-reverse HVP via ``jvp(grad(fn), x, v)`` requires the
forward-mode tape from Item 5c.
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff import TesseraAutodiffError


# ─────────────────────────────────────────────────────────────────────────────
# grad(fn) — JAX-style gradient transform
# ─────────────────────────────────────────────────────────────────────────────


class TestGrad:
    def test_grad_of_squared_norm(self):
        f = lambda x: ts.ops.reduce(ts.ops.mul(x, x), op="sum")
        g = ts.autodiff.grad(f)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(g(x), 2 * x, rtol=1e-12)

    def test_grad_of_quartic(self):
        """∂(sum x^4)/∂x = 4 x^3."""
        f = lambda x: ts.ops.reduce(
            ts.ops.mul(ts.ops.mul(x, x), ts.ops.mul(x, x)), op="sum"
        )
        g = ts.autodiff.grad(f)
        x = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(g(x), 4 * x ** 3, rtol=1e-10)

    def test_grad_argnums_int_returns_array(self):
        """Default int argnums returns a single ndarray, not a tuple."""
        f = lambda a, b: ts.ops.reduce(ts.ops.mul(a, b), op="sum")
        g = ts.autodiff.grad(f, argnums=0)
        a = np.array([1.0, 2.0]); b = np.array([3.0, 4.0])
        out = g(a, b)
        assert isinstance(out, np.ndarray)
        np.testing.assert_allclose(out, b)

    def test_grad_argnums_tuple_returns_tuple(self):
        f = lambda a, b: ts.ops.reduce(ts.ops.mul(a, b), op="sum")
        g = ts.autodiff.grad(f, argnums=(0, 1))
        a = np.array([1.0, 2.0]); b = np.array([3.0, 4.0])
        out = g(a, b)
        assert isinstance(out, tuple) and len(out) == 2
        np.testing.assert_allclose(out[0], b)
        np.testing.assert_allclose(out[1], a)

    def test_grad_rejects_non_scalar_output(self):
        with pytest.raises(TesseraAutodiffError, match="scalar output"):
            ts.autodiff.grad(lambda x: ts.ops.matmul(x, np.eye(3, dtype=np.float64)))(
                np.eye(3, dtype=np.float64)
            )

    def test_grad_does_not_mutate_caller_state(self):
        """grad() must not leave residue in the user's parameters' .grad
        slots — it reads cotangents from the tape map directly."""
        p = ts.nn.Parameter(np.array([1.0, 2.0]))
        ts.autodiff.grad(
            lambda a: ts.ops.reduce(ts.ops.mul(a, a), op="sum")
        )(p)
        # The grad function received `p` → wrapped a fresh Parameter
        # internally, so `p.grad` should still be None on the caller's
        # side.
        assert p.grad is None


# ─────────────────────────────────────────────────────────────────────────────
# Hessian-vector product
# ─────────────────────────────────────────────────────────────────────────────


class TestHVP:
    def test_hvp_of_quadratic_form_matches_2Av(self):
        """For f(x) = x^T A x with symmetric A, H = 2A so H @ v = 2 A v."""
        A = np.array([[3.0, 1.0], [1.0, 2.0]])

        def quad(x):
            return ts.ops.reduce(
                ts.ops.matmul(ts.ops.transpose(x), ts.ops.matmul(A, x)),
                op="sum",
            )

        x = np.array([[1.0], [2.0]])
        v = np.array([[1.0], [0.0]])
        np.testing.assert_allclose(
            ts.autodiff.hvp(quad, x, v), 2 * (A @ v), atol=1e-3
        )

    def test_hvp_of_cubic_uses_x_dependent_curvature(self):
        """For f(x) = sum(x^3), H = 6 diag(x). HVP = 6 (x ⊙ v)."""
        f = lambda x: ts.ops.reduce(
            ts.ops.mul(ts.ops.mul(x, x), x), op="sum"
        )
        x = np.array([1.0, 2.0, 3.0])
        v = np.array([1.0, 1.0, 1.0])
        ref = 6 * x * v
        np.testing.assert_allclose(
            ts.autodiff.hvp(f, x, v), ref, atol=1e-3
        )

    def test_hvp_multi_arg(self):
        """f(a, b) = sum(a^2 + b^2 + a*b). H = [[2, 1], [1, 2]];
        for v=(1, 0) the HVP is (2, 1)."""
        # Avoid `mul(x, scalar_literal)` because non-array positional
        # args don't survive the tape's _make_wrapper recording — that's
        # a pre-existing tape edge case orthogonal to higher-order
        # autodiff (and pinned by Item 4's existing v1 contract).
        f = lambda a, b: ts.ops.reduce(
            ts.ops.add(
                ts.ops.add(ts.ops.mul(a, a), ts.ops.mul(b, b)),
                ts.ops.mul(a, b),
            ),
            op="sum",
        )
        a = np.array([1.0])
        b = np.array([1.0])
        v_a = np.array([1.0])
        v_b = np.array([0.0])
        hvp_a, hvp_b = ts.autodiff.hvp(f, (a, b), (v_a, v_b))
        np.testing.assert_allclose(hvp_a, [2.0], atol=1e-3)
        np.testing.assert_allclose(hvp_b, [1.0], atol=1e-3)


# ─────────────────────────────────────────────────────────────────────────────
# elementwise_grad
# ─────────────────────────────────────────────────────────────────────────────


class TestElementwiseGrad:
    def test_silu_derivative(self):
        d = ts.autodiff.elementwise_grad(ts.ops.silu)
        x = np.array([-1.0, 0.0, 1.0, 2.0])
        s = 1.0 / (1.0 + np.exp(-x))
        ref = s + x * s * (1 - s)
        np.testing.assert_allclose(d(x), ref, rtol=1e-5)

    def test_relu_derivative(self):
        d = ts.autodiff.elementwise_grad(ts.ops.relu)
        x = np.array([-1.0, 0.5, 2.0])
        np.testing.assert_array_equal(d(x), [0.0, 1.0, 1.0])

    def test_tanh_derivative_at_zero_is_one(self):
        d = ts.autodiff.elementwise_grad(ts.ops.tanh)
        np.testing.assert_allclose(d(np.array([0.0])), [1.0], rtol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# retain_graph — repeated backward
# ─────────────────────────────────────────────────────────────────────────────


class TestRetainGraph:
    def test_default_backward_consumed_after_first_call(self):
        x = ts.nn.Parameter(np.array([1.0, 2.0]))
        with ts.autodiff.tape() as t:
            y = ts.ops.reduce(ts.ops.mul(x, x), op="sum")
            t.backward(y)
            with pytest.raises(TesseraAutodiffError, match="twice"):
                t.backward(y)

    def test_retain_graph_allows_multiple_backward(self):
        """Run backward with two different cotangent seeds — used by
        jacrev (Item 5b) which seeds one basis vector per output dim."""
        np.random.seed(0)
        a = np.eye(3, dtype=np.float64)
        b = np.array([[1.0], [2.0], [3.0]])
        with ts.autodiff.tape() as t:
            y = ts.ops.matmul(a, b)  # 3x1 vector
            # Seed e_0
            t.backward(
                y, cotangent=np.array([[1.0], [0.0], [0.0]]),
                retain_graph=True,
            )
            cotan_e0 = dict(t.cotangent)
            # Re-seed e_1
            t.backward(
                y, cotangent=np.array([[0.0], [1.0], [0.0]]),
                retain_graph=True,
            )
            cotan_e1 = dict(t.cotangent)
        # Both backwards completed without raising.
        assert id(b) in cotan_e0 or any(
            isinstance(k, int) for k in cotan_e0.keys()
        )

    def test_accumulate_param_grad_false_skips_param_writes(self):
        """grad() uses this path so it doesn't leak into user params."""
        p = ts.nn.Parameter(np.array([1.0, 2.0]))
        with ts.autodiff.tape() as t:
            y = ts.ops.reduce(ts.ops.mul(p, p), op="sum")
            t.backward(y, accumulate_param_grad=False)
        assert p.grad is None
