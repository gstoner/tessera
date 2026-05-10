"""Tests for S-series sprint S2 — reductions, stability primitives,
numeric helpers, comparisons.

For each primitive we check:
  - the numpy-reference output matches the canonical numpy call
  - VJPs are registered (where applicable)
  - VJPs match a numerical gradient on a small input
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff.vjp import _VJPS, get_vjp


# Helper: numerical gradient for VJP correctness checks.
def _numeric_grad(fn, x, eps=1e-4):
    g = np.zeros_like(x, dtype=np.float64)
    x = x.astype(np.float64).copy()
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + eps
        f_plus = fn(x)
        x[idx] = orig - eps
        f_minus = fn(x)
        x[idx] = orig
        diff = np.asarray(f_plus) - np.asarray(f_minus)
        g[idx] = diff.sum() / (2 * eps)
        it.iternext()
    return g


# ── reductions ─────────────────────────────────────────────────────────────


def test_mean_matches_numpy():
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(ts.ops.mean(x), np.mean(x))
    np.testing.assert_array_equal(
        ts.ops.mean(x, axis=0, keepdims=True), np.mean(x, axis=0, keepdims=True)
    )


def test_var_std_match_numpy_with_ddof():
    x = np.arange(20.0).reshape(4, 5)
    np.testing.assert_allclose(ts.ops.var(x, ddof=0), np.var(x, ddof=0))
    np.testing.assert_allclose(ts.ops.std(x, axis=1, ddof=1),
                               np.std(x, axis=1, ddof=1))


def test_argmax_and_argmin_match_numpy():
    x = np.array([[5, 1, 3], [0, 7, 2]])
    np.testing.assert_array_equal(ts.ops.argmax(x, axis=-1),
                                   np.argmax(x, axis=-1))
    np.testing.assert_array_equal(ts.ops.argmin(x, axis=-1),
                                   np.argmin(x, axis=-1))


def test_cumsum_and_cumprod_match_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_array_equal(ts.ops.cumsum(x), np.cumsum(x))
    np.testing.assert_array_equal(ts.ops.cumprod(x), np.cumprod(x))


def test_amax_amin_match_numpy():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(ts.ops.amax(x, axis=1), np.max(x, axis=1))
    np.testing.assert_array_equal(ts.ops.amin(x, axis=0), np.min(x, axis=0))


def test_prod_matches_numpy():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    np.testing.assert_allclose(ts.ops.prod(x), 24.0)


# ── stability primitives ───────────────────────────────────────────────────


def test_logsumexp_is_stable_for_large_inputs():
    x = np.array([1000.0, 1001.0, 1002.0])
    result = ts.ops.logsumexp(x)
    # Without stability shift this would overflow; reference value is
    # 1002 + log(1 + e^-1 + e^-2) ≈ 1002.4076.
    np.testing.assert_allclose(result, 1002.4076, atol=1e-3)


def test_log_softmax_sums_to_zero_in_log_space_after_exp():
    x = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    out = ts.ops.log_softmax(x, axis=-1)
    # exp(log_softmax) should sum to 1.
    np.testing.assert_allclose(np.exp(out).sum(axis=-1), [1.0, 1.0], atol=1e-7)


def test_log1p_expm1_match_numpy():
    x = np.array([1e-6, 1e-3, 1.0, 5.0])
    np.testing.assert_allclose(ts.ops.log1p(x), np.log1p(x))
    np.testing.assert_allclose(ts.ops.expm1(x), np.expm1(x))


def test_softplus_is_stable_for_negative_and_large_inputs():
    x = np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0])
    out = ts.ops.softplus(x)
    # softplus(x) = log(1 + exp(x)). Reference values:
    #   softplus(-1000) ≈ 0      (exp(-1000) underflows; result is exact 0)
    #   softplus(-1)    ≈ 0.3133
    #   softplus(0)     ≈ 0.6931
    #   softplus(1)     ≈ 1.3133
    #   softplus(1000)  = 1000.0  (1 + exp(1000) ≈ exp(1000) -> log = 1000)
    expected = np.array([0.0, 0.31326169, 0.69314718, 1.31326169, 1000.0])
    np.testing.assert_allclose(out, expected, atol=1e-6)
    assert np.isfinite(out).all(), "softplus must not produce nan/inf at extremes"


def test_sigmoid_safe_no_overflow_at_extremes():
    x = np.array([-1000.0, 0.0, 1000.0])
    out = ts.ops.sigmoid_safe(x)
    np.testing.assert_allclose(out, [0.0, 0.5, 1.0], atol=1e-7)


# ── numeric helpers + comparisons ──────────────────────────────────────────


def test_clamp_min_and_max():
    x = np.array([-1.0, 0.5, 2.0])
    np.testing.assert_array_equal(ts.ops.clamp(x, min=0.0, max=1.0),
                                   [0.0, 0.5, 1.0])


def test_where_selects_branchwise():
    cond = np.array([True, False, True])
    x = np.array([1, 2, 3])
    y = np.array([10, 20, 30])
    np.testing.assert_array_equal(ts.ops.where(cond, x, y), [1, 20, 3])


def test_minimum_maximum_elementwise():
    a = np.array([1, 5, 3])
    b = np.array([2, 4, 6])
    np.testing.assert_array_equal(ts.ops.minimum(a, b), [1, 4, 3])
    np.testing.assert_array_equal(ts.ops.maximum(a, b), [2, 5, 6])


def test_isnan_isinf_isfinite():
    x = np.array([0.0, np.nan, np.inf, -np.inf, 1.0])
    np.testing.assert_array_equal(ts.ops.isnan(x), [False, True, False, False, False])
    np.testing.assert_array_equal(ts.ops.isinf(x), [False, False, True, True, False])
    np.testing.assert_array_equal(ts.ops.isfinite(x), [True, False, False, False, True])


def test_sign_and_absolute():
    x = np.array([-2.0, 0.0, 3.0])
    np.testing.assert_array_equal(ts.ops.sign(x), [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(ts.ops.absolute(x), [2.0, 0.0, 3.0])


@pytest.mark.parametrize("op,expected", [
    ("eq", [True, False, False]),
    ("ne", [False, True, True]),
    ("lt", [False, True, False]),
    ("le", [True, True, False]),
    ("gt", [False, False, True]),
    ("ge", [True, False, True]),
])
def test_comparisons(op, expected):
    a = np.array([1, 1, 3])
    b = np.array([1, 2, 2])
    fn = getattr(ts.ops, op)
    np.testing.assert_array_equal(fn(a, b), expected)


# ── VJP coverage ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "mean", "prod", "amax", "amin", "var", "std", "cumsum",
    "logsumexp", "log_softmax", "log1p", "expm1", "softplus", "sigmoid_safe",
    "clamp", "where", "absolute", "minimum", "maximum",
])
def test_s2_primitive_has_registered_vjp(name):
    """All differentiable S2 primitives must register a VJP."""
    assert get_vjp(name) is not None, f"VJP not registered for {name}"


def test_mean_vjp_matches_numerical_gradient():
    x = np.arange(12.0).reshape(3, 4) + 0.1
    dout = np.array(1.5)
    grad, = get_vjp("mean")(dout, x, axis=None, keepdims=False)
    expected = _numeric_grad(lambda v: float(dout) * np.mean(v), x)
    np.testing.assert_allclose(grad, expected, atol=1e-3)


def test_logsumexp_vjp_matches_numerical_gradient():
    x = np.array([[1.0, 2.0, 3.0], [0.5, -1.0, 2.0]])
    dout = np.array([1.0, 1.0])
    grad, = get_vjp("logsumexp")(dout, x, axis=-1, keepdims=False)
    expected = _numeric_grad(
        lambda v: (dout * np.log(np.sum(np.exp(v), axis=-1))).sum(), x
    )
    np.testing.assert_allclose(grad, expected, atol=1e-3)


def test_log_softmax_vjp_matches_numerical_gradient():
    x = np.array([1.0, 2.0, 3.0])
    dout = np.array([0.5, -1.0, 0.25])
    grad, = get_vjp("log_softmax")(dout, x, axis=-1)
    expected = _numeric_grad(
        lambda v: float((dout * (v - np.log(np.sum(np.exp(v))))).sum()), x
    )
    np.testing.assert_allclose(grad, expected, atol=1e-3)


def test_softplus_vjp_matches_sigmoid():
    x = np.array([-2.0, -0.5, 0.0, 1.0, 5.0])
    dout = np.ones_like(x)
    grad, = get_vjp("softplus")(dout, x)
    expected = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_allclose(grad, expected, atol=1e-7)


def test_clamp_vjp_zeros_grad_outside_range():
    x = np.array([-2.0, 0.5, 1.5, 3.0])
    dout = np.ones_like(x)
    grad, = get_vjp("clamp")(dout, x, min=0.0, max=1.0)
    np.testing.assert_array_equal(grad, [0.0, 1.0, 0.0, 0.0])


def test_where_vjp_routes_grad_through_predicate():
    cond = np.array([True, False, True])
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([10.0, 20.0, 30.0])
    dout = np.array([100.0, 200.0, 300.0])
    grad_cond, grad_x, grad_y = get_vjp("where")(dout, cond, x, y)
    assert grad_cond is None  # Boolean predicate is not differentiable.
    np.testing.assert_array_equal(grad_x, [100.0, 0.0, 300.0])
    np.testing.assert_array_equal(grad_y, [0.0, 200.0, 0.0])


def test_minimum_vjp_routes_to_smaller_branch():
    x = np.array([1.0, 5.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    dout = np.ones_like(x)
    grad_x, grad_y = get_vjp("minimum")(dout, x, y)
    np.testing.assert_array_equal(grad_x, [1.0, 0.0, 1.0])
    np.testing.assert_array_equal(grad_y, [0.0, 1.0, 0.0])
