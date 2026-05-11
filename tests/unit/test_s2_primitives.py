"""Tests for S-series sprint S2 — reductions, stability primitives,
numeric helpers, comparisons.

For each primitive we check:
  - the numpy-reference output matches the canonical numpy call
  - VJPs are registered (where applicable)
  - VJPs match a numerical gradient on a small input
"""

from __future__ import annotations

import math

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
    np.testing.assert_array_equal(ts.ops.cummax(x), np.maximum.accumulate(x))
    np.testing.assert_array_equal(ts.ops.cummin(x[::-1]), np.minimum.accumulate(x[::-1]))


def test_amax_amin_match_numpy():
    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    np.testing.assert_array_equal(ts.ops.amax(x, axis=1), np.max(x, axis=1))
    np.testing.assert_array_equal(ts.ops.amin(x, axis=0), np.min(x, axis=0))
    np.testing.assert_array_equal(ts.ops.max(x, axis=1), np.max(x, axis=1))
    np.testing.assert_array_equal(ts.ops.min(x, axis=0), np.min(x, axis=0))


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
    np.testing.assert_array_equal(ts.ops.abs(x), [2.0, 0.0, 3.0])


def test_s2_scalar_math_matches_numpy_and_math_refs():
    x = np.array([0.25, 0.5, 1.5], dtype=np.float64)
    y = np.array([1.5, 2.0, 3.0], dtype=np.float64)
    trig = np.array([-0.5, 0.0, 0.5], dtype=np.float64)

    np.testing.assert_allclose(ts.ops.sub(y, x), np.subtract(y, x))
    np.testing.assert_allclose(ts.ops.div(y, x), np.divide(y, x))
    np.testing.assert_allclose(ts.ops.floor_div(y, x), np.floor_divide(y, x))
    np.testing.assert_allclose(ts.ops.mod(y, x), np.mod(y, x))
    np.testing.assert_allclose(ts.ops.exp(x), np.exp(x))
    np.testing.assert_allclose(ts.ops.log(x), np.log(x))
    np.testing.assert_allclose(ts.ops.sqrt(x), np.sqrt(x))
    np.testing.assert_allclose(ts.ops.rsqrt(x), 1.0 / np.sqrt(x))
    np.testing.assert_allclose(ts.ops.pow(x, y), np.power(x, y))
    np.testing.assert_allclose(ts.ops.cos(trig), np.cos(trig))
    np.testing.assert_allclose(ts.ops.tan(trig), np.tan(trig))
    np.testing.assert_allclose(ts.ops.sinh(trig), np.sinh(trig))
    np.testing.assert_allclose(ts.ops.cosh(trig), np.cosh(trig))
    np.testing.assert_allclose(ts.ops.asin(trig), np.arcsin(trig))
    np.testing.assert_allclose(ts.ops.acos(trig), np.arccos(trig))
    np.testing.assert_allclose(ts.ops.atan(trig), np.arctan(trig))
    np.testing.assert_allclose(ts.ops.atan2(y, x), np.arctan2(y, x))


def test_s2_erf_erfc_lgamma_match_math_refs():
    x = np.array([-1.5, -0.25, 0.0, 0.75, 2.0], dtype=np.float64)
    positive = np.array([0.25, 1.0, 3.5], dtype=np.float64)

    np.testing.assert_allclose(
        ts.ops.erf(x), np.array([math.erf(v) for v in x], dtype=np.float64)
    )
    np.testing.assert_allclose(
        ts.ops.erfc(x), np.array([math.erfc(v) for v in x], dtype=np.float64)
    )
    np.testing.assert_allclose(
        ts.ops.lgamma(positive),
        np.array([math.lgamma(v) for v in positive], dtype=np.float64),
    )


def test_s2_digamma_matches_known_values():
    gamma = 0.5772156649015329
    x = np.array([0.5, 1.0, 2.0, 4.0], dtype=np.float64)
    expected = np.array([
        -gamma - 2.0 * math.log(2.0),
        -gamma,
        1.0 - gamma,
        1.0 + 0.5 + (1.0 / 3.0) - gamma,
    ])
    np.testing.assert_allclose(ts.ops.digamma(x), expected, atol=1e-8)


def test_s2_numeric_rounding_helpers_match_numpy():
    x = np.array([-1.75, -0.25, 0.25, 1.75])
    np.testing.assert_allclose(ts.ops.reciprocal(np.array([0.5, 2.0])), [2.0, 0.5])
    np.testing.assert_array_equal(ts.ops.floor(x), np.floor(x))
    np.testing.assert_array_equal(ts.ops.ceil(x), np.ceil(x))
    np.testing.assert_array_equal(ts.ops.round(x), np.round(x))
    np.testing.assert_array_equal(ts.ops.trunc(x), np.trunc(x))


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


def test_s2_logical_ops_match_numpy():
    a = np.array([True, True, False, False])
    b = np.array([True, False, True, False])
    np.testing.assert_array_equal(ts.ops.logical_and(a, b), np.logical_and(a, b))
    np.testing.assert_array_equal(ts.ops.logical_or(a, b), np.logical_or(a, b))
    np.testing.assert_array_equal(ts.ops.logical_not(a), np.logical_not(a))
    np.testing.assert_array_equal(ts.ops.logical_xor(a, b), np.logical_xor(a, b))


def test_s2_bitwise_ops_match_numpy():
    a = np.array([0b1100, 0b1010, 0b0110], dtype=np.int32)
    b = np.array([0b1010, 0b0101, 0b0011], dtype=np.int32)
    np.testing.assert_array_equal(ts.ops.bitwise_and(a, b), np.bitwise_and(a, b))
    np.testing.assert_array_equal(ts.ops.bitwise_or(a, b), np.bitwise_or(a, b))
    np.testing.assert_array_equal(ts.ops.bitwise_xor(a, b), np.bitwise_xor(a, b))
    np.testing.assert_array_equal(ts.ops.bitwise_not(a), np.bitwise_not(a))


# ── tensor algebra + indexing ───────────────────────────────────────────────


def test_s2_shape_view_ops_match_numpy():
    x = np.arange(24.0).reshape(2, 3, 4)

    np.testing.assert_array_equal(ts.ops.reshape(x, (3, 8)), np.reshape(x, (3, 8)))
    np.testing.assert_array_equal(ts.ops.view(x, (4, 6)), np.reshape(x, (4, 6)))
    np.testing.assert_array_equal(ts.ops.flatten(x, start_axis=1), x.reshape(2, 12))
    np.testing.assert_array_equal(ts.ops.squeeze(x[:, :1, :], axis=1), np.squeeze(x[:, :1, :], axis=1))
    np.testing.assert_array_equal(ts.ops.unsqueeze(x, axis=1), np.expand_dims(x, axis=1))
    np.testing.assert_array_equal(ts.ops.permute(x, (2, 0, 1)), np.transpose(x, (2, 0, 1)))
    np.testing.assert_array_equal(ts.ops.broadcast(np.array([1.0, 2.0, 3.0]), (2, 3)),
                                  np.broadcast_to([1.0, 2.0, 3.0], (2, 3)))
    np.testing.assert_array_equal(ts.ops.expand(np.array([[1.0], [2.0]]), (2, 3)),
                                  np.broadcast_to([[1.0], [2.0]], (2, 3)))


def test_s2_sequence_shape_ops_match_numpy():
    a = np.arange(6).reshape(2, 3)
    b = np.arange(6, 12).reshape(2, 3)
    x = np.arange(12).reshape(3, 4)

    np.testing.assert_array_equal(ts.ops.cat((a, b), axis=0), np.concatenate((a, b), axis=0))
    np.testing.assert_array_equal(ts.ops.stack((a, b), axis=1), np.stack((a, b), axis=1))
    assert all(np.array_equal(got, exp) for got, exp in zip(ts.ops.split(x, 3, axis=0), np.array_split(x, 3, axis=0)))
    assert all(np.array_equal(got, exp) for got, exp in zip(ts.ops.chunk(x, 2, axis=1), np.array_split(x, 2, axis=1)))
    np.testing.assert_array_equal(ts.ops.pad(a, ((1, 0), (0, 2)), constant_values=-1),
                                  np.pad(a, ((1, 0), (0, 2)), constant_values=-1))
    np.testing.assert_array_equal(ts.ops.tile(a, (2, 1)), np.tile(a, (2, 1)))
    np.testing.assert_array_equal(ts.ops.repeat(a, 2, axis=1), np.repeat(a, 2, axis=1))
    np.testing.assert_array_equal(ts.ops.roll(a, shift=1, axis=0), np.roll(a, shift=1, axis=0))
    np.testing.assert_array_equal(ts.ops.flip(a, axis=1), np.flip(a, axis=1))


def test_s2_dynamic_slice_and_update_are_functional():
    x = np.arange(20).reshape(4, 5)
    update = np.full((2, 3), -1)

    np.testing.assert_array_equal(ts.ops.dynamic_slice(x, (1, 1), (2, 3)), x[1:3, 1:4])
    np.testing.assert_array_equal(ts.ops.slice(x, (1, 1), (2, 3)), x[1:3, 1:4])
    np.testing.assert_array_equal(ts.ops.select(x, 2, axis=0), x[2])
    out = ts.ops.dynamic_update_slice(x, update, (1, 2))
    expected = x.copy()
    expected[1:3, 2:5] = update
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(x, np.arange(20).reshape(4, 5))


def test_s2_indexing_sorting_ops_match_numpy():
    x = np.array([[3.0, 1.0, 2.0], [9.0, 8.0, 7.0], [4.0, 6.0, 5.0]])
    indices = np.array([2, 0])

    np.testing.assert_array_equal(ts.ops.take(x, indices, axis=0), np.take(x, indices, axis=0))
    np.testing.assert_array_equal(ts.ops.index_select(x, indices, axis=1), np.take(x, indices, axis=1))
    np.testing.assert_array_equal(ts.ops.nonzero(x > 5), np.nonzero(x > 5))
    values, value_indices = ts.ops.top_k(x, k=2, axis=1)
    np.testing.assert_array_equal(values, np.array([[3.0, 2.0], [9.0, 8.0], [6.0, 5.0]]))
    np.testing.assert_array_equal(value_indices, np.array([[0, 2], [0, 1], [1, 2]]))
    np.testing.assert_array_equal(ts.ops.sort(x, axis=1), np.sort(x, axis=1))
    np.testing.assert_array_equal(ts.ops.sort(x, axis=1, descending=True), np.flip(np.sort(x, axis=1), axis=1))
    np.testing.assert_array_equal(ts.ops.argsort(x, axis=1), np.argsort(x, axis=1))


def test_s2_scatter_family_is_functional_and_matches_reference():
    x = np.zeros((4, 2), dtype=np.float64)
    indices = np.array([2, 1, 2])
    updates = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])

    set_expected = x.copy()
    set_expected[indices] = updates
    add_expected = x.copy()
    np.add.at(add_expected, indices, updates)

    np.testing.assert_array_equal(ts.ops.scatter(x, indices, updates, axis=0), set_expected)
    np.testing.assert_array_equal(ts.ops.index_update(x, indices, updates, axis=0), set_expected)
    np.testing.assert_array_equal(ts.ops.scatter_add(x, indices, updates, axis=0), add_expected)
    np.testing.assert_array_equal(ts.ops.scatter_reduce(x, indices, updates, axis=0, reduce="sum"), add_expected)
    np.testing.assert_array_equal(x, np.zeros((4, 2), dtype=np.float64))


# ── VJP coverage ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "mean", "prod", "amax", "amin", "max", "min", "var", "std", "cumsum",
    "logsumexp", "log_softmax", "log1p", "expm1", "softplus", "sigmoid_safe",
    "sub", "div", "exp", "log", "sqrt", "rsqrt", "pow", "cos", "tan",
    "sinh", "cosh", "asin", "acos", "atan", "atan2", "erf", "erfc",
    "lgamma", "digamma", "reciprocal", "clamp", "where", "absolute", "minimum", "maximum",
    "reshape", "view", "flatten", "squeeze", "unsqueeze", "permute",
    "broadcast", "expand", "cat", "stack", "split", "chunk", "pad", "tile",
    "repeat", "roll", "flip", "dynamic_slice", "dynamic_update_slice",
    "slice", "select",
    "take", "index_select", "scatter", "scatter_add", "scatter_reduce", "index_update",
])
def test_s2_primitive_has_registered_vjp(name):
    """All differentiable S2 primitives must register a VJP."""
    assert get_vjp(name) is not None, f"VJP not registered for {name}"


@pytest.mark.parametrize("name", [
    # Pure discontinuities — no closed-form or STE rule yet
    "floor", "ceil", "round", "trunc",
    # Comparisons (boolean output)
    "eq", "ne", "lt", "le", "gt", "ge",
    # Logical / bitwise (boolean / integer output)
    "logical_and", "logical_or", "logical_not", "logical_xor",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    # Predicates (boolean output)
    "isnan", "isinf", "isfinite",
    # Index-producing ops (integer output)
    "argmax", "argmin", "nonzero", "top_k", "sort", "argsort",
])
def test_s2_non_differentiable_or_discontinuous_ops_skip_vjp(name):
    """Ops whose outputs are not in the differentiable manifold (integers,
    booleans, indices) have no VJP — calling them on the gradient path
    raises ``TesseraAutodiffError``.

    Sprint A (2026-05-11): `floor_div` / `mod` / `sign` / `cumprod` were
    moved OUT of this list — they're piecewise-constant or have non-trivial
    Jacobians, but each got an STE-style or closed-form VJP that the
    registry can flip to `complete`.  See `test_s2_ste_or_closed_form_vjp_ops`
    for the positive coverage.
    """
    assert get_vjp(name) is None


@pytest.mark.parametrize("name", ["floor_div", "mod", "sign", "cumprod"])
def test_s2_ste_or_closed_form_vjp_ops_have_registered_vjp(name):
    """Sprint A (2026-05-11): piecewise-constant ops (`floor_div`/`mod`/
    `sign`) get STE-style zero VJPs; `cumprod` gets a closed-form ratio-
    cumsum VJP.  The registry flips these from `vjp = planned` to `complete`
    once they're registered."""
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


@pytest.mark.parametrize("name,fn", [
    ("exp", lambda v: np.exp(v).sum()),
    ("log", lambda v: np.log(v).sum()),
    ("sqrt", lambda v: np.sqrt(v).sum()),
    ("rsqrt", lambda v: (1.0 / np.sqrt(v)).sum()),
    ("cos", lambda v: np.cos(v).sum()),
    ("tan", lambda v: np.tan(v).sum()),
    ("sinh", lambda v: np.sinh(v).sum()),
    ("cosh", lambda v: np.cosh(v).sum()),
    ("asin", lambda v: np.arcsin(v).sum()),
    ("acos", lambda v: np.arccos(v).sum()),
    ("atan", lambda v: np.arctan(v).sum()),
    ("erf", lambda v: np.array([math.erf(float(x)) for x in v]).sum()),
    ("erfc", lambda v: np.array([math.erfc(float(x)) for x in v]).sum()),
    ("lgamma", lambda v: np.array([math.lgamma(float(x)) for x in v]).sum()),
    ("digamma", lambda v: ts.ops.digamma(v).sum()),
    ("reciprocal", lambda v: np.reciprocal(v).sum()),
])
def test_unary_scalar_math_vjp_matches_numerical_gradient(name, fn):
    x = np.array([0.25, 0.5, 0.75], dtype=np.float64)
    if name in {"asin", "acos", "atan", "cos", "tan", "sinh", "cosh", "erf", "erfc"}:
        x = np.array([-0.25, 0.0, 0.25], dtype=np.float64)
    if name in {"lgamma", "digamma"}:
        x = np.array([0.75, 1.5, 3.0], dtype=np.float64)
    dout = np.ones_like(x)
    grad, = get_vjp(name)(dout, x)
    expected = _numeric_grad(fn, x)
    np.testing.assert_allclose(grad, expected, atol=1e-3)


def test_binary_scalar_math_vjps_match_numerical_gradient():
    x = np.array([0.5, 1.0, 1.5], dtype=np.float64)
    y = np.array([1.25, 2.0, 2.5], dtype=np.float64)
    dout = np.ones_like(x)

    grad_x, grad_y = get_vjp("sub")(dout, x, y)
    np.testing.assert_allclose(grad_x, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(grad_y, [-1.0, -1.0, -1.0])

    grad_x, grad_y = get_vjp("div")(dout, x, y)
    np.testing.assert_allclose(grad_x, 1.0 / y)
    np.testing.assert_allclose(grad_y, -x / (y * y))

    grad_x, grad_y = get_vjp("pow")(dout, x, y)
    expected_x = _numeric_grad(lambda v: np.power(v, y).sum(), x)
    expected_y = _numeric_grad(lambda v: np.power(x, v).sum(), y)
    np.testing.assert_allclose(grad_x, expected_x, atol=1e-3)
    np.testing.assert_allclose(grad_y, expected_y, atol=1e-3)

    grad_y, grad_x = get_vjp("atan2")(dout, y, x)
    expected_y = _numeric_grad(lambda v: np.arctan2(v, x).sum(), y)
    expected_x = _numeric_grad(lambda v: np.arctan2(y, v).sum(), x)
    np.testing.assert_allclose(grad_y, expected_y, atol=1e-3)
    np.testing.assert_allclose(grad_x, expected_x, atol=1e-3)


def test_shape_vjps_route_cotangents_back_to_original_shape():
    x = np.arange(6.0).reshape(2, 3)
    dout = np.ones((3, 2), dtype=np.float64)

    grad, = get_vjp("reshape")(dout, x, shape=(3, 2))
    np.testing.assert_array_equal(grad, np.ones_like(x))

    grad, = get_vjp("permute")(np.ones((3, 2)), x, axes=(1, 0))
    np.testing.assert_array_equal(grad, np.ones_like(x))

    grad, = get_vjp("broadcast")(np.ones((4, 2, 3)), x, shape=(4, 2, 3))
    np.testing.assert_array_equal(grad, np.full_like(x, 4.0))

    padded = ts.ops.pad(x, ((1, 1), (2, 0)))
    grad, = get_vjp("pad")(np.ones_like(padded), x, pad_width=((1, 1), (2, 0)))
    np.testing.assert_array_equal(grad, np.ones_like(x))


def test_sequence_shape_vjps_concatenate_and_split_cotangents():
    a = np.ones((2, 3), dtype=np.float64)
    b = np.full((1, 3), 2.0, dtype=np.float64)
    cat_grad, = get_vjp("cat")(np.ones((3, 3)), (a, b), axis=0)
    np.testing.assert_array_equal(cat_grad[0], np.ones_like(a))
    np.testing.assert_array_equal(cat_grad[1], np.ones_like(b))

    stack_grad, = get_vjp("stack")(np.ones((2, 2, 3)), (a, a), axis=0)
    np.testing.assert_array_equal(stack_grad[0], np.ones_like(a))
    np.testing.assert_array_equal(stack_grad[1], np.ones_like(a))

    split_grad, = get_vjp("split")((np.ones((1, 3)), np.ones((2, 3))), np.zeros((3, 3)), axis=0)
    np.testing.assert_array_equal(split_grad, np.ones((3, 3)))


def test_dynamic_slice_and_update_vjps():
    x = np.arange(12.0).reshape(3, 4)
    dout_slice = np.ones((2, 2), dtype=np.float64)
    grad, = get_vjp("dynamic_slice")(dout_slice, x, start_indices=(1, 1), slice_sizes=(2, 2))
    expected = np.zeros_like(x)
    expected[1:3, 1:3] = 1.0
    np.testing.assert_array_equal(grad, expected)

    grad, = get_vjp("slice")(dout_slice, x, start_indices=(1, 1), slice_sizes=(2, 2))
    np.testing.assert_array_equal(grad, expected)

    grad, = get_vjp("select")(np.ones(4), x, index=1, axis=0)
    expected_select = np.zeros_like(x)
    expected_select[1] = 1.0
    np.testing.assert_array_equal(grad, expected_select)

    update = np.ones((2, 2), dtype=np.float64)
    dout_update = np.arange(12.0).reshape(3, 4)
    grad_x, grad_update = get_vjp("dynamic_update_slice")(dout_update, x, update, start_indices=(1, 1))
    expected_x = dout_update.copy()
    expected_x[1:3, 1:3] = 0.0
    np.testing.assert_array_equal(grad_x, expected_x)
    np.testing.assert_array_equal(grad_update, dout_update[1:3, 1:3])


def test_take_and_scatter_vjps():
    x = np.arange(8.0).reshape(4, 2)
    indices = np.array([2, 0, 2])
    updates = np.ones((3, 2), dtype=np.float64)
    dout = np.full((3, 2), 2.0, dtype=np.float64)

    grad_x, grad_indices = get_vjp("take")(dout, x, indices, axis=0)
    expected = np.zeros_like(x)
    np.add.at(expected, indices, dout)
    np.testing.assert_array_equal(grad_x, expected)
    assert grad_indices is None

    dout_scatter = np.arange(8.0).reshape(4, 2)
    grad_x, grad_indices, grad_updates = get_vjp("scatter_add")(dout_scatter, x, indices, updates, axis=0)
    np.testing.assert_array_equal(grad_x, dout_scatter)
    assert grad_indices is None
    np.testing.assert_array_equal(grad_updates, dout_scatter[indices])

    grad_x, _, grad_updates = get_vjp("scatter")(dout_scatter, x, indices, updates, axis=0)
    expected_x = dout_scatter.copy()
    expected_x[indices] = 0.0
    np.testing.assert_array_equal(grad_x, expected_x)
    np.testing.assert_array_equal(grad_updates, dout_scatter[indices])
