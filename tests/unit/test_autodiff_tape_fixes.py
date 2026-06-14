"""Regression tests for the autodiff v1-tape fixes (see COMPILER_AUDIT).

E3: scalar/0-d tape-link break — reductions feeding later ops keep their grad.
E1: reduce(op="mean") forward + VJP.
E2: clip min/max kwarg aliases (forward + backward STE).
F1: ops.minimum/maximum with a python scalar operand backprop correctly.
F2: ops.mul(scalar_tensor, python_float) carries the factor into the gradient.
"""

from __future__ import annotations

import numpy as np

import tessera as ts
from tessera import ops


def _np(x):
    return x.numpy() if hasattr(x, "numpy") else np.asarray(x)


def _grad_of(loss_fn, W):
    W._grad = None
    with ts.autodiff.tape() as t:
        loss = loss_fn()
        t.backward(loss)
    return (None if W.grad is None else W.grad.numpy()), float(_np(loss))


# ── E3: scalar-valued intermediates keep the gradient chain ──────────────────

def test_e3_scalar_reduce_then_mul_keeps_grad():
    rng = np.random.default_rng(0)
    W = ts.nn.Parameter(rng.standard_normal((4, 3)).astype(np.float32))
    X = rng.standard_normal((5, 4)).astype(np.float32)
    neg = np.full((), -0.5, np.float32)  # array, not python float

    g, _ = _grad_of(
        lambda: ops.mul(ops.reduce(ops.mul(ops.gemm(X, W), ops.gemm(X, W)), op="sum"), neg),
        W,
    )
    assert g is not None and np.all(np.isfinite(g))


def test_e3_scalar_reduce_then_exp_keeps_grad():
    rng = np.random.default_rng(1)
    W = ts.nn.Parameter(rng.standard_normal((3, 3)).astype(np.float32))
    X = rng.standard_normal((4, 3)).astype(np.float32)
    scale = np.full((), 0.01, np.float32)
    g, _ = _grad_of(lambda: ops.exp(ops.mul(ops.reduce(ops.gemm(X, W), op="sum"), scale)), W)
    assert g is not None and np.all(np.isfinite(g))


# ── E1: reduce(op="mean") ────────────────────────────────────────────────────

def test_e1_reduce_mean_forward():
    x = np.random.default_rng(2).standard_normal((4, 6)).astype(np.float32)
    np.testing.assert_allclose(_np(ops.reduce(x, op="mean")), np.mean(x), rtol=1e-6)
    np.testing.assert_allclose(
        _np(ops.reduce(x, op="mean", axis=1)), np.mean(x, axis=1), rtol=1e-6
    )


def test_e1_reduce_mean_grad_matches_numerical():
    rng = np.random.default_rng(3)
    W = ts.nn.Parameter(rng.standard_normal((5, 4)).astype(np.float32))
    X = rng.standard_normal((6, 5)).astype(np.float32)

    def loss():
        return ops.reduce(ops.gemm(X, W), op="mean")

    g, _ = _grad_of(loss, W)
    # Numerical gradient via central difference on a few entries.
    base = W.numpy().copy()
    eps = 1e-3
    for (i, j) in [(0, 0), (2, 1), (4, 3)]:
        wp = base.copy(); wp[i, j] += eps
        wm = base.copy(); wm[i, j] -= eps
        fp = np.mean(X @ wp); fm = np.mean(X @ wm)
        num = (fp - fm) / (2 * eps)
        assert abs(g[i, j] - num) < 1e-2


def test_e1_reduce_rejects_unsupported_op():
    import pytest
    with pytest.raises(ValueError):
        ops.reduce(np.ones((3,), np.float32), op="prod")


# ── E2: clip min/max aliases ─────────────────────────────────────────────────

def test_e2_clip_aliases_forward_match_canonical():
    x = np.array([-2.0, 0.5, 3.0], np.float32)
    np.testing.assert_array_equal(
        _np(ops.clip(x, min=0.0, max=1.0)), _np(ops.clip(x, min_val=0.0, max_val=1.0))
    )
    np.testing.assert_array_equal(_np(ops.clip(x, min=0.0, max=1.0)), np.clip(x, 0.0, 1.0))


def test_e2_clip_alias_backward_ste():
    # STE: grad passes only where strictly inside [min, max].
    W = ts.nn.Parameter(np.array([[-2.0, 0.5, 3.0]], np.float32))
    g, _ = _grad_of(lambda: ops.reduce(ops.clip(ops.mul(W, np.ones((1, 3), np.float32)),
                                                min=0.0, max=1.0), op="sum"), W)
    np.testing.assert_array_equal(g, np.array([[0.0, 1.0, 0.0]], np.float32))


# ── F1: minimum/maximum with a python scalar operand ─────────────────────────

def test_f1_minimum_scalar_backprops():
    rng = np.random.default_rng(0)
    W = ts.nn.Parameter(rng.standard_normal((4, 3)).astype(np.float32))
    X = rng.standard_normal((5, 4)).astype(np.float32)
    g_min, _ = _grad_of(lambda: ops.reduce(ops.minimum(ops.gemm(X, W), 0.5), op="sum"), W)
    g_max, _ = _grad_of(lambda: ops.reduce(ops.maximum(ops.gemm(X, W), 0.5), op="sum"), W)
    assert g_min is not None and np.all(np.isfinite(g_min))
    assert g_max is not None and np.all(np.isfinite(g_max))


def test_f1_minimum_scalar_gradient_is_correct():
    # min(z, c): grad flows only where z < c. Build z = X @ W, check the gate.
    W = ts.nn.Parameter(np.array([[1.0]], np.float32))
    X = np.array([[0.2], [0.9]], np.float32)  # z = [0.2, 0.9]; clip at 0.5
    g, _ = _grad_of(lambda: ops.reduce(ops.minimum(ops.mul(X, W) if False else ops.gemm(X, W), 0.5), op="sum"), W)
    # d/dW sum(min(X*W, 0.5)) = sum over rows where X*W < 0.5 of X.
    # At W=1: row0 (0.2<0.5) contributes 0.2; row1 (0.9>=0.5) contributes 0.
    assert np.isclose(g[0, 0], 0.2, atol=1e-5)


# ── F2: mul by a python float carries the factor ─────────────────────────────

def test_f2_mul_scalar_carries_factor():
    W = ts.nn.Parameter(np.array([[1.0, 2.0]], np.float32))
    g, _ = _grad_of(lambda: ops.mul(ops.reduce(W, op="sum"), -3.0), W)
    np.testing.assert_array_equal(g, np.array([[-3.0, -3.0]], np.float32))


def test_f2_mul_array_operand_still_correct():
    W = ts.nn.Parameter(np.array([[1.0, 2.0]], np.float32))
    g, _ = _grad_of(lambda: ops.reduce(ops.mul(W, np.array([[2.0, 2.0]], np.float32)), op="sum"), W)
    np.testing.assert_array_equal(g, np.array([[2.0, 2.0]], np.float32))
