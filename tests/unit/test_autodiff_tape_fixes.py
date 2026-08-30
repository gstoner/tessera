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


def test_keyword_spelled_operand_is_recorded_as_operand():
    """PR #604 review (P1), the general class: any op whose second operand
    is spelled by keyword (`ops.mul(x, y=y)`) left it in kwargs, so the
    record never saw it while the rule bound it by name and answered for
    it — "VJP returned 2 cotangents, expected 1". `promote_operand_kwargs`
    routes a kwarg into the positional operand list iff its name is the
    rule's next positional slot AND the forward's parameter at that index
    (config like `eps`/`axis` is keyword-only in the rules and can never
    be promoted)."""
    import numpy as _np

    from tessera import ops
    from tessera.autodiff.tape import tape

    rng = _np.random.default_rng(41)
    x = rng.standard_normal(5)
    y = rng.standard_normal(5)
    with tape() as t:
        loss = ops.sum(ops.mul(x, y=y))
    t.backward(loss)
    _np.testing.assert_array_equal(t.cotangent[id(y)], x)
    _np.testing.assert_array_equal(t.cotangent[id(x)], y)


def test_fused_gemm_bias_and_residual_are_real_tape_operands():
    A = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    B = np.array([[0.5, -1.0], [2.0, 0.25]], dtype=np.float64)
    bias = np.array([0.2, -0.4], dtype=np.float64)
    residual = np.arange(4, dtype=np.float64).reshape(2, 2)

    dbias = ts.autodiff.grad(
        lambda b: ops.reduce(ops.gemm(A, B, bias=b), op="sum")
    )(bias)
    dresidual = ts.autodiff.grad(
        lambda r: ops.reduce(ops.gemm(A, B, residual=r), op="sum")
    )(residual)
    np.testing.assert_array_equal(dbias, np.array([2.0, 2.0]))
    np.testing.assert_array_equal(dresidual, np.ones_like(residual))


def test_fused_gemm_bias_and_residual_jvps_are_not_dropped():
    A = np.eye(2, dtype=np.float64)
    B = np.array([[2.0, -1.0], [0.5, 3.0]], dtype=np.float64)
    bias = np.array([0.1, -0.2], dtype=np.float64)
    residual = np.zeros((2, 2), dtype=np.float64)

    _, dbias = ts.autodiff.jvp(
        lambda b: ops.gemm(A, B, bias=b),
        (bias,), (np.ones_like(bias),),
    )
    _, dresidual = ts.autodiff.jvp(
        lambda r: ops.gemm(A, B, residual=r),
        (residual,), (np.ones_like(residual),),
    )
    np.testing.assert_array_equal(dbias, np.ones((2, 2)))
    np.testing.assert_array_equal(dresidual, np.ones((2, 2)))


def test_qr_r_component_tape_gradient_uses_r_cotangent_route():
    A = np.array(
        [[2.0, -0.5, 0.3], [0.4, 1.7, -0.2], [0.1, 0.6, 1.4]],
        dtype=np.float64,
    )

    def squared_r_norm(matrix):
        _, R = ops.qr(matrix)
        return ops.reduce(ops.mul(R, R), op="sum")

    # ||R||_F^2 == ||A||_F^2 for QR, hence the exact gradient is 2A.
    np.testing.assert_allclose(
        ts.autodiff.grad(squared_r_norm)(A), 2.0 * A,
        rtol=1.0e-10, atol=1.0e-10,
    )


# ── Aliased op output: the entry's adjoint is consumed, not re-accumulated ───

def test_identity_returning_op_does_not_double_upstream_gradients():
    """`ops.clamp(y)` with no bounds returns `y` itself, so the clamp tape entry
    shares its output_id with its input's array_id. Reading the adjoint and then
    accumulating the passthrough cotangent into that same key used to double
    every gradient upstream of the alias."""
    x = np.array([1.0, 2.0])

    def with_alias(v):
        return ops.reduce(ops.clamp(ops.mul(v, 2.0)), op="sum")

    def without_alias(v):
        return ops.reduce(ops.mul(v, 2.0), op="sum")

    expected = np.array([2.0, 2.0])
    np.testing.assert_allclose(ts.autodiff.grad(with_alias)(x), expected)
    # The alias must not change the answer.
    np.testing.assert_allclose(
        ts.autodiff.grad(with_alias)(x), ts.autodiff.grad(without_alias)(x)
    )


def test_aliased_output_still_accumulates_real_fan_out():
    """Consuming the adjoint must not drop a genuine second use of the value:
    `y` feeds both the alias and the addition, so both paths must contribute."""
    x = np.array([1.0, 2.0])

    def fan_out(v):
        y = ops.mul(v, 2.0)
        return ops.reduce(ops.add(y, ops.clamp(y)), op="sum")

    # d/dv sum(y + y) with y = 2v  ->  4 per element.
    np.testing.assert_allclose(ts.autodiff.grad(fan_out)(x), np.array([4.0, 4.0]))


# ── the forward-signature cache serves every signature question ─────────────

def test_kwarg_operand_promotion_does_not_reparse_signatures_per_call():
    """`promote_operand_kwargs.has_none_default` called `inspect.signature`
    directly, bypassing the cache two functions above it, so every taped call
    carrying a kwarg re-parsed the forward — ~7 us on this Mac, about a third
    of the call, repeated identically every iteration of a training loop
    (2026-08-29 review, P2)."""
    import importlib
    import inspect

    # `tessera.autodiff.tape` the name is the context manager, not the module.
    tape_mod = importlib.import_module("tessera.autodiff.tape")

    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 8))
    B = rng.standard_normal((8, 8))

    calls = []
    real = inspect.signature

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    # `activation` makes the rule's unfilled `bias` slot the one this path
    # asks about, which is what enters `has_none_default`.
    with ts.autodiff.tape():
        ops.gemm(A, B, activation="relu")       # warm every cache entry

    original = tape_mod.inspect.signature
    try:
        tape_mod.inspect.signature = counting
        with ts.autodiff.tape():
            for _ in range(50):
                ops.gemm(A, B, activation="relu")
    finally:
        tape_mod.inspect.signature = original

    assert len(calls) == 0, (
        f"{len(calls)} signature parses for 50 warm taped calls — the cache "
        f"is being bypassed again"
    )
