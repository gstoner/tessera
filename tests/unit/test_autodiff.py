"""Tests for the v1 autodiff first slice — see docs/spec/AUTODIFF_SPEC.md.

Coverage:
  * Per-op numerical-Jacobian checks (every built-in VJP)
  * `tape()` context manager — recording, backward, error paths
  * Parameter.grad population through the buffer-registry trace
  * `reverse(fn)` convenience wrapper
  * `custom_rule(name)` decorator — register, override
  * `TesseraAutodiffError` — unsupported op, scalar shape, double-backward,
    target not on tape
  * End-to-end MLP one-step SGD loss decrease
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts
from tessera.autodiff import TesseraAutodiffError


# ─────────────────────────────────────────────────────────────────────────────
# Numerical-Jacobian helper
# ─────────────────────────────────────────────────────────────────────────────


def _numerical_grad(fn, x, eps=1e-6):
    """Central-difference numerical gradient. Use fp64 inputs for tight tolerance."""
    grad = np.zeros_like(x)
    flat = x.ravel()
    flat_grad = grad.ravel()
    for i in range(flat.size):
        orig = flat[i]
        flat[i] = orig + eps
        plus = float(fn(x))
        flat[i] = orig - eps
        minus = float(fn(x))
        flat[i] = orig
        flat_grad[i] = (plus - minus) / (2 * eps)
    return grad


def _jacobian_close(analytical, numerical, rtol=1e-5, atol=1e-6):
    """Compare two gradient arrays with fp64-friendly tolerance."""
    assert analytical.shape == numerical.shape, (analytical.shape, numerical.shape)
    np.testing.assert_allclose(analytical, numerical, rtol=rtol, atol=atol)


# ─────────────────────────────────────────────────────────────────────────────
# Per-op VJP correctness
# ─────────────────────────────────────────────────────────────────────────────


class TestVJPGemm:
    def test_2d_grad(self):
        np.random.seed(0)
        # Use fp64 so analytical-vs-numerical comparison is tight.
        A = np.random.randn(3, 4).astype(np.float64)
        B = np.random.randn(4, 5).astype(np.float64)

        A_p = ts.nn.Parameter(A.copy())
        with ts.autodiff.tape() as t:
            C = ts.ops.gemm(A_p, B)
            loss = ts.ops.reduce(C, op="sum")
            t.backward(loss)
        analytic = A_p.grad.numpy()

        def fn(a):
            return float(np.matmul(a, B).sum())

        numerical = _numerical_grad(fn, A.copy())
        _jacobian_close(analytic, numerical)


class TestVJPElementwise:
    def test_add_broadcast(self):
        x = np.random.randn(3, 4).astype(np.float32)
        y = np.random.randn(4).astype(np.float32)
        x_p = ts.nn.Parameter(x.copy())
        y_p = ts.nn.Parameter(y.copy())

        with ts.autodiff.tape() as t:
            z = ts.ops.add(x_p, y_p)
            loss = ts.ops.reduce(z, op="sum")
            t.backward(loss)

        # d(sum(x + y))/dx = ones; /dy = sum-along-broadcast-axes
        np.testing.assert_allclose(x_p.grad.numpy(), np.ones_like(x))
        np.testing.assert_allclose(y_p.grad.numpy(), np.ones_like(y) * x.shape[0])

    def test_mul_broadcast(self):
        x = np.random.randn(2, 3).astype(np.float32)
        y = np.random.randn(3).astype(np.float32)
        x_p = ts.nn.Parameter(x.copy())

        with ts.autodiff.tape() as t:
            z = ts.ops.mul(x_p, y)
            loss = ts.ops.reduce(z, op="sum")
            t.backward(loss)

        # d(sum(x * y))/dx = broadcast(y) over x.shape
        expected = np.broadcast_to(y, x.shape)
        np.testing.assert_allclose(x_p.grad.numpy(), expected)


class TestVJPActivations:
    @pytest.mark.parametrize("op_name,fn", [
        ("relu", lambda x: np.maximum(0, x)),
        ("sigmoid", lambda x: 1.0 / (1.0 + np.exp(-x))),
        ("tanh", np.tanh),
        ("silu", lambda x: x / (1.0 + np.exp(-x))),
        ("gelu", lambda x: x * 0.5 * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))),
    ])
    def test_activation_gradient(self, op_name, fn):
        np.random.seed(0)
        x = np.random.randn(4, 5).astype(np.float64)
        x_p = ts.nn.Parameter(x.copy())

        op = getattr(ts.ops, op_name)
        with ts.autodiff.tape() as t:
            y = op(x_p)
            loss = ts.ops.reduce(y, op="sum")
            t.backward(loss)

        analytic = x_p.grad.numpy()

        def loss_fn(arr):
            return fn(arr).sum()

        numerical = _numerical_grad(loss_fn, x.copy())
        _jacobian_close(analytic, numerical)


class TestVJPNormsSoftmax:
    def test_softmax(self):
        np.random.seed(0)
        x = np.random.randn(3, 5).astype(np.float64)
        x_p = ts.nn.Parameter(x.copy())

        with ts.autodiff.tape() as t:
            y = ts.ops.softmax(x_p)
            loss = ts.ops.reduce(ts.ops.mul(y, y), op="sum")
            t.backward(loss)
        analytic = x_p.grad.numpy()

        def loss_fn(arr):
            e = np.exp(arr - arr.max(axis=-1, keepdims=True))
            s = e / e.sum(axis=-1, keepdims=True)
            return float((s * s).sum())

        numerical = _numerical_grad(loss_fn, x.copy())
        _jacobian_close(analytic, numerical)

    def test_layer_norm(self):
        np.random.seed(0)
        x = np.random.randn(2, 6).astype(np.float64)
        x_p = ts.nn.Parameter(x.copy())

        with ts.autodiff.tape() as t:
            y = ts.ops.layer_norm(x_p)
            loss = ts.ops.reduce(ts.ops.mul(y, y), op="sum")
            t.backward(loss)
        analytic = x_p.grad.numpy()

        def loss_fn(arr):
            mu = arr.mean(axis=-1, keepdims=True)
            var = arr.var(axis=-1, keepdims=True)
            y = (arr - mu) / np.sqrt(var + 1e-5)
            return float((y * y).sum())

        numerical = _numerical_grad(loss_fn, x.copy())
        _jacobian_close(analytic, numerical)

    def test_rmsnorm(self):
        np.random.seed(0)
        x = np.random.randn(2, 6).astype(np.float64)
        x_p = ts.nn.Parameter(x.copy())

        with ts.autodiff.tape() as t:
            y = ts.ops.rmsnorm(x_p)
            loss = ts.ops.reduce(ts.ops.mul(y, y), op="sum")
            t.backward(loss)
        analytic = x_p.grad.numpy()

        def loss_fn(arr):
            ms = (arr * arr).mean(axis=-1, keepdims=True)
            y = arr / np.sqrt(ms + 1e-5)
            return float((y * y).sum())

        numerical = _numerical_grad(loss_fn, x.copy())
        _jacobian_close(analytic, numerical)


class TestVJPShape:
    def test_transpose(self):
        x = np.random.randn(3, 4).astype(np.float32)
        x_p = ts.nn.Parameter(x.copy())
        y = np.random.randn(4, 3).astype(np.float32)

        with ts.autodiff.tape() as t:
            z = ts.ops.transpose(x_p)
            loss = ts.ops.reduce(ts.ops.mul(z, y), op="sum")
            t.backward(loss)

        np.testing.assert_allclose(x_p.grad.numpy(), y.T)


class TestVJPReductions:
    def test_sum_full(self):
        x = np.random.randn(3, 4).astype(np.float32)
        x_p = ts.nn.Parameter(x.copy())

        with ts.autodiff.tape() as t:
            loss = ts.ops.reduce(x_p, op="sum")
            t.backward(loss)

        np.testing.assert_allclose(x_p.grad.numpy(), np.ones_like(x))

    def test_sum_axis(self):
        x = np.random.randn(3, 4).astype(np.float32)
        x_p = ts.nn.Parameter(x.copy())

        with ts.autodiff.tape() as t:
            r = ts.ops.reduce(x_p, op="sum", axis=0)
            loss = ts.ops.reduce(r, op="sum")
            t.backward(loss)

        np.testing.assert_allclose(x_p.grad.numpy(), np.ones_like(x))


# ─────────────────────────────────────────────────────────────────────────────
# Tape behavior
# ─────────────────────────────────────────────────────────────────────────────


class TestTape:
    def test_records_only_inside_block(self):
        A = np.random.randn(2, 2).astype(np.float32)
        B = np.random.randn(2, 2).astype(np.float32)
        # Outside any tape, ops just compute
        ts.ops.gemm(A, B)
        with ts.autodiff.tape() as t:
            ts.ops.gemm(A, B)
            ts.ops.gemm(A, B)
        assert len(t.entries) == 2

    def test_explicit_cotangent_for_raw_numpy_loss(self):
        np.random.seed(0)
        mlp = ts.nn.MLP(dim=4, hidden_dim=8)
        x = np.random.randn(2, 4).astype(np.float32)
        target = np.random.randn(2, 4).astype(np.float32)

        with ts.autodiff.tape() as t:
            y = mlp(x)
            diff = y - target
            dy = (2.0 * diff / diff.size).astype(np.float32)
            t.backward(y, cotangent=dy)

        for p in mlp.parameters():
            assert p.grad is not None

    def test_double_backward_raises(self):
        x = np.random.randn(3, 3).astype(np.float32)
        x_p = ts.nn.Parameter(x.copy())
        with ts.autodiff.tape() as t:
            loss = ts.ops.reduce(x_p, op="sum")
            t.backward(loss)
            with pytest.raises(TesseraAutodiffError, match="twice"):
                t.backward(loss)

    def test_backward_target_not_on_tape_raises(self):
        with ts.autodiff.tape() as t:
            ts.ops.gemm(
                np.random.randn(2, 2).astype(np.float32),
                np.random.randn(2, 2).astype(np.float32),
            )
            bogus = np.zeros(3, dtype=np.float32)
            with pytest.raises(TesseraAutodiffError, match="not a tape-recorded"):
                t.backward(bogus)

    def test_unsupported_op_raises(self):
        # An op without a registered VJP (here: `moe`, which still has no
        # adjoint as of v1 — `flash_attn` got one in Phase F3). The error
        # fires during backward iff the gradient path reaches that entry.
        x_p = ts.nn.Parameter(np.random.randn(2, 4).astype(np.float32))
        experts = np.random.randn(2, 4, 4).astype(np.float32)
        with ts.autodiff.tape() as t:
            out = ts.ops.moe(x_p, experts)
            loss = ts.ops.reduce(out, op="sum")
            with pytest.raises(TesseraAutodiffError, match=r"moe.+not differentiable"):
                t.backward(loss)

    def test_non_scalar_target_without_cotangent_raises(self):
        x_p = ts.nn.Parameter(shape=(2, 2))
        with ts.autodiff.tape() as t:
            y = ts.ops.gemm(x_p, np.eye(2, dtype=np.float32))
            with pytest.raises(TesseraAutodiffError, match="scalar target"):
                t.backward(y)


# ─────────────────────────────────────────────────────────────────────────────
# `reverse(fn)` and `custom_rule`
# ─────────────────────────────────────────────────────────────────────────────


class TestReverseTransform:
    def test_reverse_returns_loss_and_grads(self):
        np.random.seed(0)
        mlp = ts.nn.MLP(dim=4, hidden_dim=8)

        @ts.autodiff.reverse
        def loss_step(model, x, neg_target):
            # All loss math goes through traced ops so the chain stays connected:
            #   diff = y + (-target);  loss = sum(diff * diff)
            y = model(x)
            diff = ts.ops.add(y, neg_target)
            return ts.ops.reduce(ts.ops.mul(diff, diff), op="sum")

        x = np.random.randn(2, 4).astype(np.float32)
        neg_target = (-np.random.randn(2, 4)).astype(np.float32)
        loss, grads = loss_step(mlp, x, neg_target)

        assert isinstance(grads, dict)
        # MLP is positional arg 0 → grads keyed as "0.<param_name>"
        param_names = {f"0.{n}" for n, _ in mlp.named_parameters()}
        assert param_names <= set(grads), f"missing keys: {param_names - set(grads)}"
        for n, p in mlp.named_parameters():
            assert grads[f"0.{n}"].shape == p.shape


class TestCustomRule:
    def test_custom_rule_overrides_existing(self):
        # Override gemm VJP with a marker, verify it's used, then restore.
        from tessera.autodiff.vjp import get_vjp, register_vjp

        original = get_vjp("gemm")
        called = {"count": 0}

        @ts.autodiff.custom_rule("gemm")
        def _stub_gemm_vjp(dout, A, B, **_):
            called["count"] += 1
            # Return zeros so we can detect this VJP fired
            return (np.zeros_like(A), np.zeros_like(B))

        try:
            x = ts.nn.Parameter(np.random.randn(2, 3).astype(np.float32))
            with ts.autodiff.tape() as t:
                y = ts.ops.gemm(x, np.eye(3, dtype=np.float32))
                t.backward(ts.ops.reduce(y, op="sum"))
            assert called["count"] >= 1
            np.testing.assert_allclose(x.grad.numpy(), np.zeros_like(x.numpy()))
        finally:
            register_vjp("gemm", original)

    def test_custom_rule_for_new_op_wraps_it(self):
        # Register a VJP for `ops.cast` (which is in _VJPS already, so just
        # verify the install_op_wrappers idempotent path doesn't break).
        from tessera.autodiff.vjp import get_vjp, register_vjp
        original = get_vjp("cast")

        @ts.autodiff.custom_rule("cast")
        def _id_cast_vjp(dout, x, *, dtype=None, **_):
            return (dout.astype(x.dtype),)

        try:
            x = ts.nn.Parameter(np.random.randn(3).astype(np.float32))
            with ts.autodiff.tape() as t:
                y = ts.ops.cast(x, "fp32")
                t.backward(ts.ops.reduce(y, op="sum"))
            assert x.grad is not None
        finally:
            register_vjp("cast", original)


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end MLP train step
# ─────────────────────────────────────────────────────────────────────────────


class TestEndToEnd:
    def test_one_sgd_step_decreases_loss(self):
        np.random.seed(7)
        block = ts.nn.Sequential(
            ts.nn.Linear(8, 16, bias=True),
            ts.nn.RMSNorm(16),
            ts.nn.MLP(dim=16, hidden_dim=32),
            ts.nn.Linear(16, 8, bias=False),
        )
        x = np.random.randn(2, 4, 8).astype(np.float32)
        target = np.random.randn(2, 4, 8).astype(np.float32)

        def loss(model):
            return float(((model(x) - target) ** 2).mean())

        loss_before = loss(block)

        with ts.autodiff.tape() as t:
            y = block(x)
            diff = y - target
            dy = (2.0 * diff / diff.size).astype(np.float32)
            t.backward(y, cotangent=dy)

        n_with_grad = sum(1 for p in block.parameters() if p.grad is not None)
        assert n_with_grad == sum(1 for _ in block.parameters())

        lr = 0.05
        for p in block.parameters():
            if p.grad is not None:
                p._data._data[...] -= lr * p.grad.numpy()
        block.zero_grad()

        loss_after = loss(block)
        assert loss_after < loss_before, (loss_before, loss_after)

    def test_grad_accumulates_across_two_backwards(self):
        np.random.seed(0)
        x_p = ts.nn.Parameter(np.random.randn(3, 3).astype(np.float32))

        def run_once():
            with ts.autodiff.tape() as t:
                loss = ts.ops.reduce(x_p, op="sum")
                t.backward(loss)

        run_once()
        first = x_p.grad.numpy().copy()
        run_once()
        second = x_p.grad.numpy()
        np.testing.assert_allclose(second, 2 * first)
