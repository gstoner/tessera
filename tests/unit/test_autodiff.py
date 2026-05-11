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


class TestSiluMul:
    """Stage 2a — fused silu-and-multiply primitive that backs the SwiGLU
    decomposition. Forward correctness + per-input numerical-Jacobian VJP
    check + ops.swiglu chain decomposition on the tape."""

    def test_forward_matches_reference(self):
        np.random.seed(0)
        a = np.random.randn(3, 5).astype(np.float64)
        b = np.random.randn(3, 5).astype(np.float64)
        out = ts.ops.silu_mul(a, b)
        expected = (a / (1.0 + np.exp(-a))) * b
        np.testing.assert_allclose(out, expected)

    def test_vjp_matches_numerical_jacobian(self):
        np.random.seed(0)
        a = np.random.randn(3, 5).astype(np.float64)
        b = np.random.randn(3, 5).astype(np.float64)
        a_p = ts.nn.Parameter(a.copy())
        b_p = ts.nn.Parameter(b.copy())

        with ts.autodiff.tape() as t:
            y = ts.ops.silu_mul(a_p, b_p)
            loss = ts.ops.reduce(y, op="sum")
            t.backward(loss)

        def loss_a(arr):
            return float(((arr / (1.0 + np.exp(-arr))) * b).sum())

        def loss_b(arr):
            return float(((a / (1.0 + np.exp(-a))) * arr).sum())

        _jacobian_close(a_p.grad.numpy(), _numerical_grad(loss_a, a.copy()))
        _jacobian_close(b_p.grad.numpy(), _numerical_grad(loss_b, b.copy()))

    def test_swiglu_decomposes_on_tape(self):
        """ops.swiglu must record matmul → matmul → silu_mul → matmul on the
        tape, not a single 'swiglu' entry — that's what lets the Schedule IR
        fusion recognizer match the chain."""
        np.random.seed(0)
        x = np.random.randn(2, 4).astype(np.float64)
        Wg = np.random.randn(4, 8).astype(np.float64)
        Wu = np.random.randn(4, 8).astype(np.float64)
        Wd = np.random.randn(8, 4).astype(np.float64)

        with ts.autodiff.tape() as t:
            ts.ops.swiglu(x, Wg, Wu, Wd)

        op_names = [e.op for e in t.entries]
        assert "swiglu" not in op_names, (
            f"ops.swiglu must decompose, not be a single tape entry: {op_names}"
        )
        # Three matmuls (gate, up, down) and one silu_mul, in that order.
        assert op_names == ["gemm", "gemm", "silu_mul", "gemm"], op_names

    def test_swiglu_end_to_end_bptt(self):
        """All three weight matrices receive correct gradients through the
        decomposed chain."""
        np.random.seed(0)
        x = np.random.randn(2, 4).astype(np.float64)
        Wg = np.random.randn(4, 8).astype(np.float64)
        Wu = np.random.randn(4, 8).astype(np.float64)
        Wd = np.random.randn(8, 4).astype(np.float64)

        Wg_p = ts.nn.Parameter(Wg.copy())
        Wu_p = ts.nn.Parameter(Wu.copy())
        Wd_p = ts.nn.Parameter(Wd.copy())

        with ts.autodiff.tape() as t:
            y = ts.ops.swiglu(x, Wg_p, Wu_p, Wd_p)
            loss = ts.ops.reduce(y, op="sum")
            t.backward(loss)

        def loss_at(W, idx):
            Ws = [Wg.copy(), Wu.copy(), Wd.copy()]
            Ws[idx] = W
            gate = x @ Ws[0]
            up = x @ Ws[1]
            h = (gate / (1.0 + np.exp(-gate))) * up
            return float((h @ Ws[2]).sum())

        for W_p, idx in [(Wg_p, 0), (Wu_p, 1), (Wd_p, 2)]:
            ng = _numerical_grad(lambda W: loss_at(W, idx), W_p._data._data.copy())
            _jacobian_close(W_p.grad.numpy(), ng)


class TestTheme9UtilityOps:
    """Theme 9 — utility tensor ops (gather, clip, masked_fill, arange).
    Forward correctness + numerical-Jacobian VJPs for the differentiable
    ones. arange is non-differentiable (no tensor input)."""

    def test_arange_single_arg(self):
        np.testing.assert_array_equal(ts.ops.arange(5), np.arange(5, dtype=np.float32))

    def test_arange_with_start_stop_step(self):
        np.testing.assert_array_equal(
            ts.ops.arange(2, 8, 2), np.arange(2, 8, 2, dtype=np.float32)
        )

    def test_arange_int_dtype(self):
        out = ts.ops.arange(0, 5, dtype="i64")
        assert out.dtype == np.int64
        np.testing.assert_array_equal(out, np.arange(5, dtype=np.int64))

    def test_gather_axis_0(self):
        x = np.arange(12, dtype=np.float64).reshape(3, 4)
        idx = np.array([0, 2, 1, 0])
        np.testing.assert_array_equal(ts.ops.gather(x, idx, axis=0), x[idx])

    def test_gather_axis_1(self):
        x = np.arange(12, dtype=np.float64).reshape(3, 4)
        idx = np.array([2, 0, 3])
        np.testing.assert_array_equal(ts.ops.gather(x, idx, axis=1), x[:, idx])

    def test_gather_vjp_repeated_indices(self):
        """Repeated indices in `idx` accumulate gradients via np.add.at —
        the canonical gather adjoint."""
        np.random.seed(0)
        xv = np.random.randn(5, 3).astype(np.float64)
        xp = ts.nn.Parameter(xv.copy())
        idx = np.array([2, 0, 2, 4])  # idx 2 appears twice → grad accumulates
        with ts.autodiff.tape() as t:
            g = ts.ops.gather(xp, idx, axis=0)
            loss = ts.ops.reduce(ts.ops.mul(g, g), op="sum")
            t.backward(loss)
        ng = _numerical_grad(
            lambda a: float(((np.take(a, idx, axis=0)) ** 2).sum()), xv.copy()
        )
        _jacobian_close(xp.grad.numpy(), ng)

    def test_clip_forward_both_bounds(self):
        y = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        np.testing.assert_array_equal(
            ts.ops.clip(y, min_val=-1.0, max_val=1.0), [-1, -1, 0, 1, 1]
        )

    def test_clip_forward_single_bound(self):
        y = np.array([-2.0, 0.0, 2.0])
        np.testing.assert_array_equal(
            ts.ops.clip(y, max_val=0.5), [-2.0, 0.0, 0.5]
        )
        np.testing.assert_array_equal(
            ts.ops.clip(y, min_val=-0.5), [-0.5, 0.0, 2.0]
        )

    def test_clip_vjp_strict_bounds(self):
        """Gradient is 1 strictly inside (min, max), 0 elsewhere — matches
        PyTorch's `torch.clamp` and the central-difference Jacobian."""
        np.random.seed(0)
        xv = np.random.randn(6).astype(np.float64) * 2
        xp = ts.nn.Parameter(xv.copy())
        with ts.autodiff.tape() as t:
            c = ts.ops.clip(xp, min_val=-1.0, max_val=1.0)
            loss = ts.ops.reduce(ts.ops.mul(c, c), op="sum")
            t.backward(loss)
        ng = _numerical_grad(
            lambda a: float((np.clip(a, -1.0, 1.0) ** 2).sum()), xv.copy()
        )
        _jacobian_close(xp.grad.numpy(), ng)

    def test_masked_fill_forward(self):
        z = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        m = np.array([[True, False, True], [False, True, False]])
        np.testing.assert_array_equal(
            ts.ops.masked_fill(z, m, value=0.0),
            [[0.0, 2.0, 0.0], [4.0, 0.0, 6.0]],
        )

    def test_masked_fill_broadcast(self):
        z = np.zeros((2, 3))
        m = np.array([True, False, True])  # broadcast across rows
        out = ts.ops.masked_fill(z, m, value=-1.0)
        np.testing.assert_array_equal(out, [[-1.0, 0.0, -1.0], [-1.0, 0.0, -1.0]])

    def test_masked_fill_vjp(self):
        np.random.seed(0)
        xv = np.random.randn(2, 4).astype(np.float64)
        xp = ts.nn.Parameter(xv.copy())
        mask = np.array([[True, False, True, False], [False, True, False, True]])
        with ts.autodiff.tape() as t:
            mf = ts.ops.masked_fill(xp, mask, value=0.0)
            loss = ts.ops.reduce(ts.ops.mul(mf, mf), op="sum")
            t.backward(loss)
        ng = _numerical_grad(
            lambda a: float((np.where(mask, 0.0, a) ** 2).sum()), xv.copy()
        )
        _jacobian_close(xp.grad.numpy(), ng)


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
        # An op without a registered VJP. The error fires during backward
        # iff the gradient path reaches that entry.
        # History of sentinels (each migrated as its VJP landed):
        #   - `moe`        v1 → F3-moe shipped 2026-05-09
        #   - `cholesky`   v1 → long-tail closure 2026-05-10 (Murray)
        # Current sentinel: `cumprod` — gradient through `cumprod` requires
        # a non-trivial reverse-cumprod construction still on the
        # `vjp = planned` list.
        x = np.array([1.0, 2.0, 3.0])
        x_p = ts.nn.Parameter(x.copy())
        with ts.autodiff.tape() as t:
            out = ts.ops.cumprod(x_p)
            loss = ts.ops.reduce(out, op="sum")
            with pytest.raises(
                TesseraAutodiffError, match=r"cumprod.+not differentiable",
            ):
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
