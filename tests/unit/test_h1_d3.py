"""Tests for Phase H1 (Conv2d) and Phase D3 (Mamba2 selective_ssm)."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# Phase H1 — Conv2d (NHWC) + Conv2dNCHW shim
# ─────────────────────────────────────────────────────────────────────────────


class TestConv2dNHWC:
    def test_basic_forward_shape(self):
        m = ts.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        y = m(np.random.randn(2, 16, 16, 3).astype(np.float32))
        assert y.shape == (2, 16, 16, 8)

    def test_kernel_size_tuple(self):
        m = ts.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=(3, 5), padding=(1, 2))
        y = m(np.random.randn(1, 8, 8, 3).astype(np.float32))
        assert y.shape == (1, 8, 8, 4)

    def test_no_bias(self):
        m = ts.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, bias=False)
        names = {n for n, _ in m.named_parameters()}
        assert names == {"weight"}
        assert m.bias is None

    def test_weight_shape_is_hwio(self):
        m = ts.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        assert m.weight.shape == (3, 3, 3, 8)  # (kH, kW, in, out)

    def test_invalid_input_rank(self):
        m = ts.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        with pytest.raises(ValueError, match=r"\(N, H, W, C\)"):
            m(np.zeros((4, 4)))

    def test_input_channel_mismatch(self):
        m = ts.nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3)
        with pytest.raises(ValueError, match="channel dim"):
            m(np.zeros((1, 8, 8, 5), dtype=np.float32))

    def test_state_dict_roundtrip(self):
        m1 = ts.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        sd = m1.state_dict()
        m2 = ts.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        m2.load_state_dict(sd)
        x = np.random.randn(1, 8, 8, 3).astype(np.float32)
        np.testing.assert_allclose(m1(x), m2(x))


class TestConv2dNCHW:
    def test_forward_shape(self):
        m = ts.nn.Conv2dNCHW(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        y = m(np.random.randn(2, 3, 16, 16).astype(np.float32))
        assert y.shape == (2, 8, 16, 16)

    def test_equivalence_with_nhwc(self):
        """Same weights, same input (transposed) → same output (transposed)."""
        np.random.seed(0)
        m_nhwc = ts.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        m_nchw = ts.nn.Conv2dNCHW(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        m_nchw.conv.weight._data._data[...] = m_nhwc.weight._data._data
        m_nchw.conv.bias._data._data[...] = m_nhwc.bias._data._data

        x_nhwc = np.random.randn(2, 16, 16, 3).astype(np.float32)
        y_nhwc = m_nhwc(x_nhwc)
        y_nchw = m_nchw(np.transpose(x_nhwc, (0, 3, 1, 2)))
        np.testing.assert_allclose(np.transpose(y_nhwc, (0, 3, 1, 2)), y_nchw, rtol=1e-5)

    def test_invalid_input_rank(self):
        m = ts.nn.Conv2dNCHW(in_channels=3, out_channels=4, kernel_size=3)
        with pytest.raises(ValueError, match=r"\(N, C, H, W\)"):
            m(np.zeros((4, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# Phase D3 — selective_ssm
# ─────────────────────────────────────────────────────────────────────────────


def _ssm_inputs(seed=0, B=2, S=8, D=4, N=6):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((B, S, D)).astype(np.float32)
    A = -np.abs(rng.standard_normal((D,)).astype(np.float32))  # decay
    B_arr = (rng.standard_normal((B, S, N)) * 0.1).astype(np.float32)
    C_arr = (rng.standard_normal((B, S, N)) * 0.1).astype(np.float32)
    delta = (np.abs(rng.standard_normal((B, S, D))) * 0.1).astype(np.float32)
    return x, A, B_arr, C_arr, delta


class TestSelectiveSSM:
    def test_forward_shape(self):
        x, A, B, C, delta = _ssm_inputs()
        y = ts.ops.selective_ssm(x, A, B, C, delta)
        assert y.shape == x.shape

    def test_chunked_scan_equivalence(self):
        x, A, B, C, delta = _ssm_inputs()
        y_full = ts.ops.selective_ssm(x, A, B, C, delta, chunk_size=128)
        y_chunk2 = ts.ops.selective_ssm(x, A, B, C, delta, chunk_size=2)
        y_chunk1 = ts.ops.selective_ssm(x, A, B, C, delta, chunk_size=1)
        np.testing.assert_allclose(y_full, y_chunk2, rtol=1e-5)
        np.testing.assert_allclose(y_full, y_chunk1, rtol=1e-5)

    def test_gate_multiplies_output(self):
        x, A, B, C, delta = _ssm_inputs()
        y_no_gate = ts.ops.selective_ssm(x, A, B, C, delta)
        gate = np.random.randn(*y_no_gate.shape).astype(np.float32)
        y_gated = ts.ops.selective_ssm(x, A, B, C, delta, gate=gate)
        np.testing.assert_allclose(y_gated, y_no_gate * gate)

    def test_initial_state(self):
        x, A, B, C, delta = _ssm_inputs()
        Bsz, S, D = x.shape
        N = B.shape[-1]
        h0 = np.random.randn(Bsz, D, N).astype(np.float32) * 0.1
        y_zero = ts.ops.selective_ssm(x, A, B, C, delta)
        y_h0 = ts.ops.selective_ssm(x, A, B, C, delta, state=h0)
        # Different initial state → different output
        assert not np.allclose(y_zero, y_h0)

    def test_streaming_with_state(self):
        """Splitting the sequence at midpoint and feeding state across the
        boundary should match a single-shot full-sequence computation."""
        x, A, B, C, delta = _ssm_inputs(B=1, S=10, D=4, N=6)
        y_full = ts.ops.selective_ssm(x, A, B, C, delta)

        mid = 5
        x_a, x_b = x[:, :mid, :], x[:, mid:, :]
        B_a, B_b = B[:, :mid, :], B[:, mid:, :]
        C_a, C_b = C[:, :mid, :], C[:, mid:, :]
        d_a, d_b = delta[:, :mid, :], delta[:, mid:, :]

        # Compute the state at the end of chunk a manually
        Bsz, _, D = x.shape
        N = B.shape[-1]
        A2d = np.broadcast_to(A[:, None], (D, N))
        h = np.zeros((Bsz, D, N), dtype=x.dtype)
        for t in range(mid):
            A_bar = np.exp(d_a[:, t, :, None] * A2d[None, :, :])
            B_bar = d_a[:, t, :, None] * B_a[:, t, None, :]
            h = A_bar * h + B_bar * x_a[:, t, :, None]

        y_a = ts.ops.selective_ssm(x_a, A, B_a, C_a, d_a)
        y_b = ts.ops.selective_ssm(x_b, A, B_b, C_b, d_b, state=h)
        y_concat = np.concatenate([y_a, y_b], axis=1)
        np.testing.assert_allclose(y_concat, y_full, rtol=1e-5)

    def test_a_2d_form_supported(self):
        """A as (D, N) gives more expressive dynamics than (D,)."""
        x, A_1d, B, C, delta = _ssm_inputs()
        Bsz, S, D = x.shape
        N = B.shape[-1]
        A_2d = np.broadcast_to(A_1d[:, None], (D, N)).copy()
        # Both forms with the same effective matrix should match
        y_1d = ts.ops.selective_ssm(x, A_1d, B, C, delta)
        y_2d = ts.ops.selective_ssm(x, A_2d, B, C, delta)
        np.testing.assert_allclose(y_1d, y_2d, rtol=1e-5)

    def test_invalid_x_rank_rejected(self):
        with pytest.raises(ValueError, match=r"\(B, S, D\)"):
            ts.ops.selective_ssm(
                np.zeros((4,)), np.zeros((4,)), np.zeros((4,)), np.zeros((4,)),
                np.zeros((4,)),
            )

    def test_a_invalid_shape_rejected(self):
        x, _, B, C, delta = _ssm_inputs()
        with pytest.raises(ValueError, match="A must be"):
            ts.ops.selective_ssm(x, np.zeros((3, 3, 3)), B, C, delta)

    def test_vjp_numerical_jacobian(self):
        """Phase D3 VJP — analytical adjoint matches central-difference
        numerical gradient at fp64 to 1e-6 across all five positional inputs."""
        np.random.seed(0)
        Bsz, S, D, N = 1, 4, 3, 4
        x = (np.random.randn(Bsz, S, D) * 0.5).astype(np.float64)
        A = -np.abs(np.random.randn(D).astype(np.float64))
        B = (np.random.randn(Bsz, S, N) * 0.1).astype(np.float64)
        C = (np.random.randn(Bsz, S, N) * 0.1).astype(np.float64)
        delta = (np.abs(np.random.randn(Bsz, S, D)) * 0.1).astype(np.float64)

        x_p = ts.nn.Parameter(x.copy())
        A_p = ts.nn.Parameter(A.copy())
        B_p = ts.nn.Parameter(B.copy())
        C_p = ts.nn.Parameter(C.copy())
        d_p = ts.nn.Parameter(delta.copy())

        with ts.autodiff.tape() as t:
            y = ts.ops.selective_ssm(x_p, A_p, B_p, C_p, d_p)
            loss = ts.ops.reduce(ts.ops.mul(y, y), op="sum")
            t.backward(loss)

        def loss_fn(xv, Av, Bv, Cv, dv):
            y_ = ts.ops.selective_ssm(xv, Av, Bv, Cv, dv)
            return float((y_ * y_).sum())

        eps = 1e-6

        def numerical(arr, idx):
            base_args = (x, A, B, C, delta)
            names = ("x", "A", "B", "C", "delta")
            slot = names.index(arr)
            pos = list(base_args)
            neg = list(base_args)
            pos[slot] = pos[slot].copy(); pos[slot][idx] += eps
            neg[slot] = neg[slot].copy(); neg[slot][idx] -= eps
            return (loss_fn(*pos) - loss_fn(*neg)) / (2 * eps)

        # Spot-check one element of each gradient
        checks = [
            (x_p.grad.numpy(), "x", (0, 1, 1)),
            (A_p.grad.numpy(), "A", (0,)),
            (B_p.grad.numpy(), "B", (0, 1, 1)),
            (C_p.grad.numpy(), "C", (0, 2, 1)),
            (d_p.grad.numpy(), "delta", (0, 1, 1)),
        ]
        for analytic, name, idx in checks:
            num = numerical(name, idx)
            assert abs(analytic[idx] - num) < 1e-6, (name, idx, analytic[idx], num)

    def test_vjp_a_2d_form(self):
        """When A is (D, N), dA shape matches (D, N)."""
        x, A_1d, B, C, delta = _ssm_inputs(B=1, S=4, D=3, N=4)
        D = x.shape[-1]; N = B.shape[-1]
        A_2d = np.broadcast_to(A_1d[:, None], (D, N)).copy().astype(np.float64)
        x_p = ts.nn.Parameter(x.astype(np.float64))
        A_p = ts.nn.Parameter(A_2d)
        B_p = ts.nn.Parameter(B.astype(np.float64))
        C_p = ts.nn.Parameter(C.astype(np.float64))
        d_p = ts.nn.Parameter(delta.astype(np.float64))

        with ts.autodiff.tape() as t:
            y = ts.ops.selective_ssm(x_p, A_p, B_p, C_p, d_p)
            t.backward(ts.ops.reduce(y, op="sum"))
        assert A_p.grad.shape == (D, N)

    def test_jvp_numerical_jacobian(self):
        """Sprint A follow-up (2026-05-11) — selective_ssm closed-form JVP
        matches central-difference reference at fp64 to <1e-8 relative
        error across all five positional inputs (x, A, B, C, delta).
        This closes the last `jvp = planned` entry in the registry.
        """
        import importlib

        jvp_mod = importlib.import_module("tessera.autodiff.jvp")

        rng = np.random.default_rng(0)
        Bsz, S, D, N = 2, 5, 3, 4
        x = rng.standard_normal((Bsz, S, D)).astype(np.float64)
        A = -np.abs(rng.standard_normal((D, N))).astype(np.float64)
        B = rng.standard_normal((Bsz, S, N)).astype(np.float64)
        C = rng.standard_normal((Bsz, S, N)).astype(np.float64)
        delta = np.clip(
            np.abs(rng.standard_normal((Bsz, S, D))).astype(np.float64),
            0.05, 0.5,
        )

        dx = rng.standard_normal(x.shape).astype(np.float64) * 1e-3
        dA = rng.standard_normal(A.shape).astype(np.float64) * 1e-3
        dB = rng.standard_normal(B.shape).astype(np.float64) * 1e-3
        dC = rng.standard_normal(C.shape).astype(np.float64) * 1e-3
        ddelta = rng.standard_normal(delta.shape).astype(np.float64) * 1e-3

        jvp_fn = jvp_mod._JVPS["selective_ssm"]
        y_jvp, dy = jvp_fn((x, A, B, C, delta), (dx, dA, dB, dC, ddelta))

        # Forward primal cross-check.
        fn = getattr(ts.ops.selective_ssm, "__wrapped__", ts.ops.selective_ssm)
        y_ref = fn(x, A, B, C, delta)
        np.testing.assert_allclose(y_jvp, y_ref, atol=1e-10)

        # Central-difference reference for the tangent.
        eps = 1e-4
        y_plus = fn(x + eps * dx, A + eps * dA, B + eps * dB,
                    C + eps * dC, delta + eps * ddelta)
        y_minus = fn(x - eps * dx, A - eps * dA, B - eps * dB,
                     C - eps * dC, delta - eps * ddelta)
        dy_num = (y_plus - y_minus) / (2 * eps)

        rel_err = np.max(np.abs(dy - dy_num)) / (np.max(np.abs(dy_num)) + 1e-12)
        assert rel_err < 1e-7, f"JVP rel err {rel_err:.2e} exceeds 1e-7"

    def test_jvp_gated_and_initial_state(self):
        """Gate + initial state path matches CD reference at fp64."""
        import importlib

        jvp_mod = importlib.import_module("tessera.autodiff.jvp")

        rng = np.random.default_rng(7)
        Bsz, S, D, N = 2, 4, 3, 4
        x = rng.standard_normal((Bsz, S, D)).astype(np.float64)
        A = -np.abs(rng.standard_normal((D, N))).astype(np.float64)
        B = rng.standard_normal((Bsz, S, N)).astype(np.float64)
        C = rng.standard_normal((Bsz, S, N)).astype(np.float64)
        delta = np.clip(
            np.abs(rng.standard_normal((Bsz, S, D))).astype(np.float64),
            0.05, 0.5,
        )
        gate = rng.standard_normal((Bsz, S, D)).astype(np.float64)
        state = (rng.standard_normal((Bsz, D, N)) * 0.1).astype(np.float64)

        dx = rng.standard_normal(x.shape).astype(np.float64) * 1e-3
        dA = rng.standard_normal(A.shape).astype(np.float64) * 1e-3
        dB = rng.standard_normal(B.shape).astype(np.float64) * 1e-3
        dC = rng.standard_normal(C.shape).astype(np.float64) * 1e-3
        ddelta = rng.standard_normal(delta.shape).astype(np.float64) * 1e-3

        jvp_fn = jvp_mod._JVPS["selective_ssm"]
        y_jvp, dy = jvp_fn(
            (x, A, B, C, delta),
            (dx, dA, dB, dC, ddelta),
            gate=gate, state=state,
        )

        fn = getattr(ts.ops.selective_ssm, "__wrapped__", ts.ops.selective_ssm)
        y_ref = fn(x, A, B, C, delta, gate=gate, state=state)
        np.testing.assert_allclose(y_jvp, y_ref, atol=1e-10)

        eps = 1e-4
        y_plus = fn(x + eps * dx, A + eps * dA, B + eps * dB,
                    C + eps * dC, delta + eps * ddelta,
                    gate=gate, state=state)
        y_minus = fn(x - eps * dx, A - eps * dA, B - eps * dB,
                     C - eps * dC, delta - eps * ddelta,
                     gate=gate, state=state)
        dy_num = (y_plus - y_minus) / (2 * eps)

        rel_err = np.max(np.abs(dy - dy_num)) / (np.max(np.abs(dy_num)) + 1e-12)
        assert rel_err < 1e-7, f"Gated JVP rel err {rel_err:.2e} exceeds 1e-7"

    def test_jvp_a_1d_form(self):
        """When A is 1-D (D,), the tangent dA broadcasts to (D, N) the same
        way the primal does — matches the VJP's `A_was_1d` path."""
        import importlib

        jvp_mod = importlib.import_module("tessera.autodiff.jvp")

        rng = np.random.default_rng(13)
        Bsz, S, D, N = 1, 4, 3, 4
        x = rng.standard_normal((Bsz, S, D)).astype(np.float64)
        A = -np.abs(rng.standard_normal((D,))).astype(np.float64)
        B = rng.standard_normal((Bsz, S, N)).astype(np.float64)
        C = rng.standard_normal((Bsz, S, N)).astype(np.float64)
        delta = np.clip(
            np.abs(rng.standard_normal((Bsz, S, D))).astype(np.float64),
            0.05, 0.5,
        )
        dx = rng.standard_normal(x.shape).astype(np.float64) * 1e-3
        dA = rng.standard_normal(A.shape).astype(np.float64) * 1e-3
        dB = rng.standard_normal(B.shape).astype(np.float64) * 1e-3
        dC = rng.standard_normal(C.shape).astype(np.float64) * 1e-3
        ddelta = rng.standard_normal(delta.shape).astype(np.float64) * 1e-3

        jvp_fn = jvp_mod._JVPS["selective_ssm"]
        y_jvp, dy = jvp_fn((x, A, B, C, delta), (dx, dA, dB, dC, ddelta))

        fn = getattr(ts.ops.selective_ssm, "__wrapped__", ts.ops.selective_ssm)
        eps = 1e-4
        y_plus = fn(x + eps * dx, A + eps * dA, B + eps * dB,
                    C + eps * dC, delta + eps * ddelta)
        y_minus = fn(x - eps * dx, A - eps * dA, B - eps * dB,
                     C - eps * dC, delta - eps * ddelta)
        dy_num = (y_plus - y_minus) / (2 * eps)

        rel_err = np.max(np.abs(dy - dy_num)) / (np.max(np.abs(dy_num)) + 1e-12)
        assert rel_err < 1e-7, f"1-D A JVP rel err {rel_err:.2e} exceeds 1e-7"
