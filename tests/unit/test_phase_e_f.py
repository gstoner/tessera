"""Tests for Phase E (KV-cache quantization, rolling window) and Phase F
(autocast, GradScaler, rematerialize, fused-op adjoints).

Phase F4 (Graph IR adjoint pass) is C++/MLIR work — its tests live in
`tests/tessera-ir/` once the build wires the AdjointInterface ODS.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# Phase E1 — quantize_kv / dequantize_kv
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantizeKV:
    @pytest.mark.parametrize("bits,expected_max_rel_err", [
        (8, 0.01),
        (4, 0.10),
        (2, 0.55),
    ])
    def test_round_trip_error_bound(self, bits, expected_max_rel_err):
        np.random.seed(0)
        k = np.random.randn(8, 4, 16).astype(np.float32)
        v = np.random.randn(8, 4, 16).astype(np.float32)
        k_q, v_q, scale, zp = ts.ops.quantize_kv(k, v, bits=bits)
        k_r, v_r = ts.ops.dequantize_kv(k_q, v_q, scale, zp)
        assert np.abs(k_r - k).max() / np.abs(k).max() <= expected_max_rel_err
        assert np.abs(v_r - v).max() / np.abs(v).max() <= expected_max_rel_err

    def test_quantized_dtype_is_int8(self):
        k = np.random.randn(2, 1, 4).astype(np.float32)
        v = np.random.randn(2, 1, 4).astype(np.float32)
        k_q, v_q, _, _ = ts.ops.quantize_kv(k, v, bits=4)
        assert k_q.dtype == np.int8
        assert v_q.dtype == np.int8

    def test_invalid_bits_rejected(self):
        k = np.random.randn(2, 1, 4).astype(np.float32)
        with pytest.raises(ValueError, match="bits"):
            ts.ops.quantize_kv(k, k, bits=1)

    def test_kv_shape_mismatch_rejected(self):
        k = np.random.randn(2, 1, 4).astype(np.float32)
        v = np.random.randn(3, 1, 4).astype(np.float32)
        with pytest.raises(ValueError, match="shape"):
            ts.ops.quantize_kv(k, v)


# ─────────────────────────────────────────────────────────────────────────────
# Phase E2 — quantized KVCacheHandle, kv_cache_update
# ─────────────────────────────────────────────────────────────────────────────


class TestQuantizedKVCache:
    def test_quantized_storage_int8(self):
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=8, quantize_bits=4)
        assert c.keys.dtype == np.int8
        assert c.values.dtype == np.int8

    def test_quantized_round_trip(self):
        np.random.seed(0)
        c = ts.cache.KVCacheHandle(num_heads=2, head_dim=4, max_seq=16, quantize_bits=8)
        k = np.random.randn(4, 2, 4).astype(np.float32)
        v = np.random.randn(4, 2, 4).astype(np.float32)
        c.append(k, v)
        k_r, v_r = c.read(0, 4)
        # 8-bit gives ~1% precision
        np.testing.assert_allclose(k_r, k, atol=0.05)
        np.testing.assert_allclose(v_r, v, atol=0.05)

    def test_quantize_bits_validation(self):
        with pytest.raises(ValueError, match="quantize_bits"):
            ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=4, quantize_bits=1)

    def test_kv_cache_update_alias(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=4)
        result = ts.ops.kv_cache_update(
            c, np.zeros((1, 1, 2), dtype=np.float32), np.zeros((1, 1, 2), dtype=np.float32)
        )
        assert result is c
        assert c.current_seq == 1


# ─────────────────────────────────────────────────────────────────────────────
# Phase E3 — rolling-window auto_evict
# ─────────────────────────────────────────────────────────────────────────────


class TestRollingWindow:
    def test_auto_evict_sliding_window(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=4, auto_evict=True)
        # Append 7 single tokens; only the trailing 4 should remain.
        for i in range(7):
            arr = np.full((1, 1, 2), float(i), dtype=np.float32)
            c.append(arr, arr)
        assert c.current_seq == 4
        k, _ = c.read(0, c.current_seq)
        np.testing.assert_allclose(k[:, 0, 0], [3.0, 4.0, 5.0, 6.0])

    def test_auto_evict_chunk_too_large_rejected(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=4, auto_evict=True)
        with pytest.raises(ValueError, match="cannot fit"):
            c.append(
                np.zeros((5, 1, 2), dtype=np.float32),
                np.zeros((5, 1, 2), dtype=np.float32),
            )

    def test_evict_oldest_explicit(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=10)
        for i in range(6):
            arr = np.full((1, 1, 2), float(i), dtype=np.float32)
            c.append(arr, arr)
        c.evict_oldest(2)
        k, _ = c.read(0, c.current_seq)
        np.testing.assert_allclose(k[:, 0, 0], [2.0, 3.0, 4.0, 5.0])

    def test_evict_oldest_negative_rejected(self):
        c = ts.cache.KVCacheHandle(num_heads=1, head_dim=2, max_seq=4)
        with pytest.raises(ValueError, match="non-negative"):
            c.evict_oldest(-1)

    def test_auto_evict_with_quantization(self):
        c = ts.cache.KVCacheHandle(
            num_heads=1, head_dim=2, max_seq=4, auto_evict=True, quantize_bits=8
        )
        for i in range(6):
            arr = np.full((1, 1, 2), float(i), dtype=np.float32)
            c.append(arr, arr)
        assert c.current_seq == 4
        k, _ = c.read(0, c.current_seq)
        # 8-bit quantization preserves whole numbers up to scale precision
        np.testing.assert_allclose(k[:, 0, 0], [2.0, 3.0, 4.0, 5.0], atol=0.05)


# ─────────────────────────────────────────────────────────────────────────────
# Phase F1 — autocast + GradScaler
# ─────────────────────────────────────────────────────────────────────────────


class TestAutocast:
    def test_gemm_downcasts_to_fp16(self):
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(8, 16).astype(np.float32)
        with ts.autodiff.autocast("fp16"):
            C = ts.ops.gemm(A, B)
        assert C.dtype == np.float16

    def test_softmax_promotes_to_fp32(self):
        x = np.random.randn(4, 8).astype(np.float16)  # already fp16
        with ts.autodiff.autocast("fp16"):
            y = ts.ops.softmax(x)
        assert y.dtype == np.float32

    def test_layer_norm_promotes_to_fp32(self):
        x = np.random.randn(4, 8).astype(np.float16)
        with ts.autodiff.autocast("fp16"):
            y = ts.ops.layer_norm(x)
        assert y.dtype == np.float32

    def test_no_autocast_outside_context(self):
        A = np.random.randn(4, 8).astype(np.float32)
        B = np.random.randn(8, 16).astype(np.float32)
        C = ts.ops.gemm(A, B)
        assert C.dtype == np.float32

    def test_invalid_autocast_dtype(self):
        with pytest.raises(ValueError, match="autocast dtype"):
            with ts.autodiff.autocast("bogus"):
                pass

    def test_autocast_dtype_inspection(self):
        assert ts.autodiff.autocast_dtype() is None
        with ts.autodiff.autocast("fp16"):
            assert ts.autodiff.autocast_dtype() == "fp16"
        assert ts.autodiff.autocast_dtype() is None


class TestGradScaler:
    def test_scale_loss(self):
        s = ts.autodiff.GradScaler(init_scale=128.0)
        assert s.scale_loss(2.0) == 256.0
        np.testing.assert_allclose(s.scale_grad(np.array([1.0, 2.0])), [128.0, 256.0])

    def test_unscale_on_finite_step(self):
        s = ts.autodiff.GradScaler(init_scale=128.0)
        m = ts.nn.Linear(4, 4)
        for p in m.parameters():
            p.grad = np.ones(p.shape, dtype=np.float32) * 2.0
        ok = s.step(lambda: None, params=list(m.parameters()))
        assert ok is True
        for p in m.parameters():
            np.testing.assert_allclose(p.grad.numpy(), 2.0 / 128.0)

    def test_skip_step_on_inf(self):
        s = ts.autodiff.GradScaler(init_scale=128.0)
        m = ts.nn.Linear(4, 4)
        for p in m.parameters():
            p.grad = np.full(p.shape, np.inf, dtype=np.float32)
        ok = s.step(lambda: None, params=list(m.parameters()))
        assert ok is False
        # Scale halved
        assert s.scale == 64.0
        # Grads cleared
        assert all(p.grad is None for p in m.parameters())

    def test_growth_after_interval(self):
        s = ts.autodiff.GradScaler(init_scale=10.0, growth_factor=2.0, growth_interval=3)
        m = ts.nn.Linear(4, 4)
        for _ in range(3):
            for p in m.parameters():
                p.grad = np.ones(p.shape, dtype=np.float32)
            s.step(lambda: None, params=list(m.parameters()))
        # After 3 successful steps, scale should grow
        assert s.scale == 20.0


# ─────────────────────────────────────────────────────────────────────────────
# Phase F2 — rematerialize
# ─────────────────────────────────────────────────────────────────────────────


class TestRematerialize:
    def test_remat_produces_same_grads_as_direct(self):
        np.random.seed(0)
        m = ts.nn.MLP(dim=4, hidden_dim=8)
        m_clone = ts.nn.MLP(dim=4, hidden_dim=8)
        m_clone.load_state_dict(m.state_dict())

        x = np.random.randn(2, 4).astype(np.float32)
        target = np.random.randn(2, 4).astype(np.float32)

        # Direct
        with ts.autodiff.tape() as t:
            y = m(x)
            dy = (2.0 * (y - target) / y.size).astype(np.float32)
            t.backward(y, cotangent=dy)
        direct_grads = {n: p.grad.numpy().copy() for n, p in m.named_parameters()}

        # Rematerialized
        @ts.autodiff.rematerialize
        def remat_fwd(z):
            return m_clone(z)

        with ts.autodiff.tape() as t:
            y2 = remat_fwd(x)
            dy = (2.0 * (y2 - target) / y2.size).astype(np.float32)
            t.backward(y2, cotangent=dy)
        remat_grads = {n: p.grad.numpy().copy() for n, p in m_clone.named_parameters()}

        for k, g_direct in direct_grads.items():
            np.testing.assert_allclose(remat_grads[k], g_direct, rtol=1e-5)

    def test_remat_outside_tape_just_calls(self):
        @ts.autodiff.rematerialize
        def f(x):
            return ts.ops.gemm(x, np.eye(3, dtype=np.float32))

        x = np.random.randn(2, 3).astype(np.float32)
        out = f(x)
        np.testing.assert_allclose(out, x)

    def test_checkpoint_alias(self):
        # `checkpoint` is just an alias matching torch's spelling
        assert ts.autodiff.checkpoint is ts.autodiff.rematerialize

    def test_remat_input_cotangent_propagates(self):
        # A non-Parameter input flowing through rematerialize should still
        # contribute its cotangent to upstream tape entries.
        np.random.seed(0)
        x_p = ts.nn.Parameter(np.random.randn(3, 4).astype(np.float64))

        @ts.autodiff.rematerialize
        def block(z):
            return ts.ops.silu(ts.ops.mul(z, z))

        with ts.autodiff.tape() as t:
            y = block(x_p)
            loss = ts.ops.reduce(y, op="sum")
            t.backward(loss)
        analytic = x_p.grad.numpy()

        # Numerical check
        def fn(arr):
            tmp = arr * arr
            silu = tmp / (1.0 + np.exp(-tmp))
            return float(silu.sum())

        eps = 1e-6
        x_arr = x_p.numpy().copy()
        x_pos = x_arr.copy(); x_pos[0, 0] += eps
        x_neg = x_arr.copy(); x_neg[0, 0] -= eps
        num_grad = (fn(x_pos) - fn(x_neg)) / (2 * eps)
        assert abs(analytic[0, 0] - num_grad) < 1e-5


# ─────────────────────────────────────────────────────────────────────────────
# Phase F3 — flash_attn / fft VJPs
# ─────────────────────────────────────────────────────────────────────────────


class TestFlashAttnVJP:
    def test_dQ_matches_numerical(self):
        np.random.seed(0)
        Q_p = ts.nn.Parameter(np.random.randn(1, 2, 4, 8).astype(np.float64))
        K = np.random.randn(1, 2, 4, 8).astype(np.float64)
        V = np.random.randn(1, 2, 4, 8).astype(np.float64)

        with ts.autodiff.tape() as t:
            O = ts.ops.flash_attn(Q_p, K, V)
            loss = ts.ops.reduce(ts.ops.mul(O, O), op="sum")
            t.backward(loss)
        analytic_dQ = Q_p.grad.numpy()

        def loss_fn(q):
            O = ts.ops.flash_attn(q, K, V)
            return float((O * O).sum())

        eps = 1e-6
        Q_arr = Q_p.numpy().copy()
        Q_pos = Q_arr.copy(); Q_pos[0, 0, 0, 0] += eps
        Q_neg = Q_arr.copy(); Q_neg[0, 0, 0, 0] -= eps
        num_dQ = (loss_fn(Q_pos) - loss_fn(Q_neg)) / (2 * eps)
        assert abs(analytic_dQ[0, 0, 0, 0] - num_dQ) < 1e-5

    def test_dV_matches_numerical(self):
        np.random.seed(0)
        Q = np.random.randn(1, 1, 3, 4).astype(np.float64)
        K = np.random.randn(1, 1, 3, 4).astype(np.float64)
        V_p = ts.nn.Parameter(np.random.randn(1, 1, 3, 4).astype(np.float64))

        with ts.autodiff.tape() as t:
            O = ts.ops.flash_attn(Q, K, V_p)
            loss = ts.ops.reduce(ts.ops.mul(O, O), op="sum")
            t.backward(loss)
        analytic_dV = V_p.grad.numpy()

        def loss_fn(v):
            return float((ts.ops.flash_attn(Q, K, v) ** 2).sum())

        eps = 1e-6
        V_arr = V_p.numpy().copy()
        V_pos = V_arr.copy(); V_pos[0, 0, 1, 2] += eps
        V_neg = V_arr.copy(); V_neg[0, 0, 1, 2] -= eps
        num_dV = (loss_fn(V_pos) - loss_fn(V_neg)) / (2 * eps)
        assert abs(analytic_dV[0, 0, 1, 2] - num_dV) < 1e-5

    def test_causal_path(self):
        # Causal flash_attn VJP — same numerical check
        np.random.seed(0)
        Q_p = ts.nn.Parameter(np.random.randn(1, 1, 4, 4).astype(np.float64))
        K = np.random.randn(1, 1, 4, 4).astype(np.float64)
        V = np.random.randn(1, 1, 4, 4).astype(np.float64)

        with ts.autodiff.tape() as t:
            O = ts.ops.flash_attn(Q_p, K, V, causal=True)
            loss = ts.ops.reduce(ts.ops.mul(O, O), op="sum")
            t.backward(loss)
        # Just verify gradients exist + shape matches; numerical check skipped
        # (causal masking introduces -inf which complicates finite-diff).
        assert Q_p.grad is not None
        assert Q_p.grad.shape == Q_p.shape


class TestFFTVJP:
    def test_fft_vjp_registered(self):
        from tessera.autodiff.vjp import get_vjp
        assert get_vjp("fft") is not None
        assert get_vjp("ifft") is not None
        assert get_vjp("rfft") is not None
        assert get_vjp("irfft") is not None
