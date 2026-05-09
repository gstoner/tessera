"""Tests for Phase C + D streaming kernels.

Covers:
  * `tessera.nn.BatchNorm1d` (Phase C1) — train/eval modes, running stats
  * `tessera.nn.KVCache` (Phase C2) — Module wrapper around KVCacheHandle
  * `tessera.ops.depthwise_conv1d` (Phase D1) — non-causal/causal/streaming + VJP
  * `tessera.ops.online_softmax` (Phase D2) — single-chunk + streaming via state helper
  * `tessera.nn.DynamicDepthwiseConv1d` (Phase D4)
"""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


# ─────────────────────────────────────────────────────────────────────────────
# C1 — BatchNorm1d
# ─────────────────────────────────────────────────────────────────────────────


class TestBatchNorm1d:
    def test_construction_registers_params_and_buffers(self):
        bn = ts.nn.BatchNorm1d(num_features=4)
        names_p = sorted(n for n, _ in bn.named_parameters())
        names_b = sorted(n for n, _ in bn.named_buffers())
        assert names_p == ["bias", "weight"]
        assert names_b == ["num_batches_tracked", "running_mean", "running_var"]

    def test_train_normalizes_per_channel(self):
        bn = ts.nn.BatchNorm1d(num_features=4)
        x = np.random.randn(8, 4).astype(np.float32) * 5 + 3  # non-standard distribution
        y = bn(x)
        # After BN (with default affine: weight=1, bias=0), per-channel mean ≈ 0, std ≈ 1
        np.testing.assert_allclose(y.mean(axis=0), 0.0, atol=1e-4)
        np.testing.assert_allclose(y.std(axis=0), 1.0, rtol=1e-4)

    def test_train_3d_input(self):
        bn = ts.nn.BatchNorm1d(num_features=4)
        x = np.random.randn(8, 4, 16).astype(np.float32) * 2 + 1
        y = bn(x)
        # Stats over (batch, length) per channel
        np.testing.assert_allclose(y.mean(axis=(0, 2)), 0.0, atol=1e-4)
        np.testing.assert_allclose(y.std(axis=(0, 2)), 1.0, rtol=1e-4)

    def test_running_stats_update_in_train(self):
        bn = ts.nn.BatchNorm1d(num_features=4, momentum=0.1)
        x = np.ones((8, 4), dtype=np.float32) * 2.0
        bn(x)
        # running_mean = 0.1 * batch_mean = 0.1 * 2 = 0.2 (started at 0)
        np.testing.assert_allclose(bn.running_mean.numpy(), 0.2)
        # num_batches_tracked incremented
        assert int(bn.num_batches_tracked.numpy()[0]) == 1

    def test_eval_uses_running_stats(self):
        bn = ts.nn.BatchNorm1d(num_features=4)
        x_train = np.random.randn(16, 4).astype(np.float32) + 5
        bn(x_train)  # populate running stats
        rm_before = bn.running_mean.numpy().copy()
        rv_before = bn.running_var.numpy().copy()

        bn.eval()
        x_eval = np.random.randn(2, 4).astype(np.float32)
        bn(x_eval)
        # Running stats unchanged in eval mode
        np.testing.assert_allclose(bn.running_mean.numpy(), rm_before)
        np.testing.assert_allclose(bn.running_var.numpy(), rv_before)

    def test_invalid_input_shape(self):
        bn = ts.nn.BatchNorm1d(num_features=4)
        with pytest.raises(ValueError, match="\\(N, C\\) or \\(N, C, L\\)"):
            bn(np.random.randn(4))

    def test_channel_mismatch(self):
        bn = ts.nn.BatchNorm1d(num_features=4)
        with pytest.raises(ValueError, match="channel dim"):
            bn(np.random.randn(8, 5).astype(np.float32))

    def test_no_affine(self):
        bn = ts.nn.BatchNorm1d(num_features=4, affine=False)
        names_p = list(n for n, _ in bn.named_parameters())
        assert names_p == []
        x = np.random.randn(8, 4).astype(np.float32)
        y = bn(x)
        assert y.shape == x.shape

    def test_state_dict_roundtrip(self):
        bn1 = ts.nn.BatchNorm1d(num_features=4)
        bn1(np.random.randn(8, 4).astype(np.float32))  # populate stats
        sd = bn1.state_dict()

        bn2 = ts.nn.BatchNorm1d(num_features=4)
        bn2.load_state_dict(sd)
        np.testing.assert_allclose(bn2.running_mean.numpy(), bn1.running_mean.numpy())
        np.testing.assert_allclose(bn2.running_var.numpy(), bn1.running_var.numpy())


# ─────────────────────────────────────────────────────────────────────────────
# C2 — KVCache Module
# ─────────────────────────────────────────────────────────────────────────────


class TestKVCacheModule:
    def test_initial_state(self):
        cache = ts.nn.KVCache(num_heads=2, head_dim=4, max_seq=16)
        assert cache.current_seq == 0

    def test_forward_appends_and_returns_full(self):
        cache = ts.nn.KVCache(num_heads=2, head_dim=4, max_seq=16)
        k1 = np.random.randn(3, 2, 4).astype(np.float32)
        v1 = np.random.randn(3, 2, 4).astype(np.float32)
        K, V = cache(k1, v1)
        assert K.shape == (3, 2, 4)

        k2 = np.random.randn(2, 2, 4).astype(np.float32)
        v2 = np.random.randn(2, 2, 4).astype(np.float32)
        K, V = cache(k2, v2)
        # Returned K/V cover all 5 tokens cumulatively
        assert K.shape == (5, 2, 4)
        np.testing.assert_allclose(K[:3], k1)
        np.testing.assert_allclose(K[3:], k2)

    def test_reset(self):
        cache = ts.nn.KVCache(num_heads=2, head_dim=4, max_seq=16)
        cache(
            np.ones((3, 2, 4), dtype=np.float32),
            np.ones((3, 2, 4), dtype=np.float32),
        )
        cache.reset()
        assert cache.current_seq == 0


# ─────────────────────────────────────────────────────────────────────────────
# D1 — depthwise_conv1d
# ─────────────────────────────────────────────────────────────────────────────


class TestDepthwiseConv1d:
    def test_non_causal_with_padding(self):
        x = np.random.randn(2, 4, 16).astype(np.float64)
        w = np.random.randn(4, 3).astype(np.float64)
        y = ts.ops.depthwise_conv1d(x, w, kernel_size=3, padding=1)
        assert y.shape == (2, 4, 16)

    def test_causal_preserves_length(self):
        x = np.random.randn(2, 4, 8).astype(np.float64)
        w = np.random.randn(4, 5).astype(np.float64)
        y = ts.ops.depthwise_conv1d(x, w, kernel_size=5, causal=True)
        assert y.shape == (2, 4, 8)

    def test_causal_no_future_leakage(self):
        # Output at time t should only depend on inputs ≤ t
        x = np.zeros((1, 1, 8), dtype=np.float64)
        x[0, 0, 5] = 1.0  # only spike at index 5
        w = np.ones((1, 3), dtype=np.float64)
        y = ts.ops.depthwise_conv1d(x, w, kernel_size=3, causal=True)
        # Output[0..4] should be 0 (no spike yet); output[5..7] should reflect the spike
        np.testing.assert_allclose(y[0, 0, :5], 0.0)
        assert y[0, 0, 5] != 0.0

    def test_streaming_matches_single_shot(self):
        x = np.random.randn(2, 4, 16).astype(np.float64)
        w = np.random.randn(4, 3).astype(np.float64)
        y_full = ts.ops.depthwise_conv1d(x, w, kernel_size=3, causal=True)

        x_a, x_b = x[..., :8], x[..., 8:]
        y_a = ts.ops.depthwise_conv1d(x_a, w, kernel_size=3, causal=True)
        state = x_a[..., -2:]
        y_b = ts.ops.depthwise_conv1d(x_b, w, kernel_size=3, state=state)
        np.testing.assert_allclose(np.concatenate([y_a, y_b], axis=-1), y_full)

    def test_invalid_weight_shape(self):
        x = np.random.randn(2, 4, 8).astype(np.float64)
        w = np.random.randn(3, 3).astype(np.float64)  # wrong channel
        with pytest.raises(ValueError, match="weight shape"):
            ts.ops.depthwise_conv1d(x, w, kernel_size=3, causal=True)

    def test_invalid_state_shape(self):
        x = np.random.randn(2, 4, 8).astype(np.float64)
        w = np.random.randn(4, 3).astype(np.float64)
        bad_state = np.zeros((2, 4, 5), dtype=np.float64)  # K-1 should be 2, got 5
        with pytest.raises(ValueError, match="state shape"):
            ts.ops.depthwise_conv1d(x, w, kernel_size=3, state=bad_state)

    def test_vjp_numerical(self):
        # Numerical gradient check at fp64
        np.random.seed(0)
        x_p = ts.nn.Parameter(np.random.randn(1, 2, 6).astype(np.float64))
        w_p = ts.nn.Parameter(np.random.randn(2, 3).astype(np.float64))

        with ts.autodiff.tape() as t:
            y = ts.ops.depthwise_conv1d(x_p, w_p, kernel_size=3, causal=True)
            loss = ts.ops.reduce(ts.ops.mul(y, y), op="sum")
            t.backward(loss)

        analytic_x = x_p.grad.numpy()
        analytic_w = w_p.grad.numpy()

        def _loss(arr_x, arr_w):
            y_ = np.zeros_like(arr_x)
            x_full = np.pad(arr_x, ((0, 0), (0, 0), (2, 0)))
            for k in range(3):
                y_ += x_full[..., k:k + 6] * arr_w[None, :, k:k + 1]
            return float((y_ * y_).sum())

        eps = 1e-6
        # Spot-check one element of x grad
        x_arr = x_p.numpy().copy()
        x0 = x_arr.copy(); x0[0, 0, 3] += eps
        x1 = x_arr.copy(); x1[0, 0, 3] -= eps
        numerical = (_loss(x0, w_p.numpy()) - _loss(x1, w_p.numpy())) / (2 * eps)
        assert abs(analytic_x[0, 0, 3] - numerical) < 1e-4, (analytic_x[0, 0, 3], numerical)

        # Spot-check one weight gradient element
        w_arr = w_p.numpy().copy()
        w0 = w_arr.copy(); w0[1, 2] += eps
        w1 = w_arr.copy(); w1[1, 2] -= eps
        numerical_w = (_loss(x_p.numpy(), w0) - _loss(x_p.numpy(), w1)) / (2 * eps)
        assert abs(analytic_w[1, 2] - numerical_w) < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# D2 — online_softmax
# ─────────────────────────────────────────────────────────────────────────────


class TestOnlineSoftmax:
    def test_single_chunk_matches_softmax(self):
        x = np.random.randn(3, 8).astype(np.float64)
        np.testing.assert_allclose(ts.ops.online_softmax(x), ts.ops.softmax(x))

    def test_streaming_two_chunks(self):
        x = np.random.randn(3, 16).astype(np.float64)
        ref = ts.ops.softmax(x)

        x_a, x_b = x[..., :8], x[..., 8:]
        state_a = ts.ops.online_softmax_state(x_a)
        y_b = ts.ops.online_softmax(x_b, state=state_a)
        np.testing.assert_allclose(y_b, ref[..., 8:], rtol=1e-5)

    def test_streaming_three_chunks(self):
        x = np.random.randn(3, 18).astype(np.float64)
        ref = ts.ops.softmax(x)
        x1, x2, x3 = x[..., :5], x[..., 5:11], x[..., 11:]
        s1 = ts.ops.online_softmax_state(x1)
        s2 = ts.ops.online_softmax_state(x2, state=s1)
        y3 = ts.ops.online_softmax(x3, state=s2)
        np.testing.assert_allclose(y3, ref[..., 11:], rtol=1e-5)

    def test_state_helper_independent_of_axis(self):
        x = np.random.randn(4, 5, 6).astype(np.float64)
        m, s = ts.ops.online_softmax_state(x, axis=1)
        assert m.shape == (4, 1, 6)

    def test_vjp_single_chunk(self):
        x_p = ts.nn.Parameter(np.random.randn(2, 5).astype(np.float64))
        with ts.autodiff.tape() as t:
            y = ts.ops.online_softmax(x_p)
            loss = ts.ops.reduce(ts.ops.mul(y, y), op="sum")
            t.backward(loss)
        # Same VJP as standard softmax for the single-chunk path
        x_p2 = ts.nn.Parameter(x_p.numpy().copy())
        with ts.autodiff.tape() as t2:
            y2 = ts.ops.softmax(x_p2)
            loss2 = ts.ops.reduce(ts.ops.mul(y2, y2), op="sum")
            t2.backward(loss2)
        np.testing.assert_allclose(x_p.grad.numpy(), x_p2.grad.numpy(), rtol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# D4 — DynamicDepthwiseConv1d
# ─────────────────────────────────────────────────────────────────────────────


class TestDynamicDepthwiseConv1d:
    def test_non_streaming_forward(self):
        m = ts.nn.DynamicDepthwiseConv1d(channels=4, kernel_size=3, causal=True)
        x = np.random.randn(2, 4, 16).astype(np.float32)
        y = m(x)
        assert y.shape == (2, 4, 16)
        # No buffer in non-streaming mode
        assert list(m.named_buffers()) == []

    def test_streaming_buffer_present_and_non_persistent(self):
        m = ts.nn.DynamicDepthwiseConv1d(channels=4, kernel_size=3, streaming=True)
        named = list(m.named_buffers())
        assert len(named) == 1
        name, buf = named[0]
        assert name == "_state"
        assert buf.persistent is False

    def test_streaming_matches_single_shot(self):
        # Compare: single-call vs. two-call streaming on the same weights
        np.random.seed(7)
        m_full = ts.nn.DynamicDepthwiseConv1d(channels=4, kernel_size=3, causal=True)
        m_s = ts.nn.DynamicDepthwiseConv1d(channels=4, kernel_size=3, causal=True, streaming=True)
        m_s.weight._data._data[...] = m_full.weight._data._data

        x = np.random.randn(2, 4, 12).astype(np.float32)
        y_full = m_full(x)

        y_a = m_s(x[..., :7])
        y_b = m_s(x[..., 7:])
        y_concat = np.concatenate([y_a, y_b], axis=-1)
        np.testing.assert_allclose(y_full, y_concat, atol=1e-5)

    def test_reset_state_drops_history(self):
        m = ts.nn.DynamicDepthwiseConv1d(channels=2, kernel_size=3, streaming=True)
        x = np.random.randn(1, 2, 8).astype(np.float32)
        y_first = m(x)
        m.reset_state()
        y_second = m(x)
        np.testing.assert_allclose(y_first, y_second)

    def test_state_dict_excludes_buffer(self):
        m = ts.nn.DynamicDepthwiseConv1d(channels=2, kernel_size=3, streaming=True)
        sd = m.state_dict()
        assert "_state" not in sd  # non-persistent
        assert "weight" in sd

    def test_kernel_size_one_skips_state(self):
        # K=1 → state has shape (N, C, 0); no buffer needed
        m = ts.nn.DynamicDepthwiseConv1d(channels=2, kernel_size=1, streaming=True)
        assert list(m.named_buffers()) == []

    def test_invalid_input_rank(self):
        m = ts.nn.DynamicDepthwiseConv1d(channels=2, kernel_size=3)
        with pytest.raises(ValueError, match="\\(N, C, L\\)"):
            m(np.zeros((2, 4)))
