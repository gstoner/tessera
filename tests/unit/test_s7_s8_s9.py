"""S7/S8/S9 standalone model-layer, conformance, and quantization coverage."""

from __future__ import annotations

import numpy as np

import tessera as ts


def test_s7_linear_general_lora_and_norms():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    w = np.arange(20, dtype=np.float32).reshape(4, 5) / 10.0
    np.testing.assert_allclose(ts.nn.linear_general(x, w), np.tensordot(x, w, axes=((-1,), (0,))))

    a = np.ones((4, 2), dtype=np.float32)
    b = np.ones((2, 5), dtype=np.float32) * 0.5
    y = ts.nn.lora_linear(x, w, a, b, alpha=2.0)
    expected = x @ w + ((x @ a) @ b)
    np.testing.assert_allclose(y, expected)

    gn_x = np.arange(32, dtype=np.float32).reshape(2, 4, 4)
    gn = ts.nn.group_norm(gn_x, num_groups=2)
    grouped = gn.reshape(2, 2, 2, 4)
    np.testing.assert_allclose(grouped.mean(axis=(2, 3)), 0.0, atol=1e-6)
    np.testing.assert_allclose(grouped.var(axis=(2, 3)), 1.0, atol=1e-5)

    wn = ts.nn.weight_norm(np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32), axis=0)
    np.testing.assert_allclose(np.sqrt((wn * wn).sum(axis=1)), 1.0)


def test_s7_conv_pool_recurrent_and_attention_helpers():
    x = np.arange(5, dtype=np.float32).reshape(1, 1, 5)
    weight = np.array([[[1.0, 2.0, 1.0]]], dtype=np.float32)
    np.testing.assert_allclose(ts.nn.conv1d(x, weight, padding=1), [[[1.0, 4.0, 8.0, 12.0, 11.0]]])

    pooled = ts.nn.max_pool(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4), kernel_size=2)
    np.testing.assert_array_equal(pooled, [[[[5.0, 7.0], [13.0, 15.0]]]])
    avg = ts.nn.adaptive_pool(np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4), output_size=(2, 2))
    np.testing.assert_allclose(avg, [[[[2.5, 4.5], [10.5, 12.5]]]])

    rnn_x = np.array([[1.0, -1.0]], dtype=np.float32)
    h = np.array([[0.5, 0.25]], dtype=np.float32)
    W = np.eye(2, dtype=np.float32)
    np.testing.assert_allclose(ts.nn.simple_rnn_cell(rnn_x, h, W, W), np.tanh(rnn_x + h))

    xs = np.arange(6, dtype=np.float32).reshape(3, 2)
    fwd, bwd = ts.nn.bidirectional_scan(lambda carry, item: carry + item, np.zeros(2), np.zeros(2), xs)
    np.testing.assert_array_equal(fwd[-1], xs.sum(axis=0))
    np.testing.assert_array_equal(bwd[0], xs.sum(axis=0))

    bias = ts.nn.alibi(num_heads=2, seq_len=4)
    assert bias.shape == (2, 4, 4)
    q = np.ones((1, 4, 3, 2), dtype=np.float32)
    k = np.ones((1, 2, 3, 2), dtype=np.float32)
    v = np.ones((1, 2, 3, 2), dtype=np.float32)
    assert ts.nn.gqa_attention(q, k, v, num_query_heads=4, num_kv_heads=2).shape == q.shape
    assert ts.nn.mqa_attention(q, k[:, :1], v[:, :1]).shape == q.shape


def test_s7_stateful_layer_wrappers():
    lg = ts.nn.LinearGeneral(4, 3)
    assert lg(np.ones((2, 4), dtype=np.float32)).shape == (2, 3)

    conv = ts.nn.Conv1d(1, 2, 3, padding=1)
    assert conv(np.ones((2, 1, 5), dtype=np.float32)).shape == (2, 2, 5)

    gru = ts.nn.GRUCell(3, 4)
    assert gru(np.ones((2, 3), dtype=np.float32), np.zeros((2, 4), dtype=np.float32)).shape == (2, 4)


def test_s9_int_quantization_observer_fake_quant_and_grad_scaler():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    q8, scale8, zp8 = ts.quantize_int8(x)
    assert q8.dtype == np.int8
    np.testing.assert_allclose(ts.dequantize_int8(q8, scale8, zp8), x, atol=float(scale8))

    q4, scale4, zp4 = ts.quantize_int4(x)
    assert q4.min() >= -7 and q4.max() <= 7
    np.testing.assert_allclose(ts.dequantize_int4(q4, scale4, zp4), x, atol=float(scale4))

    fq = ts.fake_quantize(x, num_bits=4)
    assert fq.dtype == x.dtype

    obs = ts.calibration_observer().observe(x).observe(np.array([3.0], dtype=np.float32))
    scale, zp = obs.calculate_qparams(num_bits=8)
    assert scale > 0.0 and zp == 0

    grads, new_scale, tracker, should_step = ts.grad_scaler_step({"w": np.array([8.0])}, 8.0, growth_tracker=1, growth_interval=2)
    np.testing.assert_array_equal(grads["w"], [1.0])
    assert new_scale == 16.0 and tracker == 0 and should_step


def test_s8_tiny_recurrent_conformance_forward_backward_rng_state():
    def recurrent_loss(xs):
        def step(carry, item):
            carry = ts.ops.tanh(ts.ops.add(carry, item))
            return carry, carry

        final, ys = ts.scan(step, np.array([0.0], dtype=np.float64), xs)
        return ts.ops.add(ts.ops.reduce(ts.ops.mul(final, final), op="sum"),
                          ts.ops.reduce(ts.ops.mul(ys, ys), op="sum"))

    xs = np.array([[0.1], [0.2], [0.3]], dtype=np.float64)
    value, grad = ts.value_and_grad(recurrent_loss)(xs)
    assert np.asarray(value).shape == ()
    assert grad.shape == xs.shape

    key = ts.rng.RNGKey.from_seed(7, name="diffusion")
    noisy = ts.rng.normal(key, (2, 3))
    state = key.to_state()
    restored = ts.rng.RNGKey.from_state(state)
    np.testing.assert_array_equal(noisy, ts.rng.normal(restored, (2, 3)))


def test_s8_tiny_diffusion_and_attention_conformance_smoke():
    x = np.linspace(-1.0, 1.0, 12, dtype=np.float32).reshape(1, 2, 6)
    kernel = np.ones((2, 1, 3), dtype=np.float32) / 3.0
    h = ts.nn.conv1d(x, kernel, padding=1, groups=2)
    h = ts.nn.group_norm(h, num_groups=2)
    loss = np.mean((h - x) ** 2)
    assert np.isfinite(loss)

    tokens = np.arange(24, dtype=np.float32).reshape(1, 3, 8) / 10.0
    proj = np.ones((8, 8), dtype=np.float32) / 8.0
    mixed = ts.nn.linear_general(tokens, proj)
    attn = ts.nn.multi_head_attention(mixed, mixed, mixed, num_heads=2)
    assert attn.shape == tokens.shape
