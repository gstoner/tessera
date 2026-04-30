from __future__ import annotations

import numpy as np

import tessera


def test_ops_registry_contains_reference_ops():
    names = tessera.ops.registry.list()
    for name in ("gemm", "conv2d", "flash_attn", "fft", "ifft", "rfft", "irfft", "dct", "spectral_conv"):
        assert name in names


def test_ops_registry_reference_dispatch():
    a = np.ones((2, 2), dtype=np.float32)
    b = np.ones((2, 2), dtype=np.float32)
    out = tessera.ops.registry.dispatch("gemm", a, b, prefer_runtime=False)
    np.testing.assert_allclose(out, a @ b)


def test_ops_registry_accepts_runtime_kernel_registration():
    def kernel(x):
        return x + 1

    tessera.ops.register_runtime_kernel("unit_test_increment", kernel, backend="unit")
    out = tessera.ops.registry.dispatch("unit_test_increment", np.array([1]))
    np.testing.assert_array_equal(out, np.array([2]))


def test_flash_attn_runtime_supports_causal_and_seeded_dropout():
    q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    k = q.copy()
    v = np.array([[[[1.0, 10.0], [100.0, 1000.0]]]], dtype=np.float32)

    plain = tessera.ops.flash_attn(q, k, v, causal=False)
    causal = tessera.ops.flash_attn(q, k, v, causal=True)
    assert plain.shape == causal.shape == v.shape
    assert not np.allclose(plain, causal)
    np.testing.assert_allclose(causal[..., 0, :], v[..., 0, :])

    dropped_a = tessera.ops.flash_attn(q, k, v, dropout_p=0.25, seed=123)
    dropped_b = tessera.ops.flash_attn(q, k, v, dropout_p=0.25, seed=123)
    np.testing.assert_allclose(dropped_a, dropped_b)


def test_flash_attn_runtime_rejects_invalid_dropout_probability():
    q = np.zeros((1, 1, 2, 2), dtype=np.float32)

    with np.testing.assert_raises(ValueError):
        tessera.ops.flash_attn(q, q, q, dropout_p=1.0)


def test_fft_runtime_helpers_match_numpy():
    x = np.arange(8.0)
    xf = np.fft.fft(x)
    rxf = np.fft.rfft(x)

    np.testing.assert_allclose(tessera.ops.ifft(xf), np.fft.ifft(xf))
    np.testing.assert_allclose(tessera.ops.rfft(x), rxf)
    np.testing.assert_allclose(tessera.ops.irfft(rxf, n=x.size), np.fft.irfft(rxf, n=x.size))


def test_graph_ir_recognizes_new_operator_names():
    @tessera.jit
    def spectral_kernel(x, w):
        y = tessera.ops.fft(x)
        z = tessera.ops.dct(y, type=2)
        return tessera.ops.spectral_conv(z, w)

    ir = spectral_kernel.ir_text()
    assert "tessera.fft" in ir
    assert "tessera.dct" in ir
    assert "tessera.spectral_conv" in ir
