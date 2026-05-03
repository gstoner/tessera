from __future__ import annotations

import numpy as np

import tessera
from tessera.compiler.graph_ir import _OpExtractor
from tessera.compiler.matmul_pipeline import SUPPORTED_CPU_OPS
from tessera.compiler.op_catalog import GRAPH_OP_MAP, OP_SPECS


def test_ops_registry_contains_reference_ops():
    names = tessera.ops.registry.list()
    for name in OP_SPECS:
        assert name in names


def test_op_catalog_is_consistent_across_frontend_and_cpu():
    assert set(_OpExtractor._OP_MAP) == set(OP_SPECS)
    assert set(_OpExtractor._OP_MAP.values()) == set(SUPPORTED_CPU_OPS)
    assert GRAPH_OP_MAP["kv_cache_append"] == "tessera.kv_cache.append"


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


def test_jit_cpu_executes_reconciled_numpy_ops():
    @tessera.jit
    def op_chain(x, w):
        y = tessera.ops.layer_norm(x)
        z = tessera.ops.gelu(y)
        c = tessera.ops.cast(z, dtype="fp32")
        f = tessera.ops.fft(c)
        d = tessera.ops.dct(f)
        return tessera.ops.spectral_conv(d, w)

    x = np.arange(6, dtype=np.float32).reshape(2, 3)
    w = np.array([1.0, 0.5], dtype=np.float32)
    out = op_chain(x, w)
    assert op_chain.uses_compiled_path
    assert out.shape[-1] == 4


def test_jit_cpu_executes_conv2d_nhwc_reference():
    @tessera.jit
    def conv(x, w):
        return tessera.ops.conv2d(x, w, stride=1, padding=0)

    x = np.arange(9, dtype=np.float32).reshape(1, 3, 3, 1)
    w = np.ones((2, 2, 1, 1), dtype=np.float32)
    expected = np.array([[[[8.0], [12.0]], [[20.0], [24.0]]]], dtype=np.float32)
    np.testing.assert_allclose(conv(x, w), expected)
    assert conv.uses_compiled_path


def test_jit_cpu_executes_seeded_dropout_and_collective_stubs():
    @tessera.jit
    def dropped(x):
        y = tessera.ops.dropout(x, p=0.25, seed=7)
        return tessera.ops.all_reduce(y)

    x = np.ones((4,), dtype=np.float32)
    np.testing.assert_allclose(dropped(x), dropped(x))
    assert dropped.uses_compiled_path


def test_jit_cpu_executes_flash_attention_reference():
    @tessera.jit
    def flash(q, k, v):
        return tessera.ops.flash_attn(q, k, v, causal=True)

    q = np.array([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
    v = np.array([[[[1.0, 10.0], [100.0, 1000.0]]]], dtype=np.float32)
    out = flash(q, q, v)
    assert out.shape == v.shape
    np.testing.assert_allclose(out[..., 0, :], v[..., 0, :])
