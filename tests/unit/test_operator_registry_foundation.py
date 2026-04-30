from __future__ import annotations

import numpy as np

import tessera


def test_ops_registry_contains_reference_ops():
    names = tessera.ops.registry.list()
    for name in ("gemm", "conv2d", "flash_attn", "fft", "dct", "spectral_conv"):
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
