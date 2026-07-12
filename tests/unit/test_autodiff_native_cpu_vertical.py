"""Native CPU vertical proof for compiler-generated paired autodiff."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


@ts.jit(target="cpu", autodiff="reverse", wrt=("x", "w"))
def _tanh_matmul(x, w):
    return ts.ops.tanh(ts.ops.matmul(x, w))


@ts.jit(target="cpu", autodiff="reverse", wrt=("x", "w"))
def _sigmoid_matmul(x, w):
    return ts.ops.sigmoid(ts.ops.matmul(x, w))


def _inputs():
    rng = np.random.default_rng(17)
    return (
        rng.standard_normal((4, 8)).astype(np.float32),
        rng.standard_normal((8, 16)).astype(np.float32),
    )


def test_unannotated_autodiff_jit_specializes_from_call_values():
    x, w = _inputs()
    text = _tanh_matmul.specialized_autodiff_ir(x, w)
    assert "tensor<4x8xf32>" in text
    assert "tensor<8x16xf32>" in text
    assert "tensor<4x16xf32>" in text
    assert "tensor<*x?>" not in text
    # A second call with the same signature reuses the specialized module.
    assert _tanh_matmul._specialized_autodiff_module((x, w), {}) is \
        _tanh_matmul._specialized_autodiff_module((x, w), {})


def _native_tools_available() -> bool:
    from tessera import _jit_boundary as jb

    return jb._find_tessera_opt() is not None and jb.is_available()


@pytest.mark.parametrize(
    "compiled,activation_grad",
    [
        (_tanh_matmul, lambda a: 1.0 - a * a),
        (_sigmoid_matmul, lambda a: a * (1.0 - a)),
    ],
)
@pytest.mark.skipif(
    not _native_tools_available(),
    reason="requires built tessera-opt and libtessera_jit",
)
def test_compiler_generated_backward_launches_natively_and_matches_oracle(
    compiled, activation_grad
):
    from tessera import _jit_boundary as jb

    x, w = _inputs()
    seed = np.ones((4, 16), dtype=np.float32)
    before = jb.invocation_count()
    dx, dw = compiled.native_backward(x, w, out_cotangents=seed)
    assert jb.invocation_count() == before + 1
    assert compiled.last_backward_execution == {
        "compiler_path": "cpu_autodiff_matmul_llvm_jit",
        "execution_kind": "native_cpu",
        "execution_mode": "mlir_llvm_jit",
        "invocation_delta": 1,
    }

    raw = x @ w
    activation = np.tanh(raw) if compiled is _tanh_matmul else 1.0 / (1.0 + np.exp(-raw))
    dm = seed * activation_grad(activation)
    np.testing.assert_allclose(dx, dm @ w.T, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(dw, x.T @ dm, rtol=1e-4, atol=1e-5)
