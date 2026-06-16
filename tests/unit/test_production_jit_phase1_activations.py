"""Phase 1 Sprint 1.6 — activation tests (docs/spec/PRODUCTION_COMPILER_PLAN.md).

Unary math family: relu / sigmoid / tanh / silu / gelu(tanh-approx). Each lowers
to a single linalg.generic with a per-scalar body over arith + math.{exp,tanh}.

numpy oracle + unfakeable invocation-counter advance. Skips when libtessera_jit
is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def _np_sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _np_gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(0.7978845608028654 * (x + 0.044715 * x**3)))


_ACTS = {
    "relu": (jb.jit_relu, lambda x: np.maximum(x, 0.0)),
    "sigmoid": (jb.jit_sigmoid, _np_sigmoid),
    "tanh": (jb.jit_tanh, np.tanh),
    "silu": (jb.jit_silu, lambda x: x * _np_sigmoid(x)),
    "gelu": (jb.jit_gelu, _np_gelu_tanh),
}


@pytest.mark.parametrize("name", sorted(_ACTS))
@pytest.mark.parametrize("shape", [(8,), (3, 5), (2, 3, 4)])
def test_activation_matches_numpy_oracle(name, shape):
    fn, ref = _ACTS[name]
    rng = np.random.default_rng(abs(hash((name, shape))) & 0xFFFF)
    x = (rng.standard_normal(shape) * 2.5).astype(np.float32)
    out = fn(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, ref(x.astype(np.float64)), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("name", sorted(_ACTS))
def test_activation_executed_the_compiled_function(name):
    fn, _ = _ACTS[name]
    x = np.linspace(-3, 3, 16, dtype=np.float32)
    before = jb.invocation_count()
    fn(x)
    assert jb.invocation_count() == before + 1


def test_relu_zeros_negatives_exactly():
    x = np.array([-2.0, -0.5, 0.0, 0.5, 2.0], dtype=np.float32)
    np.testing.assert_array_equal(jb.jit_relu(x), np.array([0, 0, 0, 0.5, 2.0], np.float32))


def test_activation_f16_now_executes():
    # Phase 4: f16 is native on M1 Max NEON (ARMv8.2-A FP16) and supported.
    before = jb.invocation_count()
    out = jb.jit_gelu(np.ones((4,), dtype=np.float16))
    assert jb.invocation_count() == before + 1
    assert np.asarray(out).dtype == np.float16


def test_activation_rejects_unsupported_dtype():
    # f64 is native on M1 but not yet wired into the boundary table → reject.
    with pytest.raises(jb.TesseraJitError):
        jb.jit_gelu(np.ones((4,), dtype=np.float64))
