"""Phase 1 Sprint 1.4 — normalization tests (docs/spec/PRODUCTION_COMPILER_PLAN.md).

rmsnorm / layer_norm over the innermost axis (unweighted). Both compose the
Sprint 1.3 machinery: mean-reductions + broadcast-binary + math.sqrt. Proves the
broadcast primitive (`emitBroadcastBinary`) generalizes beyond softmax.

Every op: numpy oracle + unfakeable invocation-counter advance. Skips when
libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def _np_rmsnorm(x, eps):
    ms = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(ms + eps)).astype(np.float32)


def _np_layer_norm(x, eps):
    xd = x.astype(np.float64)
    mu = np.mean(xd, axis=-1, keepdims=True)
    var = np.mean((xd - mu) ** 2, axis=-1, keepdims=True)
    return ((xd - mu) / np.sqrt(var + eps)).astype(np.float32)


@pytest.mark.parametrize("shape", [(2, 4), (3, 5, 7), (1, 16), (8,)])
def test_jit_rmsnorm_matches_numpy_oracle(shape):
    rng = np.random.default_rng(abs(hash(("rms", shape))) & 0xFFFF)
    x = (rng.standard_normal(shape) * 2.0).astype(np.float32)
    out = jb.jit_rmsnorm(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, _np_rmsnorm(x, 1e-5), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("shape", [(2, 4), (3, 5, 7), (1, 16), (8,)])
def test_jit_layer_norm_matches_numpy_oracle(shape):
    rng = np.random.default_rng(abs(hash(("ln", shape))) & 0xFFFF)
    x = (rng.standard_normal(shape) * 2.0 + 1.0).astype(np.float32)
    out = jb.jit_layer_norm(x)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, _np_layer_norm(x, 1e-5), rtol=1e-4, atol=1e-4)


def test_jit_layer_norm_zero_mean_unit_var():
    rng = np.random.default_rng(5)
    x = (rng.standard_normal((4, 64)) * 5.0 + 3.0).astype(np.float32)
    out = jb.jit_layer_norm(x, eps=1e-6)
    # Normalized rows: ~zero mean, ~unit variance.
    np.testing.assert_allclose(out.mean(axis=-1), np.zeros(4), atol=1e-4)
    np.testing.assert_allclose(out.var(axis=-1), np.ones(4), rtol=2e-3, atol=2e-3)


def test_jit_norm_custom_eps_is_honored():
    x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
    big_eps = 10.0
    np.testing.assert_allclose(
        jb.jit_rmsnorm(x, eps=big_eps), _np_rmsnorm(x, big_eps), rtol=1e-5, atol=1e-5
    )
    np.testing.assert_allclose(
        jb.jit_layer_norm(x, eps=big_eps),
        _np_layer_norm(x, big_eps),
        rtol=1e-5,
        atol=1e-5,
    )


@pytest.mark.parametrize("fn", [jb.jit_rmsnorm, jb.jit_layer_norm])
def test_jit_norm_executed_the_compiled_function(fn):
    x = np.arange(12, dtype=np.float32).reshape(3, 4) + 1.0
    before = jb.invocation_count()
    fn(x)
    assert jb.invocation_count() == before + 1


@pytest.mark.parametrize("fn", [jb.jit_rmsnorm, jb.jit_layer_norm])
def test_jit_norm_rejects_non_f32(fn):
    with pytest.raises(jb.TesseraJitError):
        fn(np.ones((4, 4), dtype=np.float16))
