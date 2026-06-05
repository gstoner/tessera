"""Phase 1 Sprint 1.3 — softmax + div tests (docs/spec/PRODUCTION_COMPILER_PLAN.md).

Softmax is the first composite op: it stacks the reduction machinery (max, sum
over an axis) with broadcast-binary elementwise (x - max, e / sum) and a
math.exp unary. The broadcast — a reduced (...) tensor applied against the full
(..., N) input via an affine map that drops the reduced axis — is the
load-bearing new capability. Also covers the elementwise `div` family addition.

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


def _np_softmax(x, axis):
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


# ── softmax ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "shape,axis",
    [((2, 4), -1), ((2, 4), 1), ((2, 4), 0), ((3, 5, 7), 2), ((3, 5, 7), 1), ((8,), 0)],
)
def test_jit_softmax_matches_numpy_oracle(shape, axis):
    rng = np.random.default_rng(abs(hash((shape, axis))) & 0xFFFF)
    x = (rng.standard_normal(shape) * 3.0).astype(np.float32)  # wide range
    out = jb.jit_softmax(x, axis)
    expect = _np_softmax(x, axis).astype(np.float32)
    assert out.shape == x.shape
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)


def test_jit_softmax_rows_sum_to_one():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((4, 16)).astype(np.float32)
    out = jb.jit_softmax(x, axis=-1)
    np.testing.assert_allclose(out.sum(axis=-1), np.ones(4), rtol=1e-5, atol=1e-5)


def test_jit_softmax_numerically_stable_on_large_values():
    # Large inputs would overflow a naive exp(x); the max-subtract must save it.
    x = np.array([[1000.0, 1001.0, 1002.0]], dtype=np.float32)
    out = jb.jit_softmax(x, axis=-1)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, _np_softmax(x, -1), rtol=1e-5, atol=1e-5)


def test_jit_softmax_executed_the_compiled_function():
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    before = jb.invocation_count()
    jb.jit_softmax(x, axis=1)
    assert jb.invocation_count() == before + 1


@pytest.mark.parametrize(
    "x,axis",
    [
        (np.ones((4, 4), np.float16), -1),  # f16 outside envelope
        (np.ones((4, 4), np.float32), 5),   # axis out of range
    ],
)
def test_jit_softmax_rejects_out_of_envelope(x, axis):
    with pytest.raises(jb.TesseraJitError):
        jb.jit_softmax(np.asarray(x), axis)


# ── div family addition ────────────────────────────────────────────────────


def test_jit_div_matches_numpy_oracle():
    rng = np.random.default_rng(11)
    a = rng.standard_normal((3, 5)).astype(np.float32)
    b = (rng.standard_normal((3, 5)) + 2.0).astype(np.float32)  # avoid /0
    out = jb.jit_div(a, b)
    np.testing.assert_allclose(out, a / b, rtol=1e-6, atol=1e-6)


def test_jit_div_executed():
    a = np.full((2, 3), 6.0, dtype=np.float32)
    b = np.full((2, 3), 2.0, dtype=np.float32)
    before = jb.invocation_count()
    out = jb.jit_div(a, b)
    assert jb.invocation_count() == before + 1
    np.testing.assert_array_equal(out, np.full((2, 3), 3.0, dtype=np.float32))
