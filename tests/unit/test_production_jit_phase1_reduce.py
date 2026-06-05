"""Phase 1 Sprint 1.2 — reduction tests (docs/spec/PRODUCTION_COMPILER_PLAN.md).

First production-lane op whose result rank differs from its input rank. Proves:
* sum/max/min/mean match numpy (oracle),
* the unfakeable invocation counter advances (the MLIR/LLVM lane executed),
* rank-change descriptor packing is correct across axes/ranks,
* out-of-envelope inputs raise (no silent numpy fallback).

Skips only when libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)

_REDUCERS = {
    "sum": (jb.jit_sum, np.sum),
    "max": (jb.jit_amax, np.max),
    "min": (jb.jit_amin, np.min),
    "mean": (jb.jit_mean, np.mean),
}


@pytest.mark.parametrize("kind", sorted(_REDUCERS))
@pytest.mark.parametrize("shape,axis", [((4, 5), 0), ((4, 5), 1), ((2, 3, 4), 1)])
def test_jit_reduce_matches_numpy_oracle(kind, shape, axis):
    fn, ref = _REDUCERS[kind]
    rng = np.random.default_rng(hash((kind, shape, axis)) & 0xFFFF)
    a = rng.standard_normal(shape).astype(np.float32)
    out = fn(a, axis)
    expect = ref(a, axis=axis).astype(np.float32)
    assert out.shape == expect.shape  # rank actually collapsed
    np.testing.assert_allclose(out, expect, rtol=1e-5, atol=1e-5)


def test_jit_reduce_result_rank_is_one_less():
    a = np.ones((2, 3, 4), dtype=np.float32)
    assert jb.jit_sum(a, axis=0).shape == (3, 4)
    assert jb.jit_sum(a, axis=1).shape == (2, 4)
    assert jb.jit_sum(a, axis=2).shape == (2, 3)


def test_jit_reduce_negative_axis():
    rng = np.random.default_rng(1)
    a = rng.standard_normal((3, 7)).astype(np.float32)
    np.testing.assert_allclose(
        jb.jit_amax(a, axis=-1), np.max(a, axis=-1), rtol=1e-6, atol=1e-6
    )


@pytest.mark.parametrize("kind", sorted(_REDUCERS))
def test_jit_reduce_executed_the_compiled_function(kind):
    fn, _ = _REDUCERS[kind]
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    before = jb.invocation_count()
    fn(a, axis=1)
    assert jb.invocation_count() == before + 1


@pytest.mark.parametrize(
    "a,axis,kind",
    [
        (np.ones((5,), np.float32), 0, "sum"),       # rank-1 -> rank-0 result: rejected
        (np.ones((4, 4), np.float16), 0, "sum"),     # f16 outside envelope
        (np.ones((4, 4), np.float32), 5, "sum"),     # axis out of range
        (np.ones((4, 4), np.float32), 0, "median"),  # unknown kind
    ],
)
def test_jit_reduce_rejects_out_of_envelope(a, axis, kind):
    with pytest.raises(jb.TesseraJitError):
        jb.jit_reduce(np.asarray(a), axis, kind)
