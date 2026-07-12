"""Phase 1 Sprint 1.10 — compilation cache (docs/spec/PRODUCTION_COMPILER_PLAN.md;
S14 direction).

Parse→lower→JIT is expensive and deterministic in the MLIR text, so device_verified_jit
handles are cached. The decisive proof is the C++ compile-counter: a repeated
same-shape call does NOT advance it (cache hit), while each invoke still runs and
advances the invocation-counter. Correctness is unchanged.

Skips when libtessera_jit is not built.
"""

import numpy as np
import pytest

from tessera import _jit_boundary as jb

pytestmark = pytest.mark.skipif(
    not jb.is_available(),
    reason="libtessera_jit not built; run `ninja -C build tessera_jit`",
)


def test_repeated_same_shape_call_compiles_once():
    jb.clear_cache()
    a = np.ones((4, 4), np.float32)
    b = np.full((4, 4), 2.0, np.float32)

    c0, i0 = jb.compile_count(), jb.invocation_count()
    out1 = jb.jit_add(a, b)
    out2 = jb.jit_add(a, b)
    out3 = jb.jit_add(a + 1, b)  # same shape/op, different data
    c1, i1 = jb.compile_count(), jb.invocation_count()

    assert c1 - c0 == 1, "three same-(op,shape) calls must compile exactly once"
    assert i1 - i0 == 3, "but each call still executes (invokes)"
    np.testing.assert_array_equal(out1, out2)
    np.testing.assert_allclose(out3, (a + 1) + b)


def test_different_shapes_compile_separately():
    jb.clear_cache()
    c0 = jb.compile_count()
    jb.jit_add(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32))
    jb.jit_add(np.ones((3, 3), np.float32), np.ones((3, 3), np.float32))
    jb.jit_add(np.ones((2, 2), np.float32), np.ones((2, 2), np.float32))  # cache hit
    assert jb.compile_count() - c0 == 2  # two distinct shapes -> two compiles


def test_distinct_ops_same_shape_compile_separately():
    jb.clear_cache()
    a = np.full((4,), 6.0, np.float32)
    b = np.full((4,), 2.0, np.float32)
    c0 = jb.compile_count()
    jb.jit_add(a, b)
    jb.jit_mul(a, b)
    jb.jit_div(a, b)
    jb.jit_add(a, b)  # hit
    assert jb.compile_count() - c0 == 3  # add, mul, div — distinct modules


def test_cache_size_and_clear():
    jb.clear_cache()
    assert jb.cache_size() == 0
    jb.jit_add(np.ones((5,), np.float32), np.ones((5,), np.float32))
    jb.jit_mul(np.ones((5,), np.float32), np.ones((5,), np.float32))
    assert jb.cache_size() == 2
    jb.clear_cache()
    assert jb.cache_size() == 0


def test_cache_can_be_disabled():
    jb.clear_cache()
    jb.set_cache_enabled(False)
    try:
        a = np.ones((4,), np.float32)
        c0 = jb.compile_count()
        jb.jit_add(a, a)
        jb.jit_add(a, a)
        assert jb.compile_count() - c0 == 2  # no cache -> recompiles each call
    finally:
        jb.set_cache_enabled(True)


def test_cache_preserves_correctness_across_ops():
    jb.clear_cache()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 8)).astype(np.float32)
    # Run a few cached ops twice; results must be identical and correct.
    for _ in range(2):
        np.testing.assert_allclose(jb.jit_softmax(x), jb.jit_softmax(x))
        np.testing.assert_allclose(
            jb.jit_layer_norm(x), jb.jit_layer_norm(x), rtol=1e-6
        )
