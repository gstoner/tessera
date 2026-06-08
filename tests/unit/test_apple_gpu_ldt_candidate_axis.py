"""LDT candidate-axis ops on Metal — popcount + count_nonzero.

Promotes the two LDT candidate-axis primitives from the numpy-fallback
(metal_artifact) lane to metal_runtime via dedicated MSL kernels:
  * popcount       — the MSL `popcount` intrinsic, one thread per element.
  * count_nonzero  — innermost-axis nonzero count, one thread per outer row.

Closes 2 of the lattice-reasoning benchmark's apple_gpu artifact_only rows.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(
    not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")


def test_sentinel_and_envelope():
    from tessera import _apple_gpu_dispatch as agd
    assert agd._SENTINEL_SYMBOL == "tessera_apple_gpu_count_nonzero_lastaxis_f32"
    for op in ("tessera.popcount", "tessera.count_nonzero"):
        assert op in _driver._APPLE_GPU_RUNTIME_OPS
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS


@gpu
@pytest.mark.parametrize("vals", [
    [0, 1, 2, 3, 7, 8, 255, 256, 1023], [0], [2**31 - 1],
])
def test_gpu_popcount_matches_reference(vals):
    b = np.array(vals, np.int32)
    expect = np.array([int(v & 0xFFFFFFFF).bit_count() for v in b.astype(np.uint32)])
    np.testing.assert_array_equal(agb.gpu_popcount(b), expect)


@gpu
def test_gpu_popcount_preserves_shape():
    rng = np.random.default_rng(0)
    b = rng.integers(0, 2**20, size=(3, 5)).astype(np.int32)
    got = agb.gpu_popcount(b)
    assert got.shape == b.shape
    np.testing.assert_array_equal(
        got, np.vectorize(lambda v: int(v).bit_count())(b))


@gpu
@pytest.mark.parametrize("shape", [(2, 4), (5, 9), (3, 4, 6), (1, 8)])
def test_gpu_count_nonzero_lastaxis(shape):
    rng = np.random.default_rng(sum(shape))
    x = (rng.standard_normal(shape) * (rng.random(shape) > 0.4)).astype(np.float32)
    np.testing.assert_array_equal(
        agb.gpu_count_nonzero_lastaxis(x), np.count_nonzero(x, axis=-1))


# ── @jit execution mode flips to metal_runtime ─────────────────────────────── #
@gpu
def test_popcount_jit_metal_runtime():
    @ts.jit(target="apple_gpu")
    def f(b):
        return ts.ops.popcount(b)
    b = np.array([0, 1, 3, 7, 255], np.int32)
    np.testing.assert_array_equal(np.asarray(f(b)), [0, 1, 2, 3, 8])
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_count_nonzero_jit_metal_runtime():
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.count_nonzero(x, axis=-1)
    x = np.array([[0.0, 1.0, 2.0, 0.0], [0.0, 0.0, 0.0, 5.0]], np.float32)
    np.testing.assert_array_equal(np.asarray(f(x)), [2, 1])
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_count_nonzero_non_lastaxis_falls_back_correct():
    # axis=0 isn't the dedicated kernel's case, but must still be correct.
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.count_nonzero(x, axis=0)
    x = np.array([[0.0, 1.0], [2.0, 0.0]], np.float32)
    np.testing.assert_array_equal(np.asarray(f(x)), [1, 1])
