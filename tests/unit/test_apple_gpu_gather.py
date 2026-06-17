"""Apple GPU gather — the second structural data-mover on a real MPSGraph kernel.

tessera.gather now runs on Metal via gatherWithUpdatesTensor:axis:0 (was the numpy
fallback): rows of a 2D table indexed by int32 ids — the embedding / attention-
index case. v1 envelope is axis-0 + 2D table (other axes / N-D tables fall back to
np.take). Each data-mover added here widens what the general per_op_metal residency
gate accepts (e.g. an embedding lookup mid-program now stays GPU-resident).
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import apple_gpu_envelope as env

DARWIN = sys.platform == "darwin"
_RNG = np.random.default_rng(20260617)


def test_gather_is_a_first_class_gpu_op():
    assert env.lane_for("tessera.gather") == "gather"
    assert "tessera.gather" in env.runtime_ops()


@pytest.mark.parametrize("idx_shape", [(5,), (2, 3), (4,)])
def test_gather_handler_matches_numpy(idx_shape):
    """Embedding-style row gather (Metal on Darwin, host fallback elsewhere) vs
    numpy, over 1D and multi-dim index shapes."""
    from tessera.runtime import _apple_gpu_dispatch_gather
    table = _RNG.standard_normal((10, 8)).astype(np.float32)
    ids = _RNG.integers(0, 10, size=idx_shape).astype(np.int32)
    out = _apple_gpu_dispatch_gather("tessera.gather", [table, ids], {}, np)
    np.testing.assert_array_equal(np.asarray(out), table[ids])


def test_gather_handles_negative_indices():
    from tessera.runtime import _apple_gpu_dispatch_gather
    table = _RNG.standard_normal((6, 4)).astype(np.float32)
    ids = np.array([-1, 0, -2, 3], np.int32)
    out = _apple_gpu_dispatch_gather("tessera.gather", [table, ids], {}, np)
    np.testing.assert_array_equal(np.asarray(out), table[ids])


def test_gather_f16_matches_numpy():
    from tessera.runtime import _apple_gpu_dispatch_gather
    table = _RNG.standard_normal((8, 5)).astype(np.float16)
    ids = np.array([7, 0, 3, 3, 1], np.int32)
    out = _apple_gpu_dispatch_gather("tessera.gather", [table, ids], {}, np)
    np.testing.assert_array_equal(np.asarray(out), table[ids])


def test_jit_gather_matches_numpy():
    @ts.jit(target="apple_gpu")
    def f(table, ids):
        return ts.ops.gather(table, ids)

    table = _RNG.standard_normal((10, 8)).astype(np.float32)
    ids = _RNG.integers(0, 10, size=(4,)).astype(np.int32)
    np.testing.assert_array_equal(np.asarray(f(table, ids)), table[ids])


@pytest.mark.skipif(not DARWIN, reason="GPU-residency check requires Metal (Darwin)")
def test_gather_runs_on_metal_no_fallback():
    from tessera.compiler import apple_gpu_coverage as cov

    @ts.jit(target="apple_gpu")
    def f(table, ids):
        return ts.ops.gather(table, ids)

    table = _RNG.standard_normal((10, 8)).astype(np.float32)
    ids = _RNG.integers(0, 10, size=(4,)).astype(np.int32)
    assert cov.fallback_histogram(lambda: f(table, ids)) == {}
