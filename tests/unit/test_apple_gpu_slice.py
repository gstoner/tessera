"""Apple GPU slice — the fourth structural data-mover on a real MPSGraph kernel.

tessera.slice now runs on Metal via sliceTensor:starts:ends:strides: (was the
numpy fallback): a StableHLO dynamic-slice — ``x[starts[i] : starts[i]+sizes[i]]``
(stride 1) over an N-D input. The KV-cache-window / chunking case. v1 envelope is a
static per-axis window where ``start_indices`` / ``slice_sizes`` are length-rank int
lists; a rank mismatch or out-of-bounds window falls back to numpy. Each data-mover
added here widens what the general per_op_metal residency gate accepts.

Landing this required a frontend fix that is the *mirror* of cat's: slice's two
trailing positional args are index/size *lists* (ints), not tensors, so the AST
GraphIRBuilder must bind them as attributes (``_POSITIONAL_ATTR_PARAMS``) rather
than flatten them into operands — otherwise they were dropped as "%?" operands and
the op never reached a kernel.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

import tessera as ts
from tessera.compiler import apple_gpu_envelope as env

DARWIN = sys.platform == "darwin"
_RNG = np.random.default_rng(20260617)


def test_slice_is_a_first_class_gpu_op():
    assert env.lane_for("tessera.slice") == "slice"
    assert "tessera.slice" in env.runtime_ops()


@pytest.mark.parametrize("starts,sizes", [
    ([0, 0], [4, 6]),   # full
    ([1, 2], [2, 3]),   # interior window
    ([3, 5], [1, 1]),   # corner
])
def test_slice_handler_matches_numpy_2d(starts, sizes):
    from tessera.runtime import _apple_gpu_dispatch_slice
    x = _RNG.standard_normal((4, 6)).astype(np.float32)
    out = _apple_gpu_dispatch_slice(
        "tessera.slice", [x], {"start_indices": starts, "slice_sizes": sizes}, np)
    ref = x[starts[0]:starts[0] + sizes[0], starts[1]:starts[1] + sizes[1]]
    np.testing.assert_array_equal(np.asarray(out), ref)


def test_slice_handler_matches_numpy_3d():
    from tessera.runtime import _apple_gpu_dispatch_slice
    x = _RNG.standard_normal((2, 5, 4)).astype(np.float32)
    out = _apple_gpu_dispatch_slice(
        "tessera.slice", [x], {"start_indices": [0, 1, 0], "slice_sizes": [2, 3, 4]}, np)
    np.testing.assert_array_equal(np.asarray(out), x[0:2, 1:4, 0:4])


def test_slice_f16_matches_numpy():
    from tessera.runtime import _apple_gpu_dispatch_slice
    x = _RNG.standard_normal((6, 8)).astype(np.float16)
    out = _apple_gpu_dispatch_slice(
        "tessera.slice", [x], {"start_indices": [2, 1], "slice_sizes": [3, 4]}, np)
    np.testing.assert_array_equal(np.asarray(out), x[2:5, 1:5])


def test_slice_out_of_bounds_falls_back():
    """An over-range window takes the numpy path (consistent reference)."""
    from tessera.runtime import _apple_gpu_dispatch_slice
    x = _RNG.standard_normal((4, 4)).astype(np.float32)
    out = _apple_gpu_dispatch_slice(
        "tessera.slice", [x], {"start_indices": [3, 3], "slice_sizes": [4, 4]}, np)
    np.testing.assert_array_equal(np.asarray(out), x[3:7, 3:7])


def test_jit_slice_matches_numpy():
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.slice(x, [1, 2], [2, 3])

    x = _RNG.standard_normal((4, 6)).astype(np.float32)
    np.testing.assert_allclose(np.asarray(f(x)), x[1:3, 2:5], rtol=1e-6, atol=1e-6)


def test_jit_slice_is_native_gpu():
    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.slice(x, [0, 1], [3, 4])

    x = _RNG.standard_normal((4, 6)).astype(np.float32)
    f(x)
    assert f.execution_kind == "native_gpu"


@pytest.mark.skipif(not DARWIN, reason="GPU-residency check requires Metal (Darwin)")
def test_slice_runs_on_metal_no_fallback():
    from tessera.compiler import apple_gpu_coverage as cov

    @ts.jit(target="apple_gpu")
    def f(x):
        return ts.ops.slice(x, [1, 1], [2, 4])

    x = _RNG.standard_normal((4, 6)).astype(np.float32)
    assert cov.fallback_histogram(lambda: f(x)) == {}


@pytest.mark.skipif(not DARWIN, reason="GPU-residency check requires Metal (Darwin)")
def test_slice_compounds_with_matmul_per_op_metal():
    """matmul -> slice: windowing a matmul output keeps the chain GPU-resident."""
    from tessera.compiler import apple_gpu_coverage as cov

    @ts.jit(target="apple_gpu")
    def chain(a, b):
        return ts.ops.slice(ts.ops.matmul(a, b), [0, 1], [2, 3])

    a = _RNG.standard_normal((3, 4)).astype(np.float32)
    b = _RNG.standard_normal((4, 6)).astype(np.float32)
    out = np.asarray(chain(a, b))
    np.testing.assert_allclose(out, (a @ b)[0:2, 1:4], rtol=1e-4, atol=1e-4)
    assert chain.execution_kind == "native_gpu"
    assert cov.fallback_histogram(lambda: chain(a, b)) == {}
