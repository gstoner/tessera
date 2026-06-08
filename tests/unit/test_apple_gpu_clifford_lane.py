"""Apple GPU routing for the canonical ``tessera.ops.clifford_*`` GA lane.

The GA Multivector lane (``tessera.ga.*``) is projected onto the canonical
``tessera.ops`` surface as flat 8-coefficient Cl(3,0) wrappers. This suite locks
that a ``@jit(target="apple_gpu")`` clifford call is classified ``metal_runtime``
(the op is in the apple_gpu envelope and its dispatcher routes through the GA
lane, which hits the cl30 MSL kernels), and is numerically correct.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.skipif(not _GPU, reason="apple_gpu runtime / libtessera_jit unavailable")

_CLIFFORD_GRAPH_OPS = [
    "tessera.clifford_geometric_product",
    "tessera.clifford_wedge",
    "tessera.clifford_left_contraction",
    "tessera.clifford_inner",
    "tessera.clifford_rotor_sandwich",
    "tessera.clifford_reverse",
    "tessera.clifford_grade_involution",
    "tessera.clifford_conjugate",
    "tessera.clifford_grade_projection",
    "tessera.clifford_norm",
    "tessera.clifford_norm_squared",
]


def test_clifford_ops_in_apple_gpu_envelope():
    for op in _CLIFFORD_GRAPH_OPS:
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_CLIFFORD_OPS, op


def test_driver_and_runtime_clifford_envelopes_match():
    assert _driver._APPLE_GPU_CLIFFORD_OPS == _runtime._APPLE_GPU_CLIFFORD_OPS


@gpu
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_geometric_product_jit_metal_runtime(seed):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def gp(a, b):
        return ts.ops.clifford_geometric_product(a, b)

    out = np.asarray(gp(a, b))
    ref = np.asarray(ts.ops.clifford_geometric_product(a, b))
    np.testing.assert_allclose(out, ref, atol=1e-5)
    assert gp.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_norm_jit_metal_runtime():
    rng = np.random.default_rng(3)
    a = rng.standard_normal(8).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def nm(a):
        return ts.ops.clifford_norm(a)

    out = np.asarray(nm(a))
    np.testing.assert_allclose(out, float(ts.ops.clifford_norm(a)), atol=1e-5)
    assert nm.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_grade_projection_jit_metal_runtime():
    rng = np.random.default_rng(4)
    a = rng.standard_normal(8).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def gpr(a):
        return ts.ops.clifford_grade_projection(a, grade=2)

    out = np.asarray(gpr(a))
    np.testing.assert_allclose(out, np.asarray(ts.ops.clifford_grade_projection(a, grade=2)), atol=1e-5)
    assert gpr.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
