"""Apple GPU routing for the canonical ``tessera.ops.ebm_*`` lane.

The tensor-clean EBM subset is projected onto the canonical ``tessera.ops``
surface. This suite locks the apple_gpu envelope membership and that a
``@jit(target="apple_gpu")`` EBM call is classified ``metal_runtime`` (its
dispatcher routes through the EBM lane → EBM MSL kernels) and is numerically
correct.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import _apple_gpu_backend as agb
from tessera import _jit_boundary as jb
from tessera import runtime as _runtime
from tessera.compiler import driver as _driver

_GPU = agb.is_available() and jb.is_available()
gpu = pytest.mark.hardware_apple_gpu

_EBM_GRAPH_OPS = [
    "tessera.ebm_energy_quadratic",
    "tessera.ebm_self_verify",
    "tessera.ebm_refinement",
    "tessera.ebm_inner_step",
]


def test_ebm_ops_in_apple_gpu_envelope():
    for op in _EBM_GRAPH_OPS:
        assert op in _driver._APPLE_GPU_RUNTIME_OPS, op
        assert op in _runtime._APPLE_GPU_RUNTIME_OPS, op
        assert op in _driver._APPLE_GPU_EBM_OPS, op


def test_driver_and_runtime_ebm_envelopes_match():
    assert _driver._APPLE_GPU_EBM_OPS == _runtime._APPLE_GPU_EBM_OPS


@gpu
def test_energy_quadratic_jit_metal_runtime():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4)).astype(np.float32)
    y = rng.standard_normal((3, 4)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def f(x, y):
        return ts.ops.ebm_energy_quadratic(x, y)

    np.testing.assert_allclose(np.asarray(f(x, y)), np.asarray(ts.ops.ebm_energy_quadratic(x, y)), atol=1e-4)
    assert f.runtime_artifact().metadata["execution_mode"] == "metal_runtime"


@gpu
def test_refinement_and_inner_step_jit_metal_runtime():
    rng = np.random.default_rng(1)
    y0 = rng.standard_normal((2, 3)).astype(np.float32)
    grad = rng.standard_normal((2, 3)).astype(np.float32)

    @ts.jit(target="apple_gpu")
    def refine(y0, g):
        return ts.ops.ebm_refinement(y0, g, eta=0.1, T=3)

    np.testing.assert_allclose(np.asarray(refine(y0, grad)), np.asarray(ts.ops.ebm_refinement(y0, grad, eta=0.1, T=3)), atol=1e-4)
    assert refine.runtime_artifact().metadata["execution_mode"] == "metal_runtime"

    @ts.jit(target="apple_gpu")
    def step(y, g):
        return ts.ops.ebm_inner_step(y, g, eta=0.2)

    np.testing.assert_allclose(np.asarray(step(y0, grad)), np.asarray(ts.ops.ebm_inner_step(y0, grad, eta=0.2)), atol=1e-4)
    assert step.runtime_artifact().metadata["execution_mode"] == "metal_runtime"
