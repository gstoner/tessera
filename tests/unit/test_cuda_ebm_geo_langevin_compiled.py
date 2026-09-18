"""The EBM sphere Langevin front door on CUDA: one cooperative sm_120 kernel
through the row-program emitter (mirrors test_rocm_ebm_geo_langevin_compiled)."""
from __future__ import annotations

import os

import numpy as np
import pytest

from tessera.rng import RNGKey
import importlib

energy = importlib.import_module("tessera.ebm.energy")
from tessera.ebm.geo_sampling import sphere_langevin_step
from tessera.compiler.scheduled_matmul import find_tessera_opt


def _reference_step(x, grad, noise, eta, noise_scale):
    x, grad, noise = (np.asarray(a, np.float64) for a in (x, grad, noise))
    gt = grad - np.dot(grad, x) * x
    nt = noise - np.dot(noise, x) * x
    y = x - eta * gt + noise_scale * nt
    n = np.linalg.norm(y)
    return (x if n < 1e-12 else y / n).astype(np.float32)


@pytest.mark.parametrize("rows,features", [(1, 8), (3, 33)])
def test_sphere_row_program_lowers_to_one_kernel(rows, features):
    """Host-free half: the module is one row-program kernel with a lane per
    feature, three ordered reductions and the rounding-explicit sqrt."""
    from tessera.compiler.native_row_program import declared_math, row_program_kernel, sphere_langevin_step_module
    compiler = find_tessera_opt()
    if compiler is None:
        pytest.skip("production tessera-opt unavailable")
    kernel, lanes = row_program_kernel(sphere_langevin_step_module(rows, features),
                                       entry="sphere_langevin_step", backend="nvidia", compiler=compiler)
    assert lanes >= features  # one lane per feature, rounded up to the emitter's block
    assert kernel.count("gpu.func ") == 1
    assert "math.sqrt" in declared_math(kernel) or "__nv_fsqrt_rn" in kernel
    assert "arith.select" in kernel and "arith.cmpf" in kernel


def _sm120_or_skip():
    from tests._support.environment import nvidia_cuda_tool, nvidia_gpu_is_plausibly_present
    if os.environ.get("TESSERA_SM120_DEVICE_PROOF") != "1":
        pytest.skip("explicit sm_120 owning-device gate (TESSERA_SM120_DEVICE_PROOF=1)")
    if not nvidia_gpu_is_plausibly_present() or nvidia_cuda_tool("ptxas") is None or find_tessera_opt() is None:
        pytest.skip("requires the RTX 5070 host with the CUDA toolkit and tessera-opt")


def _cuda_only_spy(monkeypatch):
    """Force the other lanes off and count the steps the CUDA front door served."""
    for name in ("_try_apple_gpu_sphere_langevin_step_f32", "_try_x86_ebm_affine_langevin_step_f32",
                 "_try_rocm_ebm_affine_langevin_step_f32"):
        monkeypatch.setattr(energy, name, lambda *a, **k: None)
    orig = energy._try_cuda_gpu_sphere_langevin_step_f32
    hits: list[int] = []

    def spy(*a, **k):
        r = orig(*a, **k)
        if r is not None:
            hits.append(1)
        return r
    monkeypatch.setattr(energy, "_try_cuda_gpu_sphere_langevin_step_f32", spy)
    return hits


@pytest.mark.hardware_nvidia
@pytest.mark.parametrize("d", [16, 33, 1024])
def test_sm120_sphere_step_matches_the_numpy_formula(d):
    _sm120_or_skip()
    rng = np.random.default_rng(d)
    x = rng.standard_normal(d).astype(np.float32)
    x = (x / np.linalg.norm(x)).astype(np.float32)
    grad = rng.standard_normal(d).astype(np.float32)
    noise = rng.standard_normal(d).astype(np.float32)
    out = energy._try_cuda_gpu_sphere_langevin_step_f32(x, grad, noise, 0.02, 0.2)
    assert out is not None, energy._CUDA_SPHERE_STATE.get(d)
    assert out.dtype == np.float32 and out.shape == (d,)
    np.testing.assert_allclose(out, _reference_step(x, grad, noise, 0.02, 0.2), rtol=2e-5, atol=2e-6)
    assert abs(float(np.linalg.norm(out)) - 1.0) < 1e-5
    # The retraction guard: a step that lands at the origin keeps the state.
    zero_y = energy._try_cuda_gpu_sphere_langevin_step_f32(x, np.zeros(d, np.float32), np.zeros(d, np.float32), 1.0, 0.0)
    np.testing.assert_array_equal(zero_y, x)


@pytest.mark.hardware_nvidia
def test_sm120_sphere_chain_runs_every_step_on_the_device(monkeypatch):
    _sm120_or_skip()
    hits = _cuda_only_spy(monkeypatch)
    d, steps = 16, 12
    rng = np.random.default_rng(3)
    x = rng.standard_normal(d).astype(np.float32)
    x = (x / np.linalg.norm(x)).astype(np.float32)
    grad_fn = lambda v: np.asarray(v, np.float32)  # noqa: E731
    energy_fn = lambda v: 0.5 * float((np.asarray(v) ** 2).sum())  # noqa: E731
    key = RNGKey.from_seed(0)
    state = x
    for _ in range(steps):
        state, key = sphere_langevin_step(state, energy_fn, eta=0.02, temperature=1.0, rng_key=key, grad_fn=grad_fn)
        assert state.dtype == np.float32
        assert abs(float(np.linalg.norm(state)) - 1.0) < 1e-4
    assert len(hits) == steps
    # The numpy reference chain under the same key agrees to f32 precision.
    monkeypatch.setattr(energy, "_try_cuda_gpu_sphere_langevin_step_f32", lambda *a, **k: None)
    ref, key = x, RNGKey.from_seed(0)
    for _ in range(steps):
        ref, key = sphere_langevin_step(ref, energy_fn, eta=0.02, temperature=1.0, rng_key=key, grad_fn=grad_fn)
    np.testing.assert_allclose(state, ref, rtol=2e-4, atol=2e-5)
