"""Public @jit native_backward binds normalization graphs to target VJP ABIs."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "gamma"))
def _x86_rmsnorm(x, gamma):
    return ts.ops.rmsnorm(x, gamma=gamma, eps=1.0e-5)


@ts.jit(target="x86", autodiff="reverse", wrt=("x", "gamma", "beta"))
def _x86_layernorm(x, gamma, beta):
    return ts.ops.layer_norm(x, gamma=gamma, beta=beta, eps=1.0e-5)


@ts.jit(target="x86", autodiff="reverse", wrt=("x",))
def _x86_layernorm_x_only(x, gamma, beta):
    return ts.ops.layer_norm(x, gamma=gamma, beta=beta, eps=1.0e-5)


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "gamma"))
def _rocm_rmsnorm(x, gamma):
    return ts.ops.rmsnorm(x, gamma=gamma, eps=1.0e-5)


@ts.jit(target="rocm", autodiff="reverse", wrt=("x", "gamma", "beta"))
def _rocm_layernorm(x, gamma, beta):
    return ts.ops.layer_norm(x, gamma=gamma, beta=beta, eps=1.0e-5)


def _reference(x, gamma, dy, *, layer):
    xf = np.asarray(x, np.float32)
    gf = np.asarray(gamma, np.float32)
    dyf = np.asarray(dy, np.float32)
    if layer:
        mean = xf.mean(axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(((xf - mean) ** 2).mean(axis=-1, keepdims=True)
                            + 1.0e-5)
        z = (xf - mean) * inv
    else:
        inv = 1.0 / np.sqrt((xf * xf).mean(axis=-1, keepdims=True) + 1.0e-5)
        z = xf * inv
    dz = dyf * gf
    mean_dz = dz.mean(axis=-1, keepdims=True) if layer else 0.0
    dx = inv * (dz - mean_dz - z * (dz * z).mean(axis=-1, keepdims=True))
    return dx, (dyf * z).sum(axis=0), dyf.sum(axis=0)


@pytest.mark.parametrize("compiled,layer,path", [
    (_x86_rmsnorm, False, "x86_rmsnorm_bwd_compiled"),
    (_x86_layernorm, True, "x86_layer_norm_bwd_compiled"),
])
def test_x86_public_native_backward(compiled, layer, path):
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime unavailable")
    rng = np.random.default_rng(313 + int(layer))
    x = rng.standard_normal((7, 33)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, 33).astype(np.float32)
    beta = rng.standard_normal(33).astype(np.float32)
    dy = rng.standard_normal(x.shape).astype(np.float32)
    args = (x, gamma, beta) if layer else (x, gamma)
    gradients = compiled.native_backward(*args, out_cotangents=dy)
    expected = _reference(x, gamma, dy, layer=layer)
    wanted = expected if layer else expected[:2]
    for actual, reference in zip(gradients, wanted):
        np.testing.assert_allclose(actual, reference, atol=8e-5, rtol=8e-5)
    assert compiled.last_backward_execution["compiler_path"] == path
    assert compiled.last_backward_execution["evidence_target"] == "x86_avx512"


def test_public_native_backward_honors_wrt_subset():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime unavailable")
    rng = np.random.default_rng(315)
    x = rng.standard_normal((3, 17)).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, 17).astype(np.float32)
    beta = rng.standard_normal(17).astype(np.float32)
    dy = rng.standard_normal(x.shape).astype(np.float32)
    gradients = _x86_layernorm_x_only.native_backward(
        x, gamma, beta, out_cotangents=dy)
    assert len(gradients) == 1
    expected, _, _ = _reference(x, gamma, dy, layer=True)
    np.testing.assert_allclose(gradients[0], expected, atol=8e-5, rtol=8e-5)


@pytest.mark.parametrize("compiled,layer,path", [
    (_rocm_rmsnorm, False, "rocm_rmsnorm_bwd_compiled"),
    (_rocm_layernorm, True, "rocm_layer_norm_bwd_compiled"),
])
def test_rocm_public_native_backward(compiled, layer, path):
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/GPU unavailable")
    rng = np.random.default_rng(317 + int(layer))
    x = rng.standard_normal((5, 37)).astype(np.float16)
    gamma = rng.uniform(0.5, 1.5, 37).astype(np.float16)
    beta = rng.standard_normal(37).astype(np.float16)
    dy = rng.standard_normal(x.shape).astype(np.float16)
    args = (x, gamma, beta) if layer else (x, gamma)
    gradients = compiled.native_backward(*args, out_cotangents=dy)
    expected = _reference(x, gamma, dy, layer=layer)
    wanted = expected if layer else expected[:2]
    for actual, reference in zip(gradients, wanted):
        np.testing.assert_allclose(np.asarray(actual, np.float32), reference,
                                   atol=2e-2, rtol=2e-2)
    assert compiled.last_backward_execution["compiler_path"] == path
    assert compiled.last_backward_execution["evidence_target"] == "rocm_gfx1151"
