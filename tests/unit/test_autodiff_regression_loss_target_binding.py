"""Public paired ABI binding for native regression-loss gradients."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


@ts.jit(target="x86", autodiff="reverse", wrt=("prediction", "target"))
def _x86_huber(prediction, target):
    return ts.ops.huber_loss(prediction, target, delta=0.75, reduction="mean")


@ts.jit(target="rocm", autodiff="reverse", wrt=("prediction", "target"))
def _rocm_smooth_l1(prediction, target):
    return ts.ops.smooth_l1_loss(
        prediction, target, beta=0.5, reduction="none")


@ts.jit(target="x86", autodiff="reverse", wrt=("parameter", "gradient"))
def _x86_sgd(parameter, gradient):
    return ts.ops.sgd(parameter, gradient, lr=0.125)


@ts.jit(target="rocm", autodiff="reverse", wrt=("parameter", "gradient"))
def _rocm_sgd(parameter, gradient):
    return ts.ops.sgd(parameter, gradient, lr=0.125)


def test_x86_public_huber_backward_runs_avx512():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime unavailable")
    errors = np.asarray(
        [-1.5, -0.75, -0.2, 0.0, 0.2, 0.75, 1.5], np.float32)
    target = np.zeros_like(errors)
    seed = np.asarray(1.25, np.float32)
    dp, dt = _x86_huber.native_backward(
        errors, target, out_cotangents=seed)
    local = np.where(np.abs(errors) <= 0.75, errors, 0.75 * np.sign(errors))
    expected = local * np.float32(seed / errors.size)
    np.testing.assert_allclose(dp, expected, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(dt, -expected, atol=1e-7, rtol=1e-7)
    assert _x86_huber.last_backward_execution["compiler_path"] == (
        "x86_regression_loss_bwd_compiled")


def test_rocm_public_smooth_l1_backward_runs_gfx1151():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/GPU unavailable")
    errors = np.asarray(
        [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0], np.float32)
    target = np.zeros_like(errors)
    seed = np.linspace(0.5, 1.1, errors.size, dtype=np.float32)
    dp, dt = _rocm_smooth_l1.native_backward(
        errors, target, out_cotangents=seed)
    local = np.where(np.abs(errors) < 0.5, errors / 0.5, np.sign(errors))
    expected = local * seed
    np.testing.assert_allclose(dp, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(dt, -expected, atol=2e-6, rtol=2e-6)
    assert _rocm_smooth_l1.last_backward_execution["compiler_path"] == (
        "rocm_regression_loss_bwd_compiled")


@pytest.mark.parametrize(
    "compiled,target,evidence",
    [
        (_x86_sgd, "x86", "x86_avx512"),
        (_rocm_sgd, "rocm", "rocm_gfx1151"),
    ],
)
def test_public_sgd_backward_composes_native_optimizer(
        compiled, target, evidence):
    from tessera import runtime as rt
    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime unavailable")
    if target == "rocm" and (
        rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available()
    ):
        pytest.skip("ROCm compiler/GPU unavailable")
    parameter = np.linspace(-1.0, 1.0, 35, dtype=np.float32).reshape(5, 7)
    gradient = np.ones_like(parameter)
    seed = np.linspace(0.25, 1.25, 35, dtype=np.float32).reshape(5, 7)
    dp, dg = compiled.native_backward(
        parameter, gradient, out_cotangents=seed)
    np.testing.assert_allclose(dp, seed, atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(dg, -0.125 * seed, atol=1e-7, rtol=1e-7)
    assert compiled.last_backward_execution["implementation"] == "dedicated"
    assert compiled.last_backward_execution["evidence_target"] == evidence
