"""Exact target bindings for compiler-owned training-series adjoints."""

from __future__ import annotations

import numpy as np
import pytest

import tessera as ts


@ts.jit(target="x86", autodiff="reverse", wrt=("logits", "target"))
def _x86_bce(logits, target):
    return ts.ops.binary_cross_entropy_loss(logits, target, reduction="mean")


@ts.jit(target="rocm", autodiff="reverse", wrt=("logits", "target"))
def _rocm_bce(logits, target):
    return ts.ops.binary_cross_entropy_loss(logits, target, reduction="none")


@ts.jit(target="x86", autodiff="reverse", wrt=("logits",))
def _x86_cross_entropy(logits, target):
    return ts.ops.cross_entropy_loss(
        logits, target, reduction="mean", axis=1, ignore_index=-7,
        label_smoothing=0.2)


@ts.jit(target="rocm", autodiff="reverse", wrt=("logits",))
def _rocm_label_smoothed_cross_entropy(logits, target):
    return ts.ops.label_smoothed_cross_entropy(
        logits, target, smoothing=0.15, reduction="none",
        axis=-1, ignore_index=-9)


@ts.jit(
    target="x86", autodiff="reverse", wrt=("param", "grad", "velocity"))
def _x86_momentum(param, grad, velocity):
    return ts.ops.momentum(
        param, grad, velocity, lr=0.075, momentum=0.8)


@ts.jit(
    target="rocm", autodiff="reverse", wrt=("param", "grad", "velocity"))
def _rocm_nesterov(param, grad, velocity):
    return ts.ops.nesterov(
        param, grad, velocity, lr=0.04, momentum=0.7)


def _stable_sigmoid(x):
    out = np.empty_like(x, dtype=np.float32)
    nonnegative = x >= 0.0
    out[nonnegative] = 1.0 / (1.0 + np.exp(-x[nonnegative]))
    exp_x = np.exp(x[~nonnegative])
    out[~nonnegative] = exp_x / (1.0 + exp_x)
    return out


def test_x86_bce_backward_runs_avx512():
    from tessera import runtime as rt

    if not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime unavailable")
    logits = np.asarray(
        [-100.0, -7.0, -0.25, 0.0, 0.25, 7.0, 100.0], np.float32
    )
    target = np.asarray([0.0, 1.0, 0.25, 0.5, 0.75, 0.0, 1.0], np.float32)
    seed = np.asarray(1.75, np.float32)
    dz, dt = _x86_bce.native_backward(
        logits, target, out_cotangents=seed
    )
    scale = np.float32(seed / logits.size)
    np.testing.assert_allclose(
        dz, (_stable_sigmoid(logits) - target) * scale,
        atol=2e-7, rtol=2e-7,
    )
    np.testing.assert_allclose(dt, -logits * scale, atol=2e-7, rtol=2e-7)
    assert _x86_bce.last_backward_execution["compiler_path"] == (
        "x86_binary_loss_bwd_compiled"
    )


def test_rocm_bce_backward_runs_gfx1151():
    from tessera import runtime as rt

    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/GPU unavailable")
    logits = np.linspace(-9.0, 9.0, 35, dtype=np.float32).reshape(5, 7)
    target = np.linspace(0.0, 1.0, 35, dtype=np.float32).reshape(5, 7)
    seed = np.linspace(0.25, 1.25, 35, dtype=np.float32).reshape(5, 7)
    dz, dt = _rocm_bce.native_backward(
        logits, target, out_cotangents=seed
    )
    np.testing.assert_allclose(
        dz, (_stable_sigmoid(logits) - target) * seed,
        atol=3e-6, rtol=3e-6,
    )
    np.testing.assert_allclose(dt, -logits * seed, atol=3e-6, rtol=3e-6)
    assert _rocm_bce.last_backward_execution["compiler_path"] == (
        "rocm_binary_loss_bwd_compiled"
    )


def _class_gradient(logits, targets, smoothing, ignore_index):
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= np.sum(probabilities, axis=-1, keepdims=True)
    classes = logits.shape[-1]
    distribution = np.full_like(
        logits, smoothing / (classes - 1), dtype=np.float32)
    valid = targets != ignore_index
    safe = np.where(valid, targets, 0)
    np.put_along_axis(
        distribution, safe[..., None], np.float32(1.0 - smoothing), axis=-1)
    return np.where(valid[..., None], probabilities - distribution, 0.0)


def test_x86_class_loss_backward_handles_axis_ignore_and_smoothing():
    from tessera import runtime as rt

    if not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime unavailable")
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(4, 6, 3)).astype(np.float32)
    targets = rng.integers(0, 6, size=(4, 3), dtype=np.int64)
    targets[1, 2] = -7
    seed = np.asarray(1.25, np.float32)
    (actual,) = _x86_cross_entropy.native_backward(
        logits, targets, out_cotangents=seed)
    moved = np.moveaxis(logits, 1, -1)
    expected = _class_gradient(moved, targets, 0.2, -7)
    expected *= seed / np.count_nonzero(targets != -7)
    expected = np.moveaxis(expected, -1, 1)
    np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)


def test_rocm_label_smoothed_backward_handles_ragged_runtime_rows():
    from tessera import runtime as rt

    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/GPU unavailable")
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(5, 7)).astype(np.float32)
    targets = rng.integers(0, 7, size=(5,), dtype=np.int64)
    targets[3] = -9
    seed = np.linspace(0.5, 1.0, 5, dtype=np.float32)
    (actual,) = _rocm_label_smoothed_cross_entropy.native_backward(
        logits, targets, out_cotangents=seed)
    expected = _class_gradient(logits, targets, 0.15, -9) * seed[:, None]
    np.testing.assert_allclose(actual, expected, atol=5e-6, rtol=5e-6)


def _momentum_vjp(dp, dv, lr, momentum, nesterov):
    from_param = -lr * dp
    dg = (1.0 + momentum if nesterov else 1.0) * from_param + dv
    dvelocity = momentum * (
        (momentum if nesterov else 1.0) * from_param + dv)
    return dp, dg, dvelocity


def test_x86_momentum_backward_runs_one_avx512_launch():
    from tessera import runtime as rt

    if not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime unavailable")
    rng = np.random.default_rng(17)
    values = [rng.normal(size=(5, 19)).astype(np.float32) for _ in range(3)]
    dp = rng.normal(size=(5, 19)).astype(np.float32)
    dv = rng.normal(size=(5, 19)).astype(np.float32)
    actual = _x86_momentum.native_backward(
        *values, out_cotangents=(dp, dv))
    expected = _momentum_vjp(dp, dv, 0.075, 0.8, False)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=2e-7, rtol=2e-7)


def test_rocm_nesterov_backward_runs_one_gfx1151_launch():
    from tessera import runtime as rt

    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("ROCm compiler/GPU unavailable")
    rng = np.random.default_rng(19)
    values = [rng.normal(size=(7, 13)).astype(np.float32) for _ in range(3)]
    dp = rng.normal(size=(7, 13)).astype(np.float32)
    dv = rng.normal(size=(7, 13)).astype(np.float32)
    actual = _rocm_nesterov.native_backward(
        *values, out_cotangents=(dp, dv))
    expected = _momentum_vjp(dp, dv, 0.04, 0.7, True)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(got, want, atol=2e-6, rtol=2e-6)
