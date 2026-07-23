"""Exact-host compiled RMSNorm/LayerNorm backward for ROCm and AVX-512."""

from __future__ import annotations

import numpy as np
import pytest


def _artifact(rt, target, op_name, *, affine, beta=False, eps=1e-5):
    operands = ["x"]
    if affine:
        operands.append("gamma")
    if beta:
        operands.append("beta")
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": (
            f"{target}_layer_norm_bwd_compiled"
            if op_name == "tessera.layer_norm"
            else f"{target}_rmsnorm_bwd_compiled"
        ),
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "executable": True,
        "autodiff_phase": "backward",
        "out_cotangent": "dy",
        "arg_names": operands + ["dy"],
        "output_names": ["dx"] + (["dgamma"] if affine else [])
                        + (["dbeta"] if beta else []),
        "ops": [{
            "op_name": op_name,
            "result": "y",
            "operands": operands,
            "kwargs": {"eps": eps},
        }],
    })


def _reference(x, dy, op_name, eps, gamma=None, beta=False):
    x = np.asarray(x, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float32)
    if op_name == "tessera.layer_norm":
        mean = np.mean(x, axis=-1, keepdims=True)
        inv = 1.0 / np.sqrt(
            np.mean((x - mean) ** 2, axis=-1, keepdims=True) + eps)
        z = (x - mean) * inv
    else:
        inv = 1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
        z = x * inv
    dz = dy * (np.asarray(gamma, dtype=np.float32) if gamma is not None else 1.0)
    mean_dz = (np.mean(dz, axis=-1, keepdims=True)
               if op_name == "tessera.layer_norm" else 0.0)
    dx = inv * (dz - mean_dz - z * np.mean(dz * z, axis=-1, keepdims=True))
    result = [dx]
    if gamma is not None:
        axes = tuple(range(x.ndim - 1))
        result.append(np.sum(dy * z, axis=axes))
    if beta:
        axes = tuple(range(x.ndim - 1))
        result.append(np.sum(dy, axis=axes))
    return tuple(result)


def _target_or_skip(target):
    from tessera import runtime as rt
    if target == "rocm":
        if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
            pytest.skip("ROCm compiler/GPU unavailable")
    elif not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime unavailable")
    return rt


@pytest.mark.parametrize("target", ["x86", "rocm"])
@pytest.mark.parametrize("op_name", ["tessera.rmsnorm", "tessera.layer_norm"])
@pytest.mark.parametrize("shape", [(3, 17), (2, 5, 64), (7, 300)])
@pytest.mark.parametrize("affine", [False, True])
def test_compiled_norm_backward_f32(target, op_name, shape, affine):
    rt = _target_or_skip(target)
    rng = np.random.default_rng(211 + len(shape) + shape[-1] + len(op_name))
    x = rng.standard_normal(shape).astype(np.float32)
    dy = rng.standard_normal(shape).astype(np.float32)
    gamma = rng.uniform(0.5, 1.5, shape[-1]).astype(np.float32)
    beta = rng.standard_normal(shape[-1]).astype(np.float32)
    has_beta = affine and op_name == "tessera.layer_norm"
    inputs = [x] + ([gamma] if affine else []) + ([beta] if has_beta else []) + [dy]
    actual = rt.launch(
        _artifact(rt, target, op_name, affine=affine, beta=has_beta), tuple(inputs)
    )["output"]
    expected = _reference(
        x, dy, op_name, 1e-5, gamma if affine else None, has_beta)
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got, np.float32), want,
                                   atol=7e-5, rtol=7e-5)


@pytest.mark.parametrize("dtype,tol", [(np.float16, 2e-2), ("bf16", 8e-2)])
@pytest.mark.parametrize("op_name", ["tessera.rmsnorm", "tessera.layer_norm"])
def test_rocm_affine_norm_backward_low_precision(dtype, tol, op_name):
    rt = _target_or_skip("rocm")
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(227 + len(op_name))
    shape = (5, 37)
    x = rng.standard_normal(shape).astype(dtype)
    dy = rng.standard_normal(shape).astype(dtype)
    gamma = rng.uniform(0.5, 1.5, shape[-1]).astype(dtype)
    beta = rng.standard_normal(shape[-1]).astype(dtype)
    has_beta = op_name == "tessera.layer_norm"
    inputs = [x, gamma] + ([beta] if has_beta else []) + [dy]
    actual = rt.launch(
        _artifact(rt, "rocm", op_name, affine=True, beta=has_beta), tuple(inputs)
    )["output"]
    expected = _reference(x, dy, op_name, 1e-5, gamma, has_beta)
    for got, want in zip(actual, expected):
        np.testing.assert_allclose(np.asarray(got, np.float32), want,
                                   atol=tol, rtol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float16, "bf16"])
@pytest.mark.parametrize("op_name", ["tessera.rmsnorm", "tessera.layer_norm"])
def test_rocm_affine_gradients_are_bitwise_reproducible(dtype, op_name):
    rt = _target_or_skip("rocm")
    if dtype == "bf16":
        dtype = pytest.importorskip("ml_dtypes").bfloat16
    rng = np.random.default_rng(233 + len(op_name))
    shape = (257, 129)
    x = rng.standard_normal(shape).astype(dtype)
    dy = rng.standard_normal(shape).astype(dtype)
    gamma = rng.uniform(0.5, 1.5, shape[-1]).astype(dtype)
    beta = rng.standard_normal(shape[-1]).astype(dtype)
    has_beta = op_name == "tessera.layer_norm"
    inputs = (x, gamma, beta, dy) if has_beta else (x, gamma, dy)
    artifact = _artifact(
        rt, "rocm", op_name, affine=True, beta=has_beta,
    )
    baseline = rt.launch(artifact, inputs)["output"][1:]
    for _ in range(8):
        repeated = rt.launch(artifact, inputs)["output"][1:]
        assert len(repeated) == len(baseline)
        for got, first in zip(repeated, baseline):
            assert np.array_equal(got, first)


@pytest.mark.parametrize("target", ["x86", "rocm"])
def test_compiled_layernorm_backward_large_offset_is_stable(target):
    rt = _target_or_skip(target)
    rng = np.random.default_rng(239)
    x = (1.0e4 + rng.standard_normal((4, 128))).astype(np.float32)
    dy = rng.standard_normal(x.shape).astype(np.float32)
    (actual,) = rt.launch(
        _artifact(rt, target, "tessera.layer_norm", affine=False), (x, dy)
    )["output"]
    (expected,) = _reference(x, dy, "tessera.layer_norm", 1e-5)
    np.testing.assert_allclose(actual, expected, atol=3e-3, rtol=3e-3)


def test_rocm_backward_cache_is_shape_and_affine_independent():
    rt = _target_or_skip("rocm")
    rt._rocm_norm_bwd_hsaco_cache.clear()
    rng = np.random.default_rng(241)
    for shape, affine in [((2, 17), False), ((7, 64), True), ((3, 5, 300), True)]:
        x = rng.standard_normal(shape).astype(np.float32)
        dy = rng.standard_normal(shape).astype(np.float32)
        gamma = np.ones(shape[-1], np.float32)
        inputs = (x, gamma, dy) if affine else (x, dy)
        rt.launch(
            _artifact(rt, "rocm", "tessera.rmsnorm", affine=affine), inputs)
    assert len(rt._rocm_norm_bwd_hsaco_cache) == 1


def test_backward_contract_requires_phase_and_cotangent():
    from tessera import runtime as rt
    artifact = _artifact(
        rt, "x86", "tessera.rmsnorm", affine=False)
    artifact.metadata.pop("autodiff_phase")
    with pytest.raises(ValueError, match="autodiff_phase"):
        rt._execute_x86_compiled_norm_backward(
            artifact, (np.ones((2, 4), np.float32),) * 2)
