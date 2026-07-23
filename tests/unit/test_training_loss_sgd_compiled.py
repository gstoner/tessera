"""Exact one-launch regression-loss VJP → SGD coverage on AVX-512/gfx1151."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


def _artifact(rt, target: str, kind: str, reduction: str):
    parameter = 0.75 if kind == "huber" else 0.5
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_training_loss_sgd_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "arg_names": ["prediction", "target", "dy", "parameter"],
        "output_names": ["new_parameter", "d_target"],
        "ops": [{
            "op_name": "tessera.training.loss_sgd",
            "result": ["new_parameter", "d_target"],
            "operands": ["prediction", "target", "dy", "parameter"],
            "kwargs": {
                "kind": kind,
                "parameter": parameter,
                "reduction": reduction,
                "lr": 0.125,
            },
        }],
    })


def _expected(prediction, target, dy, parameter, kind, reduction):
    error = prediction - target
    transition = 0.75 if kind == "huber" else 0.5
    if kind == "mse":
        local = 2.0 * error
        expected_target = None
    elif kind == "bce":
        local = np.where(
            prediction >= 0.0,
            1.0 / (1.0 + np.exp(-prediction)),
            np.exp(prediction) / (1.0 + np.exp(prediction)),
        ) - target
        expected_target = -prediction
    elif kind == "mae":
        local = np.sign(error)
        expected_target = None
    elif kind == "huber":
        local = np.where(
            np.abs(error) <= transition,
            error,
            transition * np.sign(error),
        )
        expected_target = None
    else:
        local = np.where(
            np.abs(error) < transition,
            error / transition,
            np.sign(error),
        )
        expected_target = None
    scale = np.float32(
        1.0 / prediction.size if reduction == "mean" else 1.0
    )
    gradient = local * dy * scale
    target_gradient = (
        expected_target * dy * scale
        if expected_target is not None else -gradient
    )
    return parameter - np.float32(0.125) * gradient, target_gradient


@pytest.mark.parametrize("target", ["x86", "rocm"])
@pytest.mark.parametrize("kind", ["mse", "bce", "mae", "huber", "smooth_l1"])
@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_training_loss_sgd_matches_unfused_contract(target, kind, reduction):
    from tessera import runtime as rt

    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime library is unavailable")
    if target == "rocm":
        if rt._tessera_opt_path() is None:
            pytest.skip("LLVM 23 tessera-opt is unavailable")
        if not rt._rocm_wmma_runtime_available():
            pytest.skip("gfx1151 HIP runtime is unavailable")
    boundary_errors = np.asarray([
        -1.5, -0.75, -0.5, -0.0, 0.0, 0.5, 0.75, 1.5,
        -0.1, 0.1, -2.0, 2.0,
    ], np.float32)
    # Sixty values exercise both SIMD/full-wave bodies and scalar/tail paths.
    errors = np.tile(boundary_errors, 5).reshape(3, 20)
    target_value = np.zeros_like(errors)
    prediction = target_value + errors
    parameter = np.linspace(-1.0, 1.0, errors.size, dtype=np.float32).reshape(
        errors.shape
    )
    dy = (
        np.linspace(0.25, 1.5, errors.size, dtype=np.float32).reshape(
            errors.shape
        )
        if reduction == "none"
        else np.asarray(0.75, np.float32)
    )
    result = rt.launch(
        _artifact(rt, target, kind, reduction),
        (prediction, target_value, dy, parameter),
    )
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == f"{target}_training_loss_sgd_compiled"
    new_parameter, target_gradient = result["output"]
    expected_parameter, expected_target = _expected(
        prediction, target_value, dy, parameter, kind, reduction
    )
    tolerance = 2e-5 if target == "rocm" else 2e-6
    np.testing.assert_allclose(
        new_parameter, expected_parameter, atol=tolerance, rtol=tolerance
    )
    np.testing.assert_allclose(
        target_gradient, expected_target, atol=tolerance, rtol=tolerance
    )


def test_rocm_training_sgd_codegen_has_two_stores_and_runtime_lr():
    opt = Path(os.environ.get(
        "TESSERA_OPT",
        Path(__file__).resolve().parents[2]
        / "build/tools/tessera-opt/tessera-opt",
    ))
    if not opt.is_file():
        pytest.skip("LLVM 23 tessera-opt is unavailable")
    directive = (
        'module {\n  "tessera_rocm.pointwise_loss"() '
        '{name = "training_sgd", dtype = "f32", kind = 2 : i64, '
        'param = 7.5e-1 : f32, training_sgd = true, '
        'reduction = "mean"} : () -> ()\n}\n'
    )
    generated = subprocess.run(
        [str(opt), "-", "--generate-rocm-pointwise-loss-kernel"],
        input=directive, capture_output=True, text=True, check=False,
    )
    assert generated.returncode == 0, generated.stderr
    assert "gpu.func @training_sgd" in generated.stdout
    assert generated.stdout.count("memref.store") == 2
    # Four inputs, two outputs, N, scale, and runtime lr.
    signature = generated.stdout.split("gpu.func @training_sgd", 1)[1].split(
        "kernel", 1
    )[0]
    assert signature.count("memref<?xf32>") == 6
    assert signature.count("f32") >= 8


def test_rocm_training_sgd_dynamic_shapes_share_hsaco_identity():
    from tessera import runtime as rt

    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("LLVM 23 ROCm exact-device path is unavailable")
    rt._rocm_training_loss_sgd_hsaco_cache.clear()
    for shape in ((7, 19), (3, 5, 11)):
        prediction = np.linspace(-1.0, 1.0, int(np.prod(shape)),
                                 dtype=np.float32).reshape(shape)
        target = np.zeros(shape, dtype=np.float32)
        parameter = np.ones(shape, dtype=np.float32)
        result = rt.launch(
            _artifact(rt, "rocm", "smooth_l1", "mean"),
            (prediction, target, np.asarray(1.0, np.float32), parameter),
        )
        assert result["ok"] is True, result.get("reason")
    assert len(rt._rocm_training_loss_sgd_hsaco_cache) == 1
