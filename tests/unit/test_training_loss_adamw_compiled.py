"""Exact fused loss-VJP → AdamW coverage on AVX-512 and gfx1151."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess

import numpy as np
import pytest


def _artifact(rt, target: str, kind: str, reduction: str, step: int = 7):
    parameter = 0.75 if kind == "huber" else 0.5
    return rt.RuntimeArtifact(metadata={
        "target": target,
        "compiler_path": f"{target}_training_loss_adamw_compiled",
        "executable": True,
        "execution_kind": "native_gpu" if target == "rocm" else "native_cpu",
        "arg_names": [
            "prediction", "target", "dy", "parameter", "moment1", "moment2",
        ],
        "output_names": [
            "new_parameter", "new_moment1", "new_moment2", "d_target",
        ],
        "ops": [{
            "op_name": "tessera.training.loss_adamw",
            "result": [
                "new_parameter", "new_moment1", "new_moment2", "d_target",
            ],
            "operands": [
                "prediction", "target", "dy", "parameter",
                "moment1", "moment2",
            ],
            "kwargs": {
                "kind": kind,
                "parameter": parameter,
                "reduction": reduction,
                "lr": 0.002,
                "beta1": 0.8,
                "beta2": 0.95,
                "eps": 1.0e-7,
                "weight_decay": 0.01,
                "step": step,
            },
        }],
    })


def _loss_gradient(prediction, target, dy, kind, reduction):
    error = prediction - target
    transition = 0.75 if kind == "huber" else 0.5
    if kind == "mse":
        local = 2.0 * error
        target_local = None
    elif kind == "bce":
        local = np.where(
            prediction >= 0.0,
            1.0 / (1.0 + np.exp(-prediction)),
            np.exp(prediction) / (1.0 + np.exp(prediction)),
        ) - target
        target_local = -prediction
    elif kind == "mae":
        local = np.sign(error)
        target_local = None
    elif kind == "huber":
        local = np.where(
            np.abs(error) <= transition,
            error,
            transition * np.sign(error),
        )
        target_local = None
    else:
        local = np.where(
            np.abs(error) < transition,
            error / transition,
            np.sign(error),
        )
        target_local = None
    scale = np.float32(
        1.0 / prediction.size if reduction == "mean" else 1.0
    )
    gradient = (local * dy * scale).astype(np.float32)
    target_gradient = (
        target_local * dy * scale
        if target_local is not None else -gradient
    ).astype(np.float32)
    return gradient, target_gradient


@pytest.mark.parametrize("target", ["x86", "rocm"])
@pytest.mark.parametrize("kind", ["mse", "bce", "mae", "huber", "smooth_l1"])
@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_training_loss_adamw_matches_unfused_contract(
    target, kind, reduction
):
    from tessera import runtime as rt

    if target == "x86" and not rt._x86_elementwise_available():
        pytest.skip("AVX-512 runtime library is unavailable")
    if target == "rocm":
        if rt._tessera_opt_path() is None:
            pytest.skip("LLVM 23 tessera-opt is unavailable")
        if not rt._rocm_wmma_runtime_available():
            pytest.skip("gfx1151 HIP runtime is unavailable")
    errors = np.tile(np.asarray([
        -1.5, -0.75, -0.5, -0.0, 0.0, 0.5, 0.75, 1.5,
        -0.1, 0.1, -2.0, 2.0,
    ], np.float32), 5).reshape(3, 20)
    target_value = np.zeros_like(errors)
    prediction = target_value + errors
    parameter = np.linspace(
        -1.0, 1.0, errors.size, dtype=np.float32
    ).reshape(errors.shape)
    moment1 = np.linspace(
        -0.2, 0.2, errors.size, dtype=np.float32
    ).reshape(errors.shape)
    moment2 = np.linspace(
        0.01, 0.3, errors.size, dtype=np.float32
    ).reshape(errors.shape)
    dy = (
        np.linspace(0.25, 1.5, errors.size, dtype=np.float32).reshape(
            errors.shape
        )
        if reduction == "none"
        else np.asarray(0.75, np.float32)
    )
    result = rt.launch(
        _artifact(rt, target, kind, reduction),
        (prediction, target_value, dy, parameter, moment1, moment2),
    )
    assert result["ok"] is True, result.get("reason")
    assert result["compiler_path"] == (
        f"{target}_training_loss_adamw_compiled"
    )
    new_parameter, new_moment1, new_moment2, target_gradient = result[
        "output"
    ]
    gradient, expected_target = _loss_gradient(
        prediction, target_value, dy, kind, reduction
    )
    beta1, beta2, step = 0.8, 0.95, 7
    expected_m1 = beta1 * moment1 + (1.0 - beta1) * gradient
    expected_m2 = beta2 * moment2 + (1.0 - beta2) * gradient * gradient
    update = (
        (expected_m1 / (1.0 - beta1**step))
        / (np.sqrt(expected_m2 / (1.0 - beta2**step)) + 1.0e-7)
        + 0.01 * parameter
    )
    expected_parameter = parameter - 0.002 * update
    tolerance = 2.0e-5 if target == "rocm" else 8.0e-6
    for actual, expected in (
        (new_parameter, expected_parameter),
        (new_moment1, expected_m1),
        (new_moment2, expected_m2),
        (target_gradient, expected_target),
    ):
        np.testing.assert_allclose(
            actual, expected, atol=tolerance, rtol=tolerance
        )


def test_rocm_training_adamw_codegen_has_four_stores_and_runtime_state():
    opt = Path(os.environ.get(
        "TESSERA_OPT",
        Path(__file__).resolve().parents[2]
        / "build/tools/tessera-opt/tessera-opt",
    ))
    if not opt.is_file():
        pytest.skip("LLVM 23 tessera-opt is unavailable")
    directive = (
        'module {\n  "tessera_rocm.pointwise_loss"() '
        '{name = "training_adamw", dtype = "f32", kind = 2 : i64, '
        'param = 7.5e-1 : f32, training_adamw = true, '
        'reduction = "mean"} : () -> ()\n}\n'
    )
    generated = subprocess.run(
        [str(opt), "-", "--generate-rocm-pointwise-loss-kernel"],
        input=directive, capture_output=True, text=True, check=False,
    )
    assert generated.returncode == 0, generated.stderr
    assert "gpu.func @training_adamw" in generated.stdout
    assert generated.stdout.count("memref.store") == 4
    signature = generated.stdout.split(
        "gpu.func @training_adamw", 1
    )[1].split("kernel", 1)[0]
    assert signature.count("memref<?xf32>") == 10


def test_rocm_training_adamw_dynamic_shapes_share_hsaco_identity():
    from tessera import runtime as rt

    if rt._tessera_opt_path() is None or not rt._rocm_wmma_runtime_available():
        pytest.skip("LLVM 23 ROCm exact-device path is unavailable")
    rt._rocm_training_loss_adamw_hsaco_cache.clear()
    for shape in ((7, 19), (3, 5, 11)):
        size = int(np.prod(shape))
        prediction = np.linspace(
            -1.0, 1.0, size, dtype=np.float32
        ).reshape(shape)
        target = np.zeros(shape, dtype=np.float32)
        parameter = np.ones(shape, dtype=np.float32)
        moment1 = np.zeros(shape, dtype=np.float32)
        moment2 = np.full(shape, 0.1, dtype=np.float32)
        result = rt.launch(
            _artifact(rt, "rocm", "smooth_l1", "mean"),
            (
                prediction, target, np.asarray(1.0, np.float32), parameter,
                moment1, moment2,
            ),
        )
        assert result["ok"] is True, result.get("reason")
    assert len(rt._rocm_training_loss_adamw_hsaco_cache) == 1
