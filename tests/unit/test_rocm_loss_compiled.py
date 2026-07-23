"""Compiler-generated pointwise regression losses on gfx1151 — the ROCm mirror
of the x86_loss lane. The `tessera_rocm.pointwise_loss` directive expands (via
generate-rocm-pointwise-loss-kernel) into a flat per-element kernel computing
mse/mae/huber/smooth_l1/log_cosh; the runtime reduces (none/mean/sum). exp/log1p
lower through math->ROCDL.

Reachable through `runtime.launch()` via `compiler_path="rocm_loss_compiled"`.
Validated vs tessera.losses on gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import os
import numpy as np
import pytest

from tessera import losses


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built (ninja -C build tessera-opt)")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_loss_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["pred", "target"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["pred", "target"], "kwargs": kwargs}],
    })


_REF = {
    "tessera.mse_loss": (losses.mse_loss, {}),
    "tessera.loss.mse": (losses.mse_loss, {}),
    "tessera.mae_loss": (losses.mae_loss, {}),
    "tessera.huber_loss": (losses.huber_loss, {"delta": 1.0}),
    "tessera.smooth_l1_loss": (losses.smooth_l1_loss, {"beta": 1.0}),
    "tessera.log_cosh_loss": (losses.log_cosh_loss, {}),
}


@pytest.mark.parametrize("op_name", list(_REF))
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("shape", [(64,), (8, 33)])
def test_loss_matches_reference(op_name, reduction, shape):
    rt = _rocm_or_skip()
    ref_fn, params = _REF[op_name]
    rng = np.random.default_rng(5 + len(shape) + int(np.prod(shape)))
    pred = (rng.standard_normal(shape) * 2).astype(np.float32)
    target = (rng.standard_normal(shape) * 2).astype(np.float32)
    kwargs = dict(params)
    kwargs["reduction"] = reduction
    res = rt.launch(_artifact(rt, op_name, kwargs), (pred, target))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_loss_compiled"
    ref = np.asarray(ref_fn(pred, target, **{**params, "reduction": reduction}),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


def test_canonical_dynamic_mse_reuses_shape_independent_hsaco():
    rt = _rocm_or_skip()
    rt._rocm_pointwise_loss_hsaco_cache.clear()
    rng = np.random.default_rng(20260723)
    for shape in ((7, 19), (3, 5, 11)):
        pred = rng.standard_normal(shape).astype(np.float32)
        target = rng.standard_normal(shape).astype(np.float32)
        result = rt.launch(
            _artifact(rt, "tessera.loss.mse", {"reduction": "mean"}),
            (pred, target),
        )
        assert result["ok"] is True, result.get("reason")
        np.testing.assert_allclose(
            np.asarray(result["output"], dtype=np.float32),
            np.asarray(losses.mse_loss(pred, target), dtype=np.float32),
            atol=2e-5,
            rtol=2e-5,
        )
    assert len(rt._rocm_pointwise_loss_hsaco_cache) == 1


# ── GPU-free codegen gate (needs only tessera-opt) ───────────────────────────
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

_OPT = Path(os.environ.get(
    "TESSERA_OPT",
    Path(__file__).resolve().parents[2] / "build/tools/tessera-opt/tessera-opt",
))


@pytest.mark.parametrize("kind", [0, 1, 2, 3, 4])
def test_loss_codegen_and_lowers(kind):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    d = ('module {\n  "tessera_rocm.pointwise_loss"() {name = "pl", '
         f'dtype = "f32", kind = {kind} : i64, param = 1.0 : f32}} '
         ': () -> ()\n}\n')
    ir = subprocess.run([str(_OPT), "-", "--generate-rocm-pointwise-loss-kernel"],
                        input=d, capture_output=True, text=True)
    assert ir.returncode == 0, ir.stderr
    assert "gpu.func @pl" in ir.stdout
    low = subprocess.run(
        [str(_OPT), "-",
         "--pass-pipeline=builtin.module(generate-rocm-pointwise-loss-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=d, capture_output=True, text=True)
    assert low.returncode == 0 and "llvm." in low.stdout


@pytest.mark.parametrize("kind", [0, 1, 2, 3])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_regression_backward_codegen_and_lowers(kind, reduction):
    if not _OPT.is_file():
        pytest.skip("build tessera-opt: ninja -C build tessera-opt")
    directive = (
        'module {\n  "tessera_rocm.pointwise_loss"() {name = "pl_bwd", '
        f'dtype = "f32", kind = {kind} : i64, param = 0.75 : f32, '
        'backward = true, '
        f'reduction = "{reduction}"}} : () -> ()\n}}\n'
    )
    generated = subprocess.run(
        [str(_OPT), "-", "--generate-rocm-pointwise-loss-kernel"],
        input=directive, capture_output=True, text=True,
    )
    assert generated.returncode == 0, generated.stderr
    assert "gpu.func @pl_bwd" in generated.stdout
    assert generated.stdout.count("memref.store") == 2
    lowered = subprocess.run(
        [str(_OPT), "-",
         "--pass-pipeline=builtin.module(generate-rocm-pointwise-loss-kernel,"
         "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
         "reconcile-unrealized-casts))"],
        input=directive, capture_output=True, text=True,
    )
    assert lowered.returncode == 0, lowered.stderr
    assert "llvm." in lowered.stdout


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
@pytest.mark.parametrize("shape", [(263,), (7, 37)])
@pytest.mark.parametrize("op_name,param_kw,param", [
    ("tessera.loss.mse", None, 1.0),
    ("tessera.loss.mae", None, 1.0),
    ("tessera.loss.huber", "delta", 0.75),
    ("tessera.loss.smooth_l1", "beta", 0.5),
])
def test_regression_backward_matches_reference(
        reduction, shape, op_name, param_kw, param):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(20260724 + len(shape))
    prediction = rng.standard_normal(shape).astype(np.float32)
    target = rng.standard_normal(shape).astype(np.float32)
    dy = (rng.standard_normal(shape).astype(np.float32)
          if reduction == "none" else np.asarray(0.75, np.float32))
    artifact = rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_regression_loss_bwd_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "execution_mode": "hip_runtime",
        "autodiff_phase": "backward",
        "out_cotangent": "dy",
        "arg_names": ["prediction", "target", "dy"],
        "output_names": ["d_prediction", "d_target"],
        "ops": [{
            "op_name": op_name,
            "result": "loss",
            "operands": ["prediction", "target"],
            "kwargs": {
                "reduction": reduction,
                **({param_kw: param} if param_kw else {}),
            },
        }],
    })
    launched = rt.launch(artifact, (prediction, target, dy))
    assert launched["ok"] is True, launched.get("reason")
    d_prediction, d_target = launched["output"]
    error = prediction - target
    if op_name.endswith(".mse"):
        local = 2.0 * error
    elif op_name.endswith(".mae"):
        local = np.sign(error)
    elif op_name.endswith(".huber"):
        local = np.where(
            np.abs(error) <= param, error, param * np.sign(error))
    else:
        local = np.where(
            np.abs(error) < param, error / param, np.sign(error))
    scale = np.float32(1.0 / prediction.size if reduction == "mean" else 1.0)
    expected = local * dy * scale
    np.testing.assert_allclose(d_prediction, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_allclose(d_target, -expected, atol=2e-5, rtol=2e-5)
