"""Apple GPU pointwise-regression loss lane.

Parity with ``test_x86_structured_compute_compiled`` / the ROCm loss lane: mse,
mae, huber, smooth_l1, and log_cosh reach an executable ``apple_gpu`` path via
``runtime.launch()`` and match ``tessera.losses``. The residual (pred-target)
and none/mean/sum reduction compose on the MPSGraph binary + reduce lanes (with
a numpy fallback when Metal is unavailable); the piecewise/transcendental middle
for huber/smooth_l1/log_cosh is host-side.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses
from tessera import runtime as rt
from tests._support.apple import assert_native_apple_gpu, assert_reference_cpu


def _artifact(op_name, operands, kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu",
        "compiler_path": "apple_gpu_loss_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": list(operands),
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": list(operands),
            "kwargs": dict(kwargs or {}),
        }],
    })


def _launch(op_name, names, args, kwargs=None):
    res = rt.launch(_artifact(op_name, names, kwargs), args)
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_loss_compiled"
    return res


def _output(op_name, names, args, kwargs=None):
    return _launch(op_name, names, args, kwargs)["output"]


def _pair(seed):
    rng = np.random.default_rng(seed)
    pred = rng.standard_normal((4, 5)).astype(np.float32)
    target = rng.standard_normal((4, 5)).astype(np.float32)
    return pred, target


def test_apple_gpu_mse_mae_match_reference():
    pred, target = _pair(0xA10)
    for reduction in ("none", "mean", "sum"):
        np.testing.assert_allclose(
            _output("tessera.mse_loss", ("p", "t"), (pred, target),
                    {"reduction": reduction}),
            losses.mse_loss(pred, target, reduction=reduction),
            atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            _output("tessera.mae_loss", ("p", "t"), (pred, target),
                    {"reduction": reduction}),
            losses.mae_loss(pred, target, reduction=reduction),
            atol=1e-5, rtol=1e-5)


def test_apple_gpu_huber_smooth_l1_match_reference():
    pred, target = _pair(0xA11)
    for delta in (0.5, 1.0, 2.0):
        np.testing.assert_allclose(
            _output("tessera.huber_loss", ("p", "t"), (pred, target),
                    {"delta": delta, "reduction": "mean"}),
            losses.huber_loss(pred, target, delta=delta, reduction="mean"),
            atol=1e-5, rtol=1e-5)
    for beta in (0.5, 1.0):
        np.testing.assert_allclose(
            _output("tessera.smooth_l1_loss", ("p", "t"), (pred, target),
                    {"beta": beta, "reduction": "sum"}),
            losses.smooth_l1_loss(pred, target, beta=beta, reduction="sum"),
            atol=1e-5, rtol=1e-5)


def test_apple_gpu_log_cosh_matches_reference():
    pred, target = _pair(0xA12)
    for reduction in ("none", "mean", "sum"):
        np.testing.assert_allclose(
            _output("tessera.log_cosh_loss", ("p", "t"), (pred, target),
                    {"reduction": reduction}),
            losses.log_cosh_loss(pred, target, reduction=reduction),
            atol=1e-5, rtol=1e-5)


def test_apple_gpu_loss_broadcasts_scalar_target():
    pred, _ = _pair(0xA13)
    target = np.float32(0.25)
    np.testing.assert_allclose(
        _output("tessera.mse_loss", ("p", "t"), (pred, target),
                {"reduction": "mean"}),
        losses.mse_loss(pred, target, reduction="mean"),
        atol=1e-5, rtol=1e-5)


@pytest.mark.hardware_apple_gpu
def test_mse_and_mae_f32_report_native_gpu_on_metal():
    pred, target = _pair(0xC3)
    for op_name, reference in (
        ("tessera.mse_loss", losses.mse_loss),
        ("tessera.mae_loss", losses.mae_loss),
    ):
        result = _launch(op_name, ("p", "t"), (pred, target), {"reduction": "mean"})
        np.testing.assert_allclose(
            result["output"], reference(pred, target, reduction="mean"),
            atol=1e-5, rtol=1e-5)
        assert_native_apple_gpu(result, compiler_path="apple_gpu_loss_compiled")


def test_mse_forced_mpsgraph_miss_is_reference_cpu(monkeypatch):
    monkeypatch.setattr(rt, "_apple_gpu_mpsgraph_reduce_f32", lambda: None)
    pred, target = _pair(0xC4)
    for op_name, reference in (
        ("tessera.mse_loss", losses.mse_loss),
        ("tessera.mae_loss", losses.mae_loss),
    ):
        result = _launch(op_name, ("p", "t"), (pred, target), {"reduction": "mean"})
        np.testing.assert_allclose(
            result["output"], reference(pred, target, reduction="mean"),
            atol=1e-5, rtol=1e-5)
        assert_reference_cpu(result)
