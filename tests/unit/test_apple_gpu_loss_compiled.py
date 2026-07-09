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

from tessera import losses
from tessera import runtime as rt


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
    assert res["execution_kind"] == "native_gpu"
    return res["output"]


def _pair(seed):
    rng = np.random.default_rng(seed)
    pred = rng.standard_normal((4, 5)).astype(np.float32)
    target = rng.standard_normal((4, 5)).astype(np.float32)
    return pred, target


def test_apple_gpu_mse_mae_match_reference():
    pred, target = _pair(0xA10)
    for reduction in ("none", "mean", "sum"):
        np.testing.assert_allclose(
            _launch("tessera.mse_loss", ("p", "t"), (pred, target),
                    {"reduction": reduction}),
            losses.mse_loss(pred, target, reduction=reduction),
            atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(
            _launch("tessera.mae_loss", ("p", "t"), (pred, target),
                    {"reduction": reduction}),
            losses.mae_loss(pred, target, reduction=reduction),
            atol=1e-5, rtol=1e-5)


def test_apple_gpu_huber_smooth_l1_match_reference():
    pred, target = _pair(0xA11)
    for delta in (0.5, 1.0, 2.0):
        np.testing.assert_allclose(
            _launch("tessera.huber_loss", ("p", "t"), (pred, target),
                    {"delta": delta, "reduction": "mean"}),
            losses.huber_loss(pred, target, delta=delta, reduction="mean"),
            atol=1e-5, rtol=1e-5)
    for beta in (0.5, 1.0):
        np.testing.assert_allclose(
            _launch("tessera.smooth_l1_loss", ("p", "t"), (pred, target),
                    {"beta": beta, "reduction": "sum"}),
            losses.smooth_l1_loss(pred, target, beta=beta, reduction="sum"),
            atol=1e-5, rtol=1e-5)


def test_apple_gpu_log_cosh_matches_reference():
    pred, target = _pair(0xA12)
    for reduction in ("none", "mean", "sum"):
        np.testing.assert_allclose(
            _launch("tessera.log_cosh_loss", ("p", "t"), (pred, target),
                    {"reduction": reduction}),
            losses.log_cosh_loss(pred, target, reduction=reduction),
            atol=1e-5, rtol=1e-5)


def test_apple_gpu_loss_broadcasts_scalar_target():
    pred, _ = _pair(0xA13)
    target = np.float32(0.25)
    np.testing.assert_allclose(
        _launch("tessera.mse_loss", ("p", "t"), (pred, target),
                {"reduction": "mean"}),
        losses.mse_loss(pred, target, reduction="mean"),
        atol=1e-5, rtol=1e-5)
