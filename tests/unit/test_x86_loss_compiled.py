"""x86 pointwise-loss lane — mse / mae / huber / smooth_l1 / log_cosh over
(pred, target), loaded from libtessera_x86_elementwise.so. The per-element loss
runs on the AVX-512 loss kernel; reduction (none/mean/sum) on the AVX-512 reduce
kernel. The CPU lane for the S11 pointwise losses (previously reference-only).

Reachable through `runtime.launch()` via `compiler_path="x86_loss_compiled"`.
f32; validated vs the tessera.losses numpy reference at 2e-5.

Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_loss_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["pred", "target"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["pred", "target"], "kwargs": kwargs}],
    })


_REF = {
    "tessera.mse_loss": (losses.mse_loss, {}),
    "tessera.loss.mse": (losses.mse_loss, {}),
    "tessera.mae_loss": (losses.mae_loss, {}),
    "tessera.loss.mae": (losses.mae_loss, {}),
    "tessera.huber_loss": (losses.huber_loss, {"delta": 1.0}),
    "tessera.loss.huber": (losses.huber_loss, {"delta": 1.0}),
    "tessera.smooth_l1_loss": (losses.smooth_l1_loss, {"beta": 1.0}),
    "tessera.loss.smooth_l1": (losses.smooth_l1_loss, {"beta": 1.0}),
    "tessera.log_cosh_loss": (losses.log_cosh_loss, {}),
}


@pytest.mark.parametrize("op_name", list(_REF))
@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("shape", [(64,), (8, 33), (3, 5, 7)])
def test_loss_matches_reference(op_name, reduction, shape):
    rt = _x86_or_skip()
    ref_fn, params = _REF[op_name]
    rng = np.random.default_rng(5 + len(shape) + int(np.prod(shape)))
    pred = (rng.standard_normal(shape) * 2).astype(np.float32)
    target = (rng.standard_normal(shape) * 2).astype(np.float32)
    kwargs = dict(params)
    kwargs["reduction"] = reduction
    res = rt.launch(_artifact(rt, op_name, kwargs), (pred, target))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_loss_compiled"
    ref = np.asarray(ref_fn(pred, target, **{**params, "reduction": reduction}),
                     dtype=np.float32)
    out = np.asarray(res["output"]).astype(np.float32)
    np.testing.assert_allclose(out, ref, atol=2e-5, rtol=2e-5)


def test_huber_custom_delta():
    rt = _x86_or_skip()
    rng = np.random.default_rng(2)
    pred = (rng.standard_normal((64,)) * 3).astype(np.float32)
    target = (rng.standard_normal((64,)) * 3).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.huber_loss",
                              {"delta": 0.5, "reduction": "mean"}),
                    (pred, target))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(losses.huber_loss(pred, target, delta=0.5))
    np.testing.assert_allclose(np.float32(res["output"]), ref, atol=2e-5,
                               rtol=2e-5)


@pytest.mark.parametrize("tshape", [(8, 1), (1, 5), (1,), ()])
def test_loss_broadcastable_target(tshape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(hash(tshape) % 2**31)
    pred = (rng.standard_normal((8, 5)) * 2).astype(np.float32)
    target = (rng.standard_normal(tshape) * 2).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.mse_loss", {"reduction": "none"}),
                    (pred, target))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(losses.mse_loss(pred, target, reduction="none"),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


def test_loss_non_broadcastable_rejected():
    rt = _x86_or_skip()
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="do not broadcast"):
        rt._execute_x86_compiled_loss(
            _artifact(rt, "tessera.mse_loss", {}), (a, b))


def test_loss_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_loss_compiled executor"):
        rt._execute_x86_compiled_loss(
            _artifact(rt, "tessera.softmax", {}), (a, a))


@pytest.mark.parametrize("op_name,param_kw,param", [
    ("tessera.loss.mse", None, 1.0),
    ("tessera.loss.mae", None, 1.0),
    ("tessera.loss.huber", "delta", 0.75),
    ("tessera.loss.smooth_l1", "beta", 0.5),
])
@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_regression_backward_matches_reference(
        op_name, param_kw, param, reduction):
    rt = _x86_or_skip()
    errors = np.asarray([
        -2 * param, -param, -0.25 * param, 0.0, 0.25 * param,
        param, 2 * param, -0.0, 0.1, -0.1, 0.3, -0.3,
    ], np.float32).reshape(3, 4)
    target = np.zeros((3, 4), dtype=np.float32)
    prediction = target + errors
    dy = (
        np.linspace(0.5, 1.6, 12, dtype=np.float32).reshape(3, 4)
        if reduction == "none" else np.asarray(1.25, np.float32))
    kwargs = {
        "reduction": reduction,
        **({param_kw: param} if param_kw else {}),
    }
    artifact = rt.RuntimeArtifact(metadata={
        "target": "x86",
        "compiler_path": "x86_regression_loss_bwd_compiled",
        "executable": True,
        "execution_kind": "native_cpu",
        "autodiff_phase": "backward",
        "out_cotangent": "dy",
        "arg_names": ["prediction", "target", "dy"],
        "output_names": ["d_prediction", "d_target"],
        "ops": [{
            "op_name": op_name,
            "result": "loss",
            "operands": ["prediction", "target"],
            "kwargs": kwargs,
        }],
    })
    result = rt.launch(artifact, (prediction, target, dy))
    assert result["ok"] is True, result.get("reason")
    dp, dt = result["output"]
    if op_name.endswith(".mse"):
        local = 2.0 * errors
    elif op_name.endswith(".mae"):
        local = np.sign(errors)
    elif op_name.endswith(".huber"):
        local = np.where(
            np.abs(errors) <= param, errors, param * np.sign(errors))
    else:
        local = np.where(
            np.abs(errors) < param, errors / param, np.sign(errors))
    scale = 1.0 / errors.size if reduction == "mean" else 1.0
    expected = local * dy * scale
    np.testing.assert_allclose(dp, expected, atol=2e-6, rtol=2e-6)
    np.testing.assert_allclose(dt, -expected, atol=2e-6, rtol=2e-6)
