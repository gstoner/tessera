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
    "tessera.mae_loss": (losses.mae_loss, {}),
    "tessera.huber_loss": (losses.huber_loss, {"delta": 1.0}),
    "tessera.smooth_l1_loss": (losses.smooth_l1_loss, {"beta": 1.0}),
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


def test_loss_shape_mismatch_rejected():
    rt = _x86_or_skip()
    a = np.zeros((4, 8), np.float32)
    b = np.zeros((4, 9), np.float32)
    with pytest.raises(ValueError, match="matching shapes"):
        rt._execute_x86_compiled_loss(
            _artifact(rt, "tessera.mse_loss", {}), (a, b))


def test_loss_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_loss_compiled executor"):
        rt._execute_x86_compiled_loss(
            _artifact(rt, "tessera.softmax", {}), (a, a))
