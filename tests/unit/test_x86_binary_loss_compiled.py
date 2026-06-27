"""x86 binary-cross-entropy loss lane — bce / asymmetric_bce over (logits,
targets), loaded from libtessera_x86_elementwise.so. Per-element loss on the
AVX-512 binary-loss kernel (stable softplus form), reduction on the reduce
kernel. The CPU lane for these S11 losses (previously reference-only).

Reachable through `runtime.launch()` via `compiler_path="x86_binary_loss_compiled"`.
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
        "target": "x86", "compiler_path": "x86_binary_loss_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["logits", "targets"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o",
                 "operands": ["logits", "targets"], "kwargs": kwargs}],
    })


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("shape", [(64,), (8, 33), (3, 5, 7)])
def test_bce_matches_reference(reduction, shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(5 + len(shape) + int(np.prod(shape)))
    z = (rng.standard_normal(shape) * 3).astype(np.float32)
    t = rng.integers(0, 2, size=shape).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.binary_cross_entropy_loss",
                              {"reduction": reduction}), (z, t))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_binary_loss_compiled"
    ref = np.asarray(losses.binary_cross_entropy_loss(z, t, reduction=reduction),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
@pytest.mark.parametrize("pw,nw", [(1.0, 1.0), (2.0, 0.5), (0.25, 3.0)])
def test_asymmetric_bce_matches_reference(reduction, pw, nw):
    rt = _x86_or_skip()
    rng = np.random.default_rng(11 + int(pw * 10 + nw))
    z = (rng.standard_normal((6, 20)) * 4).astype(np.float32)
    t = rng.integers(0, 2, size=(6, 20)).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.asymmetric_bce",
                              {"pos_weight": pw, "neg_weight": nw,
                               "reduction": reduction}), (z, t))
    assert res["ok"] is True, res.get("reason")
    ref = np.asarray(losses.asymmetric_bce(z, t, pos_weight=pw, neg_weight=nw,
                                           reduction=reduction), dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               atol=2e-5, rtol=2e-5)


def test_asymmetric_bce_weights_one_equals_bce():
    rt = _x86_or_skip()
    rng = np.random.default_rng(3)
    z = (rng.standard_normal((64,)) * 3).astype(np.float32)
    t = rng.integers(0, 2, size=(64,)).astype(np.float32)
    a = rt.launch(_artifact(rt, "tessera.asymmetric_bce",
                            {"pos_weight": 1.0, "neg_weight": 1.0,
                             "reduction": "none"}), (z, t))
    b = rt.launch(_artifact(rt, "tessera.binary_cross_entropy_loss",
                            {"reduction": "none"}), (z, t))
    np.testing.assert_allclose(np.asarray(a["output"]).astype(np.float32),
                               np.asarray(b["output"]).astype(np.float32),
                               atol=1e-6)


def test_binary_loss_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_binary_loss_compiled executor"):
        rt._execute_x86_compiled_binary_loss(
            _artifact(rt, "tessera.mse_loss", {}), (a, a))
