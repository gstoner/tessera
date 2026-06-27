"""x86 class-axis loss lane — cross_entropy / kl / js / focal /
label_smoothed_cross_entropy / z_loss, composed from the AVX-512 transcendental
(exp/log) + reduce kernels with host class-axis structure. The CPU lane for these
S11 losses (previously reference-only on both devices).

Reachable through `runtime.launch()` via `compiler_path="x86_class_loss_compiled"`.
f32; validated vs the tessera.losses numpy (float64) reference at 2e-4.

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


def _artifact(rt, op_name, operands, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_class_loss_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs}],
    })


_TOL = dict(atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_int_targets(reduction):
    rt = _x86_or_skip()
    rng = np.random.default_rng(1 + len(reduction))
    logits = (rng.standard_normal((6, 10)) * 2).astype(np.float32)
    targets = rng.integers(0, 10, size=(6,)).astype(np.int64)
    res = rt.launch(_artifact(rt, "tessera.cross_entropy_loss",
                              ("z", "t"), {"reduction": reduction}),
                    (logits, targets))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_class_loss_compiled"
    ref = np.asarray(losses.cross_entropy_loss(logits, targets,
                                               reduction=reduction),
                     dtype=np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               **_TOL)


def test_cross_entropy_soft_targets():
    rt = _x86_or_skip()
    rng = np.random.default_rng(3)
    logits = (rng.standard_normal((4, 8)) * 2).astype(np.float32)
    soft = rng.random((4, 8)).astype(np.float32)
    soft /= soft.sum(-1, keepdims=True)
    res = rt.launch(_artifact(rt, "tessera.cross_entropy_loss",
                              ("z", "t"), {"reduction": "mean"}),
                    (logits, soft))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(losses.cross_entropy_loss(logits, soft))
    np.testing.assert_allclose(np.float32(res["output"]), ref, **_TOL)


def test_label_smoothed_cross_entropy():
    rt = _x86_or_skip()
    rng = np.random.default_rng(5)
    logits = (rng.standard_normal((5, 12)) * 2).astype(np.float32)
    targets = rng.integers(0, 12, size=(5,)).astype(np.int64)
    res = rt.launch(_artifact(rt, "tessera.label_smoothed_cross_entropy",
                              ("z", "t"), {"smoothing": 0.1}), (logits, targets))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(losses.label_smoothed_cross_entropy(logits, targets,
                                                         smoothing=0.1))
    np.testing.assert_allclose(np.float32(res["output"]), ref, **_TOL)


@pytest.mark.parametrize("gamma", [0.0, 2.0])
def test_focal_loss(gamma):
    rt = _x86_or_skip()
    rng = np.random.default_rng(7 + int(gamma))
    logits = (rng.standard_normal((6, 9)) * 2).astype(np.float32)
    targets = rng.integers(0, 9, size=(6,)).astype(np.int64)
    res = rt.launch(_artifact(rt, "tessera.focal_loss", ("z", "t"),
                              {"gamma": gamma}), (logits, targets))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(losses.focal_loss(logits, targets, gamma=gamma))
    np.testing.assert_allclose(np.float32(res["output"]), ref, **_TOL)


def test_z_loss():
    rt = _x86_or_skip()
    rng = np.random.default_rng(9)
    logits = (rng.standard_normal((4, 7, 16)) * 2).astype(np.float32)
    res = rt.launch(_artifact(rt, "tessera.z_loss", ("z",),
                              {"reduction": "mean"}), (logits,))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(losses.z_loss(logits))
    np.testing.assert_allclose(np.float32(res["output"]), ref, **_TOL)


def test_kl_divergence():
    rt = _x86_or_skip()
    rng = np.random.default_rng(11)
    p = rng.random((4, 8)).astype(np.float32) + 0.1
    p /= p.sum(-1, keepdims=True)
    q = rng.random((4, 8)).astype(np.float32) + 0.1
    q /= q.sum(-1, keepdims=True)
    res = rt.launch(_artifact(rt, "tessera.kl_divergence", ("plog", "q"),
                              {"reduction": "mean"}),
                    (np.log(p).astype(np.float32), q))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(losses.kl_divergence(np.log(p), q))
    np.testing.assert_allclose(np.float32(res["output"]), ref, **_TOL)


def test_js_divergence():
    rt = _x86_or_skip()
    rng = np.random.default_rng(13)
    p = rng.random((4, 8)).astype(np.float32) + 0.1
    p /= p.sum(-1, keepdims=True)
    q = rng.random((4, 8)).astype(np.float32) + 0.1
    q /= q.sum(-1, keepdims=True)
    res = rt.launch(_artifact(rt, "tessera.js_divergence", ("p", "q"),
                              {"reduction": "mean"}), (p, q))
    assert res["ok"] is True, res.get("reason")
    ref = np.float32(losses.js_divergence(p, q))
    np.testing.assert_allclose(np.float32(res["output"]), ref, **_TOL)


def test_class_loss_unknown_op_rejected():
    from tessera import runtime as rt
    a = np.zeros((4, 8), np.float32)
    with pytest.raises(ValueError, match="x86_class_loss_compiled executor"):
        rt._execute_x86_compiled_class_loss(
            _artifact(rt, "tessera.mse_loss", ("a",), {}), (a,))
