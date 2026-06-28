"""Compiler-composed class-axis losses on gfx1151 — the ROCm mirror of the
x86_class_loss lane. exp/log run on the rocm unary lane; class-axis structure on
the host. Reachable via `compiler_path="rocm_class_loss_compiled"`.

Validated vs tessera.losses on gfx1151. Skip-clean: tessera-opt not built/no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import losses


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _artifact(rt, op_name, operands, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_class_loss_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": list(operands), "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": list(operands),
                 "kwargs": kwargs}],
    })


_TOL = dict(atol=2e-4, rtol=2e-4)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_cross_entropy_int(reduction):
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1 + len(reduction))
    logits = (rng.standard_normal((6, 10)) * 2).astype(np.float32)
    targets = rng.integers(0, 10, size=(6,)).astype(np.int64)
    res = rt.launch(_artifact(rt, "tessera.cross_entropy_loss", ("z", "t"),
                              {"reduction": reduction}), (logits, targets))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_class_loss_compiled"
    ref = np.asarray(losses.cross_entropy_loss(logits, targets,
                                               reduction=reduction), np.float32)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32), ref,
                               **_TOL)


def test_focal_and_label_smoothed():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(5)
    logits = (rng.standard_normal((5, 12)) * 2).astype(np.float32)
    targets = rng.integers(0, 12, size=(5,)).astype(np.int64)
    r1 = rt.launch(_artifact(rt, "tessera.focal_loss", ("z", "t"),
                             {"gamma": 2.0}), (logits, targets))
    assert r1["ok"] is True, r1.get("reason")
    np.testing.assert_allclose(np.float32(r1["output"]),
                               np.float32(losses.focal_loss(logits, targets,
                                                            gamma=2.0)), **_TOL)
    r2 = rt.launch(_artifact(rt, "tessera.label_smoothed_cross_entropy",
                             ("z", "t"), {"smoothing": 0.1}), (logits, targets))
    assert r2["ok"] is True, r2.get("reason")
    np.testing.assert_allclose(
        np.float32(r2["output"]),
        np.float32(losses.label_smoothed_cross_entropy(logits, targets,
                                                       smoothing=0.1)), **_TOL)


def test_z_loss_kl_js():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(9)
    logits = (rng.standard_normal((4, 7, 16)) * 2).astype(np.float32)
    rz = rt.launch(_artifact(rt, "tessera.z_loss", ("z",), {}), (logits,))
    assert rz["ok"] is True, rz.get("reason")
    np.testing.assert_allclose(np.float32(rz["output"]),
                               np.float32(losses.z_loss(logits)), **_TOL)
    p = rng.random((4, 8)).astype(np.float32) + 0.1
    p /= p.sum(-1, keepdims=True)
    q = rng.random((4, 8)).astype(np.float32) + 0.1
    q /= q.sum(-1, keepdims=True)
    rk = rt.launch(_artifact(rt, "tessera.kl_divergence", ("pl", "q"), {}),
                   (np.log(p).astype(np.float32), q))
    assert rk["ok"] is True, rk.get("reason")
    np.testing.assert_allclose(np.float32(rk["output"]),
                               np.float32(losses.kl_divergence(np.log(p), q)),
                               **_TOL)
    rj = rt.launch(_artifact(rt, "tessera.js_divergence", ("p", "q"), {}), (p, q))
    assert rj["ok"] is True, rj.get("reason")
    np.testing.assert_allclose(np.float32(rj["output"]),
                               np.float32(losses.js_divergence(p, q)), **_TOL)
