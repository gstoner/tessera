"""x86 compiled-composite pooling and image affine lanes."""

from __future__ import annotations

import numpy as np
import pytest

from tessera import nn, ops


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _artifact(rt, op_name, path, operands=("x",), kwargs=None):
    return rt.RuntimeArtifact(metadata={
        "target": "x86",
        "compiler_path": path,
        "executable": True,
        "execution_kind": "native_cpu",
        "arg_names": list(operands),
        "output_name": "o",
        "ops": [{
            "op_name": op_name,
            "result": "o",
            "operands": list(operands),
            "kwargs": dict(kwargs or {}),
        }],
    })


@pytest.mark.parametrize("op_name,ref_fn,kwargs", [
    ("tessera.max_pool", nn.functional.max_pool, {"kernel_size": 2, "stride": 2}),
    ("tessera.avg_pool", nn.functional.avg_pool, {"kernel_size": (2, 3), "stride": (1, 2), "padding": (1, 0)}),
    ("tessera.min_pool", nn.functional.min_pool, {"kernel_size": 3, "stride": 1, "padding": 1}),
])
def test_x86_pooling_matches_reference(op_name, ref_fn, kwargs):
    rt = _x86_or_skip()
    rng = np.random.default_rng(21)
    x = (rng.standard_normal((2, 3, 5, 6)) * 2).astype(np.float32)
    res = rt.launch(_artifact(rt, op_name, "x86_pooling_compiled", kwargs=kwargs), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_pooling_compiled"
    np.testing.assert_allclose(res["output"], ref_fn(x, **kwargs), atol=2e-5, rtol=2e-5)


def test_x86_adaptive_pool_matches_reference():
    rt = _x86_or_skip()
    x = np.arange(2 * 3 * 5 * 7, dtype=np.float32).reshape(2, 3, 5, 7) / 13.0
    kwargs = {"output_size": (2, 3)}
    res = rt.launch(_artifact(rt, "tessera.adaptive_pool", "x86_pooling_compiled", kwargs=kwargs), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(res["output"], nn.functional.adaptive_pool(x, **kwargs), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("layout,shape", [
    ("nchw", (2, 3, 4, 5)),
    ("nhwc", (2, 4, 5, 3)),
])
def test_x86_image_normalize_matches_reference(layout, shape):
    rt = _x86_or_skip()
    rng = np.random.default_rng(33)
    x = rng.standard_normal(shape).astype(np.float32)
    kwargs = {"mean": [0.1, -0.2, 0.3], "std": [1.5, 0.7, 2.0], "layout": layout}
    res = rt.launch(_artifact(rt, "tessera.image_normalize", "x86_image_affine_compiled", kwargs=kwargs), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_image_affine_compiled"
    np.testing.assert_allclose(
        res["output"],
        np.asarray(ops.image_normalize(x, **kwargs), np.float32),
        atol=2e-5,
        rtol=2e-5,
    )
