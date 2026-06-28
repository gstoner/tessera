"""Compiler-generated group/instance/weight norm on x86 AVX-512 (P5 of
S_SERIES_GAP_CLOSURE_PLAN) — composed on the existing device lanes (no new
kernel): group/instance normalize is a row mean/var normalize = the layer_norm
kernel on a reshaped [rows, cols] view; weight_norm is a row sum-of-squares on
the reduce lane + a host divide. The device ops are UNWEIGHTED (the affine
composes separately). Reachable via `compiler_path="x86_normcompose_compiled"`.
Validated vs nn.functional. Skip-clean: libtessera_x86_elementwise.so not built.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.nn import functional as F


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_normcompose_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["a0"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["a0"],
                 "kwargs": kwargs}],
    })


def test_instance_norm():
    rt = _rt_or_skip()
    x = np.random.default_rng(0).standard_normal((2, 6, 4, 5)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.instance_norm", {"eps": 1e-5}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "x86_normcompose_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(F.instance_norm(x, eps=1e-5)),
                               atol=2e-5)


@pytest.mark.parametrize("ng", [2, 3, 6])
def test_group_norm(ng):
    rt = _rt_or_skip()
    x = np.random.default_rng(1).standard_normal((2, 6, 4, 5)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.group_norm",
                         {"num_groups": ng, "eps": 1e-5}), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(F.group_norm(x, num_groups=ng,
                                                       eps=1e-5)), atol=2e-5)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_weight_norm(axis):
    rt = _rt_or_skip()
    w = np.random.default_rng(2).standard_normal((8, 4, 3)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.weight_norm",
                         {"axis": axis, "eps": 1e-12}), (w,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(F.weight_norm(w, axis=axis,
                                                        eps=1e-12)), atol=1e-5)


def test_group_norm_bad_divisor_rejected():
    rt = _rt_or_skip()
    x = np.zeros((2, 5, 4), np.float32)
    res = rt.launch(_art(rt, "tessera.group_norm", {"num_groups": 3}), (x,))
    assert res["ok"] is False
