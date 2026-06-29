"""Compiler-generated group/instance/weight norm on gfx1151 (P5 of
S_SERIES_GAP_CLOSURE_PLAN) — composed on the COMPILER-GENERATED layer_norm (row
mean/var) + reduce (sum-of-squares) kernels; host does the reshape / divide.
The device ops are UNWEIGHTED. Reachable via
`compiler_path="rocm_normcompose_compiled"`. Validated vs nn.functional on
gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera.nn import functional as F


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_normcompose_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": ["a0"], "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": ["a0"],
                 "kwargs": kwargs}],
    })


def test_instance_norm():
    rt = _rocm_or_skip()
    x = np.random.default_rng(0).standard_normal((2, 6, 4, 5)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.instance_norm", {"eps": 1e-5}), (x,))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "rocm_normcompose_compiled"
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(F.instance_norm(x, eps=1e-5)),
                               atol=1e-4)


@pytest.mark.parametrize("ng", [2, 3, 6])
def test_group_norm(ng):
    rt = _rocm_or_skip()
    x = np.random.default_rng(1).standard_normal((2, 6, 4, 5)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.group_norm",
                         {"num_groups": ng, "eps": 1e-5}), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(F.group_norm(x, num_groups=ng,
                                                       eps=1e-5)), atol=1e-4)


@pytest.mark.parametrize("axis", [0, 1, 2, -1])
def test_weight_norm(axis):
    rt = _rocm_or_skip()
    w = np.random.default_rng(2).standard_normal((8, 4, 3)).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.weight_norm",
                         {"axis": axis, "eps": 1e-12}), (w,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(F.weight_norm(w, axis=axis,
                                                        eps=1e-12)), atol=1e-4)
