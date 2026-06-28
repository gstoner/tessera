"""x86 reduce / stable-reduce foundation — prod (new AVX-512 reduce kind),
var/std/count_nonzero (composed from the reduce kernel), and logsumexp/
log_softmax/softmax_safe/sigmoid_safe (max-shifted reduce + exp/log lane). The
x86 mirror of the ROCm reduce-foundation lane.

Reachable via `compiler_path` x86_reduce_compiled (prod) /
x86_stat_reduce_compiled / x86_stable_reduce_compiled. Validated vs numpy.
Skip-clean: libtessera_x86_elementwise.so absent.
"""

from __future__ import annotations

import numpy as np
import pytest


def _x86_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op_name, path, kwargs):
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": path,
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": ["x"], "output_name": "o",
        "ops": [{"op_name": op_name, "result": "o", "operands": ["x"],
                 "kwargs": kwargs}],
    })


@pytest.mark.parametrize("axis", [-1, 0, 1])
def test_prod(axis):
    rt = _x86_or_skip()
    rng = np.random.default_rng(1 + axis)
    x = (rng.standard_normal((4, 8, 5)) * 0.5 + 1.0).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.prod", "x86_reduce_compiled",
                         {"axis": axis}), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               np.prod(x, axis=axis).astype(np.float32),
                               atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("op,ref", [("tessera.var", np.var), ("tessera.std", np.std)])
@pytest.mark.parametrize("axis", [-1, 1])
def test_var_std(op, ref, axis):
    rt = _x86_or_skip()
    rng = np.random.default_rng(7 + axis + len(op))
    x = (rng.standard_normal((6, 16)) * 2).astype(np.float32)
    res = rt.launch(_art(rt, op, "x86_stat_reduce_compiled", {"axis": axis}),
                    (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref(x, axis=axis).astype(np.float32),
                               atol=2e-4, rtol=2e-4)


def test_count_nonzero():
    rt = _x86_or_skip()
    x = np.array([[1.0, 0.0, 2.0, 0.0], [0.0, 0.0, 3.0, 4.0]], np.float32)
    res = rt.launch(_art(rt, "tessera.count_nonzero", "x86_stat_reduce_compiled",
                         {"axis": -1}), (x,))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_array_equal(np.asarray(res["output"]).astype(np.int64),
                                  np.count_nonzero(x, axis=-1))


@pytest.mark.parametrize("axis", [-1, 1])
def test_logsumexp(axis):
    rt = _x86_or_skip()
    rng = np.random.default_rng(3 + axis)
    x = (rng.standard_normal((5, 12)) * 3).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.logsumexp", "x86_stable_reduce_compiled",
                         {"axis": axis}), (x,))
    assert res["ok"] is True, res.get("reason")
    m = np.max(x, axis=axis, keepdims=True)
    ref = (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(
        axis=axis)
    np.testing.assert_allclose(np.asarray(res["output"]).astype(np.float32),
                               ref.astype(np.float32), atol=2e-4, rtol=2e-4)


def test_log_softmax_and_safe():
    rt = _x86_or_skip()
    rng = np.random.default_rng(11)
    x = (rng.standard_normal((4, 10)) * 2).astype(np.float32)
    rl = rt.launch(_art(rt, "tessera.log_softmax", "x86_stable_reduce_compiled",
                        {"axis": -1}), (x,))
    assert rl["ok"] is True, rl.get("reason")
    z = x - np.max(x, -1, keepdims=True)
    ref_ls = z - np.log(np.sum(np.exp(z), -1, keepdims=True))
    np.testing.assert_allclose(np.asarray(rl["output"]).astype(np.float32),
                               ref_ls.astype(np.float32), atol=2e-4, rtol=2e-4)
    rs = rt.launch(_art(rt, "tessera.softmax_safe", "x86_stable_reduce_compiled",
                        {"axis": -1}), (x,))
    assert rs["ok"] is True, rs.get("reason")
    e = np.exp(z)
    np.testing.assert_allclose(np.asarray(rs["output"]).astype(np.float32),
                               (e / e.sum(-1, keepdims=True)).astype(np.float32),
                               atol=2e-4, rtol=2e-4)
    rg = rt.launch(_art(rt, "tessera.sigmoid_safe", "x86_stable_reduce_compiled",
                        {}), (x,))
    assert rg["ok"] is True, rg.get("reason")
    np.testing.assert_allclose(np.asarray(rg["output"]).astype(np.float32),
                               (1.0 / (1.0 + np.exp(-x))).astype(np.float32),
                               atol=2e-4, rtol=2e-4)
