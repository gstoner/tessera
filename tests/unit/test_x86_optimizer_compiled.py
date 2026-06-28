"""Compiler-generated fused optimizer steps on x86 AVX-512 (P3 of
S_SERIES_GAP_CLOSURE_PLAN) — sgd / momentum / adam / adamw / lion. A single flat
per-parameter update kernel parameterized by optimizer kind; the optimizer state
(m/v) and the 1-β^t bias correction are carried/computed on host. Reachable via
`compiler_path="x86_optimizer_compiled"`. Validated vs the tessera.optim
reference (multi-step). Skip-clean: x86 lib not built.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import optim


def _rt_or_skip():
    from tessera import runtime as rt
    if not rt._x86_elementwise_available():
        pytest.skip("libtessera_x86_elementwise.so not built/loadable")
    return rt


def _art(rt, op, operands, extras, kw):
    names = [f"a{i}" for i in range(len(operands))]
    kw = dict(kw)
    kw["extras"] = extras
    return rt.RuntimeArtifact(metadata={
        "target": "x86", "compiler_path": "x86_optimizer_compiled",
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names, "kwargs": kw}],
    })


SHAPE = (3, 7)


def test_adamw_multistep():
    rt = _rt_or_skip()
    rng = np.random.default_rng(1)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 6):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        res = rt.launch(_art(rt, "tessera.adamw", [p, g, m, v], ["m", "v"],
                             {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999,
                              "eps": 1e-8, "weight_decay": 0.01, "step": step}),
                        (p, g, m, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "x86_optimizer_compiled"
        pn, m, v = (np.asarray(x) for x in res["output"])
        ref_p, state = optim.adamw(p, g, state, lr=1e-3, beta1=0.9, beta2=0.999,
                                   eps=1e-8, weight_decay=0.01)
        np.testing.assert_allclose(pn, np.asarray(ref_p), atol=2e-5)
        np.testing.assert_allclose(m, np.asarray(state["m"]), atol=2e-5)
        np.testing.assert_allclose(v, np.asarray(state["v"]), atol=2e-6)
        p = pn


def test_sgd():
    rt = _rt_or_skip()
    rng = np.random.default_rng(2)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.sgd", [p, g], [], {"lr": 0.1}), (p, g))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(optim.sgd(p, g, lr=0.1)), atol=1e-6)


def test_momentum():
    rt = _rt_or_skip()
    rng = np.random.default_rng(3)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    v0 = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art(rt, "tessera.momentum", [p, g, v0], ["v"],
                         {"lr": 0.1, "momentum": 0.9}), (p, g, v0))
    assert res["ok"] is True, res.get("reason")
    rp, rst = optim.momentum(p, g, None, lr=0.1, momentum=0.9)
    pn, vn = (np.asarray(x) for x in res["output"])
    np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-6)
    np.testing.assert_allclose(vn, np.asarray(rst["velocity"]), atol=1e-6)


def test_adam():
    rt = _rt_or_skip()
    rng = np.random.default_rng(4)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    z = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art(rt, "tessera.adam", [p, g, z, z], ["m", "v"],
                         {"lr": 1e-3, "step": 1}), (p, g, z, z))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"][0]),
                               np.asarray(optim.adam(p, g, None, lr=1e-3)[0]),
                               atol=2e-5)


def test_lion():
    rt = _rt_or_skip()
    rng = np.random.default_rng(5)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    m0 = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art(rt, "tessera.lion", [p, g, m0], ["m"],
                         {"lr": 1e-4, "beta1": 0.9, "beta2": 0.99,
                          "weight_decay": 0.01}), (p, g, m0))
    assert res["ok"] is True, res.get("reason")
    rp, rst = optim.lion(p, g, None, lr=1e-4, beta1=0.9, beta2=0.99,
                         weight_decay=0.01)
    pn, mn = (np.asarray(x) for x in res["output"])
    np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-6)
    np.testing.assert_allclose(mn, np.asarray(rst["m"]), atol=1e-6)
