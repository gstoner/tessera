"""Apple GPU optimizer lane — sgd / momentum / adam / adamw / lion.

A single per-parameter update parameterized by optimizer kind, with the state
(m/v) in/out. Apple ships no device optimizer kernel, so the elementwise update
rules run on the numpy reference the x86/ROCm device kernels are matched against.
Reachable via `compiler_path="apple_gpu_optimizer_compiled"`; execution_kind is
reference_cpu (no Metal dispatch). Validated vs tessera.optim — parity with
test_x86_optimizer_compiled. Numpy path always runs (no skip).
"""

from __future__ import annotations

import numpy as np

from tessera import optim
from tessera import runtime as rt

SHAPE = (4, 5)


def _art(op, operands, extras, kw):
    names = [f"a{i}" for i in range(len(operands))]
    return rt.RuntimeArtifact(metadata={
        "target": "apple_gpu", "compiler_path": "apple_gpu_optimizer_compiled",
        "executable": True, "execution_kind": "reference_cpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names,
                 "kwargs": {**kw, "extras": extras}}]})


def _launch(op, operands, extras, kw):
    res = rt.launch(_art(op, operands, extras, kw), tuple(operands))
    assert res["ok"] is True, res.get("reason")
    assert res["compiler_path"] == "apple_gpu_optimizer_compiled"
    assert res["execution_kind"] == "reference_cpu"
    return res["output"]


def test_sgd():
    rng = np.random.default_rng(2)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    np.testing.assert_allclose(np.asarray(_launch("tessera.sgd", [p, g], [], {"lr": 0.1})),
                               np.asarray(optim.sgd(p, g, lr=0.1)), atol=1e-6)


def test_momentum():
    rng = np.random.default_rng(3)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    v0 = np.zeros(SHAPE, np.float32)
    pn, vn = (np.asarray(x) for x in _launch(
        "tessera.momentum", [p, g, v0], ["v"], {"lr": 0.1, "momentum": 0.9}))
    rp, rst = optim.momentum(p, g, None, lr=0.1, momentum=0.9)
    np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-6)
    np.testing.assert_allclose(vn, np.asarray(rst["velocity"]), atol=1e-6)


def test_adamw_multistep():
    rng = np.random.default_rng(1)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 6):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        pn, m, v = (np.asarray(x) for x in _launch(
            "tessera.adamw", [p, g, m, v], ["m", "v"],
            {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8,
             "weight_decay": 0.01, "step": step}))
        ref_p, state = optim.adamw(p, g, state, lr=1e-3, beta1=0.9, beta2=0.999,
                                   eps=1e-8, weight_decay=0.01)
        np.testing.assert_allclose(pn, np.asarray(ref_p), atol=2e-5)
        np.testing.assert_allclose(m, np.asarray(state["m"]), atol=2e-5)
        np.testing.assert_allclose(v, np.asarray(state["v"]), atol=2e-6)
        p = pn


def test_adam_multistep():
    rng = np.random.default_rng(4)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 4):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        pn, m, v = (np.asarray(x) for x in _launch(
            "tessera.adam", [p, g, m, v], ["m", "v"], {"lr": 1e-3, "step": step}))
        ref_p, state = optim.adam(p, g, state, lr=1e-3)
        np.testing.assert_allclose(pn, np.asarray(ref_p), atol=2e-5)
        p = pn


def test_lion():
    rng = np.random.default_rng(5)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    pn, mn = (np.asarray(x) for x in _launch(
        "tessera.lion", [p, g, m], ["m"],
        {"lr": 1e-4, "beta1": 0.9, "beta2": 0.99, "weight_decay": 0.01}))
    rp, rst = optim.lion(p, g, None, lr=1e-4, beta1=0.9, beta2=0.99,
                         weight_decay=0.01)
    np.testing.assert_allclose(pn, np.asarray(rp), atol=2e-6)
    np.testing.assert_allclose(mn, np.asarray(rst["m"]), atol=2e-6)
