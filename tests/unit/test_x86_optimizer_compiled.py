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

import tessera as ts
from tessera import optim
from tessera.autodiff.vjp import get_vjp


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


@ts.jit(target="x86", autodiff="reverse", wrt=("p", "g", "m"))
def _x86_lion_vjp(p, g, m):
    return ts.ops.lion(
        p, g, m, lr=1e-4, beta2=0.99, weight_decay=0.01
    )


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


def test_nesterov_multistep():
    """Look-ahead momentum: v=β·v+g ; p -= lr·(g+β·v). Multi-step vs
    optim.nesterov so the carried velocity is exercised."""
    rt = _rt_or_skip()
    rng = np.random.default_rng(7)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for _ in range(5):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        res = rt.launch(_art(rt, "tessera.nesterov", [p, g, v], ["v"],
                             {"lr": 1e-2, "momentum": 0.9}), (p, g, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "x86_optimizer_compiled"
        pn, v = (np.asarray(x) for x in res["output"])
        rp, state = optim.nesterov(p, g, state, lr=1e-2, momentum=0.9)
        np.testing.assert_allclose(pn, np.asarray(rp), atol=2e-6)
        np.testing.assert_allclose(v, np.asarray(state["velocity"]), atol=2e-6)
        p = pn


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


def test_lion_backward_stop_sign_vjp():
    _rt_or_skip()
    rng = np.random.default_rng(8)
    dp = rng.standard_normal(SHAPE).astype(np.float32)
    dm = rng.standard_normal(SHAPE).astype(np.float32)
    zeros = np.zeros(SHAPE, np.float32)
    got = _x86_lion_vjp.native_backward(
        zeros, zeros, zeros, out_cotangents=(dp, dm)
    )
    np.testing.assert_allclose(got[0], (1.0 - 1e-4 * 0.01) * dp, atol=1e-6)
    np.testing.assert_allclose(got[1], (1.0 - 0.99) * dm, atol=1e-6)
    np.testing.assert_allclose(got[2], 0.99 * dm, atol=1e-6)
    assert _x86_lion_vjp.last_backward_execution["implementation"] == (
        "family_plugin"
    )


def _adafactor_artifact(rt, *, backward: bool, factored: bool, shape):
    # Every reference below carries state step 1, so the update under test is
    # step 2.  `step` is 1-based and selects Adafactor's bias-corrected
    # second-moment decay (`optim.adafactor_decay`), exactly as it does for the
    # flat adam/adamw ABI above.
    numeric = {"lr": 1e-2, "beta2": 0.9, "eps": 1e-6, "step": 2}
    operands = ["p", "g", "row", "col"] if factored else ["p", "g", "moment"]
    names = operands + (["dy"] if backward else [])
    metadata = {
        "target": "x86",
        "compiler_path": (
            "x86_adafactor_bwd_compiled" if backward
            else "x86_adafactor_compiled"
        ),
        "executable": True, "execution_kind": "native_cpu",
        "arg_names": names, "out_cotangent": "dy",
        "ops": [{
            "op_name": "tessera.adafactor", "operands": operands,
            "kwargs": dict(numeric),
        }],
    }
    if backward:
        from tessera.compiler.stateful_training import lower_scheduled_adafactor_vjp

        scheduled = lower_scheduled_adafactor_vjp(
            target="x86",
            parameter_shape=shape,
            topology="factored" if factored else "full",
            kwargs=dict(numeric),
        )
        metadata["state_contract"] = dict(scheduled.state_contract)
        metadata["scheduled_training"] = scheduled.metadata()
        metadata.pop("ops")
    return rt.RuntimeArtifact(metadata=metadata)


def test_adafactor_factored_forward_and_backward():
    rt = _rt_or_skip()
    rng = np.random.default_rng(9)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = (0.2 * rng.standard_normal(SHAPE)).astype(np.float32)
    row = rng.uniform(0.1, 0.3, SHAPE[:-1]).astype(np.float32)
    col = rng.uniform(0.1, 0.3, SHAPE[-1]).astype(np.float32)
    dy = rng.standard_normal(SHAPE).astype(np.float32)
    forward = rt.launch(
        _adafactor_artifact(rt, backward=False, factored=True, shape=SHAPE),
        (p, g, row, col),
    )
    assert forward["ok"] is True, forward.get("reason")
    expected_forward, expected_state = optim.adafactor(
        p, g, {"v": {"row": row, "col": col, "factored": True}, "step": 1},
        lr=1e-2, beta2=0.9, eps=1e-6,
    )
    actual_p, actual_row, actual_col = (
        np.asarray(value) for value in forward["output"]
    )
    np.testing.assert_allclose(actual_p, expected_forward, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_row, expected_state["v"]["row"], atol=1e-6)
    np.testing.assert_allclose(actual_col, expected_state["v"]["col"], atol=1e-6)

    backward = rt.launch(
        _adafactor_artifact(rt, backward=True, factored=True, shape=SHAPE),
        (p, g, row, col, dy),
    )
    assert backward["ok"] is True, backward.get("reason")
    expected = get_vjp("adafactor")(
        dy, p, g, {"v": {"row": row, "col": col, "factored": True}, "step": 1},
        lr=1e-2, beta2=0.9, eps=1e-6,
    )
    dp, dg, drow, dcol = (np.asarray(value) for value in backward["output"])
    np.testing.assert_allclose(dp, expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(dg, expected[1], rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(drow, expected[2]["v"]["row"], rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(dcol, expected[2]["v"]["col"], rtol=3e-4, atol=3e-5)


def test_adafactor_factored_nan_gradient_propagates_like_reference():
    """A NaN gradient must poison every update sharing its row/col statistic,
    exactly as the reference does (np.maximum floors propagate NaN). The old
    fmax floors laundered the NaN statistic into eps, giving finite-but-wrong
    updates to the rest of the row/col (JIT-MATH-AUDIT-2026-08-23)."""
    rt = _rt_or_skip()
    rng = np.random.default_rng(11)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = (0.2 * rng.standard_normal(SHAPE)).astype(np.float32)
    g[1, 2] = np.nan
    row = rng.uniform(0.1, 0.3, SHAPE[:-1]).astype(np.float32)
    col = rng.uniform(0.1, 0.3, SHAPE[-1]).astype(np.float32)
    forward = rt.launch(
        _adafactor_artifact(rt, backward=False, factored=True, shape=SHAPE),
        (p, g, row, col),
    )
    assert forward["ok"] is True, forward.get("reason")
    expected_p, expected_state = optim.adafactor(
        p, g, {"v": {"row": row, "col": col, "factored": True}, "step": 1},
        lr=1e-2, beta2=0.9, eps=1e-6,
    )
    actual_p, actual_row, actual_col = (
        np.asarray(value) for value in forward["output"]
    )
    # The poisoned statistics themselves.
    np.testing.assert_array_equal(
        np.isnan(actual_row), np.isnan(np.asarray(expected_state["v"]["row"])))
    np.testing.assert_array_equal(
        np.isnan(actual_col), np.isnan(np.asarray(expected_state["v"]["col"])))
    # NaN pattern of the update matches the reference: full row 1 and full
    # col 2 are NaN, everything else finite and equal.
    expected_p = np.asarray(expected_p)
    np.testing.assert_array_equal(np.isnan(actual_p), np.isnan(expected_p))
    assert np.isnan(actual_p[1, :]).all() and np.isnan(actual_p[:, 2]).all()
    fin = ~np.isnan(expected_p)
    np.testing.assert_allclose(
        actual_p[fin], expected_p[fin], rtol=2e-5, atol=2e-5)


def test_adafactor_full_forward_and_backward():
    rt = _rt_or_skip()
    rng = np.random.default_rng(10)
    p = rng.standard_normal(19).astype(np.float32)
    g = (0.2 * rng.standard_normal(19)).astype(np.float32)
    moment = rng.uniform(0.1, 0.3, 19).astype(np.float32)
    dy = rng.standard_normal(19).astype(np.float32)
    forward = rt.launch(
        _adafactor_artifact(rt, backward=False, factored=False, shape=p.shape),
        (p, g, moment),
    )
    assert forward["ok"] is True, forward.get("reason")
    expected_forward, expected_state = optim.adafactor(
        p, g, {"v": {"v": moment, "factored": False}, "step": 1},
        lr=1e-2, beta2=0.9, eps=1e-6,
    )
    actual_p, actual_moment = (np.asarray(value) for value in forward["output"])
    np.testing.assert_allclose(actual_p, expected_forward, rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(actual_moment, expected_state["v"]["v"], atol=1e-6)
    backward = rt.launch(
        _adafactor_artifact(rt, backward=True, factored=False, shape=p.shape),
        (p, g, moment, dy),
    )
    assert backward["ok"] is True, backward.get("reason")
    expected = get_vjp("adafactor")(
        dy, p, g, {"v": {"v": moment, "factored": False}, "step": 1},
        lr=1e-2, beta2=0.9, eps=1e-6,
    )
    for actual, reference in zip(backward["output"], (expected[0], expected[1], expected[2]["v"]["v"])):
        np.testing.assert_allclose(actual, reference, rtol=3e-4, atol=3e-5)
