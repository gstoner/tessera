"""Mixed-precision optimizer compiler/autodiff coverage."""

from __future__ import annotations

import numpy as np

import tessera as ts
from tessera.autodiff import tape
from tessera.autodiff.jvp import get_jvp
from tessera.autodiff.vjp import get_vjp
from tessera.compiler.op_catalog import get_op_spec
from tessera.compiler.primitive_coverage import coverage_for


def _numeric_grad(fn, x, eps=1e-5):
    x = np.asarray(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"], op_flags=["readwrite"])
    while not it.finished:
        idx = it.multi_index
        plus = x.copy()
        minus = x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (fn(plus) - fn(minus)) / (2.0 * eps)
        it.iternext()
    return grad


def _numeric_jvp(fn, x, dx, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    dx = np.asarray(dx, dtype=np.float64)
    return (fn(x + eps * dx) - fn(x - eps * dx)) / (2.0 * eps)


def test_optimizer_dtype_policy_preserves_params_and_keeps_fp32_state():
    params = {
        "w": np.ones((2, 3), dtype=np.float16),
        "b": np.ones(3, dtype=np.float32),
    }
    grads = {
        "w": np.full((2, 3), 0.125, dtype=np.float16),
        "b": np.full(3, 0.25, dtype=np.float32),
    }

    next_params, state = ts.optim.adamw(params, grads, lr=0.01, master_dtype="fp32")

    assert next_params["w"].dtype == np.float16
    assert next_params["b"].dtype == np.float32
    assert state["m"]["w"].dtype == np.float32
    assert state["v"]["w"].dtype == np.float32
    assert state["master_params"]["w"].dtype == np.float32


def test_adafactor_and_lion_dtype_policy_for_reasoning_model_trees():
    params = {
        "moe": {"expert0": np.ones((4, 8), dtype=np.float16)},
        "attn": np.ones((8, 8), dtype=np.float32),
    }
    grads = {
        "moe": {"expert0": np.full((4, 8), 0.05, dtype=np.float16)},
        "attn": np.full((8, 8), 0.02, dtype=np.float32),
    }

    ada_params, ada_state = ts.optim.adafactor(params, grads, lr=0.01, master_dtype="fp32")
    lion_params, lion_state = ts.optim.lion(params, grads, lr=0.001, master_dtype="fp32")

    assert ada_params["moe"]["expert0"].dtype == np.float16
    assert ada_state["v"]["moe"]["expert0"]["row"].dtype == np.float32
    assert ada_state["master_params"]["moe"]["expert0"].dtype == np.float32
    assert lion_params["moe"]["expert0"].dtype == np.float16
    assert lion_state["m"]["moe"]["expert0"].dtype == np.float32
    assert lion_state["master_params"]["attn"].dtype == np.float32


def test_optimizer_ops_promoted_to_catalog_and_namespace():
    for name in ("adam", "adamw", "momentum", "adafactor", "lion"):
        assert get_op_spec(name) is not None
        assert hasattr(ts.ops, name)
        assert coverage_for(name).contract_status["vjp"] == "complete"
        assert coverage_for(name).contract_status["jvp"] == "complete"

    assert get_op_spec("adamw").graph_name == "tessera.adamw"
    assert get_op_spec("lion").lowering == "functional_optimizer_step"


def test_low_level_adam_tuple_output_vjp_and_jvp():
    p = np.array([1.0, 2.0], dtype=np.float64)
    g = np.array([0.1, -0.2], dtype=np.float64)
    m = np.array([0.01, -0.02], dtype=np.float64)
    v = np.array([0.001, 0.002], dtype=np.float64)
    dout = np.ones_like(p)

    d_p, d_g, d_m, d_v = get_vjp("adam")(dout, p, g, m, v, lr=0.01, step=2)
    expected_g = _numeric_grad(
        lambda gg: float(ts.ops.adam(p, gg, m, v, lr=0.01, step=2, compute_dtype="fp64", cast_updates_to_param_dtype=False)[0].sum()),
        g,
    )
    np.testing.assert_allclose(d_p, dout)
    np.testing.assert_allclose(d_g, expected_g, atol=1e-4)
    assert d_m.shape == m.shape
    assert d_v.shape == v.shape

    primals, tangents = get_jvp("adam")(
        (p, g, m, v),
        (np.ones_like(p), np.zeros_like(g), np.zeros_like(m), np.zeros_like(v)),
        lr=0.01,
        step=2,
    )
    expected_tangent = _numeric_jvp(
        lambda pp: ts.ops.adam(pp, g, m, v, lr=0.01, step=2, compute_dtype="fp64", cast_updates_to_param_dtype=False)[0],
        p,
        np.ones_like(p),
    )
    np.testing.assert_allclose(primals[0], ts.ops.adam(p, g, m, v, lr=0.01, step=2, compute_dtype="fp64", cast_updates_to_param_dtype=False)[0])
    np.testing.assert_allclose(tangents[0], expected_tangent, atol=1e-5)


def test_low_level_adam_tuple_tape_grad_from_moment_component():
    p = np.array([1.0, 2.0], dtype=np.float64)
    g = np.array([0.1, -0.2], dtype=np.float64)
    m = np.array([0.01, -0.02], dtype=np.float64)
    v = np.array([0.001, 0.002], dtype=np.float64)

    with tape() as t:
        new_m = ts.ops.adam(p, g, m, v, step=1)[1]
        t.backward(new_m, cotangent=np.ones_like(new_m))

    np.testing.assert_allclose(t.cotangent[id(g)], np.full_like(g, 0.1))
    np.testing.assert_allclose(t.cotangent[id(m)], np.full_like(m, 0.9))


def test_lion_vjp_jvp_policy_matches_stop_gradient_sign_update():
    params = np.array([1.0, -2.0], dtype=np.float64)
    grads = np.array([0.25, -0.5], dtype=np.float64)
    state = {"m": np.array([0.1, -0.2], dtype=np.float64), "step": 3}
    dout = np.ones_like(params)

    d_params, d_grads, d_state = get_vjp("lion")(
        dout,
        params,
        grads,
        state,
        lr=0.1,
        weight_decay=0.01,
    )
    np.testing.assert_allclose(d_params, np.full_like(params, 0.999))
    np.testing.assert_allclose(d_grads, np.zeros_like(grads))
    np.testing.assert_allclose(d_state["m"], np.zeros_like(state["m"]))

    _, tangent = get_jvp("lion")(
        (params, grads, state),
        (np.ones_like(params), np.ones_like(grads), {"m": np.ones_like(state["m"])}),
        lr=0.1,
        weight_decay=0.01,
    )
    np.testing.assert_allclose(tangent, np.full_like(params, 0.999))


def test_adafactor_vjp_and_jvp_match_finite_difference_for_vector_state():
    params = np.array([1.0, 2.0], dtype=np.float64)
    grads = np.array([0.2, -0.1], dtype=np.float64)
    state = {"v": {"v": np.array([0.03, 0.04], dtype=np.float64), "factored": False}, "step": 2}
    dout = np.ones_like(params)

    _, d_grads, d_state = get_vjp("adafactor")(dout, params, grads, state, lr=0.01, beta2=0.9, eps=1e-6)
    expected = _numeric_grad(
        lambda gg: float(ts.optim.adafactor(params, gg, state, lr=0.01, beta2=0.9, eps=1e-6)[0].sum()),
        grads,
    )
    np.testing.assert_allclose(d_grads, expected, atol=1e-3)
    assert d_state["v"]["v"].shape == state["v"]["v"].shape

    dparams = np.array([0.3, -0.4], dtype=np.float64)
    _, tangent = get_jvp("adafactor")(
        (params, grads, state),
        (dparams, np.zeros_like(grads), {"v": {"v": np.zeros_like(state["v"]["v"]), "factored": False}, "step": None}),
        lr=0.01,
        beta2=0.9,
        eps=1e-6,
    )
    expected_tangent = _numeric_jvp(
        lambda pp: ts.optim.adafactor(pp, grads, state, lr=0.01, beta2=0.9, eps=1e-6)[0],
        params,
        dparams,
    )
    np.testing.assert_allclose(tangent, expected_tangent, atol=1e-5)


def test_mixed_precision_grad_scaler_can_step_optimizer_closure():
    from tessera.nn import Parameter

    p = Parameter(np.array([1.0, -1.0], dtype=np.float16))
    p.grad = np.array([0.25, -0.5], dtype=np.float32)
    state = {}
    scaler = ts.autodiff.GradScaler(init_scale=2.0)

    def step():
        next_params, next_state = ts.optim.lion(p._data._data, p.grad.numpy(), state or None, lr=0.1, master_dtype="fp32")
        p._data._data[...] = next_params
        state.update(next_state)

    assert scaler.step(step, params=[p])
    assert state["m"].dtype == np.float32
    assert state["master_params"].dtype == np.float32
