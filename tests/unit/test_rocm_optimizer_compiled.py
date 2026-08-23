"""Compiler-generated fused optimizer steps on gfx1151 (P3 of
S_SERIES_GAP_CLOSURE_PLAN) — sgd / momentum / adam / adamw / lion. The Tessera
compiler GENERATES the kernel (generate-rocm-optimizer-kernel, kind StrAttr →
ROCDL → hsaco), then HIP launches it. Adafactor uses one compiler-owned
five-entry module (row moment, column moment, ordered row mean, factored
update, lower-rank full-moment update).
Reachable via
`compiler_path="rocm_optimizer_compiled"`. Validated vs tessera.optim on
gfx1151. Skip-clean: tessera-opt not built / no GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

from tessera import optim
from tessera.autodiff.vjp import get_vjp
from tests._support.compiler_tool import run_tessera_opt


def _rocm_or_skip():
    from tessera import runtime as rt
    if rt._tessera_opt_path() is None:
        pytest.skip("tessera-opt not built")
    if not rt._rocm_wmma_runtime_available():
        pytest.skip("no usable AMD GPU")
    return rt


def _art(rt, op, operands, extras, kw):
    names = [f"a{i}" for i in range(len(operands))]
    kw = dict(kw)
    kw["extras"] = extras
    return rt.RuntimeArtifact(metadata={
        "target": "rocm", "compiler_path": "rocm_optimizer_compiled",
        "executable": True, "execution_kind": "native_gpu",
        "arg_names": names, "output_name": "o",
        "ops": [{"op_name": op, "result": "o", "operands": names, "kwargs": kw}],
    })


SHAPE = (3, 7)


def test_adamw_multistep():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(1)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    m = np.zeros(SHAPE, np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for step in range(1, 4):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        res = rt.launch(_art(rt, "tessera.adamw", [p, g, m, v], ["m", "v"],
                             {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999,
                              "eps": 1e-8, "weight_decay": 0.01, "step": step}),
                        (p, g, m, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "rocm_optimizer_compiled"
        pn, m, v = (np.asarray(x) for x in res["output"])
        ref_p, state = optim.adamw(p, g, state, lr=1e-3, beta1=0.9, beta2=0.999,
                                   eps=1e-8, weight_decay=0.01)
        np.testing.assert_allclose(pn, np.asarray(ref_p), atol=1e-4)
        np.testing.assert_allclose(v, np.asarray(state["v"]), atol=1e-5)
        p = pn


def test_sgd():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(2)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    res = rt.launch(_art(rt, "tessera.sgd", [p, g], [], {"lr": 0.1}), (p, g))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"]),
                               np.asarray(optim.sgd(p, g, lr=0.1)), atol=1e-5)


def test_momentum():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(3)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    v0 = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art(rt, "tessera.momentum", [p, g, v0], ["v"],
                         {"lr": 0.1, "momentum": 0.9}), (p, g, v0))
    assert res["ok"] is True, res.get("reason")
    rp, rst = optim.momentum(p, g, None, lr=0.1, momentum=0.9)
    pn, vn = (np.asarray(x) for x in res["output"])
    np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-5)
    np.testing.assert_allclose(vn, np.asarray(rst["velocity"]), atol=1e-5)


def test_nesterov_multistep():
    """Look-ahead momentum: v=β·v+g ; p -= lr·(g+β·v). Multi-step vs
    optim.nesterov on gfx1151 so the carried velocity is exercised."""
    rt = _rocm_or_skip()
    rng = np.random.default_rng(7)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    v = np.zeros(SHAPE, np.float32)
    state = None
    for _ in range(5):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        res = rt.launch(_art(rt, "tessera.nesterov", [p, g, v], ["v"],
                             {"lr": 1e-2, "momentum": 0.9}), (p, g, v))
        assert res["ok"] is True, res.get("reason")
        assert res["compiler_path"] == "rocm_optimizer_compiled"
        pn, v = (np.asarray(x) for x in res["output"])
        rp, state = optim.nesterov(p, g, state, lr=1e-2, momentum=0.9)
        np.testing.assert_allclose(pn, np.asarray(rp), atol=1e-5)
        np.testing.assert_allclose(v, np.asarray(state["velocity"]), atol=1e-5)
        p = pn


def test_adam_and_lion():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(4)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = rng.standard_normal(SHAPE).astype(np.float32)
    z = np.zeros(SHAPE, np.float32)
    res = rt.launch(_art(rt, "tessera.adam", [p, g, z, z], ["m", "v"],
                         {"lr": 1e-3, "step": 1}), (p, g, z, z))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(np.asarray(res["output"][0]),
                               np.asarray(optim.adam(p, g, None, lr=1e-3)[0]),
                               atol=1e-4)
    res = rt.launch(_art(rt, "tessera.lion", [p, g, z], ["m"],
                         {"lr": 1e-4, "beta1": 0.9, "beta2": 0.99,
                          "weight_decay": 0.01}), (p, g, z))
    assert res["ok"] is True, res.get("reason")
    np.testing.assert_allclose(
        np.asarray(res["output"][0]),
        np.asarray(optim.lion(p, g, None, lr=1e-4, beta1=0.9, beta2=0.99,
                              weight_decay=0.01)[0]), atol=1e-5)


def test_adafactor_factored_multistep():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(12)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    row = np.zeros(SHAPE[:-1], np.float32)
    col = np.zeros(SHAPE[-1], np.float32)
    state = None
    for _ in range(3):
        g = rng.standard_normal(SHAPE).astype(np.float32)
        artifact = rt.RuntimeArtifact(metadata={
            "target": "rocm",
            "compiler_path": "rocm_adafactor_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "arg_names": ["p", "g", "row", "col"],
            "output_name": "o",
            "ops": [{
                "op_name": "tessera.adafactor",
                "result": "o",
                "operands": ["p", "g", "row", "col"],
                "kwargs": {"lr": 1e-2, "beta2": 0.9, "eps": 1e-6},
            }],
        })
        result = rt.launch(artifact, (p, g, row, col))
        assert result["ok"] is True, result.get("reason")
        assert result["compiler_path"] == "rocm_adafactor_compiled"
        p_new, row, col = (np.asarray(value) for value in result["output"])
        p_ref, state = optim.adafactor(
            p, g, state, lr=1e-2, beta2=0.9, eps=1e-6
        )
        np.testing.assert_allclose(p_new, np.asarray(p_ref), atol=2e-5)
        np.testing.assert_allclose(row, np.asarray(state["v"]["row"]), atol=1e-6)
        np.testing.assert_allclose(col, np.asarray(state["v"]["col"]), atol=1e-6)
        p = p_new


def test_adafactor_factored_nan_gradient_propagates_like_reference():
    """gfx1151: a NaN gradient must poison every update sharing its row/col
    statistic, exactly as the reference does (np.maximum floors propagate
    NaN). The old maxnumf floors laundered the NaN statistic into eps,
    giving finite-but-wrong updates to the rest of the poisoned row/col
    (JIT-MATH-AUDIT-2026-08-23)."""
    rt = _rocm_or_skip()
    rng = np.random.default_rng(13)
    p = rng.standard_normal(SHAPE).astype(np.float32)
    g = (0.2 * rng.standard_normal(SHAPE)).astype(np.float32)
    g[1, 2] = np.nan
    row = rng.uniform(0.1, 0.3, SHAPE[:-1]).astype(np.float32)
    col = rng.uniform(0.1, 0.3, SHAPE[-1]).astype(np.float32)
    artifact = rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_adafactor_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["p", "g", "row", "col"],
        "output_name": "o",
        "ops": [{
            "op_name": "tessera.adafactor",
            "result": "o",
            "operands": ["p", "g", "row", "col"],
            "kwargs": {"lr": 1e-2, "beta2": 0.9, "eps": 1e-6},
        }],
    })
    result = rt.launch(artifact, (p, g, row, col))
    assert result["ok"] is True, result.get("reason")
    p_new, row_new, col_new = (
        np.asarray(value) for value in result["output"])
    p_ref, state = optim.adafactor(
        p, g, {"v": {"row": row, "col": col, "factored": True}, "step": 1},
        lr=1e-2, beta2=0.9, eps=1e-6,
    )
    p_ref = np.asarray(p_ref)
    np.testing.assert_array_equal(
        np.isnan(row_new), np.isnan(np.asarray(state["v"]["row"])))
    np.testing.assert_array_equal(
        np.isnan(col_new), np.isnan(np.asarray(state["v"]["col"])))
    np.testing.assert_array_equal(np.isnan(p_new), np.isnan(p_ref))
    assert np.isnan(p_new[1, :]).all() and np.isnan(p_new[:, 2]).all()
    fin = ~np.isnan(p_ref)
    np.testing.assert_allclose(p_new[fin], p_ref[fin], atol=2e-5)


def test_adafactor_full_moment_vector_multistep():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(13)
    p = rng.standard_normal(19).astype(np.float32)
    moment = np.zeros_like(p)
    state = None
    for _ in range(3):
        g = rng.standard_normal(p.shape).astype(np.float32)
        artifact = rt.RuntimeArtifact(metadata={
            "target": "rocm",
            "compiler_path": "rocm_adafactor_compiled",
            "executable": True,
            "execution_kind": "native_gpu",
            "arg_names": ["p", "g", "moment"],
            "output_name": "o",
            "ops": [{
                "op_name": "tessera.adafactor",
                "result": "o",
                "operands": ["p", "g", "moment"],
                "kwargs": {"lr": 1e-2, "beta2": 0.9, "eps": 1e-6},
            }],
        })
        result = rt.launch(artifact, (p, g, moment))
        assert result["ok"] is True, result.get("reason")
        p_new, moment = (np.asarray(value) for value in result["output"])
        p_ref, state = optim.adafactor(
            p, g, state, lr=1e-2, beta2=0.9, eps=1e-6
        )
        np.testing.assert_allclose(p_new, np.asarray(p_ref), atol=2e-5)
        np.testing.assert_allclose(
            moment, np.asarray(state["v"]["v"]), atol=1e-6
        )
        p = p_new


def test_adafactor_factored_analytic_vjp_matches_directional_difference():
    rng = np.random.default_rng(17)
    p = rng.normal(size=SHAPE).astype(np.float64)
    g = rng.normal(scale=0.2, size=SHAPE).astype(np.float64)
    state = {
        "v": {
            "row": rng.uniform(0.1, 0.3, size=SHAPE[:-1]),
            "col": rng.uniform(0.1, 0.3, size=SHAPE[-1]),
            "factored": True,
        },
        "step": 2,
    }
    dout = rng.normal(size=SHAPE)
    dp, dg, ds = get_vjp("adafactor")(
        dout, p, g, state, lr=1e-2, beta2=0.9, eps=1e-6,
        compute_dtype="fp64", state_dtype="fp64",
    )
    np.testing.assert_allclose(dp, dout)
    entries = [
        (g, dg, lambda value: (p, value, state)),
        (
            state["v"]["row"],
            ds["v"]["row"],
            lambda value: (
                p,
                g,
                {"v": {**state["v"], "row": value}, "step": 2},
            ),
        ),
        (
            state["v"]["col"],
            ds["v"]["col"],
            lambda value: (
                p,
                g,
                {"v": {**state["v"], "col": value}, "step": 2},
            ),
        ),
    ]
    for value, gradient, bind in entries:
        direction = rng.normal(size=value.shape)

        def objective(candidate):
            pp, gg, ss = bind(candidate)
            out, _ = optim.adafactor(
                pp, gg, ss, lr=1e-2, beta2=0.9, eps=1e-6,
                compute_dtype="fp64", state_dtype="fp64",
            )
            return float(np.sum(np.asarray(out) * dout))

        epsilon = 1e-5
        numeric = (
            objective(value + epsilon * direction)
            - objective(value - epsilon * direction)
        ) / (2 * epsilon)
        analytic = float(np.sum(np.asarray(gradient) * direction))
        np.testing.assert_allclose(analytic, numeric, rtol=2e-4, atol=2e-5)


def test_adafactor_factored_backward_executes_on_gfx1151():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(18)
    p = rng.normal(size=SHAPE).astype(np.float32)
    g = rng.normal(scale=0.2, size=SHAPE).astype(np.float32)
    row = rng.uniform(0.1, 0.3, size=SHAPE[:-1]).astype(np.float32)
    col = rng.uniform(0.1, 0.3, size=SHAPE[-1]).astype(np.float32)
    dy = rng.normal(size=SHAPE).astype(np.float32)
    from tessera.compiler.stateful_training import lower_scheduled_adafactor_vjp

    scheduled = lower_scheduled_adafactor_vjp(
        target="rocm_gfx1151",
        parameter_shape=SHAPE,
        topology="factored",
        kwargs={"lr": 1e-2, "beta2": 0.9, "eps": 1e-6},
    )
    artifact = rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_adafactor_bwd_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["p", "g", "row", "col", "dy"],
        "out_cotangent": "dy",
        "state_contract": dict(scheduled.state_contract),
        "scheduled_training": scheduled.metadata(),
    })
    result = rt.launch(artifact, (p, g, row, col, dy))
    assert result["ok"] is True, result.get("reason")
    dp, dg, drow, dcol = (np.asarray(value) for value in result["output"])
    expected = get_vjp("adafactor")(
        dy,
        p,
        g,
        {"v": {"row": row, "col": col, "factored": True}, "step": 1},
        lr=1e-2,
        beta2=0.9,
        eps=1e-6,
    )
    np.testing.assert_allclose(dp, expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(dg, expected[1], rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(
        drow, expected[2]["v"]["row"], rtol=3e-4, atol=3e-5
    )
    np.testing.assert_allclose(
        dcol, expected[2]["v"]["col"], rtol=3e-4, atol=3e-5
    )


def test_adafactor_full_backward_executes_on_gfx1151():
    rt = _rocm_or_skip()
    rng = np.random.default_rng(19)
    p = rng.normal(size=31).astype(np.float32)
    g = rng.normal(scale=0.2, size=p.shape).astype(np.float32)
    moment = rng.uniform(0.1, 0.3, size=p.shape).astype(np.float32)
    dy = rng.normal(size=p.shape).astype(np.float32)
    from tessera.compiler.stateful_training import lower_scheduled_adafactor_vjp

    scheduled = lower_scheduled_adafactor_vjp(
        target="rocm_gfx1151",
        parameter_shape=p.shape,
        topology="full",
        kwargs={"lr": 1e-2, "beta2": 0.9, "eps": 1e-6},
    )
    artifact = rt.RuntimeArtifact(metadata={
        "target": "rocm",
        "compiler_path": "rocm_adafactor_bwd_compiled",
        "executable": True,
        "execution_kind": "native_gpu",
        "arg_names": ["p", "g", "moment", "dy"],
        "out_cotangent": "dy",
        "state_contract": dict(scheduled.state_contract),
        "scheduled_training": scheduled.metadata(),
    })
    result = rt.launch(artifact, (p, g, moment, dy))
    assert result["ok"] is True, result.get("reason")
    dp, dg, dmoment = (np.asarray(value) for value in result["output"])
    expected = get_vjp("adafactor")(
        dy,
        p,
        g,
        {"v": {"v": moment, "factored": False}, "step": 1},
        lr=1e-2,
        beta2=0.9,
        eps=1e-6,
    )
    np.testing.assert_allclose(dp, expected[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(dg, expected[1], rtol=3e-4, atol=3e-5)
    np.testing.assert_allclose(
        dmoment, expected[2]["v"]["v"], rtol=3e-4, atol=3e-5
    )


def _opt(directive, *passes):
    """Skips when this build lacks a requested pass (see _support.compiler_tool)."""
    return run_tessera_opt(directive, *passes)


@pytest.mark.parametrize("kind", ["sgd", "momentum", "adam", "adamw", "lion"])
def test_optimizer_codegen_lowers(kind):
    d = (f'module {{\n  "tessera_rocm.optimizer"() {{name = "o", kind = "{kind}"}} '
         ': () -> ()\n}\n')
    low = _opt(d, "--pass-pipeline=builtin.module(generate-rocm-optimizer-kernel,"
               "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
               "reconcile-unrealized-casts))")
    assert low.returncode == 0 and "llvm." in low.stdout


def test_sgd_backward_codegen_lowers():
    directive = (
        'module {\n  "tessera_rocm.optimizer"() {name = "sgd_bwd", '
        'kind = "sgd", backward = true} : () -> ()\n}\n'
    )
    generated = _opt(directive, "--generate-rocm-optimizer-kernel")
    assert generated.returncode == 0, generated.stderr
    assert "gpu.func @sgd_bwd" in generated.stdout
    assert generated.stdout.count("memref.store") == 2
    lowered = _opt(
        directive,
        "--pass-pipeline=builtin.module(generate-rocm-optimizer-kernel,"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts))")
    assert lowered.returncode == 0, lowered.stderr
    assert "llvm." in lowered.stdout


@pytest.mark.parametrize("kind", ["adam", "adamw"])
def test_adam_backward_codegen_lowers(kind):
    directive = (
        'module {\n  "tessera_rocm.optimizer"() {name = "adam_bwd", '
        f'kind = "{kind}", backward = true}} : () -> ()\n}}\n'
    )
    generated = _opt(directive, "--generate-rocm-optimizer-kernel")
    assert generated.returncode == 0, generated.stderr
    assert "gpu.func @adam_bwd" in generated.stdout
    assert generated.stdout.count("memref.store") == 4


def test_lion_backward_codegen_lowers_stop_sign_vjp():
    directive = (
        'module {\n  "tessera_rocm.optimizer"() {name = "lion_bwd", '
        'kind = "lion", backward = true} : () -> ()\n}\n'
    )
    generated = _opt(directive, "--generate-rocm-optimizer-kernel")
    assert generated.returncode == 0, generated.stderr
    assert "gpu.func @lion_bwd" in generated.stdout
    assert generated.stdout.count("memref.store") == 3
    assert "math." not in generated.stdout


@pytest.mark.parametrize("backward", [False, True])
def test_adafactor_codegen_lowers_factored_program(backward):
    directive = (
        'module {\n  "tessera_rocm.optimizer"() {name = "ada", '
        f'kind = "adafactor", backward = {str(backward).lower()}}} '
        ': () -> ()\n}\n'
    )
    generated = _opt(directive, "--generate-rocm-optimizer-kernel")
    assert generated.returncode == 0, generated.stderr
    for suffix in (
        "row",
        "col",
        "mean",
        "update",
        "full",
        "bwd_mean",
        "bwd_row",
        "bwd_col",
        "bwd_finalize",
        "full_bwd",
    ):
        assert f"gpu.func @ada_{suffix}" in generated.stdout
    assert generated.stdout.count("memref.store") == 16
    lowered = _opt(
        directive,
        "--pass-pipeline=builtin.module(generate-rocm-optimizer-kernel,"
        "gpu.module(convert-scf-to-cf,convert-gpu-to-rocdl,"
        "reconcile-unrealized-casts))")
    assert lowered.returncode == 0, lowered.stderr
    assert lowered.stdout.count("llvm.func @ada_") == 10
