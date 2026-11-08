"""Tracer authority reaches execution, not only ``graph_ir``.

Before E2E-REAL-6F's executed-plan slice, a call that established tracer
authority replaced ``graph_ir`` while the CPU plan, the tessera_jit lane and
the Apple fast paths kept executing the plan compiled at decoration from the
AST module. These tests pin that a tracer-authority call executes the plan
compiled from the traced module, and that every other call executes the
decoration plan and says so.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import tessera

# ``tessera.compiler.jit`` the attribute is the decorator; this is the module.
jit_module = importlib.import_module("tessera.compiler.jit")


def _mlp():
    @tessera.jit
    def mlp(x, w):
        return tessera.ops.relu(tessera.ops.matmul(x, w))

    return mlp


def _inputs(rows: int = 4):
    rng = np.random.default_rng(rows)
    x = rng.standard_normal((rows, 8)).astype(np.float32)
    w = rng.standard_normal((8, 3)).astype(np.float32)
    return x, w


def test_tracer_authority_executes_the_traced_plan_not_the_decoration_plan(monkeypatch):
    mlp = _mlp()
    decoration_plan = mlp.cpu_plan

    plan_type = type(decoration_plan)
    original_execute = plan_type.execute
    executed = []

    def observe(plan, *args, **kwargs):
        if plan is decoration_plan:
            raise AssertionError("the AST-compiled decoration plan executed")
        executed.append(plan)
        return original_execute(plan, *args, **kwargs)

    # Force the numpy plan lane so the executed plan object is observable.
    # The plan is a frozen dataclass, so observe through its type.
    monkeypatch.setenv("TESSERA_DISABLE_CPU_JIT", "1")
    monkeypatch.setattr(plan_type, "execute", observe)
    x, w = _inputs()

    np.testing.assert_allclose(mlp(x, w), np.maximum(x @ w, 0.0), rtol=1e-5)
    assert mlp.frontend_authority == "tracer"
    assert mlp.executed_plan_frontend == "tracer"
    assert executed == [mlp.cpu_plan]
    assert mlp.cpu_plan is not decoration_plan
    traced_names = [arg.name for arg in mlp.graph_ir.functions[0].args]
    assert mlp._plan_arg_names() == traced_names
    # The artifact the tessera_jit and Apple lanes read binds the same names.
    assert mlp.runtime_artifact().metadata["arg_names"] == traced_names


def test_traced_plan_accepts_keyword_arguments_by_abi_name():
    mlp = _mlp()
    x, w = _inputs()

    np.testing.assert_allclose(mlp(x, w=w), np.maximum(x @ w, 0.0), rtol=1e-5)
    np.testing.assert_allclose(mlp(w=w, x=x), np.maximum(x @ w, 0.0), rtol=1e-5)
    assert mlp.executed_plan_frontend == "tracer"


def test_traced_plan_is_compiled_once_per_specialization(monkeypatch):
    mlp = _mlp()
    compiles = []
    original = jit_module.compile_graph_module

    def counting(*args, **kwargs):
        compiles.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(jit_module, "compile_graph_module", counting)
    small, wide = _inputs(4), _inputs(6)
    for x, w in (small, wide, small, wide):
        np.testing.assert_allclose(mlp(x, w), np.maximum(x @ w, 0.0), rtol=1e-5)

    assert len(compiles) == 2
    assert len(mlp._traced_plan_specializations) == 2


def test_a_failed_traced_compile_runs_the_decoration_plan_and_says_why(monkeypatch):
    mlp = _mlp()

    def broken(*_args, **_kwargs):
        raise RuntimeError("synthetic compile failure")

    monkeypatch.setattr(jit_module, "compile_graph_module", broken)
    x, w = _inputs()

    np.testing.assert_allclose(mlp(x, w), np.maximum(x @ w, 0.0), rtol=1e-5)
    assert mlp.frontend_authority == "tracer"
    assert mlp.executed_plan_frontend == "ast"
    assert "synthetic compile failure" in mlp.executed_plan_fallback_reason
    assert mlp.cpu_plan is mlp._decoration_compile[0]


def test_a_legacy_call_after_a_traced_one_restores_the_decoration_plan(monkeypatch):
    mlp = _mlp()
    x, w = _inputs()
    mlp(x, w)
    assert mlp.executed_plan_frontend == "tracer"

    # A non-tensor signature keeps the AST candidate; it must not inherit the
    # previous call's traced specialization.
    monkeypatch.setattr(
        mlp, "_trace_frontend_capture",
        lambda *_a, **_k: pytest.fail("non-tensor signature was traced"))
    result = mlp(x.tolist(), w)
    np.testing.assert_allclose(result, np.maximum(x @ w, 0.0), rtol=1e-5)
    assert mlp.frontend_authority == "legacy_candidate_non_tensor_signature"
    assert mlp.executed_plan_frontend == "ast"
    assert mlp.cpu_plan is mlp._decoration_compile[0]
