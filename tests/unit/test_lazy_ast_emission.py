"""The decoration-time AST Graph IR is built only when something reads it.

E2E-REAL-6F: a tracer-authority call replaces ``graph_ir`` and the executed
plan with its own, so it never needs the AST module. The decorator defers the
AST emission wherever no decoration-time consumer reads it; the first reader of
``graph_ir`` (or any field derived from it) runs the emission, and a call that
is a legacy candidate is such a reader.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera
import tessera as ts
from tessera.compiler.jit import TesseraJitError


def _body(x, y):
    return tessera.ops.relu(tessera.ops.add(x, y))


def _x() -> np.ndarray:
    return np.array([-1.0, 2.0, 0.5, -3.0], dtype=np.float32)


def _expected(x: np.ndarray) -> np.ndarray:
    return np.maximum(x + x, 0.0)


@pytest.mark.parametrize("target", [None, "cpu"])
def test_decoration_defers_the_ast_emission(target):
    fn = ts.jit(_body) if target is None else ts.jit(target=target)(_body)
    assert fn.ast_emission_pending
    # The call ABI comes from the signature, not the unbuilt module.
    assert fn.arg_names == ["x", "y"]


def test_a_tracer_authority_call_never_builds_the_ast():
    fn = ts.jit(_body)
    x = _x()
    for _ in range(2):
        np.testing.assert_allclose(fn(x, x), _expected(x), rtol=1e-6)
        assert fn.frontend_authority == "tracer"
        assert fn.executed_plan_frontend == "tracer"
        assert fn.last_fallback_reason is None
    assert fn.ast_emission_pending
    # The artifact describes the traced plan it ran.
    assert fn.runtime_artifact().metadata["function_name"] == "_body"
    assert fn.ast_emission_pending


def test_a_legacy_candidate_call_builds_it_and_both_plans_keep_running():
    fn = ts.jit(_body)
    x = _x()
    np.testing.assert_allclose(fn(x, x), _expected(x), rtol=1e-6)
    assert fn.ast_emission_pending
    # A non-tensor signature keeps the AST candidate, which needs the module.
    np.testing.assert_allclose(fn(x.tolist(), x.tolist()), _expected(x), rtol=1e-6)
    assert not fn.ast_emission_pending
    assert fn.frontend_authority == "legacy_candidate_non_tensor_signature"
    assert fn.executed_plan_frontend == "ast"
    # Back to a tensor call: the traced plan is selected again, and the AST
    # plan is restored as the decoration plan rather than lost.
    np.testing.assert_allclose(fn(x, x), _expected(x), rtol=1e-6)
    assert fn.executed_plan_frontend == "tracer"
    decoration_plan = fn._decoration_compile[0]
    assert decoration_plan is not None and decoration_plan.target_kind == "cpu"


def test_reading_graph_ir_builds_the_ast_module():
    fn = ts.jit(_body)
    module = fn.graph_ir
    assert not fn.ast_emission_pending
    assert [op.op_name for op in module.functions[0].body] == ["tessera.add", "tessera.relu"]
    assert fn._legacy_graph_ir is module
    assert fn.cpu_plan is not None and fn.compile_bundle is not None


@pytest.mark.parametrize("decorate", [
    lambda f: ts.jit(target="rocm")(f),
    lambda f: ts.jit(target="apple_cpu")(f),
    lambda f: ts.jit(target="apple_gpu")(f),
    lambda f: ts.jit(autodiff="reverse", wrt=["x"])(f),
])
def test_decoration_time_readers_keep_the_eager_emission(decorate):
    # A native target's legality check refuses unsupported ops while compiling
    # the module; the Apple targets also read it (tracer gate, package
    # recognizer, the AST-vs-traced executable comparison); a differentiation
    # request writes its intent into it.
    assert not decorate(_body).ast_emission_pending


def _drops_a_statement(x):
    y = _passthrough(x)  # dropped by the AST front end -> dangling operand
    return tessera.ops.relu(y)


def _passthrough(v):
    return v


def test_an_emission_failure_reaches_every_reader():
    fn = ts.jit(_drops_a_statement)
    x = np.array([1.0, -2.0], np.float32)
    np.testing.assert_array_equal(fn(x), np.maximum(x, 0.0))
    # ``graph_ir`` is the traced module now; the decoration plan is the AST's.
    for _ in range(2):
        with pytest.raises(TesseraJitError, match="GRAPH_IR_UNDEFINED_OPERAND"):
            _ = fn._decoration_compile
    # A failed emission leaves the call path intact for tracer calls.
    np.testing.assert_array_equal(fn(x), np.maximum(x, 0.0))

