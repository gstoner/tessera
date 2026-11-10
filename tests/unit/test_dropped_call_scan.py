"""The dropped-call guard and the unlowered-op refusal without the AST module.

E2E-REAL-6F: the native-target refusal reads the effect scan's resolved
``tessera.ops`` calls (invocation-aware), not the AST front end's dropped-call
list, and the dropped-call guard applies only to the plan the AST compiled --
a tracer-authority call executes the traced plan, which the AST's drops do not
truncate.
"""
from __future__ import annotations

import numpy as np
import pytest

import tessera
import tessera as ts
from tessera.compiler.effect_scan import scan_call_effects
from tessera.compiler.fallback import FallbackReason
from tessera.compiler.graph_ir import _OpExtractor
from tessera.compiler.jit import TesseraJitError

UNCATALOGUED = "jepa_l2_loss"


def _relu_helper(x):
    return tessera.ops.relu(x)


def _calls_a_module_helper(x):
    # The returned value is the dropped call, so the AST graph verifies but
    # returns nothing (a call nested in a tensor expression fails emission).
    return _relu_helper(tessera.ops.add(x, x))


def _x() -> np.ndarray:
    return np.array([-1.0, 2.0, 0.5, -3.0], dtype=np.float32)


def _expected(x: np.ndarray) -> np.ndarray:
    return np.maximum(x + x, 0.0)


# ── The guard protects the AST plan only ─────────────────────────────────────

def test_traced_plan_runs_a_body_whose_helper_the_ast_dropped():
    fn = tessera.jit(_calls_a_module_helper)
    # The AST front end drops the call to the helper, so its module is
    # truncated and compiles to no plan at all.
    assert "_relu_helper" in fn.graph_ir.frontend_unlowered_calls
    assert fn._decoration_compile[0] is None
    x = _x()
    np.testing.assert_allclose(np.asarray(fn(x)), _expected(x), rtol=1e-6)
    assert fn.frontend_authority == "tracer"
    assert fn.executed_plan_frontend == "tracer"
    assert fn.last_fallback_reason is None
    assert fn.cpu_plan.function_name == "_calls_a_module_helper"
    assert [op.op_name for op in fn.cpu_plan.ops] == ["tessera.add", "tessera.relu"]


def test_an_ast_plan_call_with_the_same_drop_still_runs_the_function():
    fn = tessera.jit(_calls_a_module_helper)
    x = _x()
    # A non-tensor signature keeps the AST candidate, whose plan is truncated.
    result = fn(x.tolist())
    np.testing.assert_allclose(np.asarray(result), _expected(x), rtol=1e-6)
    assert fn.executed_plan_frontend is None
    assert fn.last_fallback_reason == FallbackReason.FRONTEND_UNLOWERED_CALL


def test_a_truncated_ast_program_never_reaches_the_package_route(monkeypatch):
    fn = tessera.jit(target="apple_gpu")(_calls_a_module_helper)
    fn.dispatch_via_package = True
    fn.recognized_package = object()
    monkeypatch.setattr(
        fn, "_call_via_package",
        lambda *_a, **_k: pytest.fail("the AST package route ran"))
    monkeypatch.setattr(
        fn, "_apple_gpu_fast_call", lambda *_a, **_k: "fast path")
    x = _x()
    result = fn(x)
    # Either the traced plan reaches the fast path, or (no executable traced
    # plan on this host) the truncated AST program is replaced by the
    # function. The package route runs in neither case.
    if fn.executed_plan_frontend == "tracer":
        assert result == "fast path"
    else:
        np.testing.assert_allclose(np.asarray(result), _expected(x), rtol=1e-6)
        assert fn.last_fallback_reason == FallbackReason.FRONTEND_UNLOWERED_CALL


# ── The refusal reads the scan, not the AST drops ────────────────────────────

def _invokes_shadowed_helper(a):
    def helper(x):
        return ts.ops.jepa_l2_loss(x, x + 1)

    def outer(x):
        def helper(y):  # shadows the uncatalogued one
            return ts.ops.relu(y)

        return helper(x)

    return outer(a)


def _passes_an_anonymous_lambda(a):
    return ts.ops.relu((lambda x: ts.ops.jepa_l2_loss(x, x + 1))(a))


def _calls_twice_bound_name(a):
    def helper(x):
        return ts.ops.jepa_l2_loss(x, x + 1)

    def helper(x):  # noqa: F811 -- rebinding: either body may run
        return ts.ops.relu(x)

    return helper(a)


@pytest.mark.parametrize("body,expected", [
    (_invokes_shadowed_helper, ["relu"]),
    (_passes_an_anonymous_lambda, ["relu", UNCATALOGUED]),
    (_calls_twice_bound_name, [UNCATALOGUED, "relu"]),
])
def test_scan_ops_follow_invocation(body, expected):
    assert [name for name, _ in scan_call_effects(body).ops] == expected


def test_refusal_needs_no_ast_dropped_call_list(monkeypatch):
    def module_attr(a):
        return ts.ops.jepa_l2_loss(a, a + 1)

    monkeypatch.setattr(_OpExtractor, "resolved_unlowered_calls", lambda self: [])
    with pytest.raises(TesseraJitError, match="JIT_UNLOWERED_OP_NATIVE_TARGET"):
        ts.jit(target="apple_cpu")(module_attr)


def test_shadowed_uncatalogued_helper_is_not_refused():
    # AST emission may still fail on the nested helpers; what must not happen
    # is a refusal naming the uncatalogued op the body never reaches.
    try:
        ts.jit(target="apple_cpu")(_invokes_shadowed_helper)
    except TesseraJitError as exc:
        assert "JIT_UNLOWERED_OP_NATIVE_TARGET" not in str(exc)
