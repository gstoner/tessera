"""``@jit`` must never execute a graph the front end truncated.

The AST front end drops any call it cannot map to a Graph op -- raw numpy, a
user helper, an unresolved spelling -- and records the dotted name on
``GraphIRModule.frontend_unlowered_calls``. ``JIT_UNLOWERED_OP_NATIVE_TARGET``
(2026-10-11) refused a dropped ``tessera.ops`` call on a native target and
left every other drop as a recorded follow-up.

That follow-up was a wrong answer, not an eager fallback. For

    y = tessera.ops.relu(x)
    return np.exp(y)

the graph held ``relu`` and a bare ``return``; the default lane
(``jit_cpu_numpy``) and ``apple_cpu`` executed it and returned ``relu(x)``
with ``last_fallback_reason`` None. Now any dropped call sends the call to the
original Python function and records
``FallbackReason.FRONTEND_UNLOWERED_CALL``; ``native_required=True`` raises
instead, and so does a differentiation request, whose derivative would omit
the dropped calls.

Pinned here:

* the repro returns ``exp(relu(x))`` on every lane that decorates on this
  host, with the reason recorded;
* ``native_required=True`` raises ``TesseraNativeRequiredError`` carrying the
  reason and naming the dropped call;
* Decision #10a negative fixtures: a body of catalogued ops still runs its
  compiled lane with no fallback reason, and a dropped call inside a nested
  helper the body never invokes does not count.

No device is touched: every lane here falls back before dispatch.
"""

import numpy as np
import pytest

import tessera
from tessera import Tensor
from tessera.compiler.fallback import FallbackReason, TesseraNativeRequiredError

REASON = FallbackReason.FRONTEND_UNLOWERED_CALL


def _x() -> np.ndarray:
    return np.array([-1.0, 2.0, 0.5, -3.0], dtype=np.float32)


def _relu_then_numpy_exp(x: Tensor["N"]):
    y = tessera.ops.relu(x)
    return np.exp(y)


def _catalogued_only(x: Tensor["N"]):
    return tessera.ops.relu(x)


def _uninvoked_helper(x: Tensor["N"]):
    def unused(a):
        return np.exp(a)
    return tessera.ops.relu(x)


@pytest.mark.parametrize("target", [None, "cpu", "apple_cpu"])
def test_dropped_numpy_call_runs_the_whole_function(target):
    fn = tessera.jit(target=target)(_relu_then_numpy_exp) if target else tessera.jit(
        _relu_then_numpy_exp)
    assert "np.exp" in fn.graph_ir.frontend_unlowered_calls
    x = _x()
    np.testing.assert_allclose(np.asarray(fn(x)), np.exp(np.maximum(x, 0.0)), rtol=1e-6)
    assert fn.last_fallback_reason is REASON


def test_native_required_refuses_a_truncated_graph():
    fn = tessera.jit(native_required=True)(_relu_then_numpy_exp)
    with pytest.raises(TesseraNativeRequiredError) as info:
        fn(_x())
    assert info.value.reason is REASON
    assert "np.exp" in str(info.value)


def test_catalogued_body_keeps_its_compiled_lane():
    fn = tessera.jit(_catalogued_only)
    assert fn.graph_ir.frontend_unlowered_calls == []
    x = _x()
    np.testing.assert_allclose(np.asarray(fn(x)), np.maximum(x, 0.0))
    assert fn.last_fallback_reason is None


def test_uninvoked_helper_drop_does_not_count():
    fn = tessera.jit(_uninvoked_helper)
    assert fn.graph_ir.frontend_unlowered_calls == []
    x = _x()
    np.testing.assert_allclose(np.asarray(fn(x)), np.maximum(x, 0.0))
    assert fn.last_fallback_reason is None


# ── Native routes run before the guard (regression found on gfx1201) ───────

from tessera.compiler.nvfp4_expansion import nvfp4_expansion_policy  # noqa: E402


def _expand_with_policy_helper(codes, scales, globals_):
    return tessera.ops.nvfp4_expand(codes, scales, globals_,
        row_offsets=[0, 3, 7], numeric_policy=nvfp4_expansion_policy())


def test_native_route_runs_before_the_dropped_call_guard(monkeypatch):
    """`nvfp4_expansion_policy()` in a keyword argument is a dropped AST call,
    but the native gfx1201 route re-traces the live function and computes the
    whole body. Guarding before it sent the call to the eager oracle; the
    guard now applies only to the lanes that execute the AST graph."""
    fn = tessera.jit(target="rocm_gfx1201")(_expand_with_policy_helper)
    assert fn.graph_ir.frontend_unlowered_calls == ["nvfp4_expansion_policy"]
    monkeypatch.setattr(fn, "_try_source_attention_call", lambda args, kwargs: "native")

    def eager(*_args, **_kwargs):
        pytest.fail("a dropped AST call pre-empted a native route")

    monkeypatch.setattr(fn, "_fn", eager)
    assert fn(None, None, None) == "native"
    assert fn.last_fallback_reason is not REASON


def test_guard_still_refuses_the_ast_lanes_once_native_routes_decline():
    fn = tessera.jit(native_required=True)(_relu_then_numpy_exp)
    with pytest.raises(TesseraNativeRequiredError):
        fn(_x())
