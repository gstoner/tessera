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
    # The dropped call is the AST frontend's record; an unannotated program has
    # no graph_ir before a call (JIT_UNSPECIALIZED_SIGNATURE), so read the AST
    # oracle that records it. (Static NVFP4 extents need a uint8 annotation,
    # which the dtype gate refuses.)
    assert fn._ensure_legacy_graph_ir().frontend_unlowered_calls == [
        "nvfp4_expansion_policy"]
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


# ── Review on #1009: a real route's failed trace must decline, not raise ────

_UFUNC_SOURCE = "def f(x):\n    y = tessera.ops.exp(x)\n    return np.exp(y)\n"


@pytest.mark.parametrize("target", ["rocm_gfx1151", "rocm_gfx1201"])
def test_failed_native_trace_on_dropped_dataflow_reaches_the_fallback(target):
    """No mocks: the native descriptor route admits the AST graph, re-traces
    the function, and the tracer fails on `np.exp(<Tracer>)`. That failure is
    explained by the dropped call, so the route declines and the original
    function runs, instead of raising out of `__call__`."""
    namespace = {"tessera": tessera, "np": np}
    exec(_UFUNC_SOURCE, namespace)  # noqa: S102 -- literal test source
    fn = tessera.jit(target=target, source=_UFUNC_SOURCE)(namespace["f"])
    x = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(fn(x)), np.exp(np.exp(x)), rtol=1e-6)
    assert fn.last_fallback_reason is REASON


def test_a_trace_failure_without_a_dropped_call_still_raises(monkeypatch):
    """The decline is reserved for bodies the AST front end found incomplete;
    any other tracer failure keeps raising the plain error."""
    import importlib

    jit_module = importlib.import_module("tessera.compiler.jit")  # the module, not the decorator
    trace_module = importlib.import_module("tessera.compiler.trace")

    fn = tessera.jit(target="rocm_gfx1151")(_catalogued_only)
    assert fn.graph_ir.frontend_unlowered_calls == []

    def broken(*_args, **_kwargs):
        raise RuntimeError("tracer exploded")

    monkeypatch.setattr(trace_module, "trace", broken)
    with pytest.raises(jit_module.TesseraJitError, match="tracer frontend failed") as info:
        fn._trace_frontend_capture((_x(),), {})
    assert not isinstance(info.value, jit_module._DroppedCallTraceDecline)


def test_an_input_refusal_during_trace_is_not_swallowed_by_the_decline(monkeypatch):
    """A body with a dropped call whose trace fails for an unrelated reason
    (here an input refusal) must keep raising: only NumPy applied to a traced
    value is explained by the dropped call. Found on gfx1201, where the decline
    turned an NVFP4 input-dtype refusal into eager execution."""
    import importlib

    jit_module = importlib.import_module("tessera.compiler.jit")
    trace_module = importlib.import_module("tessera.compiler.trace")
    namespace = {"tessera": tessera, "np": np}
    # Declared signature, so the dropped call is readable before a call
    # (JIT_UNSPECIALIZED_SIGNATURE otherwise).
    source = _UFUNC_SOURCE.replace("def f(x):", 'def f(x: tessera.Tensor["N"]):')
    exec(source, namespace)  # noqa: S102 -- literal test source
    fn = tessera.jit(target="rocm_gfx1151", source=source)(namespace["f"])
    assert fn.graph_ir.frontend_unlowered_calls == ["np.exp"]

    def refuses(*_args, **_kwargs):
        raise TypeError("NVFP4 ingest codes must be uint8 rank two")

    monkeypatch.setattr(trace_module, "trace", refuses)
    with pytest.raises(jit_module.TesseraJitError, match="NVFP4 ingest codes") as info:
        fn._trace_frontend_capture((_x(),), {})
    assert not isinstance(info.value, jit_module._DroppedCallTraceDecline)


# ── A call statement whose result is unused (2026-11-23) ───────────────────
#
# ``log.append(x)`` / ``print(y)`` as a bare statement had no AST handler: the
# call vanished unrecorded, so every AST lane ran the body with the effect
# silently skipped on every call (the Apple routes included). It is now a
# recorded dropped call, so the guard above runs the original function.

_EFFECT_LOG: list = []


def _logs_then_relu(x: Tensor["N"]):
    _EFFECT_LOG.append(1)
    return tessera.ops.relu(x)


def _declares_then_relu(x: Tensor["N"]):
    """A docstring is an expression statement too; it is not a call."""
    tessera.require(tessera.constraint.Divisible("N", 4))
    return tessera.ops.relu(x)


def _catalogued_statement_then_relu(x: Tensor["N"]):
    tessera.ops.relu(x)
    return tessera.ops.relu(x)


@pytest.mark.parametrize("target", [None, "cpu", "apple_cpu"])
def test_an_unused_result_call_statement_runs_on_every_call(target):
    fn = tessera.jit(target=target)(_logs_then_relu) if target else tessera.jit(_logs_then_relu)
    assert fn.graph_ir.frontend_unlowered_calls == ["_EFFECT_LOG.append"]
    del _EFFECT_LOG[:]
    x = _x()
    for _ in range(2):
        np.testing.assert_allclose(np.asarray(fn(x)), np.maximum(x, 0.0))
    assert len(_EFFECT_LOG) == 2
    assert fn.last_fallback_reason is REASON


@pytest.mark.hardware_apple_gpu
def test_an_unused_result_call_statement_runs_on_the_apple_gpu_lane():
    fn = tessera.jit(target="apple_gpu")(_logs_then_relu)
    del _EFFECT_LOG[:]
    x = _x()
    fn(x)
    fn(x)
    assert len(_EFFECT_LOG) == 2
    assert fn.last_fallback_reason is REASON


def test_native_required_refuses_a_body_with_an_effect_statement():
    fn = tessera.jit(native_required=True)(_logs_then_relu)
    with pytest.raises(TesseraNativeRequiredError) as info:
        fn(_x())
    assert info.value.reason is REASON
    assert "_EFFECT_LOG.append" in str(info.value)


@pytest.mark.parametrize("body", [_declares_then_relu, _catalogued_statement_then_relu])
def test_declarations_docstrings_and_catalogued_statements_are_not_dropped_calls(body):
    fn = tessera.jit(body)
    assert fn.graph_ir.frontend_unlowered_calls == []
    x = _x()
    np.testing.assert_allclose(np.asarray(fn(x)), np.maximum(x, 0.0))
    assert fn.last_fallback_reason is None


# ── JIT-NATIVE-STATEMENT-EFFECT-2026-11-24: native routes replay a trace ────
# Every native route traces the body once per signature and replays the
# capture on warm calls, so a statement run for its effect ran once, while
# tracing (gfx1151, gfx1201: 1 of 3 calls), or once more than asked when the
# route traced and then declined (sm_120: 4 of 3). A body with one now
# declines every native route before any trace.

from tessera.compiler.effect_scan import scan_call_effects  # noqa: E402


def _helper_that_logs(x):
    _EFFECT_LOG.append(2)
    return tessera.ops.relu(x)


def _calls_a_logging_helper(x: Tensor["N"]):
    return _helper_that_logs(x)


def _numpy_value_and_pure_builtin_statement(x: Tensor["N"]):
    scale = np.float32(0.5)
    len(x)
    return tessera.ops.mul(x, scale)


def _uninvoked_logging_def(x: Tensor["N"]):
    def never(y):
        print(y)
    return tessera.ops.relu(x)


def test_scan_reports_statement_calls_in_the_body_and_its_helpers():
    assert scan_call_effects(_logs_then_relu).statement_calls == ("_EFFECT_LOG.append",)
    assert scan_call_effects(_calls_a_logging_helper).statement_calls == (
        "_helper_that_logs->_EFFECT_LOG.append",)


@pytest.mark.parametrize("body", [
    _numpy_value_and_pure_builtin_statement, _uninvoked_logging_def,
    _declares_then_relu, _catalogued_statement_then_relu, _relu_then_numpy_exp])
def test_scan_reports_no_statement_call_for_value_calls_and_declarations(body):
    assert scan_call_effects(body).statement_calls == ()


@pytest.mark.parametrize("target", ["nvidia_sm120", "rocm_gfx1151", "rocm_gfx1201"])
@pytest.mark.parametrize("body", [_logs_then_relu, _calls_a_logging_helper])
def test_a_statement_effect_declines_every_native_route_before_tracing(target, body, monkeypatch):
    fn = tessera.jit(target=target)(body)

    def traced(*_args, **_kwargs):
        pytest.fail("a native route traced a body it cannot replay whole")

    monkeypatch.setattr(fn, "_trace_frontend_capture", traced)
    monkeypatch.setattr(fn, "_try_source_attention_call", traced)
    monkeypatch.setattr(fn, "_try_traced_native_routes", traced)
    del _EFFECT_LOG[:]
    x = _x()
    for _ in range(3):
        np.testing.assert_allclose(np.asarray(fn(x)), np.maximum(x, 0.0))
    assert len(_EFFECT_LOG) == 3
    assert fn.last_fallback_reason is REASON


@pytest.mark.parametrize("target", ["nvidia_sm120", "rocm_gfx1201"])
def test_native_required_refuses_a_native_target_with_a_statement_effect(target):
    fn = tessera.jit(target=target, native_required=True)(_logs_then_relu)
    with pytest.raises(TesseraNativeRequiredError) as info:
        fn(_x())
    assert info.value.reason is REASON
    assert "_EFFECT_LOG.append" in str(info.value)
