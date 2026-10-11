"""A native ``@jit`` target must refuse a ``tessera.ops`` call it cannot lower.

The AST front end drops any call it cannot map to a Graph op. For a
``tessera.ops`` callable with no ``op_catalog`` OpSpec that left an empty
``func.func`` body, joined the effect to ``top`` (so no tracer captured the
call either), and ``JitFn.__call__`` ended at ``self._fn`` -- eager numpy with
``last_fallback_reason`` None and no artifact. A ``target="apple_gpu"`` call
returned the right number while compiling nothing: a hollow green.

Decoration now raises ``JIT_UNLOWERED_OP_NATIVE_TARGET`` naming the op and
the target, mirroring the source-unavailable refusal. Pinned here:

* every native target refuses, whatever spelling reaches the op (module
  attribute, package alias, bare module-level name), and a verifier failure
  the dropped call causes is reported as the dropped op;
* an apple_gpu body routed to the Phase-F tracer is left to the tracer,
  which fails loudly (``JIT_APPLE_GPU_TRACE_FAILED``) rather than eagerly;
* the refusal survives a Graph IR cache hit, whose module carries no builder
  diagnostics;
* the reference lane (no target, or ``cpu``) keeps the eager path and still
  computes the reference value;
* Decision #10a negative fixtures: a native-target body built only from
  catalogued ops is NOT refused, and neither is one whose uncatalogued op
  sits in a nested ``def``/lambda the body never invokes;
* an invoked nested helper (one or two levels, ``def`` or lambda) IS
  refused: the frontend does not inline helpers, so the helper's dropped
  calls are what would run eagerly.

No device is touched: refusals happen at decoration, and the tracer case
fails while tracing, before any dispatch.
"""

import numpy as np
import pytest

import tessera as ts
import tessera as T
from tessera.compiler import clear_graph_ir_cache, graph_ir_cache_stats
from tessera.compiler.diagnostics import JitDiagnosticCode
from tessera.compiler.jit import TesseraJitError
from tessera.compiler.op_catalog import get_op_spec
_bare_jepa = ts.ops.jepa_l2_loss

CODE = JitDiagnosticCode.UNLOWERED_OP_NATIVE_TARGET.value
NATIVE_TARGETS = ("apple_gpu", "apple_cpu", "rocm")
UNCATALOGUED = "jepa_l2_loss"


def _a() -> np.ndarray:
    return np.random.default_rng(0).standard_normal((4, 5)).astype(np.float32)


def _module_attr(a):
    return ts.ops.jepa_l2_loss(a, a + 1)


def _package_alias(a):
    return T.ops.jepa_l2_loss(a, a + 1)


def _bare_name(a):
    return _bare_jepa(a, a + 1)


def _nested_in_catalogued(a):
    return ts.ops.relu(ts.ops.jepa_l2_loss(a, a + 1))


def _catalogued_only(a):
    return ts.ops.relu(a)


def _uninvoked_nested_def(a):
    def helper(x):
        return ts.ops.jepa_l2_loss(x, x + 1)

    return ts.ops.relu(a)


def _invoked_nested_def(a):
    def helper(x):
        return ts.ops.jepa_l2_loss(x, x + 1)

    return helper(a)


def _invoked_two_level_helper(a):
    def inner(x):
        return ts.ops.jepa_l2_loss(x, x + 1)

    def outer(x):
        return inner(x)

    return outer(a)


def _uninvoked_lambda(a):
    helper = lambda x: ts.ops.jepa_l2_loss(x, x + 1)  # noqa: E731
    return ts.ops.relu(a)


def _invoked_lambda(a):
    helper = lambda x: ts.ops.jepa_l2_loss(x, x + 1)  # noqa: E731
    return helper(a)


def _brier(p, y):
    return ts.ops.brier_loss(p, y)


def _binary_brier(p, y):
    return ts.ops.binary_brier_loss(p, y)


def _ordinal_logistic(p, y):
    return ts.ops.ordinal_logistic_loss(p, y)


def _soft_binned_ece(p, y):
    return ts.ops.soft_binned_ece(p, y)


def test_fixture_op_is_uncatalogued() -> None:
    # Precondition: if this op ever gains an OpSpec, pick another tessera.ops
    # name outside op_catalog for the fixtures below.
    assert get_op_spec(UNCATALOGUED) is None
    assert callable(getattr(ts.ops, UNCATALOGUED))


@pytest.mark.parametrize("target", NATIVE_TARGETS)
@pytest.mark.parametrize(
    "body,spelling",
    [
        (_module_attr, "ts.ops.jepa_l2_loss"),
        (_package_alias, "T.ops.jepa_l2_loss"),
        (_bare_name, "_bare_jepa"),
    ],
)
def test_native_target_refuses_uncatalogued_op(body, spelling, target) -> None:
    with pytest.raises(TesseraJitError) as info:
        ts.jit(target=target)(body)
    message = str(info.value)
    assert f"[{CODE}]" in message
    assert f"tessera.ops.{UNCATALOGUED}" in message
    assert repr(spelling) in message
    assert repr(target) in message
    assert "no op_catalog OpSpec" in message


@pytest.mark.parametrize("target", ("apple_cpu", "rocm"))
def test_dangling_operand_is_reported_as_the_dropped_op(target) -> None:
    # Nested under a catalogued op, the dropped call leaves `tessera.relu`
    # with an unresolved operand and verification fails. The refusal names
    # the dropped op (the cause), not the dangling operand (the symptom).
    with pytest.raises(TesseraJitError) as info:
        ts.jit(target=target)(_nested_in_catalogued)
    assert f"[{CODE}]" in str(info.value)
    assert f"tessera.ops.{UNCATALOGUED}" in str(info.value)


def test_apple_gpu_tracer_route_is_left_to_the_tracer() -> None:
    # apple_gpu defers a body that fails AST emission to the Phase-F tracer,
    # which is its call-time authority. Decoration does not pre-empt it, and
    # the call fails loudly instead of returning an eager value.
    fn = ts.jit(target="apple_gpu")(_nested_in_catalogued)
    assert fn._needs_trace
    with pytest.raises(TesseraJitError, match="JIT_APPLE_GPU_TRACE_FAILED"):
        fn(_a())


def test_refusal_survives_graph_ir_cache_hit() -> None:
    clear_graph_ir_cache()
    try:
        with pytest.raises(TesseraJitError, match=CODE):
            ts.jit(target="apple_gpu")(_module_attr)
        hits = graph_ir_cache_stats()["hits"]
        with pytest.raises(TesseraJitError, match=CODE):
            ts.jit(target="apple_gpu")(_module_attr)
        # The second decoration must have been served from the cache, or this
        # test would not exercise the diagnostics-free path.
        assert graph_ir_cache_stats()["hits"] == hits + 1
    finally:
        clear_graph_ir_cache()


@pytest.mark.parametrize("target", (None, "cpu"))
def test_reference_lane_keeps_eager_path(target) -> None:
    a = _a()
    fn = ts.jit(_module_attr) if target is None else ts.jit(target=target)(_module_attr)
    np.testing.assert_allclose(fn(a), _module_attr(a))


@pytest.mark.parametrize("target", NATIVE_TARGETS)
def test_catalogued_body_is_not_refused(target) -> None:
    # Decision #10a: the correct output for a fully catalogued body is no
    # refusal and nothing recorded as dropped.
    fn = ts.jit(target=target)(_catalogued_only)
    assert fn.graph_ir.frontend_unlowered_calls == []


@pytest.mark.parametrize("target", NATIVE_TARGETS)
@pytest.mark.parametrize("body", (_uninvoked_nested_def, _uninvoked_lambda))
def test_uninvoked_nested_helper_is_not_refused(body, target) -> None:
    # Decision #10a: a helper the body never calls never executes, so its
    # uncatalogued op is not a reason to refuse the outer function.
    fn = ts.jit(target=target)(body)
    assert fn.graph_ir.frontend_unlowered_calls == []


@pytest.mark.parametrize("target", NATIVE_TARGETS)
@pytest.mark.parametrize(
    "body", (_invoked_nested_def, _invoked_two_level_helper, _invoked_lambda))
def test_invoked_nested_helper_is_refused(body, target) -> None:
    # The frontend does not inline helpers, so invoking one drops the call;
    # the helper's own dropped tessera.ops calls are what would run eagerly.
    with pytest.raises(TesseraJitError) as info:
        ts.jit(target=target)(body)
    assert f"[{CODE}]" in str(info.value)
    assert f"tessera.ops.{UNCATALOGUED}" in str(info.value)


@pytest.mark.parametrize("target", NATIVE_TARGETS)
@pytest.mark.parametrize(
    "body,op",
    [
        (_brier, "brier_loss"),
        (_binary_brier, "binary_brier_loss"),
        (_ordinal_logistic, "ordinal_logistic_loss"),
        (_soft_binned_ece, "soft_binned_ece"),
    ],
)
def test_catalogued_decision_head_losses_pass_the_gate(body, op, target) -> None:
    # gstoner/tessera#938 catalogued these; the gate must not refuse them.
    assert get_op_spec(op) is not None
    fn = ts.jit(target=target)(body)
    assert fn.graph_ir.frontend_unlowered_calls == []
    assert len(fn.graph_ir.functions[0].body) == 1
