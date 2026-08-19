"""A zero-function AST candidate module must not crash ``JitFn.__call__``.

``_jit_emit_graph_ir`` has two non-emitting paths that hand ``JitFn`` an empty
``GraphIRModule()``: the ``apple_gpu`` emission-failure trace-defer and the
``auto_batch`` skip. ``_establish_tracer_authority`` read
``legacy.functions[0].body`` unguarded, so either one turned an ordinary call
into a bare ``IndexError: list index out of range`` -- an unhandled interpreter
error escaping the compiler's front door.

The reachable producer is an unlowerable intermediate statement. ``_OpExtractor``
drops a statement it cannot emit (``visit_Assign`` falls through to
``generic_visit`` when ``_emit_expr`` returns ``None``), which leaves a later op
referencing a dangling operand; the function then fails verification with
``GRAPH_IR_UNDEFINED_OPERAND``. Non-apple_gpu targets re-raise that as a
``TesseraJitError`` at decoration; ``apple_gpu`` defers to the tracer and keeps
an empty module.

What is pinned here, per target:

* No target ever surfaces a raw ``IndexError`` (the regression).
* Non-apple_gpu targets raise ``TesseraJitError`` at decoration time.
* ``apple_gpu`` decorates and routes to the tracer. The tracer runs the body,
  so a construct the AST front end could not lower may well execute correctly
  there -- and when it cannot, the failure is a stable Tessera diagnostic
  naming the construct, never a raw Python exception (Decision #21).
* The Python call ABI survives the empty module. ``arg_names`` and
  ``_constraint_ir_args`` come from the signature rather than defaulting to
  empty, so keyword calls still bind and constraints are still re-checked.
* The two explicit differential gates materialize the AST oracle rather than
  rejecting the empty module, so they return a real verdict on the candidate.

Only the test that executes a kernel needs a Metal device; the error paths all
raise inside symbolic tracing, before any dispatch.
"""

import numpy as np
import pytest

import tessera as ts
from tessera import ops
from tessera.compiler.diagnostics import JitDiagnosticCode
from tessera.compiler.jit import TesseraJitError
from tessera.compiler.trace import TesseraTraceError

gpu = pytest.mark.hardware_apple_gpu

# Targets that depend on the AST Graph IR and therefore hard-fail decoration.
_IR_DEPENDENT_TARGETS = [None, "apple_cpu", "rocm"]


def _not_a_graph_op(v):
    """A plain Python callable -- not a registered Graph operation."""
    return v


def _decorate(target, fn):
    return ts.jit(fn) if target is None else ts.jit(target=target)(fn)


def _unlowerable_intermediate(x):
    y = _not_a_graph_op(x)   # dropped by _OpExtractor -> %y is never defined
    return ops.relu(y)       # -> dangling operand -> function fails verify


# --- the regression ---------------------------------------------------------- #

def test_apple_gpu_candidate_module_has_zero_functions():
    """The state that used to crash is reachable, and is what we think it is."""
    fn = _decorate("apple_gpu", _unlowerable_intermediate)
    assert fn._legacy_graph_ir is not None
    assert fn._legacy_graph_ir.functions == []
    assert fn._needs_trace is True


def test_zero_function_candidate_does_not_raise_indexerror():
    """``_establish_tracer_authority`` survives an empty candidate module.

    Called directly so the assertion is about the guard itself and does not
    depend on whether the tracer can then execute the body (or on a device).
    """
    fn = _decorate("apple_gpu", _unlowerable_intermediate)
    fn._establish_tracer_authority((np.zeros(3, np.float32),), {})
    # The decoration-time effect stands in for the body we could not read, and
    # routing records that the AST candidate was not superseded.
    assert fn.frontend_authority.startswith("legacy_candidate")


@pytest.mark.parametrize("target", _IR_DEPENDENT_TARGETS)
def test_ir_dependent_targets_raise_tessera_jit_error_at_decoration(target):
    """Never ``IndexError`` -- a Tessera diagnostic naming the failure."""
    with pytest.raises(TesseraJitError) as excinfo:
        _decorate(target, _unlowerable_intermediate)
    assert "GRAPH_IR_UNDEFINED_OPERAND" in str(excinfo.value)


@pytest.mark.parametrize("target", _IR_DEPENDENT_TARGETS + ["apple_gpu"])
def test_no_target_leaks_an_indexerror(target):
    """The end-to-end shape of the fix, decoration through call, per target."""
    try:
        fn = _decorate(target, _unlowerable_intermediate)
        fn(np.array([1.0, -2.0, 4.0], np.float32))
    except (TesseraJitError, TesseraTraceError):
        pass          # a stable Tessera diagnostic is the acceptable failure
    except IndexError as exc:                                # pragma: no cover
        pytest.fail(f"{target!r} leaked a raw IndexError: {exc}")


# --- Decision #21: the apple_gpu tracer lane reports, never crashes ---------- #

def test_tracer_failure_is_a_tessera_diagnostic_naming_the_construct():
    """A body the tracer cannot run raises a coded Tessera error, not
    ``AttributeError``, and quotes why the AST lane deferred."""

    def uses_an_unmodelled_tracer_method(x):
        y = _not_a_graph_op(x)      # unlowerable -> AST defers to the tracer
        if y.sum() > 0:             # Tracer models no `.sum()` -> AttributeError
            return ops.relu(y)
        return y

    fn = _decorate("apple_gpu", uses_an_unmodelled_tracer_method)
    with pytest.raises(TesseraJitError) as excinfo:
        fn(np.array([1.0, -2.0, 4.0], np.float32))
    message = str(excinfo.value)
    assert JitDiagnosticCode.APPLE_GPU_TRACE_FAILED.value in message
    # The construct the AST front end could not lower is named, not just the
    # dangling operand its removal left behind.
    assert "PY_FRONTEND_UNSUPPORTED" in message
    assert "not a registered Graph operation" in message
    # The underlying interpreter error is preserved, not swallowed.
    assert isinstance(excinfo.value.__cause__, AttributeError)


def test_trace_error_is_not_rewrapped():
    """``TesseraTraceError`` already names the construct and the remedy; the
    Decision #21 lift must not bury it behind a second layer."""

    @ts.jit(target="apple_gpu")
    def raw_if_on_traced_value(flag, x):
        if flag:
            return ops.silu(x)
        return ops.relu(x)

    with pytest.raises(TesseraTraceError, match="cannot branch on a traced"):
        raw_if_on_traced_value(
            np.array([1.0], np.float32), np.ones((1, 8), np.float32))


@gpu
def test_apple_gpu_tracer_executes_the_unlowerable_body_correctly():
    """The tracer RUNS the body, so a call the AST front end could not lower is
    not itself an error -- it computes the right answer."""
    fn = _decorate("apple_gpu", _unlowerable_intermediate)
    x = np.array([1.0, -2.0, 4.0], np.float32)
    np.testing.assert_allclose(fn(x), np.maximum(x, 0.0), rtol=1e-6, atol=1e-6)


# --- the call ABI survives an empty candidate module ------------------------ #

def test_call_abi_is_recovered_from_the_signature():
    """``arg_names`` is how keyword arguments are ordered and how constraints
    are re-checked; defaulting it to empty broke both silently."""
    deferred = _decorate("apple_gpu", _unlowerable_intermediate)
    ordinary = _decorate("apple_gpu", lambda x: ops.relu(x))
    assert deferred.arg_names == ["x"]
    assert deferred.arg_names == ordinary.arg_names
    assert len(deferred._constraint_ir_args) == 1


@gpu
def test_keyword_call_binds_on_a_trace_deferred_function():
    """The empty ABI made ``run_jit_traced`` collect zero arrays from a
    keyword call, so the body was invoked with no arguments at all."""
    fn = _decorate("apple_gpu", _unlowerable_intermediate)
    x = np.array([1.0, -2.0, 4.0], np.float32)
    np.testing.assert_allclose(
        fn(x=x), np.maximum(x, 0.0), rtol=1e-6, atol=1e-6)


# --- the explicit differential gates ----------------------------------------- #

def test_differential_gate_materializes_the_oracle_and_reports_a_verdict():
    """The gates read the AST candidate through ``_ensure_legacy_graph_ir``.
    An empty module is 'not materialized', not 'nothing to compare': rebuild
    it, then report what the comparison actually found.

    Here the AST front end really did drop a statement, so the honest verdict
    is a MISMATCH -- which must surface as a refusal to certify, never as a
    certificate and never as an ``IndexError``.
    """
    fn = _decorate("apple_gpu", _unlowerable_intermediate)
    assert fn._legacy_graph_ir.functions == []
    with pytest.raises(TesseraJitError, match="lacks differential parity"):
        fn.frontend_differential(np.array([1.0, -2.0, 4.0], np.float32))
    # The oracle was materialized on demand, exactly as for a tracer-authored
    # module whose candidate was never stored.
    assert len(fn._legacy_graph_ir.functions) == 1


def test_nonreexecuting_gate_materializes_the_oracle_and_reports_a_verdict():
    fn = _decorate("apple_gpu", _unlowerable_intermediate)
    with pytest.raises(TesseraJitError, match="lacks structural parity"):
        fn._frontend_nonreexecuting_certificate(
            (np.array([1.0, -2.0, 4.0], np.float32),), {},
            graph_consumers=("tessera.relu",))


def test_differential_gate_still_certifies_a_lowerable_function():
    """The recovery must not weaken the gate: a body both front ends agree on
    still produces a certificate that validates."""

    def lowerable(x):
        return ops.relu(x)

    fn = _decorate("apple_gpu", lowerable)
    certificate = fn.frontend_differential(np.array([1.0, -2.0, 4.0], np.float32))
    certificate.validate()
    contract = dict(certificate.contract)
    assert contract["structural_match"] and contract["numerical_match"]


# --- the second producer of an empty module ---------------------------------- #

def test_auto_batch_skip_is_the_other_empty_module_producer():
    """``auto_batch`` skips emission on purpose, so it reaches ``JitFn`` with
    the same empty module -- and must get the same intact call ABI. Unlike the
    trace-defer, nothing failed here, so this path had no diagnostic to hint
    that the ABI had quietly gone missing.
    """

    @ts.jit(target="apple_gpu", auto_batch=True)
    def batched(x):
        return ops.relu(x)

    assert batched._legacy_graph_ir.functions == []
    assert batched.arg_names == ["x"]
    assert len(batched._constraint_ir_args) == 1
    batched._establish_tracer_authority((np.zeros(3, np.float32),), {})


@gpu
def test_auto_batch_keyword_call_binds():
    @ts.jit(target="apple_gpu", auto_batch=True)
    def batched(x):
        return ops.relu(x)

    x = np.array([1.0, -2.0, 4.0], np.float32)
    np.testing.assert_allclose(
        batched(x=x), np.maximum(x, 0.0), rtol=1e-6, atol=1e-6)


def test_call_time_constraints_are_re_checked_on_a_trace_deferred_function():
    """Decision #4's call-time solver reads ``_constraint_ir_args``. An empty
    tuple made it return immediately, so a symbolic-dim violation on a
    trace-deferred function was never checked -- it fell through to whatever
    the tracer happened to say about the shapes downstream, if anything.
    """
    from tessera.compiler.constraints import TesseraConstraintError

    MK = ts.Tensor["M", "K"]
    KN = ts.Tensor["K", "N"]

    @ts.jit(target="apple_gpu")
    def deferred(a: MK, b: KN):
        y = _not_a_graph_op(a)
        return ops.matmul(y, b)

    assert deferred._legacy_graph_ir.functions == []
    assert [arg.dim_names for arg in deferred._constraint_ir_args] == [
        ("M", "K"), ("K", "N")]
    with pytest.raises(TesseraConstraintError, match="Inconsistent binding for dim 'K'"):
        deferred(np.ones((4, 8), np.float32), np.ones((16, 4), np.float32))


def test_authority_probe_does_not_trace_when_there_is_nothing_to_supersede():
    """``_establish_tracer_authority`` must not run the tracer on an empty
    module -- guarding the index alone is not enough.

    ``_trace_frontend_capture`` traces by RUNNING the body. On the auto_batch
    route that is a second GPU execution of the decode chain before the real
    call, which aborted inside MPSGraph rather than failing. There is also
    nothing to decide: the probe exists to let the tracer supersede an emitted
    module, and there is no emitted function here.
    """
    calls = []

    @ts.jit(target="apple_gpu", auto_batch=True)
    def batched(x):
        return ops.relu(x)

    original = type(batched)._trace_frontend_capture

    def counting(self, *args, **kwargs):                     # pragma: no cover
        calls.append(1)
        return original(self, *args, **kwargs)

    type(batched)._trace_frontend_capture = counting
    try:
        batched._establish_tracer_authority((np.zeros(3, np.float32),), {})
    finally:
        type(batched)._trace_frontend_capture = original

    assert calls == [], "the authority probe traced an empty-module function"
    assert batched.frontend_authority == "legacy_candidate_no_emitted_function"
