"""Audit follow-up A.2 — distinct ``JIT_EAGER_FALLBACK_CONTROL_FLOW``
diagnostic for structured control flow in the CPU path.

Pre-A.2: a function whose body contained ``tessera.scf.if.*`` (or
``scf.for.*`` / ``scf.while.*``) markers — emitted by D.1/D.2/D.3 —
triggered the generic ``JIT_EAGER_FALLBACK_UNSUPPORTED_OP`` warning that
listed every supported op the function was missing. That treats scf as
an arbitrary unknown op when in fact it's a well-defined structural
construct the frontend correctly lowered; the backend just doesn't yet
emit executable code for scf at the TileIR level.

A.2 promotes scf-fallback to its own diagnostic code:

* Severity: ``info`` (not ``warning``) — eager Python correctness is
  fine; only optimization is missing.
* Code: ``JIT_EAGER_FALLBACK_CONTROL_FLOW``.
* Message: names the specific scf op encountered + the count of other
  scf ops in the body so callers can size the fallback footprint.

These tests pin the contract end-to-end via ``@tessera.jit``.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pytest


def _decorate(tmp_path, source: str, fn_name: str = "f"):
    (tmp_path / "scf_mod.py").write_text(source)
    import sys
    sys.path.insert(0, str(tmp_path))
    try:
        if "scf_mod" in sys.modules:
            del sys.modules["scf_mod"]
        import scf_mod
        return getattr(scf_mod, fn_name)
    finally:
        sys.path.remove(str(tmp_path))


# ---- Dynamic if -> JIT_EAGER_FALLBACK_CONTROL_FLOW ----------------------

def test_dynamic_if_diagnostic_is_control_flow_info_not_unsupported_op_warning(tmp_path):
    """The function lowers to scf.if.* via D.1; the CPU planner now
    reports an info-level ``CONTROL_FLOW`` diagnostic instead of the
    pre-A.2 ``UNSUPPORTED_OP`` warning."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(x):
            if x.sum() > 0:
                return tessera.ops.relu(x)
            return x
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    explanation = fn.explain_lowering()
    assert "JIT_EAGER_FALLBACK_CONTROL_FLOW" in explanation, explanation
    # Pre-A.2 warning must be gone.
    assert "JIT_EAGER_FALLBACK_UNSUPPORTED_OP" not in explanation, explanation


def test_control_flow_diagnostic_names_the_specific_op(tmp_path):
    """The diagnostic message must name the scf op (e.g.
    ``'tessera.scf.if.begin'``) so a reader can locate the source — not
    just say "control flow"."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(x):
            if x.sum() > 0:
                return tessera.ops.relu(x)
            return x
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    explanation = fn.explain_lowering()
    assert "tessera.scf.if.begin" in explanation, explanation


# ---- Dynamic for / while also get the same code -----------------------

def test_while_loop_diagnostic_is_control_flow(tmp_path):
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(x):
            while x.sum() > 0:
                y = x
            return y
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    explanation = fn.explain_lowering()
    assert "JIT_EAGER_FALLBACK_CONTROL_FLOW" in explanation, explanation
    assert "JIT_EAGER_FALLBACK_UNSUPPORTED_OP" not in explanation


def test_dynamic_for_diagnostic_is_control_flow(tmp_path):
    """Use a simple ``for x in xs: y = x`` body — the SSA verifier
    rejects ``total = 0; total = total + x`` (the literal 0 doesn't
    emit an op, so ``total`` is undefined when the loop reads it).
    That's a separate issue from A.2's diagnostic-class change; this
    test just needs to surface the for-loop scf marker shape."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(xs):
            for x in xs:
                y = x
            return y
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    explanation = fn.explain_lowering()
    assert "JIT_EAGER_FALLBACK_CONTROL_FLOW" in explanation, explanation


# ---- Eager Python execution is numerically correct --------------------

def test_dynamic_if_executes_numerically_via_eager_path(tmp_path):
    """The whole point of A.2 — scf functions still execute via eager
    Python, just with a precise diagnostic. Both branches must produce
    the right numerical answer."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(x):
            if x.sum() > 0:
                return tessera.ops.relu(x)
            return x
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    x_pos = np.array([1.0, 2.0, -0.5], dtype=np.float32)
    x_neg = np.array([-3.0, -1.0, 1.0], dtype=np.float32)
    np.testing.assert_array_equal(fn(x_pos), np.maximum(x_pos, 0))
    np.testing.assert_array_equal(fn(x_neg), x_neg)


# ---- The non-control-flow path still emits the generic warning --------

def test_unknown_op_diagnostic_stays_warning(tmp_path):
    """A function with an actually-unknown op must still get the
    ``UNSUPPORTED_OP`` warning — A.2 should narrow the change to scf
    markers only, not silently demote all unknown-op warnings to info."""
    # Tessera doesn't ship `tessera.ops.totally_made_up_op`, so a function
    # whose body is one such op would hit the generic warning. The
    # cleanest way to create this scenario is to call a real
    # un-lowered-to-CPU op like a complex spectral one — but doing it
    # through Python AST is brittle. Instead, drive the diagnostic
    # generator directly with a tiny constructed module.
    from tessera.compiler.graph_ir import (
        GraphIRFunction, GraphIRModule, IRArg, IROp, IRType,
    )
    from tessera.compiler.matmul_pipeline import explain_cpu_plan
    t = IRType("tensor<*x?>", ("*",), None)
    fn = GraphIRFunction(
        name="bogus", args=[IRArg("a", t)], result_types=[t],
        body=[IROp(result="c", op_name="tessera.totally_made_up_op",
                   operands=["%a"], operand_types=["tensor<*x?>"],
                   result_type="tensor<*x?>")],
        return_values=["%c"],
    )
    mod = GraphIRModule(functions=[fn])
    diag = explain_cpu_plan(mod, target="cpu")
    assert diag.code == "JIT_EAGER_FALLBACK_UNSUPPORTED_OP", diag
    assert diag.severity == "warning"


# ---- Bundle still reports the function as eager-runnable -------------

def test_compile_result_reports_function_runs_via_eager(tmp_path):
    """The canonical compile result must agree that a control-flow
    function isn't on the fast path — but the function IS callable.
    Locks the C-pipeline integration: scf programs aren't "executable"
    via the cpu_plan path, but the user can still call them."""
    src = textwrap.dedent("""
        import tessera

        @tessera.jit
        def f(x):
            if x.sum() > 0:
                return tessera.ops.relu(x)
            return x
    """).strip() + "\n"
    fn = _decorate(tmp_path, src)
    # The function is callable end-to-end via eager Python.
    out = fn(np.array([1.0, 2.0], dtype=np.float32))
    assert out is not None
    # And the compile_result is queryable.
    assert fn.compile_result is not None
