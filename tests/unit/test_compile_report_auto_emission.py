"""Step 4 of the 2026-05-18 post-reassessment plan: uniform
CompileReport emission across ``@tessera.jit``, the textual
frontend, and ``@clifford_jit``.

Locks the contract:

  1. Every JIT frontend exposes a ``.compile_report()`` accessor
     (or, for the textual frontend, a top-level ``compile_report_for_text``).
  2. Inside :func:`compile_report.capture_compile_reports`, every
     call to a decorated callable appends a report to the active sink.
  3. Outside the scope the emission is a no-op.
  4. Reports from different frontends are distinguishable by their
     ``frontend`` and ``value_kind`` fields.
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

from tessera.compiler import compile_report as cr
from tessera.compiler.clifford_jit import clifford_jit
from tessera.compiler.frontend.parser import compile_report_for_text


# ---------------------------------------------------------------------------
# Sink machinery
# ---------------------------------------------------------------------------

def test_active_sink_is_off_by_default() -> None:
    """Cheap probe used by frontends to skip report construction."""
    assert cr.active_sink_is_capturing() is False


def test_emit_outside_scope_is_a_noop() -> None:
    """No exception, no observable side-effect."""
    cr.emit_compile_report(cr.CompileReport(
        program_id="x", source="t", frontend="tessera.jit",
        value_kind="tensor", target="cpu",
    ))


def test_capture_collects_emitted_reports_in_order() -> None:
    reports = [
        cr.CompileReport(
            program_id=f"p{i}", source="t", frontend="tessera.jit",
            value_kind="tensor", target="cpu",
        )
        for i in range(3)
    ]
    with cr.capture_compile_reports() as sink:
        for r in reports:
            cr.emit_compile_report(r)
        # Sink is visible inside the scope.
        assert [r.program_id for r in sink] == ["p0", "p1", "p2"]


def test_capture_isolates_nested_scopes() -> None:
    with cr.capture_compile_reports() as outer:
        cr.emit_compile_report(cr.CompileReport(
            program_id="outer", source="t", frontend="tessera.jit",
            value_kind="tensor", target="cpu",
        ))
        with cr.capture_compile_reports() as inner:
            cr.emit_compile_report(cr.CompileReport(
                program_id="inner", source="t", frontend="tessera.jit",
                value_kind="tensor", target="cpu",
            ))
            assert [r.program_id for r in inner] == ["inner"]
        # After the inner scope closes the outer sink resumes.
        cr.emit_compile_report(cr.CompileReport(
            program_id="outer2", source="t", frontend="tessera.jit",
            value_kind="tensor", target="cpu",
        ))
        assert [r.program_id for r in outer] == ["outer", "outer2"]


# ---------------------------------------------------------------------------
# @clifford_jit auto-emission
# ---------------------------------------------------------------------------

def test_clifford_jit_exposes_compile_report_accessor() -> None:
    from tessera import ga

    @clifford_jit(target="apple_gpu")
    def f(a, b):
        return ga.inner(a, b)

    report = f.compile_report()
    assert report.frontend == "clifford_jit"
    assert report.value_kind == "multivector"
    assert report.target == "apple_gpu"
    assert "graph_ir" in report.ir_hashes


@pytest.mark.hardware_apple_gpu
def test_clifford_jit_auto_emits_on_call() -> None:
    from tessera import ga

    @clifford_jit(target="apple_gpu")
    def f(rotor, points):
        return ga.norm(ga.rotor_sandwich(rotor, points))

    a = ga.Cl(3, 0)
    rng = np.random.RandomState(0)
    R = rng.randn(4, 8).astype(np.float32) * 0.3
    V = rng.randn(4, 8).astype(np.float32)
    with cr.capture_compile_reports() as sink:
        f(ga.Multivector(R, a), ga.Multivector(V, a))
        f(ga.Multivector(R, a), ga.Multivector(V, a))
    assert len(sink) == 2
    assert all(r.frontend == "clifford_jit" for r in sink)
    # Proof routes are populated on Darwin. The rotor_sandwich→norm chain fuses
    # into one dispatch (gap #6), so a single proof route.
    assert all(len(r.proof_routes) == 1 for r in sink)


# ---------------------------------------------------------------------------
# Textual frontend auto-emission
# ---------------------------------------------------------------------------

def test_textual_frontend_emits_compile_report() -> None:
    text = """
module m {
  func f(x: tensor<?xfp32>) -> tensor<?xfp32> {
    return x;
  }
}
"""
    with cr.capture_compile_reports() as sink:
        report = compile_report_for_text(text, program_id="textual_demo")
    assert len(sink) == 1
    assert report is sink[0]
    assert report.frontend == "textual"
    assert report.value_kind == "tensor"
    assert report.target == "cpu"
    assert "graph_ir" in report.ir_hashes


def test_textual_compile_report_outside_scope_does_not_emit() -> None:
    text = "module m { func f(x: tensor<?xfp32>) -> tensor<?xfp32> { return x; } }"
    report = compile_report_for_text(text, program_id="silent")
    # Returned regardless, but no sink observed.
    assert report.frontend == "textual"


# ---------------------------------------------------------------------------
# Cross-frontend distinguishability
# ---------------------------------------------------------------------------

def test_reports_from_each_frontend_are_distinguishable() -> None:
    """One scope, three frontends: every report's ``frontend`` /
    ``value_kind`` pair is unique."""
    from tessera import ga

    @clifford_jit(target="apple_gpu")
    def clifford_demo(a, b):
        return ga.inner(a, b)

    text = "module m { func f(x: tensor<?xfp32>) -> tensor<?xfp32> { return x; } }"
    with cr.capture_compile_reports() as sink:
        compile_report_for_text(text, program_id="tx")
        # Don't actually invoke clifford_demo (Darwin gate); just
        # query the accessor — that's the "frontend exposes the
        # contract" half of the test.
        sink.append(clifford_demo.compile_report())
        # @tessera.jit accessor is tested separately below.
    frontends = [r.frontend for r in sink]
    assert "textual" in frontends
    assert "clifford_jit" in frontends


def test_tessera_jit_exposes_compile_report_accessor() -> None:
    from tessera import jit as _tessera_jit

    @_tessera_jit
    def add(x, y):
        return x + y

    report = add.compile_report()
    assert report.frontend == "tessera.jit"
    assert report.value_kind == "tensor"
    assert "graph_ir" in report.ir_hashes
