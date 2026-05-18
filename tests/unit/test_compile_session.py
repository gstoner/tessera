"""M2 Step 1 — :class:`CompileSession` + ``compile_session()`` scope.

Locks the M2 contract:

  - Empty scope → ``value_kind == "empty"``.
  - Single-kind scope (only tensor reports, or only multivector
    reports) → the session's ``value_kind`` is that kind.
  - Mixed scope → ``value_kind == "mixed"`` and
    ``has_mixed_boundary() is True``.
  - Reports preserve their individual ``value_kind`` fields —
    the session NEVER mutates per-report state.
  - The artifact-hash index correctly groups reports by IR hash.
  - Session-level diagnostics distinguish from per-report ones.
"""

from __future__ import annotations

import pytest

from tessera.compiler.compile_report import (
    CompileReport,
    FRONTEND_CLIFFORD_JIT,
    FRONTEND_TESSERA_JIT,
    VALUE_KIND_MIXED,
    VALUE_KIND_MULTIVECTOR,
    VALUE_KIND_TENSOR,
    emit_compile_report,
)
from tessera.compiler.compile_session import (
    SESSION_VALUE_KIND_EMPTY,
    SESSION_VALUE_KIND_MIXED,
    SESSION_VALUE_KIND_MULTIVECTOR,
    SESSION_VALUE_KIND_TENSOR,
    CompileSession,
    SessionDiagnostic,
    compile_session,
)


# ---------------------------------------------------------------------------
# Empty scope
# ---------------------------------------------------------------------------

def test_empty_session_has_value_kind_empty() -> None:
    with compile_session() as session:
        pass
    assert session.value_kind() == SESSION_VALUE_KIND_EMPTY
    assert session.frontends() == set()
    assert session.targets() == set()
    assert not session.has_mixed_boundary()


# ---------------------------------------------------------------------------
# Single-kind scopes
# ---------------------------------------------------------------------------

def _tensor_report(pid: str, target: str = "cpu") -> CompileReport:
    return CompileReport(
        program_id=pid, source=f"test.{pid}",
        frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR, target=target,
        ir_hashes={"graph_ir": f"hash_{pid}"},
        target_decision={target: f"decision_for_{pid}"},
    )


def _multivector_report(pid: str, target: str = "apple_gpu") -> CompileReport:
    return CompileReport(
        program_id=pid, source=f"test.{pid}",
        frontend=FRONTEND_CLIFFORD_JIT,
        value_kind=VALUE_KIND_MULTIVECTOR, target=target,
        ir_hashes={"graph_ir": f"hash_{pid}"},
        target_decision={target: f"decision_for_{pid}"},
    )


def test_single_kind_tensor_session() -> None:
    with compile_session() as session:
        emit_compile_report(_tensor_report("a"))
        emit_compile_report(_tensor_report("b"))
    assert session.value_kind() == SESSION_VALUE_KIND_TENSOR
    assert session.frontends() == {FRONTEND_TESSERA_JIT}
    assert not session.has_mixed_boundary()


def test_single_kind_multivector_session() -> None:
    with compile_session() as session:
        emit_compile_report(_multivector_report("x"))
        emit_compile_report(_multivector_report("y"))
    assert session.value_kind() == SESSION_VALUE_KIND_MULTIVECTOR
    assert session.frontends() == {FRONTEND_CLIFFORD_JIT}
    assert not session.has_mixed_boundary()


# ---------------------------------------------------------------------------
# Mixed sessions
# ---------------------------------------------------------------------------

def test_mixed_session_is_mixed() -> None:
    """Tensor + multivector reports in the same scope → mixed."""
    with compile_session() as session:
        emit_compile_report(_tensor_report("a"))
        emit_compile_report(_multivector_report("b"))
    assert session.value_kind() == SESSION_VALUE_KIND_MIXED
    assert session.has_mixed_boundary()
    # Schema parity preserved: per-report value_kind unchanged.
    assert {r.value_kind for r in session.reports} == {
        VALUE_KIND_TENSOR, VALUE_KIND_MULTIVECTOR,
    }


def test_explicit_mixed_value_kind_makes_session_mixed() -> None:
    """A single report with value_kind='mixed' (e.g., the
    rotor_sandwich_ebt_tiny canonical) makes the whole session mixed."""
    composite = CompileReport(
        program_id="composite", source="test",
        frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_MIXED, target="apple_gpu",
    )
    with compile_session() as session:
        emit_compile_report(composite)
    assert session.value_kind() == SESSION_VALUE_KIND_MIXED


# ---------------------------------------------------------------------------
# Per-report state is preserved (Decision #15a schema parity)
# ---------------------------------------------------------------------------

def test_session_does_not_mutate_per_report_value_kind() -> None:
    """Decision #15a: value_kind is normative per-report.  The
    session must NOT flatten reports to a single kind — each
    report keeps its declared kind, even if the session-level
    reduction is mixed."""
    tensor = _tensor_report("t")
    mv = _multivector_report("m")
    with compile_session() as session:
        emit_compile_report(tensor)
        emit_compile_report(mv)
    # Per-report kinds unchanged.
    assert session.reports[0].value_kind == VALUE_KIND_TENSOR
    assert session.reports[1].value_kind == VALUE_KIND_MULTIVECTOR
    # And the session reduction is mixed without rewriting them.
    assert session.value_kind() == SESSION_VALUE_KIND_MIXED


# ---------------------------------------------------------------------------
# Artifact-hash index
# ---------------------------------------------------------------------------

def test_artifact_index_groups_reports_by_ir_hash() -> None:
    """Two reports with the same IR hash group under one key."""
    r1 = CompileReport(
        program_id="A", source="t", frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR, target="cpu",
        ir_hashes={"graph_ir": "abc"},
    )
    r2 = CompileReport(
        program_id="B", source="t", frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR, target="cpu",
        ir_hashes={"graph_ir": "abc"},
    )
    r3 = CompileReport(
        program_id="C", source="t", frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR, target="cpu",
        ir_hashes={"graph_ir": "xyz"},
    )
    with compile_session() as session:
        emit_compile_report(r1)
        emit_compile_report(r2)
        emit_compile_report(r3)
    assert "graph_ir:abc" in session.artifact_index
    assert sorted(session.artifact_index["graph_ir:abc"]) == ["A", "B"]
    assert session.artifact_index["graph_ir:xyz"] == ["C"]


def test_artifact_index_separates_per_layer_hashes() -> None:
    """A report with multiple ir_hashes layers contributes to each
    layer's bucket independently."""
    r = CompileReport(
        program_id="multi", source="t", frontend=FRONTEND_TESSERA_JIT,
        value_kind=VALUE_KIND_TENSOR, target="cpu",
        ir_hashes={"graph_ir": "h1", "tile_ir": "h2"},
    )
    with compile_session() as session:
        emit_compile_report(r)
    assert "graph_ir:h1" in session.artifact_index
    assert "tile_ir:h2" in session.artifact_index


# ---------------------------------------------------------------------------
# Target-decision cache
# ---------------------------------------------------------------------------

def test_target_decision_cache_dedups_per_program_target_pair() -> None:
    with compile_session() as session:
        emit_compile_report(_tensor_report("a", target="cpu"))
        emit_compile_report(_tensor_report("a", target="cpu"))  # duplicate
    # Single entry — second call should not overwrite.
    assert ("a", "cpu") in session.target_decisions


def test_target_decision_cache_covers_every_seen_target() -> None:
    with compile_session() as session:
        emit_compile_report(_tensor_report("a", target="cpu"))
        emit_compile_report(_tensor_report("a", target="apple_gpu"))
    assert ("a", "cpu") in session.target_decisions
    assert ("a", "apple_gpu") in session.target_decisions


# ---------------------------------------------------------------------------
# Session diagnostics
# ---------------------------------------------------------------------------

def test_emit_diagnostic_appends_with_provenance() -> None:
    with compile_session() as session:
        session.emit_diagnostic(
            message="example finding",
            code="M2_EXAMPLE",
            source_span=(42, 7),
        )
    assert len(session.diagnostics) == 1
    d = session.diagnostics[0]
    assert isinstance(d, SessionDiagnostic)
    assert d.code == "M2_EXAMPLE"
    assert d.source_span == (42, 7)
    assert "line 42, col 7" in d.format()


def test_has_errors_flag() -> None:
    with compile_session() as session:
        session.emit_diagnostic(
            severity="warning", message="just a warning", code="X",
        )
        assert not session.has_errors
        session.emit_diagnostic(message="real error", code="Y")
        assert session.has_errors


# ---------------------------------------------------------------------------
# refresh() is idempotent
# ---------------------------------------------------------------------------

def test_refresh_is_idempotent() -> None:
    with compile_session() as session:
        emit_compile_report(_tensor_report("a"))
        session.refresh()
        before = dict(session.artifact_index)
        session.refresh()
        after = dict(session.artifact_index)
    assert before == after


# ---------------------------------------------------------------------------
# Schema parity — the M2 acceptance criterion
# ---------------------------------------------------------------------------

def test_three_frontends_in_one_session_share_envelope_shape() -> None:
    """M2 acceptance: 'A GA demo and a tensor demo produce the
    same report envelope.'  We test it with all three frontends
    by emitting synthetic reports; the integration test in
    test_compile_session_integration.py exercises the real
    decorators."""
    r_jit = _tensor_report("jit_demo")
    r_textual = CompileReport(
        program_id="textual_demo", source="t",
        frontend="textual", value_kind=VALUE_KIND_TENSOR, target="cpu",
        ir_hashes={"graph_ir": "tx"},
    )
    r_clifford = _multivector_report("clifford_demo")
    with compile_session() as session:
        emit_compile_report(r_jit)
        emit_compile_report(r_textual)
        emit_compile_report(r_clifford)
    # All three reports have the same field set — the envelope.
    canonical_keys = set(r_jit.as_dict().keys())
    for r in session.reports:
        assert set(r.as_dict().keys()) == canonical_keys, (
            f"{r.frontend} envelope drift: missing/extra keys"
        )
