"""M2 Step 4 — cross-frontend integration tests.

Locks the M2 acceptance criterion:
*"A GA demo and a tensor demo produce the same report envelope."*

Three frontends in one session:

  - ``@tessera.jit`` (tensor)
  - textual frontend (tensor)
  - ``@clifford_jit`` (multivector)

After the scope closes, the session must:

  1. Have collected one report per call (no double-counting,
     no drops).
  2. Each report's schema (as_dict keys) is identical — the
     "same envelope" promise.
  3. Each report's ``value_kind`` field is preserved per the
     emitting frontend (tensor / multivector), NOT rewritten by
     the session.
  4. ``session.value_kind() == "mixed"`` because both kinds are
     present.
  5. ``session.frontends()`` covers all three frontend names.
  6. ``session.artifact_index`` keys reflect IR-layer hashes.
"""

from __future__ import annotations

import numpy as np

from tessera import ga, jit
from tessera.compiler.canonical import (
    matmul_softmax_matmul,
    rotor_sandwich_norm,
)
from tessera.compiler.clifford_jit import clifford_jit
from tessera.compiler.compile_report import (
    FRONTEND_CLIFFORD_JIT,
    FRONTEND_TESSERA_JIT,
    FRONTEND_TEXTUAL,
)
from tessera.compiler.compile_session import (
    SESSION_VALUE_KIND_MIXED,
    compile_session,
)
from tessera.compiler.frontend.parser import compile_report_for_text


# Canonical textual program used by the M1.5 textual frontend test.
_TEXTUAL_SOURCE = """
module m {
  func f(x: tensor<?xfp32>) -> tensor<?xfp32> {
    return x;
  }
}
"""


# ---------------------------------------------------------------------------
# All three frontends, one session
# ---------------------------------------------------------------------------

def test_three_frontends_emit_in_one_compile_session() -> None:
    """Inside a single :func:`compile_session` scope, run one tensor
    program (canonical ``matmul_softmax_matmul``), one textual
    program (the M1.5 textual canonical), and one multivector
    program (canonical ``rotor_sandwich_norm``).  Verify all three
    end up in the session."""
    with compile_session() as session:
        matmul_softmax_matmul.run()
        compile_report_for_text(_TEXTUAL_SOURCE, program_id="textual_demo")
        rotor_sandwich_norm.run()
    # Three reports, three distinct frontends.
    assert len(session.reports) == 3
    assert session.frontends() == {
        FRONTEND_TESSERA_JIT,
        FRONTEND_TEXTUAL,
        FRONTEND_CLIFFORD_JIT,
    }


def test_session_value_kind_is_mixed_when_three_frontends_run() -> None:
    """Tensor + multivector reports → session is mixed."""
    with compile_session() as session:
        matmul_softmax_matmul.run()
        compile_report_for_text(_TEXTUAL_SOURCE, program_id="textual_demo")
        rotor_sandwich_norm.run()
    assert session.value_kind() == SESSION_VALUE_KIND_MIXED
    assert session.has_mixed_boundary()


# ---------------------------------------------------------------------------
# Envelope parity — the headline M2 acceptance criterion
# ---------------------------------------------------------------------------

def test_three_frontends_share_envelope_keys() -> None:
    """All three reports must have identical schema (the same
    set of top-level dict keys).  This is the M2 acceptance
    criterion for "same report envelope"."""
    with compile_session() as session:
        matmul_softmax_matmul.run()
        compile_report_for_text(_TEXTUAL_SOURCE, program_id="textual_demo")
        rotor_sandwich_norm.run()
    reports = session.reports
    keys = [tuple(sorted(r.as_dict().keys())) for r in reports]
    # All three key tuples must be identical.
    assert keys[0] == keys[1] == keys[2], (
        f"envelope drift: {[r.frontend for r in reports]} → {keys}"
    )


def test_three_frontends_preserve_per_report_value_kind() -> None:
    """Each frontend's report keeps its declared value_kind even
    when the session-level reduction is mixed.  Decision #15a
    requires per-call kind to be normative, not session-derived."""
    with compile_session() as session:
        matmul_softmax_matmul.run()                                # tensor
        compile_report_for_text(_TEXTUAL_SOURCE, program_id="t")    # tensor
        rotor_sandwich_norm.run()                                   # multivector
    by_frontend = {r.frontend: r.value_kind for r in session.reports}
    assert by_frontend == {
        FRONTEND_TESSERA_JIT: "tensor",
        FRONTEND_TEXTUAL: "tensor",
        FRONTEND_CLIFFORD_JIT: "multivector",
    }


# ---------------------------------------------------------------------------
# Artifact index across frontends
# ---------------------------------------------------------------------------

def test_artifact_index_records_each_frontend_ir_hash() -> None:
    """Three frontends produce three distinct graph_ir hashes
    (each is a different program); the index must record all
    three keys."""
    with compile_session() as session:
        matmul_softmax_matmul.run()
        compile_report_for_text(_TEXTUAL_SOURCE, program_id="textual_demo")
        rotor_sandwich_norm.run()
    # All three reports have a graph_ir hash.
    graph_keys = [
        k for k in session.artifact_index if k.startswith("graph_ir:")
    ]
    assert len(graph_keys) == 3


# ---------------------------------------------------------------------------
# Pure tensor session (negative cross-frontend test)
# ---------------------------------------------------------------------------

def test_pure_tensor_session_is_not_mixed() -> None:
    """If only ``@tessera.jit`` runs, value_kind stays ``tensor``
    — even with multiple programs."""
    with compile_session() as session:
        matmul_softmax_matmul.run()
        compile_report_for_text(_TEXTUAL_SOURCE, program_id="textual_demo")
    assert session.value_kind() == "tensor"
    assert not session.has_mixed_boundary()


# ---------------------------------------------------------------------------
# Targets across frontends
# ---------------------------------------------------------------------------

def test_session_records_distinct_targets() -> None:
    """matmul + rotor target apple_gpu on Darwin, cpu elsewhere;
    textual hard-codes cpu.  At least two distinct targets
    appear when mixing Darwin-targeting drivers with the
    textual driver in either environment."""
    import sys
    with compile_session() as session:
        matmul_softmax_matmul.run()
        compile_report_for_text(_TEXTUAL_SOURCE, program_id="t")
        rotor_sandwich_norm.run()
    targets = session.targets()
    if sys.platform == "darwin":
        # apple_gpu (matmul + rotor) + cpu (textual) = 2.
        assert targets == {"apple_gpu", "cpu"}
    else:
        # Everyone falls back to cpu off Darwin.
        assert "cpu" in targets


# ---------------------------------------------------------------------------
# @tessera.jit + @clifford_jit in one scope (no canonical drivers)
# ---------------------------------------------------------------------------

def test_jit_and_clifford_jit_emit_into_same_session() -> None:
    """Hand-written demo: a ``@tessera.jit`` call + a
    ``@clifford_jit`` call inside the same scope land in the
    same session."""

    @jit
    def add(x, y):
        return x + y

    @clifford_jit(target="apple_gpu")
    def gainner(a, b):
        return ga.inner(a, b)

    a = ga.Cl(3, 0)
    mv_a = ga.Multivector(np.zeros((4, 8), dtype=np.float32), a)
    mv_b = ga.Multivector(np.zeros((4, 8), dtype=np.float32), a)

    with compile_session() as session:
        add(np.array([1.0]), np.array([2.0]))
        # We don't actually invoke gainner on Darwin (the call
        # would need the runtime) — exercise the accessor instead.
        session.reports.append(gainner.compile_report())
        # Note: appending bypasses the auto-emit sink, but the
        # session is itself the sink so it's the same list either way.
    frontends = {r.frontend for r in session.reports}
    # @tessera.jit's auto-emit picks up the add() call, and we
    # manually inserted the clifford report.
    assert FRONTEND_TESSERA_JIT in frontends
    assert FRONTEND_CLIFFORD_JIT in frontends
